"""The neon RAG bundle must carry its own neon provider source.

`castform`'s published wheels contain no `rag/corpus/neon/` package, so an eval
image that installs `castform` from PyPI cannot resolve a by-reference
`NeonSearch`. `rag_env.build_bundle` therefore captures the provider by value.
These tests pin both halves of that: the artifact leaves no castform module
behind by reference, and it loads and reaches the database driver in an
interpreter that cannot import castform at all.

No live Neon, DSN, or network: the baked DSN points at a closed local port, so a
resolved import path surfaces as a psycopg connection error rather than an
`ImportError`.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import benchmax  # noqa: E402
import castform  # noqa: E402
from benchmax.bundle import _referenced_modules, unregistered_local_refs  # noqa: E402

from rag_env import NEON_BUNDLE_LOCAL_MODULES, build_bundle  # noqa: E402

pytestmark = pytest.mark.unit

# Port 1 is reserved and never listening: psycopg fails fast without a network
# round trip, which is exactly the signal that the import path resolved.
DEAD_DSN = "postgresql://ro:pw@127.0.0.1:1/db?sslmode=disable&connect_timeout=1"

# Modules the bundle captures by value. Every castform import on their query
# path must sit at module scope; cloudpickle does not restore by-value modules
# into `sys.modules`, so an in-method import would still hit the sandbox's
# installed castform and fail there.
_CAPTURED_SOURCES = (
    "rag/corpus/neon/search.py",
    "rag/corpus/neon/query.py",
    "rag/corpus/neon/client.py",
    "rag/corpus/neon/schema.py",
    "rag/corpus/neon/credentials.py",
)

# `filter_mapper` imports psycopg at module scope, which `search.py` must stay
# importable without. It is reachable only from a request carrying a metadata
# filter, which `NeonSearch.search` never builds. See `query._resolve_filter`.
_ALLOWED_DELAYED_IMPORTS = {"castform.rag.corpus.neon.filter_mapper"}

# Child interpreter: proves the artifact alone is enough. Runs with `-S` and an
# explicit PYTHONPATH so no .pth file can put castform's source back on the path.
_CHILD = """
import importlib.util, json, sys
from pathlib import Path

result = {"castform_importable": importlib.util.find_spec("castform") is not None}
if not result["castform_importable"]:
    from benchmax.bundle import Bundle, BundleMetadata, load_bundle

    out = Path(sys.argv[1])
    env = load_bundle(
        Bundle(
            pickled=(out / "env-cls.pkl").read_bytes(),
            metadata=BundleMetadata.from_json_bytes(
                (out / "env-metadata.json").read_bytes()
            ),
        )
    )
    result["env_class"] = type(env).__name__
    result["search_params"] = env._search.get_params()
    result["client_class"] = type(env._search._get_client()).__name__
    try:
        env._search.search("gitlab handbook", mode="lexical", top_k=2)
        result["search_error"] = None
    except Exception as exc:
        result["search_error"] = f"{type(exc).__module__}.{type(exc).__name__}"
print(json.dumps(result))
"""


def _castform_module_refs(pickled: bytes) -> list[str]:
    return sorted(
        m for m in _referenced_modules(pickled) if m.split(".")[0] == "castform"
    )


def test_declared_entry_modules_are_covered_by_the_import_scan() -> None:
    guarded = {
        "castform." + path.removesuffix(".py").replace("/", ".")
        for path in _CAPTURED_SOURCES
    }
    assert set(NEON_BUNDLE_LOCAL_MODULES) <= guarded


def test_bundle_leaves_no_castform_module_by_reference() -> None:
    bundle = build_bundle(DEAD_DSN)

    assert unregistered_local_refs(bundle.pickled) == []
    assert _castform_module_refs(bundle.pickled) == []
    # The provider really is in the artifact, not merely absent from the refs.
    assert len(bundle.pickled) > 50_000
    assert "castform" not in " ".join(bundle.metadata.pip_dependencies)


def test_bundle_loads_and_reaches_the_driver_without_castform(tmp_path: Path) -> None:
    bundle = build_bundle(DEAD_DSN)
    (tmp_path / "env-cls.pkl").write_bytes(bundle.pickled)
    (tmp_path / "env-metadata.json").write_bytes(bundle.metadata.to_json_bytes())

    child = tmp_path / "load_bundle_child.py"
    child.write_text(_CHILD)
    benchmax_root = str(Path(benchmax.__path__[0]).parent)
    completed = subprocess.run(
        [sys.executable, "-S", str(child), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{sysconfig.get_paths()['purelib']}:{benchmax_root}",
        },
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    if result["castform_importable"]:
        pytest.skip(
            "castform is installed outside a .pth-based editable layout, so the "
            "child interpreter cannot be isolated from its source"
        )
    assert result["env_class"] == "SearchEnv"
    assert result["search_params"]["backend"] == "neon"
    assert result["client_class"] == "NeonClient"
    # Reaching psycopg means every captured castform import resolved.
    assert result["search_error"] == "psycopg.OperationalError"


@pytest.mark.parametrize("relative_source", _CAPTURED_SOURCES)
def test_captured_source_imports_castform_only_at_module_scope(
    relative_source: str,
) -> None:
    source_path = Path(castform.__file__).parent / relative_source
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    module_scope = {id(node) for node in tree.body}

    delayed: list[str] = []
    for parent in ast.walk(tree):
        if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(parent):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            if id(node) in module_scope or not name.startswith("castform"):
                continue
            delayed.append(f"{name} (line {node.lineno})")

    unexpected = [d for d in delayed if d.split(" ")[0] not in _ALLOWED_DELAYED_IMPORTS]
    assert unexpected == [], (
        f"{relative_source} imports castform inside a function; a by-value bundle "
        f"cannot satisfy it at runtime: {unexpected}"
    )
