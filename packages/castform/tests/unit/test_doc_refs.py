"""Keep the shipped docs and agent skills honest about the real code surface.

Every check here resolves what the prose claims against the live tree — imports
the symbols it names, parses the snippets it shows, and reads the seed
template's own argparse/config surface — so a rename or a moved boundary breaks
the docs here rather than in a user's project. The last test is the inverse
guard: names from the pre-two-package layout must not reappear in the docs.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import castform.cli.scaffold as scaffold_pkg
import pytest
from castform.cli import build_parser, setup

from ._scaffold import discover_env_class, load_module

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_REPO_ROOT = Path(__file__).resolve().parents[4]

_SKILL_NAMES = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

# The doc surface this slice owns: the published package guides plus everything
# `castform setup` copies into a scaffolded project.
DOC_FILES = {
    "readme": _REPO_ROOT / "README.md",
    "benchmax-readme": _REPO_ROOT / "packages" / "benchmax" / "README.md",
    "castform-readme": _REPO_ROOT / "packages" / "castform" / "README.md",
    "scaffold-claude": _SCAFFOLD_DIR / "CLAUDE.md",
    "scaffold-starter": _SCAFFOLD_DIR / "STARTER.md",
    **{f"skill-{name}": _SCAFFOLD_DIR / "skills" / name / "SKILL.md" for name in _SKILL_NAMES},
}

# (doc key, text the doc must contain, ``module:attr.path`` it must resolve to).
# The literal half keeps a doc from quietly dropping a reference; the dotted
# half keeps the reference pointing at something that still exists.
SYMBOL_REFS = [
    ("benchmax-readme", "benchmax.sft", "benchmax:sft"),
    ("benchmax-readme", "SftSerializationError", "benchmax.sft:SftSerializationError"),
    ("benchmax-readme", "benchmax.envs.base.content", "benchmax.envs.base:content"),
    ("benchmax-readme", "message_text", "benchmax.envs.base.content:message_text"),
    ("benchmax-readme", "content_preview", "benchmax.envs.base.content:content_preview"),
    ("benchmax-readme", "iter_image_refs", "benchmax.envs.base.content:iter_image_refs"),
    (
        "benchmax-readme",
        "image_to_data_uri",
        "benchmax.envs.base.content:image_to_data_uri",
    ),
    ("castform-readme", "upload_sft_run", "castform.platform:upload_sft_run"),
    (
        "castform-readme",
        "launch_sft_run",
        "castform.platform.client:TrainerClient.launch_sft_run",
    ),
    (
        "castform-readme",
        "castform.platform.client.SFT_LAUNCH_SUPPORTED",
        "castform.platform.client:SFT_LAUNCH_SUPPORTED",
    ),
    (
        "scaffold-claude",
        "castform.platform.client.SFT_LAUNCH_SUPPORTED",
        "castform.platform.client:SFT_LAUNCH_SUPPORTED",
    ),
    ("scaffold-claude", "upload_sft_run", "castform.platform:upload_sft_run"),
    (
        "scaffold-claude",
        "TrainerClient.launch_sft_run",
        "castform.platform.client:TrainerClient.launch_sft_run",
    ),
    ("scaffold-claude", "dump_bundle", "benchmax.bundle:dump_bundle"),
    ("skill-design-environment", "benchmax.sft", "benchmax:sft"),
    (
        "skill-design-environment",
        "benchmax.envs.base.content",
        "benchmax.envs.base:content",
    ),
    (
        "skill-design-environment",
        "message_text",
        "benchmax.envs.base.content:message_text",
    ),
    (
        "skill-design-environment",
        "content_preview",
        "benchmax.envs.base.content:content_preview",
    ),
    (
        "skill-design-environment",
        "iter_image_refs",
        "benchmax.envs.base.content:iter_image_refs",
    ),
    (
        "skill-design-environment",
        "image_to_data_uri",
        "benchmax.envs.base.content:image_to_data_uri",
    ),
    ("skill-generate-data", "load_sft_dataset", "benchmax.sft:load_sft_dataset"),
    ("skill-generate-data", "validate_sft_dataset", "benchmax.sft:validate_sft_dataset"),
    ("skill-generate-data", "canonical_jsonl", "benchmax.sft:canonical_jsonl"),
    (
        "skill-generate-data",
        "castform.traces.TracesPipeline",
        "castform.traces:TracesPipeline",
    ),
    (
        "skill-verify-environment",
        "benchmax.sft.load_sft_dataset",
        "benchmax.sft:load_sft_dataset",
    ),
    (
        "skill-verify-environment",
        "validate_sft_dataset",
        "benchmax.sft:validate_sft_dataset",
    ),
    ("skill-launch-run", "upload_sft_run", "castform.platform:upload_sft_run"),
    (
        "skill-launch-run",
        "TrainerClient.launch_sft_run",
        "castform.platform.client:TrainerClient.launch_sft_run",
    ),
    (
        "skill-launch-run",
        "castform.platform.client.SFT_LAUNCH_SUPPORTED",
        "castform.platform.client:SFT_LAUNCH_SUPPORTED",
    ),
]

# Names from the pre-two-package layout (and from the CLI verbs the restructure
# deleted). Each maps to what replaced it, so a failure says where to go.
STALE_REFERENCES = {
    r"castform validate": "the scaffold's `python main.py validate` stage",
    r"castform launch": "the scaffold's `python main.py launch` stage",
    r"castform data ": "the `castform.traces` / `castform.rag` libraries",
    r"cli/launch\.py": "the per-template scaffold `main.py`",
    r"--set model=": "a `LAUNCH_CONFIG` key",
    r"(?<!-)--model\b": "a `LAUNCH_CONFIG` key",
    r"--allow-experimental-weights": "`LAUNCH_CONFIG['allow_experimental_weights']`",
    r"PLATFORM_API_KEY": "CASTFORM_API_KEY",
    r"benchmax\.platform": "castform.platform",
    r"benchmax\.cli": "castform.cli",
    r"train_dataset\.jsonl": "train.jsonl",
    r"eval_dataset\.jsonl": "eval.jsonl",
}

_FENCE_RE = re.compile(r"^```(\w+)\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _read(key: str) -> str:
    path = DOC_FILES[key]
    assert path.is_file(), f"documented file is missing: {path}"
    return path.read_text(encoding="utf-8")


def _fences(text: str, lang: str) -> list[str]:
    return [body for fence_lang, body in _FENCE_RE.findall(text) if fence_lang == lang]


def _section(text: str, heading: str) -> str:
    """The body of a ``## heading`` section, up to the next same-level heading."""
    start = text.index(f"## {heading}")
    rest = text[start:]
    following = re.search(r"^## ", rest[3:], re.MULTILINE)
    return rest if following is None else rest[: following.start() + 3]


def _resolve(dotted: str):
    """Resolve a ``module:attr.path`` reference, or fail with what broke."""
    module_name, _, attr_path = dotted.partition(":")
    try:
        obj = importlib.import_module(module_name)
    except ImportError as error:  # pragma: no cover - only on a real regression
        raise AssertionError(f"documented module {module_name!r} is gone: {error}")
    for attr in filter(None, attr_path.split(".")):
        assert hasattr(obj, attr), f"documented name {dotted!r} lost its {attr!r}"
        obj = getattr(obj, attr)
    return obj


@pytest.fixture(scope="module")
def sft_seed():
    return load_module(_SCAFFOLD_DIR / "sft_main.py")


@pytest.mark.parametrize(
    ("doc", "mention", "dotted"),
    SYMBOL_REFS,
    ids=[f"{doc}:{dotted}" for doc, _mention, dotted in SYMBOL_REFS],
)
def test_documented_symbols_are_named_and_resolvable(doc, mention, dotted):
    assert mention in _read(doc), f"{DOC_FILES[doc]} no longer mentions {mention!r}"
    _resolve(dotted)


@pytest.mark.parametrize("doc", sorted(DOC_FILES))
def test_documented_python_snippets_parse_and_their_imports_resolve(doc):
    snippets = _fences(_read(doc), "python")
    for snippet in snippets:
        tree = ast.parse(snippet)  # a doc snippet that cannot parse is a doc bug
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for alias in node.names:
                _resolve(f"{node.module}:{alias.name}")


def test_documented_sft_row_format_loads_and_validates(tmp_path):
    """The `messages` row the generate-data skill shows must survive the real
    loader and validator — not merely look like the schema."""
    from benchmax.sft import canonical_jsonl, load_sft_dataset, validate_sft_dataset

    section = _section(_read("skill-generate-data"), "SFT — the `messages` row format")
    rows = _fences(section, "jsonl")
    assert rows, "the SFT section no longer shows a `messages` row"

    path = tmp_path / "train.jsonl"
    path.write_text("".join(rows), encoding="utf-8")
    dataset = load_sft_dataset(path)
    report = validate_sft_dataset(dataset)
    assert report.ok, [issue.message for issue in report.issues]
    assert canonical_jsonl(dataset)


def test_documented_sft_snippets_run(tmp_path, monkeypatch, sft_seed):
    """Run the generate-data SFT snippets rather than only resolving their
    imports, so a stale call — a renamed function, a dropped argument — fails
    here. Only this section's snippets are self-contained enough to execute;
    the launch-side ones would need a live platform."""
    import base64

    monkeypatch.chdir(tmp_path)
    section = _section(_read("skill-generate-data"), "SFT — the `messages` row format")
    Path("train.jsonl").write_text("".join(_fences(section, "jsonl")), encoding="utf-8")
    _, _, encoded = sft_seed._TINY_PNG_DATA_URI.partition("base64,")
    Path("figure.png").write_bytes(base64.b64decode(encoded))

    namespace: dict = {}
    for snippet in _fences(section, "python"):
        exec(compile(snippet, "<generate-data SKILL.md>", "exec"), namespace)
    assert namespace["report"].ok
    assert namespace["image_url"].startswith("data:image/png;base64,")


def test_documented_multimodal_row_shape_survives_canonicalization(tmp_path):
    """design-environment claims a content-part list is preserved byte-for-byte
    through canonicalization. Check that against the seed's own multimodal row."""
    import json

    from benchmax.sft import canonical_jsonl, load_sft_dataset

    seed = load_module(_SCAFFOLD_DIR / "sft_main.py")
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(seed._SEED_MULTIMODAL) + "\n", encoding="utf-8")

    round_tripped = json.loads(canonical_jsonl(load_sft_dataset(path)).splitlines()[0])
    assert round_tripped["messages"] == seed._SEED_MULTIMODAL["messages"]


def test_documented_seed_stages_and_flags_exist(sft_seed, capsys):
    """Every stage and flag the docs spell out is on the seed's real parser."""
    with pytest.raises(SystemExit) as exit_info:
        sft_seed.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out

    documented = _read("scaffold-claude") + _read("skill-verify-environment")
    documented += _read("skill-launch-run") + _read("scaffold-starter")
    for token in ("data", "validate", "launch", "--force", "-y", "--yes"):
        assert token in help_text, f"the sft seed no longer accepts {token!r}"
        assert token in documented, f"the docs no longer mention {token!r}"


def test_documented_dataset_filenames_match_the_seed(sft_seed):
    assert (sft_seed.TRAIN_FILE, sft_seed.EVAL_FILE) == ("train.jsonl", "eval.jsonl")
    for name in (sft_seed.TRAIN_FILE, sft_seed.EVAL_FILE):
        assert name in _read("skill-verify-environment")


def test_documented_validate_stage_exits_zero_on_the_seed_dataset(sft_seed, tmp_path, monkeypatch):
    """verify-environment promises `validate` exits 0 on a valid dataset."""
    monkeypatch.chdir(tmp_path)
    sft_seed.generate_data()
    assert sft_seed.main(["validate"]) == 0


def test_documented_config_keys_match_the_seed(sft_seed):
    """launch-run and verify-environment name specific config keys and split
    them into wire args vs locally-resolved ones. Both halves are checkable."""
    from benchmax.sft import sft_validate_kwargs

    assert sft_seed.LAUNCH_CONFIG["training_mode"] == "sft"
    assert sft_seed._LAUNCH_CONFIG_RESERVED == frozenset(
        {"type", "name", "allow_experimental_weights"}
    )
    launch_doc = _read("skill-launch-run")
    for key in sorted(sft_seed._LAUNCH_CONFIG_RESERVED) + ["training_mode", "model"]:
        assert key in launch_doc, f"launch-run no longer documents {key!r}"

    knobs = {"max_seq_len": 4096, "max_row_bytes": 1 << 20}
    assert sft_validate_kwargs(knobs) == knobs
    verify_doc = _read("skill-verify-environment")
    assert all(knob in verify_doc for knob in knobs)


def test_documented_rl_versus_sft_marker_is_real(sft_seed):
    """CLAUDE.md routes on `LAUNCH_CONFIG["training_mode"]` plus the presence of
    a `BaseEnv` subclass. Both templates must actually differ that way."""
    from benchmax.envs import Environment

    assert 'LAUNCH_CONFIG["training_mode"] == "sft"' in _read("scaffold-claude")
    assert not [
        value
        for value in vars(sft_seed).values()
        if isinstance(value, type) and issubclass(value, Environment)
    ]
    for template in ("generic_main.py", "rag_main.py"):
        rl_seed = load_module(_SCAFFOLD_DIR / template)
        assert "training_mode" not in rl_seed.LAUNCH_CONFIG
        assert discover_env_class(rl_seed) is not None


def test_documented_setup_template_choices_are_registered():
    parser = build_parser()
    (subparsers,) = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    ]
    template = subparsers.choices["setup"]._option_string_actions["--template"]
    assert "sft" in template.choices
    assert "castform setup --template sft" in _read("scaffold-claude")


def test_documented_skill_files_exist():
    """The scaffold docs point the agent at per-stage skills by path; every
    referenced skill must be packaged and registered with setup."""
    referenced = set()
    for key in ("scaffold-claude", "scaffold-starter"):
        referenced |= set(re.findall(r"\.claude/skills/([\w-]+)/SKILL\.md", _read(key)))
    assert referenced, "the scaffold docs no longer point at any skill"
    for name in referenced:
        assert name in setup._SKILLS
        assert (_SCAFFOLD_DIR / "skills" / name / "SKILL.md").is_file()


@pytest.mark.parametrize("doc", sorted(DOC_FILES))
def test_docs_carry_no_stale_references(doc):
    text = _read(doc)
    hits = [
        f"{pattern!r} (use {replacement})"
        for pattern, replacement in STALE_REFERENCES.items()
        if re.search(pattern, text)
    ]
    assert not hits, f"{DOC_FILES[doc]} carries stale references: {hits}"
