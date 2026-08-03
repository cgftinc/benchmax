from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

from benchmax.bundle import dump_bundle, load_bundle

_MAIN_PATH = Path(__file__).parents[1] / "main.py"
_SPEC = importlib.util.spec_from_file_location("turbopuffer_rag_example_main", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOCAL_MODULE_NAMES = ("data", "search", "turbopuffer_rag_env")
_PREVIOUS = {name: sys.modules.pop(name, None) for name in _LOCAL_MODULE_NAMES}
sys.path.insert(0, str(_MAIN_PATH.parent))
try:
    main = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = main
    _SPEC.loader.exec_module(main)
    _RUNTIME_MODULES = {
        name: sys.modules[name] for name in _LOCAL_MODULE_NAMES if name in sys.modules
    }
finally:
    sys.path.remove(str(_MAIN_PATH.parent))
    for _name, _module in _PREVIOUS.items():
        if _module is not None:
            sys.modules[_name] = _module
        else:
            sys.modules.pop(_name, None)


def test_runtime_bundle_excludes_castform_and_data_pipeline() -> None:
    secret = "tpuf_test_secret"
    with (
        patch.dict(sys.modules, _RUNTIME_MODULES),
        patch.object(sys, "path", [str(_MAIN_PATH.parent), *sys.path]),
    ):
        bundle = dump_bundle(
            main.TurbopufferRagEnv,
            constructor_args={
                "judge_base_url": "https://models.example/v1",
                "embedding_base_url": "https://models.example/v1",
                "api_key": secret,
            },
            pip_dependencies=main.RUNTIME_DEPENDENCIES,
        )
    env = load_bundle(bundle)

    assert env._search.available_modes == ["hybrid", "lexical", "vector"]
    assert secret.encode() in bundle.pickled
    assert b"castform.rag" not in bundle.pickled
    assert b"TpufChunkSource" not in bundle.pickled
    assert bundle.metadata.pip_dependencies == ("turbopuffer<3,>=2.6.0",)


def test_runtime_dependency_list_does_not_include_castform() -> None:
    assert all("castform" not in dependency for dependency in main.RUNTIME_DEPENDENCIES)
    assert "TPUF_API_KEY" not in os.environ or isinstance(os.environ["TPUF_API_KEY"], str)
