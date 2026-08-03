from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

from benchmax.bundle import dump_bundle, load_bundle

_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("pinecone_rag_example_main", _ROOT / "main.py")
assert _SPEC is not None and _SPEC.loader is not None
_LOCAL_MODULE_NAMES = ("data", "pinecone_rag_env", "search")
_PREVIOUS = {name: sys.modules.pop(name, None) for name in _LOCAL_MODULE_NAMES}
sys.path.insert(0, str(_ROOT))
try:
    main = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = main
    _SPEC.loader.exec_module(main)
    _RUNTIME_MODULES = {
        name: sys.modules[name] for name in _LOCAL_MODULE_NAMES if name in sys.modules
    }
finally:
    sys.path.remove(str(_ROOT))
    for _name, _module in _PREVIOUS.items():
        if _module is not None:
            sys.modules[_name] = _module
        else:
            sys.modules.pop(_name, None)


def test_runtime_bundle_excludes_castform_data_pipeline() -> None:
    secret = "pinecone_test_secret"
    with (
        patch.dict(sys.modules, _RUNTIME_MODULES),
        patch.object(sys, "path", [str(_ROOT), *sys.path]),
    ):
        bundle = dump_bundle(
            main.PineconeRagEnv,
            constructor_args={
                "judge_base_url": "https://models.example/v1",
                "embedding_base_url": "https://models.example/v1",
                "index_host": "https://index.example",
                "api_key": secret,
            },
            pip_dependencies=main.RUNTIME_DEPENDENCIES,
        )
    env = load_bundle(bundle)

    assert env._search.available_modes == ["vector"]
    assert secret.encode() in bundle.pickled
    assert b"castform.rag" not in bundle.pickled
    assert b"PineconeChunkSource" not in bundle.pickled
    assert bundle.metadata.pip_dependencies == ("pinecone<10,>=9.1.0",)
