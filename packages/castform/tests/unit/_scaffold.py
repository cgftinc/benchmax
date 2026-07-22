"""Test-only helpers for inspecting the packaged scaffold scripts."""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import sys
from pathlib import Path
from types import ModuleType

_MODULE_IDS = itertools.count()


def load_module(path: Path) -> ModuleType:
    name = f"_castform_scaffold_{path.stem}_{next(_MODULE_IDS)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def discover_env_class(module: ModuleType) -> type:
    from benchmax.envs import Environment

    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, Environment)
        and value is not Environment
        and value.__module__ == module.__name__
    ]
    assert len(classes) == 1
    return classes[0]
