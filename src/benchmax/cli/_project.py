"""Load a benchmax project (env class + datasets) from a directory.

Convention mirrors the web-app scaffold (``buildAgentContextBody``): ``main.py``
defines a single :class:`BaseEnv` subclass; ``train_dataset.jsonl`` /
``eval_dataset.jsonl`` hold one JSON object per line. ``validate`` and ``launch``
share this loader. An importable module path (``--module``) is an alternative to
``main.py`` for shipped envs / fixtures.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal


class ProjectError(Exception):
    """A project couldn't be loaded (missing main.py/dataset, or no/ambiguous env)."""


# Allowed values for the TRAINING_MODE module attribute (see _read_training_mode).
TRAINING_MODES = frozenset({"rl", "sft"})


@dataclass
class LoadedProject:
    training_mode: Literal["rl", "sft"]
    env_class: type | None  # None iff training_mode == "sft"
    train_dataset: list[dict[str, Any]]
    eval_dataset: list[dict[str, Any]]
    module: ModuleType
    from_file: (
        bool  # loaded from a main.py path (pickle env by value) vs an importable module
    )
    # Optional module-level config dicts baked into main.py so a run is
    # reproducible from the file (validate/launch read these; CLI flags override).
    launch_config: dict[str, Any]
    validate_config: dict[str, Any]
    # sft mode only; paths are never parsed here (see sft.dataset.load_sft_dataset)
    sft_train_path: Path | None = None
    sft_eval_path: Path | None = None


def row_question_and_gold(row: Any) -> tuple[object, object]:
    """``(question, gold)`` from a dataset row, across the on-disk shapes.

    Question: ``prompt`` (a chat-list prompt → its last user turn) else ``question``.
    Gold: ``ground_truth`` else ``answer``. One definition so ``runs rollout`` and
    ``validate --reward-audit`` can't drift when a dataset field is renamed (see the
    ``castform-dataset-ondisk-shapes`` note)."""
    if not isinstance(row, dict):
        return None, None
    q = row.get("prompt")
    if not q:
        q = row.get("question")
    if isinstance(q, list):  # chat-style prompt → last user turn
        q = next(
            (
                m.get("content")
                for m in reversed(q)
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
            ),
            None,
        )
    gold = row.get("ground_truth")
    if gold is None:
        gold = row.get("answer")
    return q, gold


def _read_config(module: ModuleType, name: str) -> dict[str, Any]:
    """A module-level config dict (``LAUNCH_CONFIG`` / ``VALIDATE_CONFIG``) from
    main.py — the knobs the file bakes in so the run reproduces without remembering
    CLI flags. Absent → ``{}`` (the block is optional); present-but-not-a-dict is a
    user error we fail loudly on rather than silently drop (a dropped budget wastes
    GPU with no explanation)."""
    value = getattr(module, name, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProjectError(
            f"{name} must be a dict (got {type(value).__name__}); it bakes the "
            "validate/launch knobs into main.py. Fix or remove it."
        )
    return dict(value)


def _load_module_from_file(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ProjectError(f"Could not load a module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module  # so dataclass/pickle name resolution works
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # surface the user's import/syntax error cleanly
        raise ProjectError(f"Failed to import {path.name}: {exc}") from exc
    return module


def _env_classes_defined_in(module: ModuleType) -> list[type]:
    """BaseEnv subclasses *defined in* ``module`` (imported ones are ignored)."""
    from benchmax.envs.base_env import BaseEnv

    def _is_env(obj: Any) -> bool:
        return inspect.isclass(obj) and issubclass(obj, BaseEnv) and obj is not BaseEnv

    return [
        obj
        for obj in vars(module).values()
        if _is_env(obj) and obj.__module__ == module.__name__
    ]


def _has_env_class_anywhere(module: ModuleType) -> bool:
    """True if any BaseEnv subclass is reachable in module's namespace, defined
    here or imported. Used only for the sft-mode ambiguity guard — unlike RL
    auto-discovery, an imported env class still counts as "an env is present"."""
    from benchmax.envs.base_env import BaseEnv

    return any(
        inspect.isclass(obj) and issubclass(obj, BaseEnv) and obj is not BaseEnv
        for obj in vars(module).values()
    )


def discover_env_class(module: ModuleType, explicit: str | None = None) -> type:
    """Find the env class in ``module``. With no ``explicit`` name, require exactly
    one BaseEnv subclass *defined in* the module (imported ones are ignored)."""
    from benchmax.envs.base_env import BaseEnv

    def _is_env(obj: Any) -> bool:
        return inspect.isclass(obj) and issubclass(obj, BaseEnv) and obj is not BaseEnv

    if explicit:
        for obj in vars(module).values():
            if _is_env(obj) and obj.__name__ == explicit:
                return obj
        raise ProjectError(f"No BaseEnv subclass named {explicit!r} in the module.")

    defined_here = _env_classes_defined_in(module)
    if not defined_here:
        raise ProjectError("No BaseEnv subclass defined in the module.")
    if len(defined_here) > 1:
        names = sorted(c.__name__ for c in defined_here)
        raise ProjectError(
            f"Multiple env classes {names}; pass --env-class to pick one."
        )
    return defined_here[0]


_MISSING = object()  # distinguishes "no TRAINING_MODE attribute" from "set to None"


def _read_training_mode(module: ModuleType) -> Literal["rl", "sft"]:
    """The explicit ``TRAINING_MODE`` marker — a scalar module attribute, read on
    its own (not via ``_read_config``'s dict reads). Absent -> "rl" (today's only
    mode, env required). Any value outside {"rl", "sft"} fails loudly rather than
    being silently coerced — including ``None`` and unhashable values, which must
    raise ``ProjectError`` rather than resolving to "rl" or crashing on ``in``."""
    value = getattr(module, "TRAINING_MODE", _MISSING)
    if value is _MISSING:
        return "rl"
    if isinstance(value, str) and value in TRAINING_MODES:
        return value
    raise ProjectError(
        f"invalid TRAINING_MODE {value!r} in main.py; must be one of "
        f"{sorted(TRAINING_MODES)}"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ProjectError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{path}:{n}: invalid JSON ({exc})") from exc
    if not rows:
        raise ProjectError(f"Dataset is empty: {path}")
    return rows


def load_project(
    *,
    directory: str = ".",
    run_file: str = "main.py",
    module_path: str | None = None,
    env_class_name: str | None = None,
    train_file: str = "train_dataset.jsonl",
    eval_file: str = "eval_dataset.jsonl",
    require_eval: bool = False,
) -> LoadedProject:
    """Load the env class + datasets for a project dir (or an importable module)."""
    from_file = module_path is None
    if module_path:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # missing dep, bad path, import-time error
            raise ProjectError(
                f"Could not import module {module_path!r}: {exc}"
            ) from exc
    else:
        path = Path(directory) / run_file
        if not path.exists():
            raise ProjectError(
                f"{run_file} not found in {directory!r} — run inside a project dir, "
                "or pass --module for an importable env."
            )
        module = _load_module_from_file(path)

    training_mode = _read_training_mode(module)
    base = Path(directory)

    if training_mode == "sft":
        if env_class_name is not None:
            raise ProjectError(
                "--env-class can't be combined with an sft-mode project "
                "(TRAINING_MODE = 'sft' has no env to select)."
            )
        if _has_env_class_anywhere(module):
            raise ProjectError(
                "ambiguous project: TRAINING_MODE = 'sft' but a BaseEnv subclass "
                "is also present in main.py — remove one of the two signals."
            )
        train_path = base / train_file
        if not train_path.exists():
            raise ProjectError(f"dataset not found: {train_path}")
        eval_path = base / eval_file
        sft_eval_path = eval_path if eval_path.exists() else None
        if require_eval and sft_eval_path is None:
            raise ProjectError(f"eval dataset required but not found: {eval_path}")
        return LoadedProject(
            training_mode="sft",
            env_class=None,
            train_dataset=[],
            eval_dataset=[],
            module=module,
            from_file=from_file,
            launch_config=_read_config(module, "LAUNCH_CONFIG"),
            validate_config=_read_config(module, "VALIDATE_CONFIG"),
            sft_train_path=train_path,
            sft_eval_path=sft_eval_path,
        )

    env_class = discover_env_class(module, env_class_name)
    train_dataset = _load_jsonl(base / train_file)
    eval_path = base / eval_file
    eval_dataset = _load_jsonl(eval_path) if eval_path.exists() else []
    if require_eval and not eval_dataset:
        raise ProjectError(f"Eval dataset required but not found: {eval_path}")
    return LoadedProject(
        training_mode="rl",
        env_class=env_class,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        module=module,
        from_file=from_file,
        launch_config=_read_config(module, "LAUNCH_CONFIG"),
        validate_config=_read_config(module, "VALIDATE_CONFIG"),
    )
