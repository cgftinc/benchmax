"""Slice 7 gate: the SFT/multimodal symbols and CLI flags named in the edited
scaffold skill docs (design-environment, generate-data, verify-environment,
launch-run, scaffold CLAUDE.md) actually exist, and every scaffold template
still renders cleanly.

Pre-existing harbor-proper env API references in those docs (`BaseEnv`,
`JsonlDataset`, ...) are accepted skew per the plan's Risks section and are not
gated here — only symbols/flags newly named by this slice.
"""

from __future__ import annotations

import argparse

import pytest

from benchmax.cli import build_parser, setup


def test_content_helpers_named_in_docs_are_importable():
    from benchmax.envs.base.content import (
        content_preview,
        image_to_data_uri,
        iter_image_refs,
        message_text,
    )

    assert callable(message_text)
    assert callable(content_preview)
    assert callable(iter_image_refs)
    assert callable(image_to_data_uri)


def test_sft_dataset_helpers_named_in_docs_are_importable():
    from benchmax.sft import load_sft_dataset, validate_sft_dataset

    assert callable(load_sft_dataset)
    assert callable(validate_sft_dataset)


def test_sft_schema_module_named_in_docs_is_importable():
    from benchmax.sft import schema

    assert callable(schema.validate_row)


def test_sft_launch_supported_flag_named_in_docs_is_importable():
    from benchmax.platform.client import SFT_LAUNCH_SUPPORTED

    assert SFT_LAUNCH_SUPPORTED is False


def test_training_mode_marker_named_in_docs_matches_the_real_mode_set():
    from benchmax.cli._project import TRAINING_MODES

    assert TRAINING_MODES == frozenset({"rl", "sft"})


def _subparser(name: str) -> argparse.ArgumentParser:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            return action.choices[name]
    raise AssertionError(f"could not locate the {name!r} subcommand in the parser")


def test_launch_allow_experimental_weights_flag_named_in_docs_is_registered():
    launch_parser = _subparser("launch")
    assert "--allow-experimental-weights" in launch_parser._option_string_actions


def test_setup_template_sft_choice_named_in_docs_is_registered():
    setup_parser = _subparser("setup")
    action = setup_parser._option_string_actions["--template"]
    assert "sft" in action.choices


def _ns(tmp, **kw):
    base = dict(
        dir=str(tmp),
        agent="both",
        force=False,
        skip_login=True,
        no_template=False,
        template="generic",
        verbose=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("template", ["generic", "rag", "sft"])
def test_every_template_renders_via_castform_setup(tmp_path, template):
    target = tmp_path / template
    assert setup._cmd_setup(_ns(target, template=template)) == 0
    assert (target / "main.py").exists()
    assert (target / "train_dataset.jsonl").exists()
    assert (target / "eval_dataset.jsonl").exists()
    assert (target / "CLAUDE.md").exists()
    for skill in (
        "design-environment",
        "generate-data",
        "verify-environment",
        "launch-run",
    ):
        assert (target / ".claude" / "skills" / skill / "SKILL.md").exists()
