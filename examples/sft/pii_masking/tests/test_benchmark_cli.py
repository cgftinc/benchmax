"""Tests for the benchmark CLI's approval interlocks and offline paths.

Every test here runs with no network and no credentials. That is the property
under test as much as the assertions are: the offline commands, the help text,
and every refusal path must not import a source or model client.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pii_masking import benchmark
from pii_masking.benchmark_protocol import (
    Protocol,
    build_protocol_payload,
    canonical_bytes,
)

# examples/sft — the directory that makes `pii_masking` importable.
EXAMPLES_ROOT = Path(__file__).resolve().parents[2]


def write_protocol(tmp_path):
    payload = build_protocol_payload(
        benchmark_source="openpii-validation",
        execution_environment="production",
        base_model={"model_id": "Qwen/Qwen3.5-4B"},
        training={"lora_rank": 64},
        generation={"temperature": 0},
        prompt={"system": "mask it"},
        selection={},
        scoring={},
        expected_counts={},
    )
    protocol = Protocol.from_payload(payload)
    directory = tmp_path / protocol.protocol_id
    protocol.write(directory)
    return protocol, directory


# ── offline surface ───────────────────────────────────────────────────────────
class TestOfflineSurface:
    def test_help_needs_no_optional_dependencies(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["--help"])

        assert exit_info.value.code == 0
        assert "prepare" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "command", ["prepare", "launch", "preflight-adapter", "evaluate", "score"]
    )
    def test_every_command_is_registered(self, command, capsys):
        with pytest.raises(SystemExit):
            benchmark.main([command, "--help"])

        assert command in capsys.readouterr().out

    def test_importing_the_cli_pulls_in_no_source_or_model_client(self):
        # datasets/openai are heavyweight and credentialed; the CLI must not drag
        # them in just to print help or refuse an action. Checked in a clean
        # subprocess because this process has already imported them via fixtures.
        probe = (
            "import sys; import pii_masking.benchmark; "
            "print(sorted(m for m in ('datasets', 'openai') if m in sys.modules))"
        )
        env = dict(os.environ, PYTHONPATH=str(EXAMPLES_ROOT))
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
        )

        assert result.stdout.strip() == "[]"

    def test_the_workflow_modules_load_lazily(self):
        probe = (
            "import sys; import pii_masking.benchmark; "
            "print(sorted(m for m in sys.modules if m.startswith('pii_masking.benchmark_')))"
        )
        env = dict(os.environ, PYTHONPATH=str(EXAMPLES_ROOT))
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
        )

        # Only the protocol module, which the CLI imports eagerly by design.
        assert result.stdout.strip() == "['pii_masking.benchmark_protocol']"

    def test_a_missing_subcommand_is_an_error(self):
        with pytest.raises(SystemExit) as exit_info:
            benchmark.main([])

        assert exit_info.value.code != 0


# ── interlocks ────────────────────────────────────────────────────────────────
class TestInterlocks:
    def test_prepare_refuses_without_allow_network(self, tmp_path):
        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(
                [
                    "prepare",
                    "--benchmark-source",
                    "openpii-validation",
                    "--output-root",
                    str(tmp_path),
                ]
            )

        assert "--allow-network" in str(exit_info.value)

    def test_launch_refuses_without_yes(self, tmp_path):
        _, directory = write_protocol(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["launch", "--protocol-dir", str(directory)])

        assert "--yes" in str(exit_info.value)
        assert "PAID" in str(exit_info.value)

    def test_preflight_adapter_refuses_without_yes(self, tmp_path):
        _, directory = write_protocol(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["preflight-adapter", "--protocol-dir", str(directory)])

        assert "--yes" in str(exit_info.value)
        assert "shared serving adapter cache" in str(exit_info.value)

    @pytest.mark.parametrize("phase", ["smoke", "pilot", "full"])
    def test_evaluate_refuses_without_yes(self, tmp_path, phase):
        _, directory = write_protocol(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["evaluate", "--protocol-dir", str(directory), "--phase", phase])

        assert "--yes" in str(exit_info.value)
        assert phase in str(exit_info.value)

    def test_refusal_states_that_the_flag_is_not_approval(self, tmp_path):
        _, directory = write_protocol(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["launch", "--protocol-dir", str(directory)])

        assert "does not by itself constitute approval" in str(exit_info.value)

    def test_refusal_happens_before_any_workflow_module_loads(self, tmp_path):
        """The interlock must not depend on a module that may not exist yet."""
        with pytest.raises(SystemExit) as exit_info:
            benchmark.main(["prepare", "--benchmark-source", "openpii-validation"])

        assert "unavailable in this build" not in str(exit_info.value)

    def test_an_unknown_benchmark_source_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            benchmark.main(["prepare", "--benchmark-source", "made-up", "--allow-network"])

    def test_an_unknown_phase_is_rejected_by_the_parser(self, tmp_path):
        _, directory = write_protocol(tmp_path)

        with pytest.raises(SystemExit):
            benchmark.main(
                ["evaluate", "--protocol-dir", str(directory), "--phase", "everything", "--yes"]
            )


# ── protocol immutability across commands ─────────────────────────────────────
class TestProtocolImmutability:
    @pytest.mark.parametrize(
        "argv",
        [
            ["launch", "--protocol-dir", "{d}"],
            ["preflight-adapter", "--protocol-dir", "{d}"],
            ["evaluate", "--protocol-dir", "{d}", "--phase", "pilot"],
            ["score", "--protocol-dir", "{d}"],
        ],
    )
    def test_commands_never_rewrite_the_protocol(self, tmp_path, argv, monkeypatch):
        protocol, directory = write_protocol(tmp_path)
        path = directory / "protocol.json"
        before = path.read_bytes()

        # Let the command get as far as it can without network or credentials.
        with pytest.raises(SystemExit):
            benchmark.main([token.format(d=directory) for token in argv])

        assert path.read_bytes() == before

    def test_a_command_against_a_tampered_protocol_fails(self, tmp_path, capsys):
        protocol, directory = write_protocol(tmp_path)
        path = directory / "protocol.json"
        payload = dict(protocol.payload)
        payload["training"] = {"lora_rank": 32}
        path.write_bytes(canonical_bytes(payload))

        code = benchmark.main(["score", "--protocol-dir", str(directory)])

        assert code == 1
        assert "protocol error" in capsys.readouterr().err

    def test_score_verifies_the_protocol_before_scoring(self, tmp_path, capsys):
        _, directory = write_protocol(tmp_path)
        (directory / "protocol.json").unlink()

        code = benchmark.main(["score", "--protocol-dir", str(directory)])

        assert code == 1
        assert "no protocol at" in capsys.readouterr().err
