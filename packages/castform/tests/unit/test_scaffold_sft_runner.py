"""The sft seed's `main.py` runnable stage-runner: local validate, the two
launch gates, and the validate → upload → submit ordering that reaches the
wire.

Tested against `sft_main.py` directly rather than through `castform setup`, so
a template regression is caught here and not only in the setup gate. The wire
half runs a REAL `TrainerClient` over a mock transport, so the asserted body is
the one that would be posted.
"""

from __future__ import annotations

import json
from pathlib import Path

import castform.cli.scaffold as scaffold_pkg
import httpx
import pytest
from castform.platform import client as client_module
from castform.platform.client import TrainerClient
from castform.platform.training_run import upload_sft_run

from ._scaffold import load_module

_SFT_SEED = Path(scaffold_pkg.__file__).parent / "sft_main.py"


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Load the sft seed as a module (its `__main__` block does not fire) with
    the cwd pointed at an empty project dir.

    `ensure_session` is deliberately NOT neutralized here: a blanket no-op would
    hide a login fired behind a closed gate. Tests that legitimately reach the
    wire neutralize it themselves (`_stub_launchable`).
    """
    monkeypatch.chdir(tmp_path)
    return load_module(_SFT_SEED)


def _row(text: str = "hello") -> dict:
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"re: {text}"},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeStorageClient:
    """In-memory StorageClient stand-in. Records calls; returns synthetic blob paths."""

    def __init__(self):
        self.uploads: list[tuple[str, bytes]] = []

    def upload_local_file(
        self, path: str, file_path: Path, *, expires_in_minutes: int | None = None
    ) -> dict:
        self.uploads.append((path, Path(file_path).read_bytes()))
        return {"blobPath": f"blob://{path}"}


# ── import safety + config shape ───────────────────────────────────────────────


def test_import_defines_stages_without_running(mod, tmp_path):
    """Loading the seed defines the runner but executes NO stage (the `__main__`
    block is import-safe)."""
    assert all(hasattr(mod, n) for n in ("main", "generate_data", "validate", "launch"))
    assert not (tmp_path / "train.jsonl").exists()


def test_dataset_filenames_match_setups_destinations(mod):
    """`castform setup` writes the seed data to these exact names."""
    assert (mod.TRAIN_FILE, mod.EVAL_FILE) == ("train.jsonl", "eval.jsonl")


def test_training_mode_is_a_launch_config_key(mod):
    """The mode is a config key, not a flag — the run reproduces from this file."""
    assert mod.LAUNCH_CONFIG["training_mode"] == "sft"


def test_reserved_key_filter_keeps_the_wire_keys(mod):
    """The local-only keys are stripped; `training_mode` and the optional
    per-run `model` must survive — the server reads both."""
    mod.LAUNCH_CONFIG["model"] = "Qwen/Qwen3.5-4B"
    mod.LAUNCH_CONFIG["type"] = "simple"
    mod.LAUNCH_CONFIG["name"] = "custom-run"
    mod.LAUNCH_CONFIG["allow_experimental_weights"] = True

    args = mod._launcher_args()

    assert args["training_mode"] == "sft"
    assert args["model"] == "Qwen/Qwen3.5-4B"
    assert args["num_epochs"] == mod.LAUNCH_CONFIG["num_epochs"]
    assert not {"type", "name", "allow_experimental_weights"} & set(args)


def test_launcher_args_refuses_a_foreign_training_mode(mod):
    """`launch_sft_run` would overwrite it; refuse instead of silently ignoring."""
    mod.LAUNCH_CONFIG["training_mode"] = "rl"
    with pytest.raises(mod.SftConfigError, match="training_mode"):
        mod._launcher_args()


# ── validate stage: local-only, and its exit code ──────────────────────────────


def test_validate_passes_on_the_seed_rows(mod, tmp_path, capsys):
    """The shipped seed must validate on day one: `main.py` → data → validate."""
    assert mod.main([]) == 0
    assert "validate: pass" in capsys.readouterr().out
    assert (tmp_path / "train.jsonl").exists()


def test_validate_exits_1_on_invalid_rows(mod, tmp_path, capsys):
    """A schema-invalid row is a hard stop — an agent or CI cannot read a broken
    dataset as success."""
    _write_jsonl(tmp_path / "train.jsonl", [{"messages": [{"role": "user"}]}])

    assert mod.main(["validate"]) == 1
    assert "validate: fail" in capsys.readouterr().out


def test_validate_exits_1_on_malformed_json(mod, tmp_path):
    (tmp_path / "train.jsonl").write_text("{not json\n", encoding="utf-8")
    assert mod.main(["validate"]) == 1


def test_validate_makes_no_platform_call(mod, tmp_path, monkeypatch):
    """SFT validate is a local dataset check — no rollout, no credential, no HTTP."""
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    monkeypatch.setattr(
        mod,
        "ensure_session",
        lambda *a, **k: pytest.fail("validate requested a login"),
    )
    monkeypatch.setattr(
        mod, "TrainerClient", lambda *a, **k: pytest.fail("validate opened a client")
    )

    assert mod.main(["validate"]) == 0


def test_generate_data_refuses_eval_without_train(mod, tmp_path):
    """eval-without-train is a corrupted project, not a first run."""
    _write_jsonl(tmp_path / "eval.jsonl", [_row()])
    with pytest.raises(mod.SftScaffoldError):
        mod.generate_data()
    assert mod.main(["data"]) == 1  # surfaced as a clean exit code, not a traceback


# ── launch gates ───────────────────────────────────────────────────────────────


def _forbid_platform_calls(mod, monkeypatch) -> None:
    """Turn any login, prompt, upload or launch into a test failure."""
    monkeypatch.setattr(
        mod,
        "ensure_session",
        lambda *a, **k: pytest.fail("requested a login behind a closed gate"),
    )
    monkeypatch.setattr(
        mod,
        "upload_sft_run",
        lambda **kw: pytest.fail("uploaded behind a closed gate"),
    )
    monkeypatch.setattr(
        mod,
        "TrainerClient",
        lambda *a, **k: pytest.fail("opened a trainer client behind a closed gate"),
    )
    monkeypatch.setattr(
        "builtins.input", lambda *a: pytest.fail("prompted behind a closed gate")
    )


def test_sft_launch_capability_is_still_off():
    """Pins the inert state the whole re-port merges in: the platform does not
    accept env-less sft runs yet."""
    assert client_module.SFT_LAUNCH_SUPPORTED is False


def test_real_capability_false_blocks_before_any_upload_or_post(
    mod, tmp_path, monkeypatch, capsys
):
    """With the REAL flag (False): exit 1 and ZERO side effects — no login, no
    prompt, no upload, no POST — from either entrypoint.

    The login matters as much as the upload: `ensure_session` can open an
    interactive device flow, so firing it before the capability check would make
    an unlaunchable run demand credentials.
    """
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    _write_jsonl(tmp_path / "eval.jsonl", [_row("eval")])
    _forbid_platform_calls(mod, monkeypatch)

    assert mod.launch(assume_yes=True) is None
    assert mod.main(["launch"]) == 1
    assert mod.main(["launch", "-y"]) == 1
    assert "SFT_LAUNCH_SUPPORTED is False" in capsys.readouterr().err


def test_launch_blocked_by_failing_validate(mod, tmp_path, monkeypatch, capsys):
    _write_jsonl(tmp_path / "train.jsonl", [{"messages": []}])
    _forbid_platform_calls(mod, monkeypatch)
    monkeypatch.setattr(client_module, "SFT_LAUNCH_SUPPORTED", True)

    assert mod.launch(assume_yes=True) is None
    assert "validate gate failed" in capsys.readouterr().err


def test_launch_blocked_by_the_weight_gate(mod, tmp_path, monkeypatch, capsys):
    """A weight-bearing dataset validates fine but is a separate capability."""
    weighted = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "weight": 1},
        ]
    }
    _write_jsonl(tmp_path / "train.jsonl", [weighted])
    _forbid_platform_calls(mod, monkeypatch)
    monkeypatch.setattr(client_module, "SFT_LAUNCH_SUPPORTED", True)

    assert mod.launch(assume_yes=True) is None
    assert "allow_experimental_weights" in capsys.readouterr().err


def test_weight_gate_rejects_a_non_bool_override(mod, tmp_path, monkeypatch):
    """`"false"` is truthy in Python — a typo must fail loudly, not open the gate."""
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello", "weight": 1},
                ]
            }
        ],
    )
    _forbid_platform_calls(mod, monkeypatch)
    monkeypatch.setattr(client_module, "SFT_LAUNCH_SUPPORTED", True)
    mod.LAUNCH_CONFIG["allow_experimental_weights"] = "false"

    assert mod.main(["launch"]) == 1


# ── capability-true path: validate → upload → submit, and the wire body ────────


def _stub_launchable(mod, monkeypatch) -> tuple[list[str], dict]:
    """Open the capability gate and record the call order plus the posted body.

    Real `upload_sft_run` (over a fake storage client) and a real
    `TrainerClient` (over a mock transport) — only the I/O edges are faked.
    """
    monkeypatch.setattr(client_module, "SFT_LAUNCH_SUPPORTED", True)
    order: list[str] = []
    captured: dict = {"logins": 0}
    storage = FakeStorageClient()

    def fake_ensure_session(*args, **kwargs):
        captured["logins"] += 1

    real_validate = mod.validate_sft_dataset

    def recording_validate(*args, **kwargs):
        order.append("validate")
        return real_validate(*args, **kwargs)

    def recording_upload(**kwargs):
        order.append("upload")
        return upload_sft_run(**kwargs, storage_client=storage)

    def handler(request: httpx.Request) -> httpx.Response:
        order.append("submit")
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "sft-run-1"})

    def make_client() -> TrainerClient:
        client = TrainerClient(api_key="test-key", base_url="https://example.invalid")
        client._http_client = httpx.Client(
            base_url="https://example.invalid",
            transport=httpx.MockTransport(handler),
        )
        return client

    monkeypatch.setattr(mod, "ensure_session", fake_ensure_session)
    monkeypatch.setattr(mod, "validate_sft_dataset", recording_validate)
    monkeypatch.setattr(mod, "upload_sft_run", recording_upload)
    monkeypatch.setattr(mod, "TrainerClient", make_client)
    captured["storage"] = storage
    return order, captured


def test_capability_true_runs_validate_then_upload_then_submit(
    mod, tmp_path, monkeypatch
):
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    _write_jsonl(tmp_path / "eval.jsonl", [_row("eval")])
    order, captured = _stub_launchable(mod, monkeypatch)

    assert mod.launch(assume_yes=True) == "sft-run-1"
    assert order == ["validate", "upload", "submit"]
    assert captured["logins"] == 1  # the open path still authenticates
    # both splits reached storage before the run was submitted
    assert [key.rsplit("/", 1)[-1] for key, _ in captured["storage"].uploads] == [
        "train.jsonl",
        "eval.jsonl",
    ]


def test_capability_true_posts_nested_training_mode(mod, tmp_path, monkeypatch):
    """The mode must sit inside `args` — a top-level one is silently ignored by
    the platform and would fall through to an RL run."""
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    _write_jsonl(tmp_path / "eval.jsonl", [_row("eval")])
    mod.LAUNCH_CONFIG["model"] = "Qwen/Qwen3.5-4B"
    _, captured = _stub_launchable(mod, monkeypatch)

    assert mod.launch(assume_yes=True) == "sft-run-1"
    body = captured["body"]
    assert "/train/runs/launch" in captured["url"]
    assert "training_mode" not in body
    assert body["args"]["training_mode"] == "sft"
    assert body["args"]["model"] == "Qwen/Qwen3.5-4B"
    assert body["args"]["train_dataset_path"].endswith("/train.jsonl")
    assert body["args"]["eval_dataset_path"].endswith("/eval.jsonl")
    assert body["name"] == mod._run_name()


def test_train_only_project_omits_eval_from_the_wire(mod, tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    _, captured = _stub_launchable(mod, monkeypatch)

    assert mod.launch(assume_yes=True) == "sft-run-1"
    assert "eval_dataset_path" not in captured["body"]["args"]
    assert len(captured["storage"].uploads) == 1


def test_launch_declined_at_confirm_uploads_nothing(mod, tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "train.jsonl", [_row()])
    order, captured = _stub_launchable(mod, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    assert mod.launch() is None
    assert order == ["validate"]
    assert not captured["storage"].uploads
    assert captured["logins"] == 0  # declining never costs a login
