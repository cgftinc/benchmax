"""Unit tests for castform.platform.sft upload and the SFT launch wire format."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from benchmax.sft import SFT_DATASET_FORMAT, SftDataset, SftDatasetError
from castform.platform import (
    SftTrainingConfig,
    TrainerClient,
    UploadedSftAssets,
    upload_sft_assets,
)
from castform.platform.sft import (
    LONG_CONTEXT_LADDER,
    MAX_EVAL_ROWS,
    sft_assets_digest,
)


def _dataset() -> SftDataset:
    return SftDataset.from_rows(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    )


class FakeStorageClient:
    """Records upload_file calls; returns synthetic blob metadata."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes, str]] = []

    def upload_file(self, path: str, content: bytes, mime_type: str, **kwargs: Any) -> dict:
        self.uploads.append((path, content, mime_type))
        return {
            "blobPath": path,
            "uploadUrl": f"https://example.invalid/{path}",
            "expiresAt": "2099-01-01T00:00:00Z",
            "willOverwrite": False,
        }


class TestUploadSftAssets:
    def test_uploads_canonical_bytes_to_content_addressed_prefix(self) -> None:
        dataset = _dataset()
        payload = dataset.to_jsonl_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        storage = FakeStorageClient()

        uploaded = upload_sft_assets(
            dataset=dataset,
            run_name="support-sft",
            storage_client=storage,  # type: ignore[arg-type]
        )

        assert storage.uploads == [
            (f"datasets/support-sft/{digest[:16]}/train.jsonl", payload, "application/jsonl")
        ]
        assert uploaded == UploadedSftAssets(
            dataset_path=f"datasets/support-sft/{digest[:16]}",
            dataset_format="benchmax-sft-v1",
            content_digest=digest,
            row_count=1,
        )
        assert uploaded.dataset_format == SFT_DATASET_FORMAT
        # No eval set was uploaded, so nothing marks one.
        assert uploaded.eval_digest is None
        assert uploaded.eval_row_count is None

    def test_equivalent_datasets_share_a_prefix(self) -> None:
        storage = FakeStorageClient()
        first = upload_sft_assets(
            dataset=_dataset(),
            run_name="r",
            storage_client=storage,  # type: ignore[arg-type]
        )
        second = upload_sft_assets(
            dataset=_dataset(),
            run_name="r",
            storage_client=storage,  # type: ignore[arg-type]
        )
        assert first == second

    def test_rejects_non_dataset_without_any_request(self) -> None:
        storage = FakeStorageClient()
        for bad in (None, [{"messages": []}], b"{}", "train.jsonl"):
            with pytest.raises(TypeError, match="SftDataset"):
                upload_sft_assets(
                    dataset=bad,  # type: ignore[arg-type]
                    run_name="r",
                    storage_client=storage,  # type: ignore[arg-type]
                )
        assert storage.uploads == []

    def test_invalid_rows_fail_before_upload(self) -> None:
        storage = FakeStorageClient()
        with pytest.raises(SftDatasetError):
            upload_sft_assets(
                dataset=SftDataset.from_rows([{"messages": []}]),
                run_name="r",
                storage_client=storage,  # type: ignore[arg-type]
            )
        assert storage.uploads == []

    def test_unsafe_run_name_rejected_without_upload(self) -> None:
        storage = FakeStorageClient()
        with pytest.raises(ValueError, match="storage path segment"):
            upload_sft_assets(
                dataset=_dataset(),
                run_name="bad name?",
                storage_client=storage,  # type: ignore[arg-type]
            )
        assert storage.uploads == []


class TestSftTrainingConfig:
    def test_defaults_match_public_argument_table(self) -> None:
        assert SftTrainingConfig().as_args() == {
            "num_epochs": 1,
            "learning_rate": 1e-5,
            "max_context_tokens": 8192,
            "save_interval": 20,
            "seed": 42,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_epochs": 0},
            {"num_epochs": 101},
            {"num_epochs": True},
            {"num_epochs": 1.0},
            {"learning_rate": 0},
            {"learning_rate": -1e-5},
            {"learning_rate": 0.2},
            {"learning_rate": float("nan")},
            {"learning_rate": True},
            {"max_context_tokens": 255},
            {"max_context_tokens": 8193},
            {"max_context_tokens": 16384},
            {"max_context_tokens": False},
            {"save_interval": 0},
            {"save_interval": 10_001},
            {"seed": -1},
            {"seed": 2_147_483_648},
            {"seed": True},
        ],
    )
    def test_out_of_range_values_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SftTrainingConfig(**kwargs)

    def test_boundary_values_accepted(self) -> None:
        SftTrainingConfig(
            num_epochs=100,
            learning_rate=0.1,
            max_context_tokens=256,
            save_interval=10_000,
            seed=2_147_483_647,
        )

    def test_unset_fields_omitted_from_as_args(self) -> None:
        # Unset knobs must vanish from the payload, not serialize as null.
        args = SftTrainingConfig(adam_beta2=0.95).as_args()
        assert args["adam_beta2"] == 0.95
        for absent in (
            "lr_decay_style",
            "min_lr",
            "warmup_ratio",
            "grad_clip",
            "lora_rank",
            "global_batch_size",
        ):
            assert absent not in args

    def test_global_batch_size_presence_tracked(self) -> None:
        assert "global_batch_size" not in SftTrainingConfig().as_args()
        for gbs in (4, 8, 12, 64):
            assert SftTrainingConfig(global_batch_size=gbs).as_args()["global_batch_size"] == gbs

    def test_v28_shape_serializes_completely(self) -> None:
        assert SftTrainingConfig(
            learning_rate=6e-5,
            lr_decay_style="cosine",
            min_lr=2e-5,
            warmup_ratio=0.05,
            adam_beta2=0.95,
            grad_clip=0.5,
            lora_rank=64,
        ).as_args() == {
            "num_epochs": 1,
            "learning_rate": 6e-5,
            "max_context_tokens": 8192,
            "save_interval": 20,
            "seed": 42,
            "lr_decay_style": "cosine",
            "min_lr": 2e-5,
            "warmup_ratio": 0.05,
            "adam_beta2": 0.95,
            "grad_clip": 0.5,
            "lora_rank": 64,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr_decay_style": "linear"},
            {"min_lr": -1e-6},
            # A floor at or above the peak makes the schedule meaningless.
            {"learning_rate": 1e-5, "min_lr": 1e-5},
            {"warmup_ratio": 0.51},
            {"warmup_ratio": -0.01},
            {"adam_beta2": 0.89},
            {"adam_beta2": 1.0},
            {"grad_clip": 0},
            {"grad_clip": 10.1},
            # 128 trains but cannot be served; alpha is never a caller input.
            {"lora_rank": 128},
            {"lora_rank": 16},
            {"global_batch_size": 10},
            {"global_batch_size": 5},
            {"global_batch_size": 0},
            {"global_batch_size": 68},
        ],
    )
    def test_v11_out_of_range_values_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SftTrainingConfig(**kwargs)

    def test_client_mirror_matches_server_lora_set(self) -> None:
        for rank in (32, 64):
            assert SftTrainingConfig(lora_rank=rank).as_args()["lora_rank"] == rank


class TestLaunchSftRun:
    def _client(self, handler) -> TrainerClient:
        client = TrainerClient(api_key="test-key", base_url="https://example.invalid")
        client._http_client = httpx.Client(
            base_url="https://example.invalid",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx.MockTransport(handler),
        )
        return client

    def _assets(self) -> UploadedSftAssets:
        return UploadedSftAssets(
            dataset_path="datasets/support-sft/0123456789abcdef",
            dataset_format=SFT_DATASET_FORMAT,
            content_digest="0" * 64,
        )

    def test_posts_exact_request_shape(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(202, json={"runId": "run-sft-1", "trainingMethod": "sft"})

        run_id = self._client(handler).launch_sft_run(
            assets=self._assets(),
            name="support-sft",
            config=SftTrainingConfig(num_epochs=2, seed=7),
        )

        assert run_id == "run-sft-1"
        assert captured["url"].endswith("/v1/train/runs/sft")
        assert captured["body"] == {
            "name": "support-sft",
            "dataset": {
                "format": "benchmax-sft-v1",
                "path": "datasets/support-sft/0123456789abcdef",
            },
            "args": {
                "num_epochs": 2,
                "learning_rate": 1e-5,
                "max_context_tokens": 8192,
                "save_interval": 20,
                "seed": 7,
            },
        }

    def test_defaults_config_when_omitted(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(202, json={"runId": "run-sft-2", "trainingMethod": "sft"})

        self._client(handler).launch_sft_run(assets=self._assets(), name="n")
        assert captured["body"]["args"] == SftTrainingConfig().as_args()

    def test_rejects_untyped_assets_without_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request expected")

        client = self._client(handler)
        with pytest.raises(TypeError, match="UploadedSftAssets"):
            client.launch_sft_run(
                assets={"dataset_path": "x"},  # type: ignore[arg-type]
                name="n",
            )
        with pytest.raises(TypeError, match="SftTrainingConfig"):
            client.launch_sft_run(
                assets=self._assets(),
                name="n",
                config={"num_epochs": 1},  # type: ignore[arg-type]
            )

    def test_disabled_gate_error_surfaces(self) -> None:
        from castform.platform.exceptions import JobLaunchError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": "SFT launch is not enabled"})

        with pytest.raises(JobLaunchError, match="not enabled"):
            self._client(handler).launch_sft_run(assets=self._assets(), name="n")


class TestEvalAssets:
    """Uploading an eval set, and the marker it makes the launch send."""

    def _eval_dataset(self, count: int = 3) -> SftDataset:
        return SftDataset.from_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": f"eq{i}"},
                        {"role": "assistant", "content": f"ea{i}"},
                    ]
                }
                for i in range(count)
            ]
        )

    def test_eval_set_lands_beside_train_under_a_combined_prefix(self) -> None:
        train = SftDataset.from_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        evalset = self._eval_dataset()
        storage = FakeStorageClient()

        uploaded = upload_sft_assets(
            dataset=train,
            eval_dataset=evalset,
            run_name="support-sft",
            storage_client=storage,  # type: ignore[arg-type]
        )

        train_digest = hashlib.sha256(train.to_jsonl_bytes()).hexdigest()
        eval_digest = hashlib.sha256(evalset.to_jsonl_bytes()).hexdigest()
        prefix = f"datasets/support-sft/{sft_assets_digest(train_digest, eval_digest)[:16]}"

        assert [path for path, _, _ in storage.uploads] == [
            f"{prefix}/train.jsonl",
            f"{prefix}/eval.jsonl",
        ]
        assert uploaded.dataset_path == prefix
        assert uploaded.eval_digest == eval_digest
        assert uploaded.eval_row_count == 3
        assert uploaded.row_count == 1

    def test_adding_an_eval_set_moves_the_prefix(self) -> None:
        # The prefix pins the FULL data identity, so an overwrite at the
        # train-only prefix cannot attach an eval set to a launched run.
        train = SftDataset.from_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        train_only = upload_sft_assets(
            dataset=train,
            run_name="s",
            storage_client=FakeStorageClient(),  # type: ignore[arg-type]
        )
        with_eval = upload_sft_assets(
            dataset=train,
            eval_dataset=self._eval_dataset(),
            run_name="s",
            storage_client=FakeStorageClient(),  # type: ignore[arg-type]
        )
        assert train_only.dataset_path != with_eval.dataset_path

    def test_train_only_prefix_is_unchanged(self) -> None:
        # Train-only prefixes must stay byte-stable: with no eval set,
        # sft_assets_digest is exactly the train digest.
        train = SftDataset.from_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        digest = hashlib.sha256(train.to_jsonl_bytes()).hexdigest()
        assert sft_assets_digest(digest) == digest

    def test_combined_digest_is_framing_unambiguous(self) -> None:
        a, b = "a" * 64, "b" * 64
        assert sft_assets_digest(a, b) != sft_assets_digest(b, a)
        assert sft_assets_digest(a, b) != a

    def test_oversized_eval_set_fails_before_upload(self) -> None:
        storage = FakeStorageClient()
        train = SftDataset.from_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="above the 2048-row limit"):
            upload_sft_assets(
                dataset=train,
                eval_dataset=self._eval_dataset(MAX_EVAL_ROWS + 1),
                run_name="s",
                storage_client=storage,  # type: ignore[arg-type]
            )
        assert storage.uploads == []


class TestEvalLaunchWire:
    """What the launch sends when the uploaded assets carry an eval set."""

    def _client(self, handler) -> TrainerClient:
        client = TrainerClient(api_key="test-key", base_url="https://example.invalid")
        client._http_client = httpx.Client(
            base_url="https://example.invalid",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx.MockTransport(handler),
        )
        return client

    def _eval_assets(self) -> UploadedSftAssets:
        return UploadedSftAssets(
            dataset_path="datasets/support-sft/0123456789abcdef",
            dataset_format=SFT_DATASET_FORMAT,
            content_digest="0" * 64,
            row_count=400,
            eval_digest="1" * 64,
            eval_row_count=100,
        )

    def _capture(self, assets: UploadedSftAssets, config: SftTrainingConfig | None = None) -> dict:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(202, json={"runId": "run-1", "trainingMethod": "sft"})

        self._client(handler).launch_sft_run(assets=assets, name="support-sft", config=config)
        return captured["body"]

    def test_marker_and_train_identity_travel_together(self) -> None:
        body = self._capture(self._eval_assets())
        assert body["eval"] == {"rows": 100, "digest": "1" * 64}
        # The platform derives the pass cap and re-derives the prefix from
        # these, so they are mandatory exactly when the marker is present.
        assert body["dataset"]["digest"] == "0" * 64
        assert body["dataset"]["rows"] == 400

    def test_eval_interval_rides_args_when_sent(self) -> None:
        body = self._capture(self._eval_assets(), SftTrainingConfig(eval_interval=50))
        assert body["args"]["eval_interval"] == 50

    def test_unset_eval_interval_is_omitted_so_the_platform_derives_it(self) -> None:
        body = self._capture(self._eval_assets())
        assert "eval_interval" not in body["args"]

    def test_eval_interval_without_an_eval_set_is_refused_locally(self) -> None:
        # Fails before the request rather than earning a 400: the caller's
        # mistake is knowable here.
        assets = UploadedSftAssets(
            dataset_path="datasets/support-sft/0123456789abcdef",
            dataset_format=SFT_DATASET_FORMAT,
            content_digest="0" * 64,
            row_count=400,
        )
        with pytest.raises(ValueError, match="no eval set"):
            self._capture(assets, SftTrainingConfig(eval_interval=50))

    def test_no_eval_set_keeps_the_v1_body(self) -> None:
        assets = UploadedSftAssets(
            dataset_path="datasets/support-sft/0123456789abcdef",
            dataset_format=SFT_DATASET_FORMAT,
            content_digest="0" * 64,
            row_count=400,
        )
        body = self._capture(assets)
        assert set(body) == {"name", "dataset", "args"}
        assert set(body["dataset"]) == {"format", "path"}


class TestLongContextLadder:
    """The client-side range after the 2026-08-11 context->CP freeze.

    Two shapes, mirroring the server: a free range up to the v1 ceiling, and
    discrete rungs above it. The gating FLAG is deliberately not modelled — a
    value can be well-formed here and still be refused at launch.
    """

    def test_v1_range_still_free(self) -> None:
        for value in (256, 4096, 8192):
            assert SftTrainingConfig(max_context_tokens=value).max_context_tokens == value

    def test_every_frozen_rung_is_accepted(self) -> None:
        for value in LONG_CONTEXT_LADDER:
            assert SftTrainingConfig(max_context_tokens=value).max_context_tokens == value

    def test_off_ladder_values_above_the_ceiling_are_refused(self) -> None:
        # Not clamped: above the ceiling only calibrated rungs exist, so an
        # arbitrary value is a mistake rather than a request to round down.
        for value in (8193, 16384, 40000, 131073):
            with pytest.raises(ValueError, match="max_context_tokens"):
                SftTrainingConfig(max_context_tokens=value)

    def test_below_the_floor_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="max_context_tokens"):
            SftTrainingConfig(max_context_tokens=255)

    def test_a_long_context_config_serialises_the_value(self) -> None:
        args = SftTrainingConfig(max_context_tokens=131072, global_batch_size=10).as_args()
        assert args["max_context_tokens"] == 131072
        assert args["global_batch_size"] == 10

    def test_v1_config_as_args_is_unchanged(self) -> None:
        # The release gate: an untouched v1 config must serialise exactly as
        # before, or every existing caller's payload shifts under them.
        assert SftTrainingConfig().as_args() == {
            "num_epochs": 1,
            "learning_rate": 1e-5,
            "max_context_tokens": 8192,
            "save_interval": 20,
            "seed": 42,
        }


class TestBatchSizeAcrossContextBuckets:
    """gbs divisibility is knowable client-side only in the v1 range."""

    def test_v1_range_still_enforces_multiple_of_four(self) -> None:
        # CP is always 1 below the ceiling, so DP is always 4 and the rule is
        # certain. Keeping it preserves v1's fail-fast ergonomics.
        with pytest.raises(ValueError, match="multiple of 4"):
            SftTrainingConfig(max_context_tokens=8192, global_batch_size=10)

    def test_long_rungs_defer_divisibility_to_the_server(self) -> None:
        # Above the ceiling the platform picks CP per rung and DP narrows, so
        # batch sizes this client would have rejected become legal. Mirroring
        # the context->CP table here would let the SDK go stale against a
        # server-owned mapping.
        for context in LONG_CONTEXT_LADDER:
            cfg = SftTrainingConfig(max_context_tokens=context, global_batch_size=10)
            assert cfg.as_args()["global_batch_size"] == 10

    def test_range_still_bounded_on_long_rungs(self) -> None:
        # Relaxing divisibility is not relaxing the range.
        for bad in (0, 65, -4):
            with pytest.raises(ValueError, match="global_batch_size"):
                SftTrainingConfig(max_context_tokens=131072, global_batch_size=bad)
