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
        )
        assert uploaded.dataset_format == SFT_DATASET_FORMAT

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
        # Unset v1.1 knobs must vanish from the payload, not serialize as
        # null: the platform reads presence, and a null would look like a
        # deliberate choice overriding the model config.
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
