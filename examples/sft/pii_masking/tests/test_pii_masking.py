"""Network-free tests for the PII-masking SFT example.

All source records here are original synthetic fixtures; CI never downloads or
redistributes upstream rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from benchmax.sft import SftDataset
from castform.platform import SftTrainingConfig, UploadedSftAssets
from pii_masking import main as example

README = Path(example.__file__).parent / "README.md"


def synthetic_rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "source_text": f"Contact Person-{index} at example-{index}@synthetic.test.",
            "masked_text": f"Contact [NAME_{index}] at [EMAIL_{index}].",
            "uid": f"synthetic-{index}",
            "language": "en",
        }
        for index in range(count)
    ]


class TestFrozenConstants:
    def test_source_pinning(self) -> None:
        assert example.SOURCE_DATASET == "ai4privacy/pii-masking-openpii-1m"
        assert example.SOURCE_REVISION == "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0"
        assert example.SOURCE_SPLIT == "train"

    def test_row_limits(self) -> None:
        assert example.DEFAULT_ROWS == 256
        assert example.MAX_ROWS == 4096


class TestMapping:
    def test_maps_exact_row_shape(self) -> None:
        (source,) = synthetic_rows(1)
        assert example.map_source_row(source) == {
            "messages": [
                {"role": "system", "content": example.SYSTEM_PROMPT},
                {"role": "user", "content": source["source_text"]},
                {"role": "assistant", "content": source["masked_text"]},
            ],
            "metadata": {
                "source_dataset": "ai4privacy/pii-masking-openpii-1m",
                "source_revision": "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0",
                "source_uid": "synthetic-0",
                "language": "en",
            },
        }

    def test_mapped_rows_validate_as_sft_dataset(self) -> None:
        dataset = SftDataset.from_rows(example.build_rows(synthetic_rows(4), 4))
        assert len(dataset) == 4


class TestBuildRows:
    def test_preserves_source_order_without_filtering(self) -> None:
        rows = example.build_rows(synthetic_rows(8), 5)
        uids = [row["metadata"]["source_uid"] for row in rows]  # type: ignore[index]
        assert uids == [f"synthetic-{i}" for i in range(5)]

    def test_inspects_exactly_n_records(self) -> None:
        consumed = 0

        def counting_stream() -> Iterator[dict[str, str]]:
            nonlocal consumed
            for row in synthetic_rows(100):
                consumed += 1
                yield row

        rows = example.build_rows(counting_stream(), 7)
        assert len(rows) == 7
        assert consumed == 7

    def test_fails_when_source_ends_early(self) -> None:
        with pytest.raises(RuntimeError, match="ended after 3 records; 5 required"):
            example.build_rows(synthetic_rows(3), 5)

    def test_bounds(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 4096"):
            example.build_rows(synthetic_rows(1), 0)
        with pytest.raises(ValueError, match="between 1 and 4096"):
            example.build_rows(synthetic_rows(1), 4097)


class TestPrepare:
    def test_writes_canonical_bytes(self, tmp_path: Path) -> None:
        output = tmp_path / "train.jsonl"
        dataset = example.prepare(output, 3, synthetic_rows(3))
        assert output.read_bytes() == dataset.to_jsonl_bytes()
        assert SftDataset.from_jsonl(output).to_jsonl_bytes() == dataset.to_jsonl_bytes()

    def test_deterministic_across_runs(self, tmp_path: Path) -> None:
        first = example.prepare(tmp_path / "a.jsonl", 3, synthetic_rows(3))
        second = example.prepare(tmp_path / "b.jsonl", 3, synthetic_rows(3))
        assert first.to_jsonl_bytes() == second.to_jsonl_bytes()


class TestCli:
    def test_rows_above_hard_maximum_rejected(self) -> None:
        with pytest.raises(SystemExit):
            example.build_parser().parse_args(["--rows", "4097"])

    def test_launch_requires_run_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            example.main(["--launch"])

    def test_prepare_only_never_uploads_or_launches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(example, "stream_source_rows", lambda: iter(synthetic_rows(4)))
        monkeypatch.setattr(example, "upload_sft_assets", _fail_if_called("upload_sft_assets"))
        monkeypatch.setattr(example, "TrainerClient", _fail_if_called("TrainerClient"))
        output = tmp_path / "train.jsonl"
        assert example.main(["--rows", "4", "--output", str(output)]) == 0
        assert output.exists()

    def test_launch_wires_upload_and_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(example, "stream_source_rows", lambda: iter(synthetic_rows(4)))
        calls: dict[str, Any] = {}
        uploaded = UploadedSftAssets(
            dataset_path="datasets/pii-sft/0123456789abcdef",
            dataset_format="benchmax-sft-v1",
            content_digest="0" * 64,
        )

        def fake_upload(*, dataset: SftDataset, run_name: str) -> UploadedSftAssets:
            calls["upload"] = (dataset.to_jsonl_bytes(), run_name)
            return uploaded

        class FakeTrainerClient:
            def launch_sft_run(
                self, *, assets: UploadedSftAssets, name: str, config: SftTrainingConfig
            ) -> str:
                calls["launch"] = (assets, name, config)
                return "run-42"

        monkeypatch.setattr(example, "upload_sft_assets", fake_upload)
        monkeypatch.setattr(example, "TrainerClient", FakeTrainerClient)

        exit_code = example.main(
            [
                "--rows",
                "4",
                "--output",
                str(tmp_path / "t.jsonl"),
                "--launch",
                "--run-name",
                "pii-sft",
            ]
        )

        assert exit_code == 0
        assert calls["upload"][1] == "pii-sft"
        assert calls["launch"] == (uploaded, "pii-sft", SftTrainingConfig())


def _fail_if_called(name: str) -> Any:
    def _fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"{name} must not be called without --launch")

    return _fail


class TestReadmeAttribution:
    def test_records_source_license_revision_and_transformation(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert "ai4privacy/pii-masking-openpii-1m" in text
        assert "Ai4Privacy / Ai Suisse SA" in text
        assert "CC BY 4.0" in text
        assert "creativecommons.org/licenses/by/4.0" in text
        assert "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0" in text
        assert "transforms" in text

    def test_warns_about_cost_and_cancellation(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert "cost warning" in text
        assert "cancellation" in text
        assert "pii-masking-300k" in text
