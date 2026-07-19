from __future__ import annotations

from pathlib import Path

import pytest

from benchmax.envs import Example, JsonRow, JsonlDataset


def test_jsonl_dataset_delegates_complete_rows_in_source_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('\n{"question":"first"}\n{"question":"second","answer":2}\n')
    seen: list[JsonRow] = []

    def make_example(row: JsonRow) -> Example[JsonRow]:
        seen.append(row)
        return Example(id=f"chosen-{len(seen)}", payload=row)

    dataset = JsonlDataset(path, row_to_example=make_example)

    assert seen == [
        {"question": "first"},
        {"question": "second", "answer": 2},
    ]
    assert [example.id for example in dataset] == ["chosen-1", "chosen-2"]
    assert dataset[1].payload == {"question": "second", "answer": 2}


@pytest.mark.parametrize(
    ("contents", "error_type"),
    [
        ("{not-json}\n", ValueError),
        ("[]\n", TypeError),
    ],
)
def test_jsonl_dataset_reports_the_source_of_invalid_rows(
    tmp_path: Path,
    contents: str,
    error_type: type[Exception],
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(contents)

    with pytest.raises(error_type, match=r"train\.jsonl:1"):
        JsonlDataset(path, row_to_example=lambda row: Example(id="row", payload=row))
