"""Unit tests: deterministic re-chunk + filterable metadata derivation (no DB)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curated_rows import build_curated_rows  # noqa: E402
from handbook_corpus import (  # noqa: E402
    ROOT_SECTION,
    build_collection,
    section_metadata,
)

# A tiny handbook-shaped fixture with enough distinct, long sections to chunk and
# to yield cross-section token candidates for the curated rows.
_DOCS = {
    "engineering/development/backend.md": (
        "# Backend\n\n## Deploys\n"
        + ("The zephyrine deploy runbook covers rollout. " * 40)
        + "\n\n## Oncall\n"
        + ("Oncall escalation follows the paging ladder. " * 40)
    ),
    "engineering/frontend.md": (
        "# Frontend\n\n## Builds\n" + ("Frontend builds use the pipeline. " * 60)
    ),
    "finance/expenses.md": (
        "# Expenses\n\n## Reimbursement\n"
        + ("The zephyrine expense policy applies to travel. " * 40)
        + "\n\n## Approvals\n"
        + ("Approvals route through the finance queue. " * 40)
    ),
    "toplevel.md": "# Top\n\n" + ("A root-level page with content. " * 60),
}


def _write_fixture(root: Path) -> Path:
    for rel, text in _DOCS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_section_metadata_derivation() -> None:
    assert section_metadata("engineering/development/backend.md") == {
        "handbook_section": "engineering",
        "path_depth": 3,
    }
    assert section_metadata("finance/expenses.md") == {
        "handbook_section": "finance",
        "path_depth": 2,
    }
    # a root-level file (no directory) is tagged with the sentinel section
    assert section_metadata("toplevel.md") == {
        "handbook_section": ROOT_SECTION,
        "path_depth": 1,
    }


def test_rechunk_is_deterministic(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "docs")
    first = [c.hash for c in build_collection(root)]
    second = [c.hash for c in build_collection(root)]
    assert first == second, "re-chunk produced different hashes"
    assert len(first) == len(set(first)), "duplicate chunk hashes within one build"


def test_chunks_carry_filterable_metadata(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "docs")
    chunks = list(build_collection(root))
    sections = {dict(c.metadata)["handbook_section"] for c in chunks}
    assert {"engineering", "finance", ROOT_SECTION} <= sections
    for c in chunks:
        md = dict(c.metadata)
        assert isinstance(md["path_depth"], int)
        assert md["path_depth"] >= 1


def test_metadata_changes_hash(tmp_path: Path) -> None:
    """The section metadata is folded into the hash (so it re-hashes the corpus)."""
    from castform.rag.chunkers.markdown import MarkdownChunker

    text = _DOCS["finance/expenses.md"]
    plain = MarkdownChunker().chunk(text, file="finance/expenses.md")
    tagged = MarkdownChunker().chunk(
        text,
        file="finance/expenses.md",
        extra_metadata=section_metadata("finance/expenses.md"),
    )
    assert [c.hash for c in plain] != [c.hash for c in tagged]


def test_curated_rows_from_small_corpus(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "docs")
    collection = build_collection(root)
    # "zephyrine" appears in engineering AND finance -> a cross-section candidate.
    rows = build_curated_rows(collection, n_filter=1, n_hybrid=0)
    assert rows
    row = rows[0]
    assert row.search_mode == "lexical"
    assert row.filter_dsl is not None
    assert len(row.gold_chunk_hashes) == 1
    assert row.decoy_chunk_hashes  # decoys exist for the cross-section token


def test_curated_rows_raise_when_insufficient(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "docs")
    collection = build_collection(root)
    with pytest.raises(ValueError, match="curated candidates"):
        build_curated_rows(collection, n_filter=1000, n_hybrid=1000)
