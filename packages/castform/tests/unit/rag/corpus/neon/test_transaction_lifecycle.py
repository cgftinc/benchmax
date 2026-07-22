"""Contract #1/#7 (B5): version-lifecycle transaction seam + retention.

Retention defaults are frozen here (pass). The transaction seam and RRF fusion
are xfail skeletons that must raise NotImplementedError. The transaction test
encodes the atomicity contract: the ledger update and the view replacement must
commit or roll back together, so a failing statement leaves neither applied.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.schema import DEFAULT_RETENTION, RetentionPolicy
from castform.rag.corpus.neon.search import fuse_rrf


def test_retention_keeps_rollback_target() -> None:
    assert isinstance(DEFAULT_RETENTION, RetentionPolicy)
    # >= 2 activated retained so rollback always has a prior version.
    assert DEFAULT_RETENTION.keep_activated >= 2
    assert DEFAULT_RETENTION.keep_ready >= 1


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 4")
def test_activation_rolls_back_atomically() -> None:
    client = NeonClient(lambda: "postgresql://rw@host/db")
    # Second statement fails; the whole transaction (ledger + view swap) must
    # roll back together. Slice 4 injects the failure and asserts no partial.
    client.execute_in_transaction(["<upsert ledger active row>", "<bad view swap>"])  # type: ignore[list-item]


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_fuse_rrf_single_owner() -> None:
    fuse_rrf([["a", "b"], ["b", "c"]])
