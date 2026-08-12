"""Tests for the source adapters: normalization, lineage, and laziness.

All fixtures are original synthetic rows. Neither pinned source is contacted,
here or at import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pii_masking.benchmark_selection import (
    OPENPII_DATASET,
    TASK_PIIMB_EN,
    TASK_PIIMB_MULTI,
    SelectionError,
    lineage_key,
    normalize_openpii_row,
    normalize_piimb_row,
    row_digest,
    text_hash,
)

EXAMPLES_ROOT = Path(__file__).resolve().parents[2]


def openpii_row(**overrides):
    payload = {
        "uid": 7,
        "language": "en",
        "source_text": "call Ada at 555",
        "masked_text": "call [GIVENNAME_1] at [PHONE_1]",
    }
    payload.update(overrides)
    return payload


def piimb_row(**overrides):
    payload = {
        "uid": "p-1",
        "source_uid": 7,
        "language": "en",
        "source_text": "call Ada at 555",
        "masked_text": "call [GIVENNAME_1] at [PHONE_1]",
    }
    payload.update(overrides)
    return payload


class TestOpenPiiNormalization:
    def test_uid_becomes_a_string(self):
        assert normalize_openpii_row(openpii_row(uid=7)).uid == "7"

    def test_upstream_uid_is_the_row_uid(self):
        row = normalize_openpii_row(openpii_row(uid=7))

        assert row.upstream_uid == "7"
        assert lineage_key(row.upstream_uid) == (OPENPII_DATASET, "7")

    @pytest.mark.parametrize("field", ["uid", "language", "source_text", "masked_text"])
    def test_missing_required_fields_are_fatal(self, field):
        payload = openpii_row()
        del payload[field]

        with pytest.raises(SelectionError, match=field):
            normalize_openpii_row(payload)

    def test_payload_is_retained_for_the_row_digest(self):
        row = normalize_openpii_row(openpii_row(extra="kept"))

        assert row.payload["extra"] == "kept"
        assert row.row_digest == row_digest(row.payload)


class TestPiimbNormalization:
    def test_uid_is_task_scoped(self):
        english = normalize_piimb_row(piimb_row(), TASK_PIIMB_EN)
        multi = normalize_piimb_row(piimb_row(), TASK_PIIMB_MULTI)

        assert english.uid != multi.uid
        assert english.uid.startswith(TASK_PIIMB_EN)

    def test_lineage_maps_into_the_shared_openpii_namespace(self):
        piimb = normalize_piimb_row(piimb_row(source_uid=7), TASK_PIIMB_EN)
        openpii = normalize_openpii_row(openpii_row(uid=7))

        # Task-scoped identity differs, but lineage collides — which is what
        # makes cross-split leakage detectable at all.
        assert piimb.uid != openpii.uid
        assert lineage_key(piimb.upstream_uid) == lineage_key(openpii.upstream_uid)

    def test_missing_lineage_is_fatal(self):
        payload = piimb_row()
        del payload["source_uid"]

        with pytest.raises(SelectionError, match="source_uid"):
            normalize_piimb_row(payload, TASK_PIIMB_EN)

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_lineage_is_fatal(self, empty):
        with pytest.raises(SelectionError, match="source_uid"):
            normalize_piimb_row(piimb_row(source_uid=empty), TASK_PIIMB_EN)

    def test_same_text_across_tasks_hashes_identically(self):
        english = normalize_piimb_row(piimb_row(), TASK_PIIMB_EN)
        multi = normalize_piimb_row(piimb_row(), TASK_PIIMB_MULTI)

        assert english.text_hash == multi.text_hash


class TestHashes:
    def test_text_hash_applies_no_normalization(self):
        assert text_hash("Ada ") != text_hash("Ada")
        assert text_hash("Ada") != text_hash("ada")

    def test_text_hash_is_utf8_of_the_exact_string(self):
        import hashlib

        assert text_hash("café") == hashlib.sha256("café".encode()).hexdigest()

    def test_row_digest_is_key_order_independent(self):
        assert row_digest({"a": 1, "b": 2}) == row_digest({"b": 2, "a": 1})

    def test_row_digest_tracks_any_field_change(self):
        assert row_digest(openpii_row()) != row_digest(openpii_row(language="de"))


class TestLaziness:
    def test_importing_selection_does_not_import_datasets(self):
        probe = (
            "import sys; import pii_masking.benchmark_selection; print('datasets' in sys.modules)"
        )
        env = dict(os.environ, PYTHONPATH=str(EXAMPLES_ROOT))
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
        )

        assert result.stdout.strip() == "False"
