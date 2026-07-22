"""Contract #1: physical naming, DDL safety, and the versioned-replace lifecycle.

Identifier helpers, validation, length-safety, and lifecycle invariants are
frozen here (pass); DDL assembly and activate/rollback are xfail skeletons that
must raise NotImplementedError until Slice 2.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.schema import (
    ALLOWED_TEXT_SEARCH_CONFIGS,
    CREATE_CURRENT_POINTER_INDEX,
    DISTANCE_METRIC,
    EMBEDDING_DIM,
    MAX_IDENTIFIER_BYTES,
    PROVISIONAL_ANN_OPTIONS,
    VERSION_STATE_TRANSITIONS,
    NeonTableSpec,
    NeonVersionRecord,
    ReadGrantSpec,
    RetentionPolicy,
    activate_version_sql,
    create_table_ddl,
    index_names,
    physical_table_name,
    read_grant_sql,
    rollback_version_sql,
    validate_logical_name,
    validate_text_search_config,
    validate_version,
    view_name,
)


def test_embedding_dim_and_metric_frozen() -> None:
    assert EMBEDDING_DIM == 3072
    assert DISTANCE_METRIC == "cosine"


def test_physical_table_name() -> None:
    assert physical_table_name("mycorpus", 3) == "mycorpus__v3"


def test_index_names_are_per_version() -> None:
    names = index_names("mycorpus", 3)
    assert set(names) == {"ann", "bm25", "meta_gin", "scan", "tsv_gin"}
    assert names["scan"] == "mycorpus__v3_scan"


def test_long_names_are_length_safe() -> None:
    long_name = "corpus_" + "x" * 80
    table = physical_table_name(long_name, 12)
    assert len(table.encode()) <= MAX_IDENTIFIER_BYTES
    for name in index_names(long_name, 12).values():
        assert len(name.encode()) <= MAX_IDENTIFIER_BYTES
    assert table != physical_table_name("corpus_" + "y" * 80, 12)


def test_view_name_is_length_safe_and_stable() -> None:
    assert view_name("mycorpus") == "mycorpus"
    long_name = "corpus_" + "z" * 80
    assert len(view_name(long_name).encode()) <= MAX_IDENTIFIER_BYTES
    assert view_name(long_name) == view_name(long_name)  # deterministic


def test_validate_logical_name_rejects_bad() -> None:
    assert validate_logical_name("mycorpus") == "mycorpus"
    for bad in ("", "café", "a\tb"):
        with pytest.raises(ValueError):
            validate_logical_name(bad)


def test_validate_version_rejects_nonpositive_and_bool() -> None:
    assert validate_version(1) == 1
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            validate_version(bad)  # type: ignore[arg-type]


def test_validate_text_search_config_allowlist() -> None:
    assert validate_text_search_config("pg_catalog.english") == "pg_catalog.english"
    assert "pg_catalog.english" in ALLOWED_TEXT_SEARCH_CONFIGS
    with pytest.raises(ValueError):
        validate_text_search_config("english; DROP TABLE x")


def test_table_spec_defaults() -> None:
    spec = NeonTableSpec(logical_name="mycorpus", version=1)
    assert spec.embedding_dim == 3072
    assert spec.text_search_config == "pg_catalog.english"


def test_version_record_current_flag() -> None:
    rec = NeonVersionRecord(
        logical_name="mycorpus", version=2, state="activated", is_current=True
    )
    assert rec.is_current is True
    assert rec.retired_at is None


def test_state_transitions_frozen() -> None:
    assert VERSION_STATE_TRANSITIONS["building"] == ("ready",)
    assert set(VERSION_STATE_TRANSITIONS["ready"]) == {"activated", "retired"}
    assert VERSION_STATE_TRANSITIONS["retired"] == ()


def test_current_pointer_invariant_is_partial_unique() -> None:
    assert "UNIQUE INDEX" in CREATE_CURRENT_POINTER_INDEX
    assert "WHERE is_current" in CREATE_CURRENT_POINTER_INDEX


def test_retention_validates_minimum() -> None:
    assert RetentionPolicy().keep_activated >= 2
    with pytest.raises(ValueError):
        RetentionPolicy(keep_activated=1)
    with pytest.raises(ValueError):
        RetentionPolicy(keep_ready=0)


def test_ann_options_are_provisional() -> None:
    statuses = {opt["status"] for opt in PROVISIONAL_ANN_OPTIONS}
    assert statuses == {"provisional-primary", "provisional-fallback"}
    assert PROVISIONAL_ANN_OPTIONS[0]["access_method"] == "lakebase_ann"


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 2")
def test_create_table_ddl() -> None:
    create_table_ddl(NeonTableSpec(logical_name="mycorpus", version=1))


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 2")
def test_activate_version_sql_takes_grant() -> None:
    activate_version_sql(
        NeonTableSpec(logical_name="mycorpus", version=2),
        ReadGrantSpec(schema="corpora", view="mycorpus", ro_role="ro"),
    )


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 2")
def test_rollback_version_sql() -> None:
    rollback_version_sql("mycorpus", 1)


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 2")
def test_read_grant_sql() -> None:
    read_grant_sql(ReadGrantSpec(schema="corpora", view="mycorpus", ro_role="ro"))
