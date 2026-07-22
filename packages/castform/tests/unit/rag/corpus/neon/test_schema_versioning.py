"""Contract #1: physical table naming + versioned-replace model.

Identifier helpers are frozen here (pass); DDL assembly and activate/rollback are
xfail skeletons filled by Slice 2.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.schema import (
    DISTANCE_METRIC,
    EMBEDDING_DIM,
    NeonTableSpec,
    activate_version_sql,
    create_table_ddl,
    index_names,
    physical_table_name,
    rollback_version_sql,
)


def test_embedding_dim_and_metric_frozen() -> None:
    assert EMBEDDING_DIM == 3072
    assert DISTANCE_METRIC == "cosine"


def test_physical_table_name() -> None:
    assert physical_table_name("mycorpus", 3) == "mycorpus__v3"


def test_index_names_are_per_version() -> None:
    names = index_names("mycorpus", 3)
    assert names == {
        "hnsw": "mycorpus__v3_hnsw",
        "bm25": "mycorpus__v3_bm25",
        "meta_gin": "mycorpus__v3_meta_gin",
        "tsv_gin": "mycorpus__v3_tsv_gin",
    }


def test_table_spec_defaults() -> None:
    spec = NeonTableSpec(logical_name="mycorpus", version=1)
    assert spec.embedding_dim == 3072
    assert spec.distance_metric == "cosine"
    assert spec.text_search_config == "pg_catalog.english"


@pytest.mark.xfail(reason="DDL assembly built in Slice 2", strict=False)
def test_create_table_ddl() -> None:
    spec = NeonTableSpec(logical_name="mycorpus", version=1)
    ddl = create_table_ddl(spec)
    assert "mycorpus__v1" in ddl
    assert "vector(3072)" in ddl


@pytest.mark.xfail(reason="version activation built in Slice 2", strict=False)
def test_activate_is_single_transaction() -> None:
    spec = NeonTableSpec(logical_name="mycorpus", version=2)
    stmts = activate_version_sql(spec)
    assert any("create or replace view" in s.lower() for s in stmts)


@pytest.mark.xfail(reason="version rollback built in Slice 2", strict=False)
def test_rollback_repoints_pointer() -> None:
    stmts = rollback_version_sql("mycorpus", 1)
    assert any("mycorpus__v1" in s for s in stmts)
