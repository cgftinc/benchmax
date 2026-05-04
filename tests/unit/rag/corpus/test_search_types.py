"""Tests for search schema types: validate_search_spec_shape and required_operators."""

from __future__ import annotations

from benchmax.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
    SearchSpec,
    required_operators,
    validate_search_spec_shape,
)

# ---------------------------------------------------------------------------
# validate_search_spec_shape — lexical mode
# ---------------------------------------------------------------------------


class TestValidateLexical:
    def test_valid_lexical(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="hello")
        assert validate_search_spec_shape(spec) == []

    def test_empty_text_query(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="")
        errors = validate_search_spec_shape(spec)
        assert any("text_query" in e for e in errors)

    def test_whitespace_text_query(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="   ")
        errors = validate_search_spec_shape(spec)
        assert any("text_query" in e for e in errors)

    def test_missing_text_query(self):
        spec = SearchSpec(mode="lexical", top_k=5)
        errors = validate_search_spec_shape(spec)
        assert any("text_query" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_search_spec_shape — vector mode
# ---------------------------------------------------------------------------


class TestValidateVector:
    def test_valid_vector(self):
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2])
        assert validate_search_spec_shape(spec) == []

    def test_empty_vector_query(self):
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[])
        errors = validate_search_spec_shape(spec)
        assert any("vector_query" in e for e in errors)

    def test_missing_vector_query(self):
        spec = SearchSpec(mode="vector", top_k=5)
        errors = validate_search_spec_shape(spec)
        assert any("vector_query" in e for e in errors)

    def test_none_vector_query(self):
        spec = SearchSpec(mode="vector", top_k=5, vector_query=None)
        errors = validate_search_spec_shape(spec)
        assert any("vector_query" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_search_spec_shape — hybrid mode
# ---------------------------------------------------------------------------


class TestValidateHybrid:
    def test_valid_hybrid(self):
        spec = SearchSpec(mode="hybrid", top_k=5, text_query="hi", vector_query=[0.1])
        assert validate_search_spec_shape(spec) == []

    def test_missing_text_query(self):
        spec = SearchSpec(mode="hybrid", top_k=5, vector_query=[0.1])
        errors = validate_search_spec_shape(spec)
        assert any("hybrid" in e.lower() or "text_query" in e for e in errors)

    def test_missing_vector_query(self):
        spec = SearchSpec(mode="hybrid", top_k=5, text_query="hi")
        errors = validate_search_spec_shape(spec)
        assert any("hybrid" in e.lower() or "vector_query" in e for e in errors)

    def test_missing_both(self):
        spec = SearchSpec(mode="hybrid", top_k=5)
        errors = validate_search_spec_shape(spec)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# validate_search_spec_shape — top_k and mode
# ---------------------------------------------------------------------------


class TestValidateTopKAndMode:
    def test_top_k_zero(self):
        spec = SearchSpec(mode="lexical", top_k=0, text_query="q")
        errors = validate_search_spec_shape(spec)
        assert any("top_k" in e for e in errors)

    def test_top_k_negative(self):
        spec = SearchSpec(mode="lexical", top_k=-1, text_query="q")
        errors = validate_search_spec_shape(spec)
        assert any("top_k" in e for e in errors)

    def test_top_k_float(self):
        spec = SearchSpec(mode="lexical", top_k=3.5, text_query="q")  # type: ignore[arg-type]
        errors = validate_search_spec_shape(spec)
        assert any("top_k" in e for e in errors)

    def test_invalid_mode(self):
        spec = SearchSpec(mode="nonexistent", top_k=5, text_query="q")  # type: ignore[arg-type]
        errors = validate_search_spec_shape(spec)
        assert any("mode" in e for e in errors)

    def test_valid_returns_empty(self):
        spec = SearchSpec(mode="lexical", top_k=10, text_query="hello world")
        assert validate_search_spec_shape(spec) == []


# ---------------------------------------------------------------------------
# required_operators
# ---------------------------------------------------------------------------


class TestRequiredOperators:
    def test_none(self):
        field_ops, logical_ops = required_operators(None)
        assert field_ops == set()
        assert logical_ops == set()

    def test_single_field_predicate(self):
        pred = FieldPredicate(field="x", op="eq", value=1)
        field_ops, logical_ops = required_operators(pred)
        assert field_ops == {"eq"}
        assert logical_ops == set()

    def test_and_predicate(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                FieldPredicate(field="b", op="gte", value=5),
            )
        )
        field_ops, logical_ops = required_operators(pred)
        assert field_ops == {"eq", "gte"}
        assert logical_ops == {"and"}

    def test_nested_and_or(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                OrPredicate(
                    clauses=(
                        FieldPredicate(field="b", op="in", value=[1, 2]),
                        FieldPredicate(field="c", op="lte", value=10),
                    )
                ),
            )
        )
        field_ops, logical_ops = required_operators(pred)
        assert field_ops == {"eq", "in", "lte"}
        assert logical_ops == {"and", "or"}

    def test_not_predicate(self):
        pred = NotPredicate(clause=FieldPredicate(field="x", op="eq", value="deleted"))
        field_ops, logical_ops = required_operators(pred)
        assert field_ops == {"eq"}
        assert logical_ops == {"not"}
