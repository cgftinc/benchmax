"""content_field sugar on the Turbopuffer source — BYO namespaces
whose text attribute isn't named `content`."""

from __future__ import annotations

import pytest
from castform.rag.corpus.turbopuffer.namespace import resolve_content_attr


class TestResolveContentAttr:
    def test_field_becomes_single_attr(self):
        assert resolve_content_attr(None, "description") == ["description"]

    def test_passthrough_without_field(self):
        assert resolve_content_attr(["title", "content"], None) == [
            "title",
            "content",
        ]
        assert resolve_content_attr(None, None) is None

    def test_empty_field_is_noop(self):
        # Platform codegen passes "" for an unset optional resource field.
        assert resolve_content_attr(None, "") is None

    def test_agreeing_values_pass(self):
        assert resolve_content_attr(["description"], "description") == ["description"]

    def test_conflicting_values_raise(self):
        with pytest.raises(ValueError, match="conflicts"):
            resolve_content_attr(["title"], "description")
