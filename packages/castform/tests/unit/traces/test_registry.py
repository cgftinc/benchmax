"""Tests for the trace adapter registry."""

import pytest
from castform.traces.braintrust.adapter import BraintrustTraceAdapter
from castform.traces.registry import get_adapter_class


class TestGetAdapterClass:
    def test_braintrust_returns_class(self):
        cls = get_adapter_class("braintrust")
        assert cls is BraintrustTraceAdapter

    def test_returned_class_is_constructable(self):
        cls = get_adapter_class("braintrust")
        adapter = cls(api_key="test-key")
        assert hasattr(adapter, "connect")
        assert hasattr(adapter, "list_projects")
        assert hasattr(adapter, "fetch_traces")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown trace provider"):
            get_adapter_class("nonexistent")

    def test_error_lists_available_providers(self):
        with pytest.raises(ValueError, match="braintrust"):
            get_adapter_class("nonexistent")
