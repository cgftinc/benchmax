"""Tests for trace adapter data models and SSRF validation."""

import pytest

from benchmax.traces.adapter import (
    DetectedSystemPrompt,
    DetectedTool,
    DetectedTools,
    LangfuseCredentials,
    NormalizedTrace,
    ToolCall,
    TraceCredentials,
    TraceMessage,
    TraceProject,
    validate_provider_url,
)


class TestTraceCredentials:
    def test_repr_redacts_key(self):
        creds = TraceCredentials(api_key="sk-1234567890abcdef")
        assert "sk****" in repr(creds)
        assert "1234567890" not in repr(creds)

    def test_repr_short_key(self):
        creds = TraceCredentials(api_key="abc")
        assert "****" in repr(creds)

    def test_to_headers(self):
        creds = TraceCredentials(api_key="test-key")
        assert creds.to_headers() == {"Authorization": "Bearer test-key"}


class TestLangfuseCredentials:
    def test_repr_redacts_both_keys(self):
        creds = LangfuseCredentials(
            api_key="pk-lf-12345",
            secret_key="sk-lf-67890",
        )
        r = repr(creds)
        assert "pk****" in r
        assert "sk****" in r
        assert "12345" not in r
        assert "67890" not in r

    def test_basic_auth_headers(self):
        import base64

        creds = LangfuseCredentials(
            api_key="pk-test",
            secret_key="sk-test",
        )
        headers = creds.to_headers()
        expected = base64.b64encode(b"pk-test:sk-test").decode()
        assert headers == {"Authorization": f"Basic {expected}"}

    def test_validates_host_on_construction(self):
        with pytest.raises(ValueError, match="HTTPS"):
            LangfuseCredentials(
                api_key="pk-test",
                host="http://evil.com",
            )


class TestValidateProviderUrl:
    def test_rejects_http(self):
        with pytest.raises(ValueError, match="HTTPS"):
            validate_provider_url("http://example.com")

    def test_accepts_https(self):
        validate_provider_url("https://cloud.langfuse.com")

    def test_rejects_metadata_endpoint(self):
        with pytest.raises(ValueError, match="metadata"):
            validate_provider_url("https://169.254.169.254")

    def test_rejects_google_metadata(self):
        with pytest.raises(ValueError, match="metadata"):
            validate_provider_url("https://metadata.google.internal")

    def test_rejects_private_ip_10(self):
        with pytest.raises(ValueError, match="private/reserved"):
            validate_provider_url("https://10.0.0.1")

    def test_rejects_private_ip_172(self):
        with pytest.raises(ValueError, match="private/reserved"):
            validate_provider_url("https://172.16.0.1")

    def test_rejects_private_ip_192(self):
        with pytest.raises(ValueError, match="private/reserved"):
            validate_provider_url("https://192.168.1.1")

    def test_rejects_loopback(self):
        with pytest.raises(ValueError, match="private/reserved"):
            validate_provider_url("https://127.0.0.1")

    def test_rejects_empty_hostname(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_provider_url("https://")


class TestToolCall:
    def test_arguments_dict_parses_json(self):
        tc = ToolCall(name="search", arguments='{"query": "test"}')
        assert tc.arguments_dict() == {"query": "test"}

    def test_arguments_dict_default(self):
        tc = ToolCall(name="search")
        assert tc.arguments_dict() == {}

    def test_arguments_dict_invalid_json(self):
        tc = ToolCall(name="search", arguments="not json")
        assert tc.arguments_dict() == {}

    def test_arguments_dict_non_dict_json(self):
        tc = ToolCall(name="search", arguments="[1, 2, 3]")
        assert tc.arguments_dict() == {}


class TestTraceMessage:
    def test_to_dict_with_tool_calls(self):
        msg = TraceMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="search", arguments='{"q": "x"}', id="tc1")],
        )
        d = msg.to_dict()
        assert len(d["tool_calls"]) == 1
        tc = d["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == '{"q": "x"}'
        assert tc["id"] == "tc1"

    def test_to_dict_with_tool_response_fields(self):
        msg = TraceMessage(
            role="tool",
            content="result",
            tool_call_id="tc1",
            name="search",
        )
        d = msg.to_dict()
        assert d["tool_call_id"] == "tc1"
        assert d["name"] == "search"


class TestNormalizedTrace:
    def test_roundtrip_serialization(self):
        trace = NormalizedTrace(
            id="tr-1",
            messages=[
                TraceMessage(role="user", content="hi"),
                TraceMessage(
                    role="assistant",
                    content="hello",
                    tool_calls=[ToolCall(name="greet", arguments="{}")],
                ),
            ],
            scores={"quality": 0.9},
            metadata={"source": "test"},
            timestamp="2024-01-01T00:00:00Z",
        )
        d = trace.to_dict()
        restored = NormalizedTrace.from_dict(d)
        assert restored.id == trace.id
        assert len(restored.messages) == 2
        assert restored.messages[0].role == "user"
        assert restored.messages[1].tool_calls[0].name == "greet"
        assert restored.scores == {"quality": 0.9}
        assert restored.timestamp == "2024-01-01T00:00:00Z"

    def test_from_dict_minimal(self):
        trace = NormalizedTrace.from_dict({"id": "tr-2", "messages": []})
        assert trace.id == "tr-2"
        assert trace.messages == []
        assert trace.scores == {}

    def test_roundtrip_with_errors(self):
        trace = NormalizedTrace(
            id="tr-3",
            messages=[],
            errors=["extraction failed"],
        )
        d = trace.to_dict()
        restored = NormalizedTrace.from_dict(d)
        assert restored.errors == ["extraction failed"]
