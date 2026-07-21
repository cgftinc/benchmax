"""Tests for message content helpers."""

from __future__ import annotations

import base64

import pytest

from benchmax.envs.base.content import (
    content_preview,
    image_to_data_uri,
    iter_image_refs,
    message_text,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 16
_GIF_BYTES = b"GIF89a" + b"\x00" * 16
_WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 16


class TestMessageText:
    def test_str_content(self):
        assert message_text({"role": "user", "content": "hello"}) == "hello"

    def test_part_list_preserves_order_across_key_styles(self):
        message = {
            "role": "assistant",
            "content": [
                {"text": "first"},
                {"content": "second"},
                {"text": "third"},
            ],
        }
        assert message_text(message) == "first\nsecond\nthird"

    def test_empty_content(self):
        assert message_text({"role": "user", "content": ""}) == ""

    def test_none_content(self):
        assert message_text({"role": "user", "content": None}) == ""

    def test_missing_content_key(self):
        assert message_text({"role": "user"}) == ""

    def test_image_only_content_is_empty(self):
        message = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }
        assert message_text(message) == ""

    def test_mixed_text_and_image_content(self):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "text", "text": "what is it"},
            ],
        }
        assert message_text(message) == "look at this\nwhat is it"

    def test_non_dict_part_skipped(self):
        message = {"role": "user", "content": [42, {"text": "kept"}]}
        assert message_text(message) == "kept"

    def test_non_string_text_value_skipped(self):
        message = {"role": "assistant", "content": [{"text": 42}, {"text": "kept"}]}
        assert message_text(message) == "kept"

    def test_empty_string_text_value_preserved(self):
        message = {"role": "assistant", "content": [{"text": ""}, {"text": "kept"}]}
        assert message_text(message) == "\nkept"

    def test_unknown_part_type_skipped(self):
        # a non-text modality part must not leak its "text" key into the text
        message = {
            "role": "user",
            "content": [
                {"type": "audio", "text": "transcript"},
                {"type": "text", "text": "kept"},
            ],
        }
        assert message_text(message) == "kept"

    def test_explicit_null_type_skipped(self):
        # an explicit null type is a declared (non-text) type, not an absent key
        message = {
            "role": "user",
            "content": [
                {"type": None, "text": "x"},
                {"type": "text", "text": "kept"},
            ],
        }
        assert message_text(message) == "kept"

    def test_type_absent_text_part_still_read(self):
        # regression: a part with no "type" key keeps the lenient text/content read
        message = {
            "role": "user",
            "content": [{"text": "first"}, {"content": "second"}],
        }
        assert message_text(message) == "first\nsecond"


class TestContentPreview:
    def test_normal_truncation(self):
        result = content_preview("a" * 50, 10)
        assert result == "a" * 9 + "…"
        assert len(result) == 10

    def test_no_truncation_needed(self):
        assert content_preview("short", 10) == "short"

    def test_limit_of_one_accepted(self):
        result = content_preview("hello", 1)
        assert len(result) <= 1

    def test_truncation_marker_present_for_limit_one(self):
        assert content_preview("hello", 1) == "…"

    def test_truncation_marker_present_for_limit_two(self):
        result = content_preview("hello", 2)
        assert result == "h…"
        assert len(result) == 2

    def test_truncation_marker_present_for_limit_three(self):
        result = content_preview("hello", 3)
        assert result == "he…"
        assert len(result) == 3

    def test_limit_zero_raises(self):
        with pytest.raises(ValueError):
            content_preview("hello", 0)

    def test_limit_negative_raises(self):
        with pytest.raises(ValueError):
            content_preview("hello", -5)

    def test_image_part_renders_placeholder(self):
        content = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
        result = content_preview(content, 50)
        assert "[image]" in result

    def test_single_line_with_embedded_newlines(self):
        result = content_preview("line one\nline two\nline three", 100)
        assert "\n" not in result

    def test_single_line_truncated_still_has_no_newline(self):
        result = content_preview("line one\nline two\nline three", 10)
        assert "\n" not in result
        assert len(result) <= 10

    def test_malformed_content_never_raises(self):
        class Weird:
            def __repr__(self):
                return "<Weird object>"

        result = content_preview(Weird(), 50)
        assert isinstance(result, str)
        assert len(result) <= 50

    def test_malformed_list_item_never_raises(self):
        result = content_preview([object(), {"unexpected": "shape"}], 50)
        assert isinstance(result, str)
        assert len(result) <= 50

    def test_raising_repr_never_raises(self):
        class Unrepresentable:
            def __repr__(self):
                raise RuntimeError("boom")

        result = content_preview(Unrepresentable(), 50)
        assert isinstance(result, str)
        assert len(result) <= 50

    def test_none_content(self):
        assert content_preview(None, 10) == ""


class TestIterImageRefs:
    def test_multiple_messages_multiple_images_order_preserved(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "img1"}},
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "img2"}},
                ],
            },
            {
                "role": "assistant",
                "content": "text only, no images",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "img3"}},
                ],
            },
        ]
        assert list(iter_image_refs(messages)) == ["img1", "img2", "img3"]

    def test_str_content_message_contributes_nothing(self):
        messages = [{"role": "user", "content": "just text"}]
        assert list(iter_image_refs(messages)) == []

    def test_malformed_image_url_parts_skipped(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url"},  # missing image_url dict
                    {"type": "image_url", "image_url": {}},  # missing url
                    {"type": "image_url", "image_url": {"url": 123}},  # non-str url
                    {"type": "image_url", "image_url": "not-a-dict"},
                    {"type": "image_url", "image_url": {"url": "good"}},
                ],
            }
        ]
        assert list(iter_image_refs(messages)) == ["good"]

    def test_no_messages(self):
        assert list(iter_image_refs([])) == []


class TestImageToDataUri:
    def test_real_path(self, tmp_path):
        path = tmp_path / "tiny.png"
        path.write_bytes(_PNG_BYTES)
        result = image_to_data_uri(path)
        assert result.startswith("data:image/png;base64,")
        encoded = result.split(",", 1)[1]
        assert base64.b64decode(encoded) == _PNG_BYTES

    def test_bytes_input_directly(self):
        result = image_to_data_uri(_PNG_BYTES)
        assert result.startswith("data:image/png;base64,")

    def test_png_signature(self):
        assert image_to_data_uri(_PNG_BYTES).startswith("data:image/png;base64,")

    def test_jpeg_signature(self):
        assert image_to_data_uri(_JPEG_BYTES).startswith("data:image/jpeg;base64,")

    def test_gif_signature(self):
        assert image_to_data_uri(_GIF_BYTES).startswith("data:image/gif;base64,")

    def test_webp_signature(self):
        assert image_to_data_uri(_WEBP_BYTES).startswith("data:image/webp;base64,")

    def test_data_uri_passthrough(self):
        source = "data:image/png;base64,abc123"
        assert image_to_data_uri(source) == source

    def test_https_passthrough(self):
        source = "https://example.com/image.png"
        assert image_to_data_uri(source) == source

    def test_unrecognized_bytes_raises(self):
        with pytest.raises(ValueError):
            image_to_data_uri(b"not an image")
