"""Tests for Castform MarkdownChunker preprocessing."""

from __future__ import annotations

from castform.rag.chunkers.markdown import _preprocess_mdx


class TestPreprocessMdx:
    """Unit tests for the _preprocess_mdx function."""

    # ------------------------------------------------------------------ imports

    def test_import_lines_are_stripped(self):
        """JS/TS import statements should be removed entirely."""
        src = (
            'import MyComponent from "../components/MyComponent"\n'
            "import { Foo } from './foo'\n"
            "\nSome real content."
        )
        result = _preprocess_mdx(src)
        assert "import" not in result
        assert "Some real content." in result

    def test_import_inside_code_fence_is_preserved(self):
        """import lines inside a code fence must NOT be removed."""
        src = 'Normal prose.\n\n```js\nimport React from "react";\n```\n\nMore prose.'
        result = _preprocess_mdx(src)
        assert 'import React from "react";' in result

    # --------------------------------------------------------- self-closing JSX

    def test_jsx_self_closing_component_removed(self):
        """PascalCase self-closing JSX tags like <ComponentName /> are removed."""
        src = "Intro.\n\n<DetailSetUpReverseProxy />\n\nBody text."
        result = _preprocess_mdx(src)
        assert "<DetailSetUpReverseProxy" not in result
        assert "Body text." in result

    def test_html_void_element_preserved(self):
        """Lowercase void elements like <br /> must NOT be removed."""
        src = "Line one.<br />\nLine two."
        result = _preprocess_mdx(src)
        assert "<br />" in result

    def test_jsx_self_closing_with_props_removed(self):
        """PascalCase self-closing tags with attributes are also removed."""
        src = 'Content.\n<MyWidget foo="bar" />\nMore content.'
        result = _preprocess_mdx(src)
        assert "<MyWidget" not in result
        assert "More content." in result

    # ----------------------------------------------------- empty-block JSX

    def test_empty_block_jsx_removed(self):
        """<Wrapper>\\n</Wrapper> (only whitespace children) is removed."""
        src = "Before.\n\n<Wrapper>\n</Wrapper>\n\nAfter."
        result = _preprocess_mdx(src)
        assert "<Wrapper>" not in result
        assert "After." in result

    def test_empty_block_jsx_multiline_whitespace_removed(self):
        """<Wrapper> with multiple blank lines as children is removed."""
        src = "Text.\n<MyWrapper>\n\n\n</MyWrapper>\nMore."
        result = _preprocess_mdx(src)
        assert "<MyWrapper>" not in result
        assert "More." in result

    def test_nonempty_block_jsx_preserved(self):
        """A JSX block with actual content children is NOT removed."""
        src = "<Callout>\nThis is important.\n</Callout>"
        result = _preprocess_mdx(src)
        # The PascalCase self-closing regex won't match; empty-block regex won't
        # match either because there is non-whitespace content between the tags.
        assert "This is important." in result

    # --------------------------------------------------------- HTML comments

    def test_html_comment_removed(self):
        """HTML comments <!-- ... --> should be stripped."""
        src = "Before.\n<!-- This is a comment -->\nAfter."
        result = _preprocess_mdx(src)
        assert "<!--" not in result
        assert "After." in result

    def test_multiline_html_comment_removed(self):
        """Multi-line HTML comments are also removed."""
        src = "Start.\n<!-- \n  Multi-line\n  comment\n-->\nEnd."
        result = _preprocess_mdx(src)
        assert "<!--" not in result
        assert "End." in result

    # -------------------------------------------------- code fence preservation

    def test_code_fence_content_preserved(self):
        """Content inside ``` fences is left completely untouched."""
        src = (
            "Prose.\n\n"
            "```python\n"
            "import os\n"
            "<br />\n"
            "<!-- a comment -->\n"
            "<MyComponent />\n"
            "```\n\n"
            "More prose."
        )
        result = _preprocess_mdx(src)
        # These would normally be stripped but must survive inside the fence.
        assert "import os" in result
        assert "<br />" in result
        assert "<!-- a comment -->" in result
        assert "<MyComponent />" in result
        assert "More prose." in result

    # ------------------------------------------------- blank line collapsing

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive newlines are collapsed to at most two."""
        src = "First paragraph.\n\n\n\nSecond paragraph."
        result = _preprocess_mdx(src)
        # Should not contain 3+ consecutive newlines.
        assert "\n\n\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_two_blank_lines_unchanged(self):
        """Exactly two consecutive newlines (one blank line) are left as-is."""
        src = "First.\n\nSecond."
        result = _preprocess_mdx(src)
        assert "First.\n\nSecond." in result
