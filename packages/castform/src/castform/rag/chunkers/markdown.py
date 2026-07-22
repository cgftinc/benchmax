"""Markdown document chunker with header-aware splitting."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from castform.rag.chunkers.models import Chunk, ChunkCollection

# Matches lines that are JS/TS import statements, e.g.:
#   import Foo from "../path/to/file.mdx"
#   import { Foo, Bar } from '../module'
_MDX_IMPORT_RE = re.compile(r"^\s*import\s+.*\bfrom\s+['\"].*['\"]\s*$", re.MULTILINE)

# Matches ES module export statements at the start of a line, including multi-line blocks, e.g.:
#   export const metadata = { ... }   (single-line)
#   export const metadata = {         (multi-line — matched up to next blank line / heading / EOF)
#     title: "foo",
#   }
#   export default function ...
_MDX_EXPORT_BLOCK_RE = re.compile(
    r"^\s*export\s+(?:const|let|var|default)\s+.*?(?=\n\n|\n#|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Matches JSX self-closing component tags whose name starts with an uppercase letter
# (which distinguishes React components from HTML void elements like <br />, <hr />, <img />).
# Examples matched:   <MyComponent />   <DetailSetUpReverseProxy />
# Examples NOT matched: <br />  <hr />  <img src="x" />
_JSX_SELF_CLOSING_RE = re.compile(r"<[A-Z][A-Za-z0-9]*(?:\s+[^>]*)?\s*/\s*>")

# Matches JSX block component tags that wrap only whitespace, e.g.:
#   <MyWrapper>   </MyWrapper>
#   <MyWrapper>\n\n</MyWrapper>
# The component name must start with uppercase (React convention).
_JSX_EMPTY_BLOCK_RE = re.compile(r"<([A-Z][A-Za-z0-9]*)[^>]*>\s*</\1>", re.DOTALL)

# Matches HTML comments: <!-- ... -->
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Code fence delimiter used to find protected regions.
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _preprocess_mdx(text: str) -> str:
    """Strip MDX-specific boilerplate from a document before chunking.

    This removes content that adds no informational value to chunks:
    - JS/TS import statements (``import X from "..."``).
    - ES module export statements (``export const ...``).
    - JSX self-closing component tags whose name is PascalCase (React components).
    - JSX block component tags that contain only whitespace children.
    - HTML comments (``<!-- ... -->``).

    Content inside markdown code fences (``` blocks) is intentionally left
    untouched so that legitimate code examples are preserved.

    Args:
        text: Raw MDX (or plain Markdown) source text.

    Returns:
        Cleaned text with MDX boilerplate removed and excess blank lines
        collapsed to at most two consecutive newlines.
    """
    # Split on code-fence boundaries so we can process prose regions only.
    # Parts at even indices (0, 2, 4, …) are outside fences; odd indices are inside.
    parts = _CODE_FENCE_RE.split(text)

    cleaned_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside a code fence — reconstruct the fence delimiters and leave content alone.
            cleaned_parts.append("```" + part)
        else:
            # Outside a code fence — apply MDX stripping.
            part = _HTML_COMMENT_RE.sub("", part)
            part = _MDX_IMPORT_RE.sub("", part)
            part = _MDX_EXPORT_BLOCK_RE.sub("", part)
            part = _JSX_EMPTY_BLOCK_RE.sub("", part)
            part = _JSX_SELF_CLOSING_RE.sub("", part)
            cleaned_parts.append(part)

    result = "".join(cleaned_parts)

    # Collapse runs of more than two consecutive blank lines.
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


@dataclass
class _MutableSection:
    """Mutable intermediate representation used during chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


class MarkdownChunker:
    """Chunker for markdown documents with header-aware splitting.

    This chunker implements a 3-stage pipeline:
    1. Split by markdown headers (H1, H2, H3), preserving hierarchy in metadata
    2. Fuse adjacent short sections to avoid over-fragmentation
    3. Split remaining large sections using recursive character splitting

    Example:
        >>> chunker = MarkdownChunker(min_char=1024, max_char=2048)
        >>> chunks = chunker.chunk(markdown_text)
        >>> for chunk in chunks:
        ...     print(chunk.content, chunk.metadata_dict)
    """

    def __init__(
        self,
        min_char: int = 1024,
        max_char: int = 2048,
        chunk_overlap: int = 128,
    ) -> None:
        """Initialize the markdown chunker.

        Args:
            min_char: Minimum characters per chunk. Sections shorter than this
                will be fused with adjacent sections.
            max_char: Maximum characters per chunk. Sections longer than this
                will be split using recursive character splitting.
            chunk_overlap: Number of overlapping characters between chunks
                when splitting large sections.
        """
        self.min_char = min_char
        self.max_char = max_char
        self.chunk_overlap = chunk_overlap
        self._seen_hashes: set[str] = set()

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_char,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(
        self,
        content: str,
        file: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        preprocess_mdx: bool = False,
    ) -> list[Chunk]:
        """Chunk a markdown document.

        Args:
            content: Raw markdown text.
            file: Optional file path to include in metadata.
            extra_metadata: Optional additional metadata to include in all chunks.
            preprocess_mdx: If True, run :func:`_preprocess_mdx` on the content
                before header splitting to remove JSX/import boilerplate.
                Set automatically by :meth:`chunk_file` and :meth:`chunk_folder`
                for ``.mdx`` files.

        Returns:
            List of Chunk objects with content and metadata.
            Metadata includes header hierarchy (h1, h2, h3), index, and file if provided.
        """
        if preprocess_mdx:
            content = _preprocess_mdx(content)

        sections = self._split_by_headers(content)

        if not sections:
            if content.strip():
                sections = [_MutableSection(content=content, metadata={})]
            else:
                return []

        fused = self._fuse_short_sections(sections)
        split_sections = self._split_large_sections(fused)

        # Build final immutable chunks with all metadata
        chunks: list[Chunk] = []
        for idx, section in enumerate(split_sections):
            metadata = section.metadata.copy()
            metadata["index"] = idx
            if file is not None:
                metadata["file"] = file
            if extra_metadata:
                metadata.update(extra_metadata)
            chunk = Chunk(
                content=section.content,
                metadata=tuple(metadata.items()),
            )

            # Check for duplicate chunk hash
            if chunk.hash in self._seen_hashes:
                file_info = f" in file '{file}'" if file else ""
                raise ValueError(
                    f"Duplicate chunk detected{file_info}. "
                    f"Chunk with hash '{chunk.hash}' has already been processed. "
                    f"This may indicate duplicate content or repeated processing of the same file."
                )

            self._seen_hashes.add(chunk.hash)
            chunks.append(chunk)

        return chunks

    def reset_hash_tracking(self) -> None:
        """Reset the hash tracking to allow reprocessing of files.

        Use this if you want to reuse the same chunker instance
        and allow processing the same content again.
        """
        self._seen_hashes.clear()

    def chunk_file(self, file_path: str | Path) -> list[Chunk]:
        """Chunk a markdown file.

        ``.mdx`` files are automatically preprocessed to strip JSX/import
        boilerplate before chunking (see :func:`_preprocess_mdx`).

        Args:
            file_path: Path to the markdown file.

        Returns:
            List of Chunk objects with 'file' added to metadata.
        """
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")
        is_mdx = file_path.suffix.lower() == ".mdx"
        return self.chunk(content, file=str(file_path.name), preprocess_mdx=is_mdx)

    def chunk_folder(
        self,
        folder_path: str | Path,
        file_extensions: list[str] | str | None = None,
    ) -> ChunkCollection:
        """Chunk all markdown files in a folder recursively.

        Args:
            folder_path: Path to the folder to process.
            file_extensions: File extension(s) to filter. Can be a string or list of strings.
                Defaults to [".md", ".mdx"] if not provided.

        Returns:
            ChunkCollection with all chunks from all files, preserving file structure.
        """
        folder_path = Path(folder_path).resolve()
        all_chunks: list[Chunk] = []

        if file_extensions is None:
            file_extensions = [".md", ".mdx"]
        elif isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        files = []
        for ext in file_extensions:
            files.extend(folder_path.rglob(f"*{ext}"))

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                relative_path = str(file_path.relative_to(folder_path))
                is_mdx = file_path.suffix.lower() == ".mdx"
                chunks = self.chunk(content, file=relative_path, preprocess_mdx=is_mdx)
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return ChunkCollection(all_chunks)

    def _split_by_headers(self, content: str) -> list[_MutableSection]:
        """Split content by markdown headers."""
        splits = self._header_splitter.split_text(content)
        return [
            _MutableSection(content=doc.page_content, metadata=dict(doc.metadata)) for doc in splits
        ]

    def _fuse_short_sections(self, sections: list[_MutableSection]) -> list[_MutableSection]:
        """Fuse adjacent short sections until they reach min_char threshold."""
        if not sections:
            return []

        def single_pass(secs: list[_MutableSection]) -> list[_MutableSection]:
            if not secs:
                return []

            fused: list[_MutableSection] = []
            current = _MutableSection(
                content=secs[0].content,
                metadata=secs[0].metadata.copy(),
            )

            for i in range(1, len(secs)):
                next_section = secs[i]
                combined_length = len(current) + len(next_section) + 2

                if len(current) < self.min_char and combined_length <= self.max_char:
                    current.content = current.content + "\n\n" + next_section.content
                    for key, value in next_section.metadata.items():
                        if key not in current.metadata:
                            current.metadata[key] = value
                else:
                    fused.append(current)
                    current = _MutableSection(
                        content=next_section.content,
                        metadata=next_section.metadata.copy(),
                    )

            fused.append(current)
            return fused

        result = sections
        while True:
            fused = single_pass(result)
            if len(fused) == len(result):
                break
            result = fused

        return result

    def _split_large_sections(self, sections: list[_MutableSection]) -> list[_MutableSection]:
        """Split sections that exceed max_char."""
        result: list[_MutableSection] = []

        for section in sections:
            sub_chunks = self._text_splitter.split_text(section.content)
            for sub_chunk in sub_chunks:
                result.append(_MutableSection(content=sub_chunk, metadata=section.metadata.copy()))

        return result
