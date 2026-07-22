"""Inspector for navigating and visualizing chunk collections."""

import random
from typing import Any

from castform.rag.chunkers.models import Chunk, ChunkCollection


class ChunkInspector:
    """Inspector for navigating and visualizing ChunkCollection.

    Provides tree-based summaries and convenient sampling/reading methods.

    Example:
        >>> collection = chunker.chunk_folder("./docs")
        >>> inspector = ChunkInspector(collection)
        >>> inspector.summary(max_depth=3, max_files_per_folder=10)
        >>> inspector.sample_chunk(show_context=True)
    """

    def __init__(self, collection: ChunkCollection) -> None:
        """Initialize with a ChunkCollection.

        Args:
            collection: ChunkCollection to inspect
        """
        self.collection = collection
        self._build_tree_structure()

    def _build_tree_structure(self) -> None:
        """Build hierarchical file tree from flat file paths."""
        self.tree: dict[str, Any] = {}
        for file in self.collection.files:
            parts = file.split("/")
            current = self.tree
            for part in parts[:-1]:  # folders
                if part not in current:
                    current[part] = {}
                current = current[part]
            # Store file with chunk count
            chunks = self.collection.get_file_chunks(file)
            current[parts[-1]] = {"_chunks": len(chunks), "_file_path": file}

    def summary(self, max_depth: int = 3, max_files_per_folder: int = 10) -> None:
        """Print tree-based summary of collection.

        Args:
            max_depth: Maximum tree depth to display
            max_files_per_folder: Max files to show per folder before truncating
        """
        total_chunks = len(self.collection)
        total_files = len(self.collection.files)

        # Calculate chunk size statistics
        chunk_sizes = [len(chunk) for chunk in self.collection.chunks]
        if chunk_sizes:
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            mean_size = sum(chunk_sizes) // len(chunk_sizes)

            # Find the chunks with min and max sizes
            min_chunk = self.collection.chunks[chunk_sizes.index(min_size)]
            max_chunk = self.collection.chunks[chunk_sizes.index(max_size)]

            min_file = min_chunk.get_metadata("file", "unknown")
            min_index = min_chunk.get_metadata("index", "?")
            max_file = max_chunk.get_metadata("file", "unknown")
            max_index = max_chunk.get_metadata("index", "?")
        else:
            min_size = max_size = mean_size = 0
            min_file = max_file = "N/A"
            min_index = max_index = "N/A"

        print("ChunkCollection Summary")
        print(f"Total chunks: {total_chunks}")
        print(f"Total files: {total_files}")
        print("\nChunk Size Statistics:")
        print(f"  Min: {min_size} chars ({min_file}, chunk {min_index})")
        print(f"  Max: {max_size} chars ({max_file}, chunk {max_index})")
        print(f"  Mean: {mean_size} chars")
        print()
        print("File Structure:")
        self._print_tree(self.tree, depth=0, max_depth=max_depth, max_files=max_files_per_folder)

    def _print_tree(
        self, node: dict[str, Any], depth: int, max_depth: int, max_files: int, prefix: str = ""
    ) -> None:
        """Recursively print tree structure."""
        if depth >= max_depth:
            return

        items = list(node.items())
        files = [(k, v) for k, v in items if isinstance(v, dict) and "_chunks" in v]
        folders = [(k, v) for k, v in items if isinstance(v, dict) and "_chunks" not in v]

        # Print folders first
        for i, (name, subtree) in enumerate(folders):
            is_last_folder = (i == len(folders) - 1) and len(files) == 0
            connector = "└── " if is_last_folder else "├── "
            print(f"{prefix}{connector}{name}/")

            extension = "    " if is_last_folder else "│   "
            self._print_tree(subtree, depth + 1, max_depth, max_files, prefix + extension)

        # Print files
        displayed_files = files[:max_files]
        for i, (name, info) in enumerate(displayed_files):
            is_last = i == len(displayed_files) - 1 and i == len(files) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{name} ({info['_chunks']} chunks)")

        # Show truncation message
        if len(files) > max_files:
            print(f"{prefix}    ... {len(files) - max_files} more files")

    def read_chunk(
        self,
        chunk: Chunk,
        show_context_before: bool = False,
        show_context_after: bool = False,
        context_max_chars: int = 200,
    ) -> str:
        """Format a chunk as a string with optional context.

        Args:
            chunk: The chunk to format
            show_context_before: Whether to show chunks before
            show_context_after: Whether to show chunks after
            context_max_chars: Max chars to show for context chunks

        Returns:
            Formatted string representation of the chunk
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Length: {len(chunk)} chars")
        lines.append("=" * 80)

        show_context = show_context_before or show_context_after

        if show_context:
            ctx = self.collection.get_chunk_with_context(
                chunk,
                context_max_chars=context_max_chars,
                include_before=show_context_before,
                include_after=show_context_after,
            )

            if show_context_before:
                lines.append("\n--- Context Before ---")
                lines.append(ctx["prev_chunk_preview"])

            lines.append("\n--- Main Chunk ---")
            lines.append(ctx["chunk_content"])

            if show_context_after:
                lines.append("\n--- Context After ---")
                lines.append(ctx["next_chunk_preview"])
        else:
            lines.append(chunk.chunk_str())

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def read_file(self, file_path: str, max_chunks: int | None = None) -> str:
        """Read all chunks from a specific file.

        Args:
            file_path: Path to the file (as stored in metadata)
            max_chunks: Maximum chunks to display (None for all)

        Returns:
            Formatted string representation of the file's chunks
        """
        chunks = self.collection.get_file_chunks(file_path)

        if not chunks:
            return (
                f"File not found: {file_path}\n"
                f"Available files: {', '.join(self.collection.files[:5])}..."
            )

        lines = []
        lines.append("=" * 80)
        lines.append(f"File: {file_path}")
        lines.append(f"Total chunks: {len(chunks)}")
        lines.append("=" * 80)

        chunks_to_show = chunks[:max_chunks] if max_chunks else chunks

        for chunk in chunks_to_show:
            lines.append(str(chunk))

        if max_chunks and len(chunks) > max_chunks:
            lines.append(f"\n... {len(chunks) - max_chunks} more chunks in this file")

        return "\n".join(lines)

    def sample_chunk(
        self,
        show_context_before: bool = False,
        show_context_after: bool = False,
    ) -> str:
        """Randomly sample a chunk from the collection.

        Args:
            show_context_before: Whether to show chunks before
            show_context_after: Whether to show chunks after

        Returns:
            Formatted string representation of the sampled chunk
        """
        chunk = random.choice(self.collection.chunks)
        return self.read_chunk(
            chunk,
            show_context_before=show_context_before,
            show_context_after=show_context_after,
        )

    def sample_file(self, max_chunks: int | None = None) -> str:
        """Randomly sample a file from the collection.

        Args:
            max_chunks: Maximum chunks to display from the file

        Returns:
            Formatted string representation of the sampled file's chunks
        """
        file_path = random.choice(self.collection.files)
        return self.read_file(file_path, max_chunks=max_chunks)
