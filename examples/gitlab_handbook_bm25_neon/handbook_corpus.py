"""Deterministic GitLab-handbook checkout + re-chunk for the Neon corpus.

Ports the corpus-build half of the old ``examples/gitlab_handbook_bm25`` loader
onto the harbor ``castform.*`` layout and the Neon provider. Two things change
from the original:

* the document source is a git *sparse-checkout* of the public GitLab handbook
  pinned to an exact commit SHA (no token, fully reproducible); and
* every chunk carries FILTERABLE metadata (``handbook_section``, ``path_depth``)
  derived deterministically from the file path, so the Neon ``filter_mapper`` can
  bind ``eq`` / range predicates against it.

Ingest itself is not here — it is the provider's job
(:meth:`castform.rag.corpus.neon.source.NeonChunkSource.populate_from_chunks`);
this module produces the deterministic :class:`ChunkCollection` that ingest
consumes. Per-file chunking failures are surfaced as
:class:`castform.rag.corpus.neon.source.NeonIngestError`, never silently dropped
(B9), so the chunk set is reproducible run to run and nothing is lost unnoticed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from castform.rag.chunkers.markdown import MarkdownChunker
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.source import NeonIngestError

# Pinned public GitLab handbook source. The SHA is the gold reference commit the
# original BM25 loader used; keeping it pinned makes the checkout — and therefore
# every derived chunk hash — reproducible.
HANDBOOK_REPO_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
HANDBOOK_COMMIT = "3078d0213524f8ca0c0e3a70680a21929a9f65ff"
HANDBOOK_SUBDIR = "content/handbook"

# Neon logical corpus name the golden set is validated against. Ingest publishes a
# new physical version under this name; reads/gates bind to the active version.
LOGICAL_NAME = "gitlab_handbook_neon"

# Chunker parameters. Pinned here (not defaulted per call) because they are part
# of the corpus identity: any change re-hashes every chunk and invalidates the
# frozen golden set.
CHUNKER_VERSION = "markdown-v1"
MIN_CHARS = 1024
MAX_CHARS = 2048
OVERLAP_CHARS = 128
FILE_EXTENSIONS = (".md", ".mdx")

# Metadata key for a root-level file (one whose relative path has no directory).
ROOT_SECTION = "_root"

# Exact chunk count of the pinned corpus. The full-corpus ingest aborts unless the
# deterministic re-chunk yields exactly this many chunks, so a stray untracked file
# or a moved ref can never silently change the corpus the golden set was authored
# against.
EXPECTED_CHUNK_COUNT = 31665


@dataclass(frozen=True)
class ChunkerParams:
    """The chunk-identity parameters recorded in the provenance manifest.

    Args:
        version: Chunker version tag (bumped when chunking semantics change).
        min_chars: Minimum characters per chunk.
        max_chars: Maximum characters per chunk.
        overlap_chars: Character overlap between adjacent chunks.
        file_extensions: Extensions ingested.
    """

    version: str = CHUNKER_VERSION
    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    overlap_chars: int = OVERLAP_CHARS
    file_extensions: tuple[str, ...] = FILE_EXTENSIONS

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "overlap_chars": self.overlap_chars,
            "file_extensions": list(self.file_extensions),
        }


def section_metadata(rel_path: str) -> dict[str, object]:
    """Derive filterable metadata from a docs-root-relative path (deterministic).

    ``handbook_section`` is the first path segment (the top-level handbook area,
    e.g. ``engineering``); a file that sits directly at the docs root has no
    directory, so it is tagged :data:`ROOT_SECTION`. ``path_depth`` is the number
    of path segments including the filename (a real JSON int so numeric filter
    predicates bind), e.g. ``engineering/development/index.md`` -> section
    ``engineering``, depth ``3``.

    Args:
        rel_path: File path relative to the docs root, POSIX or OS-native.
    """
    parts = PurePosixPath(str(rel_path).replace("\\", "/")).parts
    if not parts:
        raise ValueError("empty relative path")
    section = parts[0] if len(parts) > 1 else ROOT_SECTION
    return {"handbook_section": section, "path_depth": len(parts)}


def sparse_checkout(
    dest: Path,
    *,
    repo_url: str = HANDBOOK_REPO_URL,
    commit: str = HANDBOOK_COMMIT,
    subdir: str = HANDBOOK_SUBDIR,
) -> Path:
    """Sparse-checkout ``subdir`` of ``repo_url`` at exactly ``commit`` (no token).

    Idempotent: re-running against a populated ``dest`` re-fetches and re-checks
    out the same commit. Returns the path to the checked-out docs directory
    (``dest/subdir``). Raises if the resolved HEAD does not equal ``commit`` so a
    moving ref can never silently change the corpus.

    Args:
        dest: Working directory for the shallow clone.
        repo_url: Git remote to fetch from.
        commit: Exact 40-hex commit SHA to check out.
        subdir: Repository subdirectory to materialize.
    """
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True)

    if not (dest / ".git").exists():
        git("init", "-q")
        git("remote", "add", "origin", repo_url)
        git("sparse-checkout", "init", "--cone")
    git("sparse-checkout", "set", subdir)
    git("fetch", "--depth=1", "origin", commit)
    git("checkout", "--force", "FETCH_HEAD")

    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if head != commit:
        raise NeonIngestError(f"checkout HEAD {head} != pinned commit {commit}")

    docs_dir = dest / subdir
    if not docs_dir.is_dir():
        raise NeonIngestError(f"checked-out subdir missing: {docs_dir}")
    return docs_dir


def git_tracked_docs(
    repo_dir: Path,
    subdir: str = HANDBOOK_SUBDIR,
    exts: tuple[str, ...] = FILE_EXTENSIONS,
) -> list[Path]:
    """Return the git-tracked doc files under ``subdir`` at the checked-out commit.

    Enumerates from ``git ls-tree`` rather than a working-tree walk, so an
    untracked or ignored file dropped into the checkout cannot slip into the
    corpus (the corpus is exactly the pinned tree, nothing more). Paths are
    returned absolute and sorted for a deterministic order.

    Args:
        repo_dir: The git working directory (the sparse-checkout root).
        subdir: Repository subdirectory to enumerate.
        exts: Doc extensions to keep.
    """
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            subdir,
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    lower = tuple(e.lower() for e in exts)
    return sorted(repo_dir / rel for rel in listing if rel.lower().endswith(lower))


def build_collection(
    docs_dir: Path,
    *,
    params: ChunkerParams = ChunkerParams(),
    files: list[Path] | None = None,
) -> ChunkCollection:
    """Chunk ``docs_dir`` deterministically with filterable metadata attached.

    Determinism, mirroring the provider's strict ingest path: the file list is
    sorted (``rglob`` order is filesystem dependent), one chunker instance is
    reused so the cross-file duplicate-hash guard stays effective, and the
    docs-root-*relative* path is used as ``file`` so nested files neither collide
    nor lose their section. Every chunk additionally gets ``handbook_section`` /
    ``path_depth`` via :func:`section_metadata`; because the chunk hash folds in a
    ``sort_keys`` serialization of the metadata, these keys re-hash every chunk
    deterministically. Per-file failures are collected and raised together as
    :class:`NeonIngestError` before any chunk is returned (no partial set).

    Args:
        docs_dir: Root directory of the checked-out docs.
        params: Chunker identity parameters (must match the frozen golden set).
        files: Explicit file list to chunk (e.g. from :func:`git_tracked_docs`);
            ``None`` falls back to a sorted recursive walk of ``docs_dir`` (used by
            fixtures that are not git checkouts).
    """
    root = docs_dir.resolve()
    exts = params.file_extensions
    if files is None:
        paths = sorted({p for ext in exts for p in root.rglob(f"*{ext}")})
    else:
        paths = sorted({p.resolve() for p in files})

    chunker = MarkdownChunker(
        min_char=params.min_chars,
        max_char=params.max_chars,
        chunk_overlap=params.overlap_chars,
    )
    all_chunks: list[Chunk] = []
    errors: dict[str, Exception] = {}
    for file_path in paths:
        rel = PurePosixPath(file_path.relative_to(root)).as_posix()
        try:
            content = file_path.read_text(encoding="utf-8")
            is_mdx = file_path.suffix.lower() == ".mdx"
            all_chunks.extend(
                chunker.chunk(
                    content,
                    file=rel,
                    extra_metadata=section_metadata(rel),
                    preprocess_mdx=is_mdx,
                )
            )
        except Exception as exc:  # surface, never swallow (B9)
            errors[rel] = exc

    if errors:
        detail = "; ".join(f"{name}: {exc}" for name, exc in sorted(errors.items()))
        raise NeonIngestError(f"failed to chunk {len(errors)} file(s): {detail}")
    return ChunkCollection(all_chunks)
