"""Single source of truth for the RAG corpus providers and their sandbox deps.

``PROVIDER_PIP`` mirrors the per-provider extras in ``pyproject.toml`` (keep the two
in sync — the unit test asserts the keys match the ``qa-gen --provider`` choices and
that chroma carries ``snowballstemmer``, which chromadb's BM25 needs but doesn't
declare). ``data``/``validate``/``launch`` all read their ``--provider`` choices and
install hints from here instead of repeating the SDK names procedurally.
"""

from __future__ import annotations

# Maps a provider key → the pip requirements its search SDK needs in the rollout
# sandbox. Mirrors pyproject.toml's [project.optional-dependencies] provider extras.
PROVIDER_PIP: dict[str, list[str]] = {
    "turbopuffer": ["turbopuffer>=1.16.2"],
    "pinecone": ["pinecone>=5.0.0"],
    # chromadb's BM25 embedding function needs snowballstemmer but doesn't declare it.
    "chroma": ["chromadb>=1.0.0", "snowballstemmer>=2.2.0"],
}


def provider_choices() -> list[str]:
    """The provider keys, for argparse ``choices=`` on ``--provider``."""
    return list(PROVIDER_PIP)


def install_hint(provider: str) -> str:
    """The user-facing ``Install with: pip install castform[<extra>]`` line."""
    return f"Install with: pip install castform[{provider}]"


def resolve_pip_dependencies(
    explicit: list[str] | None,
    env_class: type | None = None,
    provider: str | None = None,
) -> list[str] | None:
    """Compose the rollout-sandbox pip deps for ``validate``/``launch``.

    Merges, in order: ``--pip`` (``explicit``), the env's self-declared
    ``PIP_DEPENDENCIES`` class attribute, then ``PROVIDER_PIP[provider]`` when
    ``--provider`` was passed — de-duped preserving first-seen order. Returns
    ``None`` when nothing resolves, so the no-slot/no-provider case is the old
    ``args.pip or None`` verbatim (preserving the single ``dump_bundle`` channel).

    Read CLI-side, not sandbox-side: ``dump_bundle`` pickles the env class by value
    but the sandbox installs from ``BundleMetadata.pip_dependencies``, so the slot
    must be resolved here and fed the existing ``pip_dependencies=`` channel.
    """
    deps: list[str] = list(explicit or [])

    declared = getattr(env_class, "PIP_DEPENDENCIES", None)
    if isinstance(declared, (list, tuple)):  # guard: a list-of-str slot only
        deps.extend(d for d in declared if isinstance(d, str))

    if provider:
        deps.extend(PROVIDER_PIP.get(provider, []))

    seen: set[str] = set()
    ordered = [d for d in deps if not (d in seen or seen.add(d))]
    return ordered or None
