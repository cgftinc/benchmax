"""Wiki page builder for corpus preprocessing.

This module synthesizes scattered corpus chunks into entity-centric wiki pages
with cross-references. The resulting WikiIndex can be used to inject structured
context into QA generation prompts.

Typical usage::

    from openai import OpenAI
    from benchmax.rag.qa_generation.pipeline_config import WikiPreprocessingConfig
    from benchmax.rag.qa_generation.wiki_builder import WikiBuilder

    config = WikiPreprocessingConfig(
        enabled=True,
        model="gpt-4o-mini",
        api_key="...",
    )
    client = OpenAI(api_key=config.api_key)
    builder = WikiBuilder(config, client)

    clusters = builder.cluster_chunks(chunks, entity_patterns)
    wiki_index = builder.generate_pages(clusters, corpus_summary, corpus_description)
    context = builder.get_wiki_context(wiki_index, primary_chunk, secondary_chunks)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from benchmax.rag.qa_generation.batch_processor import batch_process_sync
from benchmax.rag.qa_generation.pipeline_config import WikiPreprocessingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WikiPage:
    """A synthesized wiki page built from multiple source chunks."""

    title: str
    content: str  # The synthesized markdown content
    source_chunk_ids: list[str]  # Hashes of chunks that contributed
    cross_links: list[str]  # Titles of related wiki pages
    entity_names: list[str]  # Entities this page covers


@dataclass
class WikiIndex:
    """Collection of wiki pages built from a corpus."""

    pages: dict[str, WikiPage] = field(default_factory=dict)  # title -> WikiPage
    chunk_to_pages: dict[str, list[str]] = field(
        default_factory=dict
    )  # chunk_hash -> list of page titles


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a technical writer synthesizing information from multiple document chunks "
    "into a concise wiki page."
)

_PAGE_PROMPT_TEMPLATE = """\
Create a wiki page about "{entity_name}" based on the following source chunks from a corpus.

Corpus context: {corpus_summary}
{language_instruction}
Source chunks:
{chunk_texts}

Requirements:
- Write a concise wiki page (max ~{max_page_tokens} tokens) synthesizing the key information about {entity_name}
- Include a brief summary at the top
- Organize information logically under subheadings if needed
- Note relationships to other concepts/entities mentioned in the chunks
- Cite chunk indices [0], [1], etc. when making claims
- Do not invent information not present in the chunks
- At the end, list "Related topics:" with entity/concept names mentioned that could be their own pages

Output the wiki page in markdown format."""  # noqa: E501


# ---------------------------------------------------------------------------
# WikiBuilder
# ---------------------------------------------------------------------------


class WikiBuilder:
    """Builds wiki pages from corpus chunks as a preprocessing step for QA generation."""

    def __init__(self, config: WikiPreprocessingConfig, client: Any) -> None:
        self.config = config
        self.client = client

    # ------------------------------------------------------------------
    # cluster_chunks
    # ------------------------------------------------------------------

    def cluster_chunks(
        self,
        chunks: list[Any],
        entity_patterns: list[Any],
        profile: Any | None = None,
    ) -> dict[str, list[Any]]:
        """Group chunks by entity co-occurrence.

        When a ``profile`` with a pre-computed ``entity_chunk_index`` is
        provided, reads from the graph instead of rescanning.  Otherwise
        falls back to the original O(E×N) scan.

        Args:
            chunks: List of Chunk objects to cluster.
            entity_patterns: EntityPattern objects (from corpus_profile.py).
            profile: Optional CorpusProfile with entity_chunk_index.

        Returns:
            dict mapping entity/concept name -> list of chunks.
        """
        # Filter to high-quality entities for clustering
        discriminative = [
            e for e in entity_patterns
            if e.quality_score > 0.75 and e.document_frequency < 0.40
        ]
        # Process best entities first so their names survive merges
        discriminative.sort(key=lambda e: e.quality_score, reverse=True)

        # Build initial clusters: entity name -> set of chunk indices
        cluster_chunk_indices: dict[str, set[int]] = {}

        if profile and getattr(profile, "entity_chunk_index", None):
            # Fast path: read from pre-computed entity-chunk graph
            hash_to_idx = {
                getattr(c, "hash", str(id(c))): i for i, c in enumerate(chunks)
            }
            for entity in discriminative:
                name_lower = entity.name.lower()
                chunk_hashes = profile.entity_chunk_index.get(name_lower, set())
                indices = {hash_to_idx[h] for h in chunk_hashes if h in hash_to_idx}
                if indices:
                    cluster_chunk_indices[entity.name] = indices
        else:
            # Fallback: original O(E×N) scan
            for entity in discriminative:
                matched_indices: set[int] = set()
                for idx, chunk in enumerate(chunks):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if entity.type == "code_pattern":
                        try:
                            if re.findall(entity.name, content):
                                matched_indices.add(idx)
                        except re.error:
                            continue
                    else:
                        if entity.name.lower() in content.lower():
                            matched_indices.add(idx)
                if matched_indices:
                    cluster_chunk_indices[entity.name] = matched_indices

        # Merge clusters that share >50% of their chunks
        merged = _merge_overlapping_clusters(cluster_chunk_indices, overlap_threshold=0.5)

        # Convert index sets back to chunk lists, applying size filters
        result: dict[str, list[Any]] = {}
        min_chunks = self.config.min_chunks_per_page
        max_per_page = self.config.max_chunks_per_page

        for name, indices in merged.items():
            n = len(indices)
            if n < min_chunks:
                continue

            sorted_indices = sorted(indices)
            if n > max_per_page:
                # Sample diversely by file to get representative chunks
                sorted_indices = _diverse_sample(
                    chunks, sorted_indices, max_per_page,
                )

            result[name] = [chunks[i] for i in sorted_indices]

        return result

    # ------------------------------------------------------------------
    # generate_pages
    # ------------------------------------------------------------------

    def generate_pages(
        self,
        clusters: dict[str, list[Any]],
        corpus_summary: str,
        corpus_description: str,
        corpus_language: str = "",
    ) -> WikiIndex:
        """Generate wiki pages from entity clusters.

        For each cluster, makes one LLM call that receives the entity name,
        corpus summary, and all chunk texts in the cluster, and generates a
        markdown wiki page with cross-references.

        Args:
            clusters: Output of cluster_chunks — entity name -> list of chunks.
            corpus_summary: High-level corpus summary string from CorpusProfile.
            corpus_description: Corpus description string from CorpusProfile.
            corpus_language: Language to write wiki pages in (e.g. "Korean"). Empty = no
                instruction (LLM will default to the language of the source chunks, which
                is often English-influenced). When set, all page content — headings,
                summaries, and body — will be written in this language.

        Returns:
            WikiIndex with all generated pages.
        """
        if not clusters:
            return WikiIndex()

        language_instruction = (
            f"\nLanguage instruction: Write the entire wiki page in {corpus_language}, "
            f"including the title, subheadings, summary, and all body text. "
            f"Use terminology consistent with the source chunks.\n"
            if corpus_language
            else ""
        )

        entity_names = list(clusters.keys())
        prompts: list[str] = []

        for entity_name in entity_names:
            entity_chunks = clusters[entity_name]
            chunk_texts = _format_chunk_texts(entity_chunks)
            prompt = _PAGE_PROMPT_TEMPLATE.format(
                entity_name=entity_name,
                corpus_summary=corpus_summary,
                chunk_texts=chunk_texts,
                max_page_tokens=self.config.max_page_tokens,
                language_instruction=language_instruction,
            )
            prompts.append(prompt)

        logger.info("Generating %d wiki pages via batch LLM calls", len(prompts))
        batch_result = batch_process_sync(
            client=self.client,
            model=self.config.model,
            prompts=prompts,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=self.config.max_page_tokens,
            desc="Generating wiki pages",
        )

        wiki_index = WikiIndex()

        for i, entity_name in enumerate(entity_names):
            response = batch_result.responses[i]
            if response is None:
                logger.warning("Wiki page generation failed for entity: %s", entity_name)
                continue

            entity_chunks = clusters[entity_name]
            source_chunk_ids = [c.hash for c in entity_chunks if hasattr(c, "hash")]
            cross_links = _extract_cross_links(response.answer, entity_names, entity_name)

            page = WikiPage(
                title=entity_name,
                content=response.answer,
                source_chunk_ids=source_chunk_ids,
                cross_links=cross_links,
                entity_names=[entity_name],
            )

            wiki_index.pages[entity_name] = page

            # Update chunk_to_pages index
            for chunk_hash in source_chunk_ids:
                wiki_index.chunk_to_pages.setdefault(chunk_hash, []).append(entity_name)

        return wiki_index

    # ------------------------------------------------------------------
    # get_wiki_context
    # ------------------------------------------------------------------

    def get_wiki_context(
        self,
        wiki_index: WikiIndex,
        primary_chunk: Any,
        secondary_chunks: list[Any],
        max_tokens: int = 2000,
    ) -> str:
        """Get relevant wiki context for a given primary chunk.

        1. Find all wiki pages that reference the primary chunk's hash.
        2. Also find pages that reference any secondary chunk's hash.
        3. Rank by relevance (pages referencing more of the task's chunks rank higher).
        4. Truncate to max_tokens.
        5. Format as markdown sections.

        Args:
            wiki_index: The WikiIndex to look up pages from.
            primary_chunk: The primary Chunk object for the task.
            secondary_chunks: Secondary Chunk objects for the task.
            max_tokens: Approximate token budget (estimated as chars / 4).

        Returns:
            Formatted wiki context string (empty string if no relevant pages).
        """
        if not wiki_index.pages:
            return ""

        # Gather chunk content for content-based matching
        primary_content = (
            primary_chunk.content
            if hasattr(primary_chunk, "content")
            else str(primary_chunk)
        ).lower()
        secondary_contents = [
            (
                c.content if hasattr(c, "content") else str(c)
            ).lower()
            for c in secondary_chunks
        ]

        # Score each page by content relevance (entity name mentions)
        # plus hash overlap as a bonus signal
        page_scores: dict[str, float] = {}
        for title, page in wiki_index.pages.items():
            score = 0.0
            for entity_name in page.entity_names:
                entity_lower = entity_name.lower()
                # Skip very short entity names to avoid false matches
                if len(entity_lower) < 3:
                    continue
                if entity_lower in primary_content:
                    score += 2.0  # primary chunk match is worth more
                for sc in secondary_contents:
                    if entity_lower in sc:
                        score += 1.0

            # Bonus for hash overlap (chunk was used to build the page)
            primary_hash = getattr(primary_chunk, "hash", "")
            if primary_hash and primary_hash in page.source_chunk_ids:
                score += 3.0
            for c in secondary_chunks:
                c_hash = getattr(c, "hash", "")
                if c_hash and c_hash in page.source_chunk_ids:
                    score += 1.5

            if score > 0:
                page_scores[title] = score

        if not page_scores:
            return ""

        # Sort by score descending, then by title for determinism
        ranked_titles = sorted(page_scores.keys(), key=lambda t: (-page_scores[t], t))

        # Accumulate pages up to token budget (rough estimate: 4 chars per token)
        char_budget = max_tokens * 4
        sections: list[str] = []
        used_chars = 0

        for title in ranked_titles:
            page = wiki_index.pages.get(title)
            if page is None:
                continue
            section = f"## {page.title}\n\n{page.content}"
            section_chars = len(section)
            if used_chars + section_chars > char_budget:
                # Truncate this section to fit remaining budget
                remaining = char_budget - used_chars
                if remaining > 100:
                    section = section[:remaining] + "\n...(truncated)"
                    sections.append(section)
                break
            sections.append(section)
            used_chars += section_chars

        if not sections:
            return ""

        return "# Relevant Wiki Context\n\n" + "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_overlapping_clusters(
    clusters: dict[str, set[int]],
    overlap_threshold: float = 0.5,
) -> dict[str, set[int]]:
    """Merge clusters that share more than overlap_threshold fraction of chunks.

    Uses a union-find-style pass: repeatedly merges the first pair that exceeds
    the threshold until no more merges are possible. The merged cluster keeps
    the name of the first (alphabetically earlier) cluster.

    Args:
        clusters: Mapping of entity name -> set of chunk indices.
        overlap_threshold: Fraction of the smaller cluster that must overlap
            to trigger a merge.

    Returns:
        New dict with merged clusters.
    """
    # Work with a mutable list of (name, index_set) pairs
    items: list[tuple[str, set[int]]] = sorted(clusters.items())

    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                name_i, set_i = items[i]
                name_j, set_j = items[j]
                intersection = len(set_i & set_j)
                smaller = min(len(set_i), len(set_j))
                if smaller == 0:
                    continue
                if intersection / smaller > overlap_threshold:
                    # Merge j into i
                    items[i] = (name_i, set_i | set_j)
                    items.pop(j)
                    merged = True
                    break
            if merged:
                break

    return dict(items)


def _diverse_sample(
    chunks: list[Any],
    indices: list[int],
    max_count: int,
) -> list[int]:
    """Sample indices diversely by source file.

    Round-robin across files to ensure coverage from different parts
    of the corpus rather than clustering on a single document.
    """
    # Group indices by file metadata
    by_file: dict[str, list[int]] = {}
    for idx in indices:
        chunk = chunks[idx]
        meta = getattr(chunk, "metadata", ())
        file_key = ""
        for k, v in (meta if isinstance(meta, tuple) else ()):
            if k == "file":
                file_key = str(v)
                break
        by_file.setdefault(file_key, []).append(idx)

    # Round-robin across files
    sampled: list[int] = []
    file_lists = list(by_file.values())
    file_cursors = [0] * len(file_lists)
    while len(sampled) < max_count:
        added_any = False
        for i, flist in enumerate(file_lists):
            if file_cursors[i] < len(flist):
                sampled.append(flist[file_cursors[i]])
                file_cursors[i] += 1
                added_any = True
                if len(sampled) >= max_count:
                    break
        if not added_any:
            break

    return sorted(sampled)


def _format_chunk_texts(chunks: list[Any]) -> str:
    """Format a list of chunks as numbered sections for the LLM prompt."""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks):
        if hasattr(chunk, "chunk_str"):
            text = chunk.chunk_str()
        elif hasattr(chunk, "content"):
            text = chunk.content
        else:
            text = str(chunk)
        parts.append(f"[{idx}]\n{text}")
    return "\n\n".join(parts)


def _extract_cross_links(
    page_content: str,
    all_entity_names: list[str],
    current_entity: str,
) -> list[str]:
    """Extract cross-links from generated page content.

    Looks for a "Related topics:" section in the generated markdown and
    extracts entity names. Also falls back to scanning all known entity
    names for mentions in the content.

    Args:
        page_content: The generated wiki page markdown.
        all_entity_names: All entity names that were clustered.
        current_entity: The entity this page is about (excluded from links).

    Returns:
        List of entity names that this page references.
    """
    cross_links: list[str] = []
    current_lower = current_entity.lower()

    # Try to parse "Related topics:" section
    related_match = re.search(
        r"related topics?[:]\s*\n(.*?)(?:\n\n|\Z)",
        page_content,
        re.IGNORECASE | re.DOTALL,
    )
    if related_match:
        related_block = related_match.group(1)
        # Each line may be "- Entity" or "* Entity" or just "Entity"
        for line in related_block.splitlines():
            line = line.strip().lstrip("-*").strip()
            if line and line.lower() != current_lower:
                cross_links.append(line)

    # Additionally, scan content for mentions of known entity names
    content_lower = page_content.lower()
    for name in all_entity_names:
        if name.lower() == current_lower:
            continue
        if name.lower() in content_lower and name not in cross_links:
            cross_links.append(name)

    return cross_links
