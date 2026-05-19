"""SearchAgentLinker — LLM-driven chunk linker using RolloutClient.

Uses an LLM via the rollout server to search the corpus for related
chunks, producing higher-quality links than heuristic-only approaches.
Falls back to MetadataChunkLinker when the LLM linker fails or is not
selected.
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import re
from typing import Any

from benchmax.bundle.bundler import bundle_env

import benchmax
from benchmax.rag.corpus.search_client import SearchClient
from benchmax.envs.postgres_search.linker_env import LinkerEnv
from benchmax.rag.qa_generation.anchor_selector import AnchorBundle
from benchmax.rag.qa_generation.pipeline_config import SearchAgentLinkerCfg
from benchmax.rag.qa_generation.metadata_linker import MetadataChunkLinker
from benchmax.platform.client import RolloutClient

logger = logging.getLogger(__name__)


def _get_content(chunk: Any) -> str:
    if hasattr(chunk, "content"):
        return str(chunk.content)
    return str(chunk)


def _get_metadata(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "metadata_dict"):
        md = chunk.metadata_dict
        return dict(md) if md else {}
    if hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (tuple, list)):
            return dict(raw)
    if isinstance(chunk, dict):
        return dict(chunk.get("metadata", {}))
    return {}


def _get_file(chunk: Any) -> str:
    md = _get_metadata(chunk)
    return str(md.get("file", "") or md.get("file_name", "") or "")


def _has_date_metadata(chunk: Any) -> bool:
    """Check if chunk has date fields usable for temporal reasoning."""
    md = _get_metadata(chunk)
    return bool(md.get("date_start") or md.get("date_end") or md.get("date"))


def _parse_date(chunk: Any) -> datetime.date | None:
    """Try to parse a date from chunk metadata."""
    md = _get_metadata(chunk)
    for key in ("date_start", "date_end", "date"):
        val = md.get(key)
        if val:
            try:
                return datetime.date.fromisoformat(str(val)[:10])
            except (ValueError, TypeError):
                continue
    return None


class SearchAgentLinker:
    """LLM-driven chunk linker backed by RolloutClient.

    Hybrid approach:
    1. Always runs MetadataChunkLinker first (zero cost).
    2. If confidence is low OR this item is selected by ``search_agent_pct``,
       falls back to the LLM linker via rollout.
    3. The LLM generates search queries; those queries are replayed against
       the source to obtain actual Chunk objects for the AnchorBundle.
    """

    def __init__(
        self,
        *,
        metadata_linker: MetadataChunkLinker,
        source: Any,
        cfg: SearchAgentLinkerCfg,
        search_agent_pct: float,
        rollout_client: RolloutClient,
        search_client: SearchClient,
        llm_model: str = "",
        llm_base_url: str = "",
        llm_api_key: str = "",
    ) -> None:
        self._metadata_linker = metadata_linker
        self._source = source
        self._cfg = cfg
        self._search_agent_pct = search_agent_pct
        self._rollout_client = rollout_client
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key

        self._env_cls_bytes: bytes | None = None
        self._env_meta_bytes: bytes | None = None
        self._env_args_bytes: bytes | None = None
        self._used_hashes: set[str] = set()
        self._prepare_env_bundle(search_client)

    # ------------------------------------------------------------------
    # Env bundling
    # ------------------------------------------------------------------

    def _prepare_env_bundle(self, search_client: SearchClient) -> None:
        """Bundle LinkerEnv with the SearchClient for rollout server use.

        If the user pre-configured env_bundle paths/files, those take
        precedence.  Otherwise, use ``bundle_env`` (same mechanism as
        ``train()``) to package LinkerEnv with its constructor args.
        """
        bundle = self._cfg.env_bundle
        if bundle.has_paths() or bundle.has_files():
            return

        env_bundle = bundle_env(
            LinkerEnv,
            local_modules=[benchmax],
            constructor_args={"search": search_client},
            validate=False,
        )
        self._env_cls_bytes = env_bundle.pickled_class
        self._env_meta_bytes = env_bundle.metadata.to_json_bytes()
        self._env_args_bytes = env_bundle.pickled_constructor_args

    def reset_used_hashes(self) -> None:
        """Clear cross-question dedup state (e.g. between batches)."""
        self._used_hashes.clear()
        self._metadata_linker.reset_used_hashes()

    # ------------------------------------------------------------------
    # ChunkLinker protocol
    # ------------------------------------------------------------------

    def link(
        self,
        primary_chunk: Any,
        *,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
        reasoning_mode: str = "",
    ) -> AnchorBundle:
        hop_count = target_hop_count or 2
        n_secondaries = hop_count - 1

        # Always run metadata linker first (zero cost).
        metadata_bundle = self._metadata_linker.link(
            primary_chunk,
            target_hop_count=target_hop_count,
            corpus_pool=corpus_pool,
            reasoning_mode=reasoning_mode,
        )

        confidence = metadata_bundle.structural_hints.get("confidence", 0.0)
        use_llm = random.random() < self._search_agent_pct or confidence < 0.5

        if not use_llm:
            return metadata_bundle

        try:
            return self._link_with_llm(primary_chunk, hop_count, n_secondaries, reasoning_mode)
        except Exception:
            logger.exception("SearchAgentLinker LLM linking failed")
            if self._cfg.fallback_to_metadata:
                metadata_bundle.structural_hints["llm_fallback"] = True
                return metadata_bundle
            raise

    # ------------------------------------------------------------------
    # LLM linking via rollout
    # ------------------------------------------------------------------

    def _link_with_llm(
        self,
        primary_chunk: Any,
        hop_count: int,
        n_secondaries: int,
        reasoning_mode: str,
    ) -> AnchorBundle:
        if not hasattr(primary_chunk, "hash"):
            return self._empty_bundle(primary_chunk, 1, "not_a_chunk")

        content = _get_content(primary_chunk)
        raw_example = {
            "prompt": content[:4000],
            "target_n": n_secondaries,
            "reasoning_mode": reasoning_mode,
        }

        result = self._run_rollout(raw_example, hop_count=hop_count)
        messages = result.get("messages", [])
        queries = _extract_queries(messages)
        evidence_chain = _extract_evidence_chain(messages)

        if not queries:
            return self._empty_bundle(primary_chunk, hop_count, "no_queries")

        # Replay queries against source to get actual Chunk objects.
        search_results = self._source.search_related(
            primary_chunk,
            queries,
            top_k=n_secondaries * 3,
        )

        # Apply post-filtering with reasoning-mode awareness.
        candidates = self._filter_candidates(
            primary_chunk,
            search_results,
            n_secondaries,
            reasoning_mode,
        )
        secondary_chunks = candidates[:n_secondaries]

        # Register used hashes for cross-question dedup.
        primary_h = getattr(primary_chunk, "hash", None)
        if primary_h:
            self._used_hashes.add(primary_h)
        for chunk in secondary_chunks:
            h = getattr(chunk, "hash", None)
            if h:
                self._used_hashes.add(h)

        actual_hop_count = len(secondary_chunks) + 1
        pre_filter_count = len(search_results)
        confidence = self._compute_confidence(
            secondary_chunks,
            n_secondaries,
            pre_filter_count,
            evidence_chain,
        )

        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=secondary_chunks,
            target_hop_count=min(hop_count, actual_hop_count),
            structural_hints={
                "linker": "search_agent_v2",
                "confidence": confidence,
                "confidence_fulfillment": (
                    min(len(secondary_chunks) / n_secondaries, 1.0)
                    if n_secondaries > 0
                    else 1.0
                ),
                "confidence_survival": (
                    len(secondary_chunks) / pre_filter_count
                    if pre_filter_count > 0
                    else 0.0
                ),
                "confidence_chain": (
                    1.0
                    if evidence_chain.get("connection")
                    else 0.3
                    if evidence_chain.get("raw")
                    else 0.0
                ),
                "queries_used": queries,
                "candidates_found": pre_filter_count,
                "reasoning_mode": reasoning_mode,
                "hop_demoted": actual_hop_count < hop_count,
                "requested_hop_count": hop_count,
                "evidence_chain": evidence_chain.get("connection", ""),
                "chunk_reasons": evidence_chain.get("chunk_reasons", []),
            },
        )

    def _run_rollout(
        self,
        raw_example: dict[str, Any],
        *,
        hop_count: int = 2,
    ) -> dict[str, Any]:
        cfg = self._cfg
        env_bundle = cfg.env_bundle

        max_turns = cfg.max_turns
        max_tool_calls = cfg.max_tool_calls
        if cfg.auto_scale_turns:
            max_turns = max(max_turns, hop_count + 1)
            max_tool_calls = max(max_tool_calls, hop_count)

        kwargs: dict[str, Any] = {
            "raw_example": raw_example,
            "llm_model": self._llm_model,
            "llm_base_url": self._llm_base_url,
            "llm_api_key": self._llm_api_key,
            "max_turns": max_turns,
            "max_tool_calls": max_tool_calls,
            "max_completion_tokens": cfg.max_completion_tokens,
            "capture_messages": True,
            "include_event_meta": False,
        }

        if env_bundle.has_paths():
            kwargs["env_cls_path"] = env_bundle.env_cls_path
            kwargs["env_metadata_path"] = env_bundle.env_metadata_path
        elif env_bundle.has_files():
            cls_bytes, meta_bytes = env_bundle.as_bytes_bundle()
            kwargs["env_cls_bytes"] = cls_bytes
            kwargs["env_metadata_bytes"] = meta_bytes
        else:
            kwargs["env_cls_bytes"] = self._env_cls_bytes
            kwargs["env_metadata_bytes"] = self._env_meta_bytes
            if self._env_args_bytes is not None:
                kwargs["env_args_bytes"] = self._env_args_bytes

        return self._rollout_client.stream_rollout(**kwargs)

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        primary_chunk: Any,
        search_results: list[dict[str, Any]],
        n_secondaries: int,
        reasoning_mode: str = "",
    ) -> list[Any]:
        primary_hash = getattr(primary_chunk, "hash", None)
        primary_file = _get_file(primary_chunk)
        primary_tokens = set(_get_content(primary_chunk).lower().split())

        # Reasoning-mode overrides.
        filter_same_file = reasoning_mode != "sequential"
        min_coherence = 0.05 if reasoning_mode == "inference" else 0.15

        seen: set[str] = set()
        scored: list[tuple[float, float, Any, set[str]]] = []

        for result in search_results:
            chunk = result.get("chunk")
            if chunk is None:
                continue

            chunk_hash = getattr(chunk, "hash", None)

            # Cross-question dedup (before local seen check).
            if chunk_hash and chunk_hash in self._used_hashes:
                continue

            if chunk_hash and (chunk_hash == primary_hash or chunk_hash in seen):
                continue

            chunk_content = _get_content(chunk)
            if len(chunk_content) < 400:
                continue

            chunk_file = _get_file(chunk)
            if (
                filter_same_file
                and primary_file
                and chunk_file
                and chunk_file == primary_file
            ):
                continue

            tokens = set(chunk_content.lower().split())

            # Coherence floor.
            if primary_tokens and tokens:
                jaccard = len(primary_tokens & tokens) / len(primary_tokens | tokens)
                if jaccard < min_coherence:
                    continue
            else:
                jaccard = 0.0

            search_score = float(result.get("max_score", 0.0) or 0.0)

            if chunk_hash:
                seen.add(chunk_hash)
            scored.append((search_score, jaccard, chunk, tokens))

        if not scored:
            return []

        # Composite ranking: 60% search relevance, 40% coherence.
        max_search = max(s for s, _, _, _ in scored) or 1.0
        ranked = sorted(
            scored,
            key=lambda x: -(0.6 * (x[0] / max_search) + 0.4 * x[1]),
        )

        # Greedy diversity selection (similarity cap 0.8).
        selected: list[Any] = []
        selected_tokens: list[set[str]] = []
        for _, _, chunk, tokens in ranked:
            too_similar = False
            for prev in selected_tokens:
                sim = len(tokens & prev) / len(tokens | prev) if tokens and prev else 0.0
                if sim > 0.8:
                    too_similar = True
                    break
            if too_similar:
                continue
            selected.append(chunk)
            selected_tokens.append(tokens)
            if len(selected) >= n_secondaries:
                break

        # Mode-specific reranking.
        if reasoning_mode == "temporal" and _has_date_metadata(primary_chunk):
            selected = _rerank_by_date_diversity(primary_chunk, selected)
        if reasoning_mode == "sequential":
            primary_md = _get_metadata(primary_chunk)
            if primary_md.get("file") and primary_md.get("index") is not None:
                selected = _rerank_by_index_proximity(
                    primary_chunk, selected, index_range=(2, 5)
                )

        return selected

    def _compute_confidence(
        self,
        secondary_chunks: list[Any],
        n_secondaries: int,
        pre_filter_count: int,
        evidence_chain: dict[str, Any],
    ) -> float:
        if n_secondaries == 0:
            return 1.0

        fulfillment = min(len(secondary_chunks) / n_secondaries, 1.0)
        survival = (
            len(secondary_chunks) / pre_filter_count if pre_filter_count > 0 else 0.0
        )
        connection = evidence_chain.get("connection", "")
        chain_raw = evidence_chain.get("raw", "")
        if connection:
            chain_score = 1.0
        elif chain_raw:
            chain_score = 0.3
        else:
            chain_score = 0.0

        return 0.4 * fulfillment + 0.3 * survival + 0.3 * chain_score

    def _empty_bundle(
        self,
        primary_chunk: Any,
        hop_count: int,
        reason: str,
    ) -> AnchorBundle:
        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=[],
            target_hop_count=1,  # No secondaries found; just the primary.
            structural_hints={
                "linker": "search_agent",
                "confidence": 0.0,
                "reason": reason,
                "requested_hop_count": hop_count,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL,
)
_EVIDENCE_CHAIN_RE = re.compile(
    r"<evidence_chain>(.*?)</evidence_chain>", re.DOTALL,
)
_CHUNK_ROLE_RE = re.compile(
    r'<chunk[^>]*role="secondary"[^>]*>(.*?)</chunk>', re.DOTALL,
)
_CONNECTION_RE = re.compile(
    r"<connection>(.*?)</connection>", re.DOTALL,
)


def _extract_queries(messages: list[dict[str, Any]]) -> list[str]:
    """Extract search queries from rollout messages.

    Handles two message formats:
    1. Structured: content is list[dict] with type="tool_use" blocks
    2. Text: content is a string with <tool_call> JSON blocks
    """
    queries: list[str] = []
    for msg in messages:
        content = msg.get("content")
        # Structured tool_use blocks
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    input_data = block.get("input", {})
                    query = input_data.get("query")
                    if query and query not in queries:
                        queries.append(query)
        # Text content with <tool_call> markers
        elif isinstance(content, str):
            for match in _TOOL_CALL_RE.finditer(content):
                try:
                    payload = json.loads(match.group(1))
                except (json.JSONDecodeError, ValueError):
                    continue
                args = payload.get("arguments", payload.get("input", {}))
                query = args.get("query")
                if query and query not in queries:
                    queries.append(query)
    return queries


def _extract_evidence_chain(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract evidence chain reasoning from the LLM's final output."""
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        match = _EVIDENCE_CHAIN_RE.search(content)
        if match:
            chain_text = match.group(1)
            chunk_reasons = _CHUNK_ROLE_RE.findall(chain_text)
            connection_match = _CONNECTION_RE.search(chain_text)
            return {
                "raw": chain_text.strip(),
                "chunk_reasons": [r.strip() for r in chunk_reasons],
                "connection": (
                    connection_match.group(1).strip() if connection_match else ""
                ),
            }
    return {}


def _rerank_by_date_diversity(
    primary_chunk: Any,
    candidates: list[Any],
) -> list[Any]:
    """Boost candidates whose date differs from the primary chunk."""
    primary_date = _parse_date(primary_chunk)
    if primary_date is None or not candidates:
        return candidates

    day_diffs: list[int] = []
    for chunk in candidates:
        chunk_date = _parse_date(chunk)
        day_diffs.append(abs((chunk_date - primary_date).days) if chunk_date else 0)

    max_diff = max(max(day_diffs), 1)
    scored: list[tuple[float, int, Any]] = []
    n = len(candidates)
    for idx, (chunk, dd) in enumerate(zip(candidates, day_diffs)):
        rank_score = 1.0 - (idx / max(n - 1, 1))
        date_score = dd / max_diff
        combined = 0.7 * rank_score + 0.3 * date_score
        scored.append((combined, idx, chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [chunk for _, _, chunk in scored]


def _rerank_by_index_proximity(
    primary_chunk: Any,
    candidates: list[Any],
    index_range: tuple[int, int],
) -> list[Any]:
    """Boost same-file candidates with index gap in the target range."""
    primary_file = _get_file(primary_chunk)
    primary_md = _get_metadata(primary_chunk)
    primary_index = primary_md.get("index")
    if not primary_file or primary_index is None or not candidates:
        return candidates

    try:
        primary_index = int(primary_index)
    except (TypeError, ValueError):
        return candidates

    min_gap, max_gap = index_range

    scored: list[tuple[int, int, Any]] = []
    for idx, chunk in enumerate(candidates):
        chunk_file = _get_file(chunk)
        chunk_md = _get_metadata(chunk)
        chunk_index = chunk_md.get("index")

        score = 0
        if chunk_file == primary_file and chunk_index is not None:
            try:
                gap = abs(int(chunk_index) - primary_index)
                if min_gap <= gap <= max_gap:
                    score = max_gap - gap + 1
            except (TypeError, ValueError):
                pass

        scored.append((score, idx, chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [chunk for _, _, chunk in scored]
