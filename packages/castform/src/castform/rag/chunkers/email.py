"""Email chunker that reconstructs reply chains before sliding-window chunking.

This chunker expects canonical message rows (see preprocess/email/schema.py).
It groups by thread_id, reconstructs reply paths from reply_to, and chunks each
path into overlapping windows.

Limitation:
- Fork-heavy threads may duplicate shared history across branch chunks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.preprocess.email.schema import (
    date_yyyy_mm_dd,
    extract_participants,
    validate_rows,
)

SHARED_PREFIX_THRESHOLD = 4
CONTEXT_TAIL_FOR_BRANCH = 3


def _format_address(address: dict) -> str:
    """Format a name/email dict as 'Name <email>' or just one field."""
    name = (address.get("name") or "").strip()
    email = (address.get("email") or "").strip()
    if name and email and name.lower() == email.lower():
        return name
    if name and email:
        return f"{name} <{email}>"
    return name or email or "unknown"


def _format_email_block(email_message: dict, reply_chain_message_index: int | None = None) -> str:
    """Format a single email as a text block for chunk content."""
    date_str = date_yyyy_mm_dd(email_message.get("date")) or "unknown date"

    from_address = email_message.get("from") if isinstance(email_message.get("from"), dict) else {}
    from_str = _format_address(from_address)

    to_addresses = email_message.get("to") if isinstance(email_message.get("to"), list) else []
    to_str = "; ".join(_format_address(address) for address in to_addresses) if to_addresses else ""

    cc_addresses = email_message.get("cc") if isinstance(email_message.get("cc"), list) else []
    cc_str = "; ".join(_format_address(address) for address in cc_addresses) if cc_addresses else ""

    order_prefix = (
        f"msg #{reply_chain_message_index} | " if reply_chain_message_index is not None else ""
    )
    header = f"[{order_prefix}{date_str} | {from_str} -> {to_str}"
    if cc_str:
        header += f" | cc: {cc_str}"
    header += "]"

    body = (email_message.get("body") or "").strip()
    return f"{header}\n{body}"


def _compact_participants_display(display_text: str, max_names: int = 3) -> str:
    """Return a short participants summary for the chunk header."""
    names = [name.strip() for name in display_text.split(",") if name.strip()]
    if not names:
        return "unknown"

    visible = names[:max_names]
    remainder = len(names) - len(visible)
    summary = ", ".join(visible)
    if remainder > 0:
        summary = f"{summary} (+{remainder} more)"
    return summary


def _coerce_date_sort_key(date_value: object, message_id: str) -> tuple[int, str, str]:
    """Return a deterministic date sort key.

    First element marks parse success so parsed datetimes sort after failures.
    """
    text = str(date_value or "").strip()
    if text:
        try:
            normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
            parsed = datetime.fromisoformat(normalized)
            return (1, parsed.isoformat(), message_id)
        except Exception:
            pass
    return (0, text, message_id)


def _normalize_thread_rows_for_reconstruction(
    email_messages: list[dict],
) -> list[dict]:
    """Coerce row shapes and derive stable reconstruction fields."""
    normalized_rows: list[dict] = []
    message_index_by_canonical_id: dict[str, int] = {}

    for row_index, original_row in enumerate(email_messages):
        row = dict(original_row)

        original_message_id = str(row.get("id") or "").strip()
        if not original_message_id:
            original_message_id = f"missing_id_{row_index}"

        canonical_message_id = original_message_id
        message_id_for_coverage = canonical_message_id
        if canonical_message_id in message_index_by_canonical_id:
            message_id_for_coverage = f"{canonical_message_id}__dup_{row_index}"
        else:
            message_index_by_canonical_id[canonical_message_id] = row_index

        parent_message_id = str(row.get("reply_to") or "").strip()

        from_address = row.get("from")
        if not isinstance(from_address, dict):
            from_address = {}

        to_addresses = row.get("to")
        if not isinstance(to_addresses, list):
            to_addresses = []

        cc_addresses = row.get("cc")
        if not isinstance(cc_addresses, list):
            cc_addresses = []

        row["id"] = canonical_message_id
        row["reply_to"] = parent_message_id
        row["from"] = from_address
        row["to"] = to_addresses
        row["cc"] = cc_addresses
        row["_reconstruct_message_id"] = message_id_for_coverage
        row["_reconstruct_parent_id"] = parent_message_id
        row["_reconstruct_date_sort_key"] = _coerce_date_sort_key(
            row.get("date"), canonical_message_id
        )
        normalized_rows.append(row)

    return normalized_rows


def _build_thread_graph(
    normalized_rows: list[dict],
) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, str]]:
    """Build message lookup and adjacency maps for reconstruction."""
    message_by_id: dict[str, dict] = {}
    canonical_row_by_message_id: dict[str, dict] = {}
    child_message_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
    parent_id_by_message_id: dict[str, str] = {}

    for row in normalized_rows:
        coverage_message_id = row["_reconstruct_message_id"]
        canonical_message_id = row["id"]
        message_by_id[coverage_message_id] = row

        if canonical_message_id not in canonical_row_by_message_id:
            canonical_row_by_message_id[canonical_message_id] = row

    # Build edges from alias rows to canonical parent targets when present.
    for coverage_message_id, row in message_by_id.items():
        parent_message_id = row["_reconstruct_parent_id"]
        canonical_message_id = row["id"]
        parent_id_by_message_id[coverage_message_id] = ""
        if not parent_message_id:
            continue

        if parent_message_id in canonical_row_by_message_id:
            parent_coverage_id = canonical_row_by_message_id[parent_message_id][
                "_reconstruct_message_id"
            ]
            child_message_ids_by_parent_id[parent_coverage_id].append(coverage_message_id)
            parent_id_by_message_id[coverage_message_id] = parent_coverage_id

    for parent_coverage_id in list(child_message_ids_by_parent_id.keys()):
        child_message_ids_by_parent_id[parent_coverage_id].sort(
            key=lambda child_id: message_by_id[child_id]["_reconstruct_date_sort_key"]
        )

    return (
        message_by_id,
        dict(child_message_ids_by_parent_id),
        parent_id_by_message_id,
    )


def _find_graph_components(
    message_by_id: dict[str, dict],
    child_message_ids_by_parent_id: dict[str, list[str]],
    parent_id_by_message_id: dict[str, str],
) -> list[list[str]]:
    """Find weakly connected components of the reply graph."""
    undirected_neighbors_by_message_id: dict[str, set[str]] = {
        message_id: set() for message_id in message_by_id
    }

    for parent_message_id, child_message_ids in child_message_ids_by_parent_id.items():
        for child_message_id in child_message_ids:
            undirected_neighbors_by_message_id[parent_message_id].add(child_message_id)
            undirected_neighbors_by_message_id[child_message_id].add(parent_message_id)

    for message_id, parent_message_id in parent_id_by_message_id.items():
        if parent_message_id:
            undirected_neighbors_by_message_id[message_id].add(parent_message_id)
            undirected_neighbors_by_message_id[parent_message_id].add(message_id)

    components: list[list[str]] = []
    visited_message_ids: set[str] = set()

    for start_message_id in sorted(message_by_id.keys()):
        if start_message_id in visited_message_ids:
            continue
        stack = [start_message_id]
        component_message_ids: list[str] = []
        visited_message_ids.add(start_message_id)

        while stack:
            current_message_id = stack.pop()
            component_message_ids.append(current_message_id)
            for neighbor_message_id in sorted(
                undirected_neighbors_by_message_id[current_message_id]
            ):
                if neighbor_message_id in visited_message_ids:
                    continue
                visited_message_ids.add(neighbor_message_id)
                stack.append(neighbor_message_id)

        components.append(sorted(component_message_ids))

    return components


def _build_leaf_path(
    leaf_message_id: str,
    parent_id_by_message_id: dict[str, str],
    message_by_id: dict[str, dict],
) -> list[str]:
    """Build root->leaf path by following parent pointers from a leaf."""
    path_message_ids_reversed: list[str] = []
    seen_message_ids: set[str] = set()
    current_message_id = leaf_message_id

    while (
        current_message_id
        and current_message_id in message_by_id
        and current_message_id not in seen_message_ids
    ):
        seen_message_ids.add(current_message_id)
        path_message_ids_reversed.append(current_message_id)
        current_message_id = parent_id_by_message_id.get(current_message_id, "")

    path_message_ids_reversed.reverse()
    return path_message_ids_reversed


def _find_cycle_nodes_in_component(
    component_message_ids: list[str],
    parent_id_by_message_id: dict[str, str],
) -> list[str]:
    """Return one deterministic cycle node set from a component, if present."""
    visit_state_by_message_id: dict[str, str] = {}

    def dfs(message_id: str, stack: list[str]) -> list[str]:
        visit_state_by_message_id[message_id] = "visiting"
        stack.append(message_id)
        parent_message_id = parent_id_by_message_id.get(message_id, "")
        if parent_message_id and parent_message_id in component_message_ids:
            parent_state = visit_state_by_message_id.get(parent_message_id, "unvisited")
            if parent_state == "unvisited":
                cycle = dfs(parent_message_id, stack)
                if cycle:
                    return cycle
            elif parent_state == "visiting":
                cycle_start_index = stack.index(parent_message_id)
                return stack[cycle_start_index:].copy()
        stack.pop()
        visit_state_by_message_id[message_id] = "visited"
        return []

    for message_id in sorted(component_message_ids):
        if visit_state_by_message_id.get(message_id, "unvisited") != "unvisited":
            continue
        cycle_nodes = dfs(message_id, [])
        if cycle_nodes:
            return sorted(set(cycle_nodes))
    return []


def _build_cycle_component_path(
    component_message_ids: list[str],
    message_by_id: dict[str, dict],
    parent_id_by_message_id: dict[str, str],
) -> list[str]:
    """Build deterministic path for a cycle-only component.

    Cycle-break strategy:
    - choose cycle start by newest date (tie: lexicographic message id)
    - follow parent pointers until repeat
    """
    cycle_message_ids = _find_cycle_nodes_in_component(
        component_message_ids, parent_id_by_message_id
    )
    if not cycle_message_ids:
        return [
            sorted(
                component_message_ids,
                key=lambda message_id: message_by_id[message_id]["_reconstruct_date_sort_key"],
            )[-1]
        ]

    cycle_start_message_id = sorted(
        cycle_message_ids,
        key=lambda message_id: (
            message_by_id[message_id]["_reconstruct_date_sort_key"],
            message_id,
        ),
        reverse=True,
    )[0]

    path_message_ids: list[str] = []
    seen_message_ids: set[str] = set()
    current_message_id = cycle_start_message_id
    while current_message_id and current_message_id not in seen_message_ids:
        seen_message_ids.add(current_message_id)
        path_message_ids.append(current_message_id)
        parent_message_id = parent_id_by_message_id.get(current_message_id, "")
        if not parent_message_id or parent_message_id not in component_message_ids:
            break
        current_message_id = parent_message_id

    path_message_ids.reverse()
    return path_message_ids


def _compact_fork_paths(
    reconstructed_paths: list[list[str]],
    message_by_id: dict[str, dict],
) -> list[list[str]]:
    """Compact shared prefixes to reduce branch duplication.

    Limitation:
    - This is a lightweight heuristic. High-fork graphs may still duplicate
      meaningful context.
    """
    if len(reconstructed_paths) < 2:
        return reconstructed_paths

    # Stable order by leaf recency; first path remains baseline.
    sorted_paths = sorted(
        reconstructed_paths,
        key=lambda path: (
            message_by_id[path[-1]]["_reconstruct_date_sort_key"] if path else (0, "", ""),
            path[-1] if path else "",
        ),
        reverse=True,
    )

    compacted_paths: list[list[str]] = [sorted_paths[0]]
    for current_path in sorted_paths[1:]:
        best_shared_prefix = 0
        for kept_path in compacted_paths:
            shared_prefix_length = 0
            for left_message_id, right_message_id in zip(current_path, kept_path):
                if left_message_id != right_message_id:
                    break
                shared_prefix_length += 1
            if shared_prefix_length > best_shared_prefix:
                best_shared_prefix = shared_prefix_length

        if best_shared_prefix > SHARED_PREFIX_THRESHOLD:
            trimmed_path = current_path[max(0, best_shared_prefix - CONTEXT_TAIL_FOR_BRANCH) :]
            compacted_paths.append(trimmed_path)
        else:
            compacted_paths.append(current_path)

    return compacted_paths


def _ensure_full_message_coverage(
    reconstructed_paths: list[list[str]],
    all_message_ids: set[str],
) -> list[list[str]]:
    """Guarantee every message appears in at least one path."""
    covered_message_ids: set[str] = set()
    for path_message_ids in reconstructed_paths:
        covered_message_ids.update(path_message_ids)

    for uncovered_message_id in sorted(all_message_ids - covered_message_ids):
        reconstructed_paths.append([uncovered_message_id])

    return reconstructed_paths


def _reconstruct_paths_from_reply_graph(
    email_messages: list[dict],
) -> list[dict]:
    """Reconstruct chunkable paths from reply graph.

    Returns:
    - list of path descriptors: {message_ids, path_type, component_index, leaf_message_id}
    """
    normalized_rows = _normalize_thread_rows_for_reconstruction(email_messages)
    (
        message_by_id,
        child_message_ids_by_parent_id,
        parent_id_by_message_id,
    ) = _build_thread_graph(normalized_rows)

    reconstructed_path_descriptors: list[dict] = []

    component_message_id_groups = _find_graph_components(
        message_by_id=message_by_id,
        child_message_ids_by_parent_id=child_message_ids_by_parent_id,
        parent_id_by_message_id=parent_id_by_message_id,
    )

    all_message_ids = set(message_by_id.keys())

    for component_index, component_message_ids in enumerate(component_message_id_groups):
        component_leaf_message_ids = [
            message_id
            for message_id in component_message_ids
            if not child_message_ids_by_parent_id.get(message_id)
        ]
        component_leaf_message_ids = sorted(
            component_leaf_message_ids,
            key=lambda message_id: message_by_id[message_id]["_reconstruct_date_sort_key"],
            reverse=True,
        )

        if component_leaf_message_ids:
            leaf_paths: list[list[str]] = []
            seen_signatures: set[tuple[str, ...]] = set()
            for leaf_message_id in component_leaf_message_ids:
                path_message_ids = _build_leaf_path(
                    leaf_message_id=leaf_message_id,
                    parent_id_by_message_id=parent_id_by_message_id,
                    message_by_id=message_by_id,
                )
                path_signature = tuple(path_message_ids)
                if path_signature in seen_signatures:
                    continue
                seen_signatures.add(path_signature)
                leaf_paths.append(path_message_ids)
                reconstructed_path_descriptors.append(
                    {
                        "message_ids": path_message_ids,
                        "path_type": "leaf_path",
                        "component_index": component_index,
                        "leaf_message_id": leaf_message_id,
                    }
                )

            compacted_leaf_paths = _compact_fork_paths(leaf_paths, message_by_id)
            # Replace leaf paths for this component with compacted versions.
            retained_descriptors: list[dict] = []
            compacted_signatures = {tuple(path) for path in compacted_leaf_paths}
            for descriptor in reconstructed_path_descriptors:
                if (
                    descriptor["component_index"] != component_index
                    or descriptor["path_type"] != "leaf_path"
                ):
                    retained_descriptors.append(descriptor)
                    continue
                if tuple(descriptor["message_ids"]) in compacted_signatures:
                    retained_descriptors.append(descriptor)
                    compacted_signatures.remove(tuple(descriptor["message_ids"]))
            # Add compacted paths not already retained.
            for compacted_path in compacted_leaf_paths:
                if any(
                    descriptor["component_index"] == component_index
                    and descriptor["path_type"] == "leaf_path"
                    and descriptor["message_ids"] == compacted_path
                    for descriptor in retained_descriptors
                ):
                    continue
                retained_descriptors.append(
                    {
                        "message_ids": compacted_path,
                        "path_type": "leaf_path",
                        "component_index": component_index,
                        "leaf_message_id": compacted_path[-1] if compacted_path else "",
                    }
                )
            reconstructed_path_descriptors = retained_descriptors
            continue

        if len(component_message_ids) == 1:
            reconstructed_path_descriptors.append(
                {
                    "message_ids": [component_message_ids[0]],
                    "path_type": "orphan_component",
                    "component_index": component_index,
                    "leaf_message_id": component_message_ids[0],
                }
            )
            continue

        cycle_path_message_ids = _build_cycle_component_path(
            component_message_ids=component_message_ids,
            message_by_id=message_by_id,
            parent_id_by_message_id=parent_id_by_message_id,
        )
        reconstructed_path_descriptors.append(
            {
                "message_ids": cycle_path_message_ids,
                "path_type": "cycle_component",
                "component_index": component_index,
                "leaf_message_id": cycle_path_message_ids[-1] if cycle_path_message_ids else "",
            }
        )

    reconstructed_paths = [
        descriptor["message_ids"] for descriptor in reconstructed_path_descriptors
    ]
    reconstructed_paths = _ensure_full_message_coverage(
        reconstructed_paths=reconstructed_paths,
        all_message_ids=all_message_ids,
    )

    # Reattach metadata to possibly-extended path set.
    path_descriptor_by_signature = {
        tuple(descriptor["message_ids"]): descriptor
        for descriptor in reconstructed_path_descriptors
    }
    final_path_descriptors: list[dict] = []
    for path_message_ids in reconstructed_paths:
        signature = tuple(path_message_ids)
        if signature in path_descriptor_by_signature:
            final_path_descriptors.append(path_descriptor_by_signature[signature])
            continue
        # Coverage fallback singleton path.
        fallback_message_id = path_message_ids[0]
        final_path_descriptors.append(
            {
                "message_ids": path_message_ids,
                "path_type": "orphan_component",
                "component_index": -1,
                "leaf_message_id": fallback_message_id,
            }
        )

    # Materialize row objects for callers.
    materialized_descriptors: list[dict] = []
    for descriptor in final_path_descriptors:
        message_ids = descriptor["message_ids"]
        materialized_descriptors.append(
            {
                **descriptor,
                "messages": [message_by_id[message_id] for message_id in message_ids],
            }
        )

    return materialized_descriptors


def _build_windows_from_path(
    path_messages: list[dict],
    max_chars: int,
    max_emails_per_chunk: int,
    overlap_emails: int,
) -> list[list[tuple[int, dict]]]:
    windows: list[list[tuple[int, dict]]] = []
    start_index = 0
    last_emitted_end_index = -1
    while start_index < len(path_messages):
        window_messages: list[tuple[int, dict]] = []
        char_count = 0

        for message_index in range(start_index, len(path_messages)):
            message_order = message_index + 1
            message_obj = path_messages[message_index]
            message_block = _format_email_block(message_obj, message_order)
            if window_messages and char_count + len(message_block) + 2 > max_chars:
                break
            if len(window_messages) >= max_emails_per_chunk:
                break
            # The first email in a window bypasses the size check above (window_messages
            # is empty so the guard is False). Truncate its body here so it can't produce
            # a chunk that exceeds max_chars regardless of how large the raw email is.
            if not window_messages and len(message_block) > max_chars:
                message_obj = dict(message_obj)
                message_obj["body"] = (message_obj.get("body") or "")[:max_chars] + " [truncated]"
                message_block = _format_email_block(message_obj, message_order)
            window_messages.append((message_order, message_obj))
            char_count += len(message_block) + 2

        end_index = start_index + len(window_messages) - 1

        # If this window does not extend coverage beyond the previous window,
        # skip it and jump to the first uncovered message.
        if end_index <= last_emitted_end_index:
            start_index = last_emitted_end_index + 1
            continue

        windows.append(window_messages)
        last_emitted_end_index = end_index

        # Stop once this window already reaches the end of the path.
        if end_index >= len(path_messages) - 1:
            break
        advance_by = max(1, len(window_messages) - overlap_emails)
        start_index += advance_by

    return windows


class EmailChunker:
    """Chunker for email threads using reply-graph-aware path windows.

    Treats each thread as a document and groups reconstructed message paths into
    overlapping windows. For forked graphs, one branch path may be produced per
    leaf, with optional prefix compaction.

    Args:
        max_emails_per_chunk: Maximum number of emails per chunk (default 10).
        max_chars: Maximum characters per chunk (default 2048).
        overlap_emails: Number of emails to overlap between adjacent chunks
            within the same path (default 2).
    """

    def __init__(
        self,
        max_emails_per_chunk: int = 10,
        max_chars: int = 2048,
        overlap_emails: int = 2,
    ) -> None:
        self.max_emails_per_chunk = max_emails_per_chunk
        self.max_chars = max_chars
        self.overlap_emails = min(overlap_emails, max_emails_per_chunk - 1)

    def chunk_thread(self, email_messages: list[dict], thread_id: str) -> list[Chunk]:
        """Chunk a single email thread into one or more Chunk objects."""
        if not email_messages:
            return []

        path_descriptors = _reconstruct_paths_from_reply_graph(
            email_messages=email_messages,
        )

        chunks: list[Chunk] = []
        for thread_path_index, path_descriptor in enumerate(path_descriptors):
            path_messages = path_descriptor["messages"]
            if not path_messages:
                continue

            subject = path_messages[0].get("subject") or ""
            windows = _build_windows_from_path(
                path_messages=path_messages,
                max_chars=self.max_chars,
                max_emails_per_chunk=self.max_emails_per_chunk,
                overlap_emails=self.overlap_emails,
            )

            chain_key = f"{thread_id}-{thread_path_index}"
            path_chunk_descriptors: list[dict[str, object]] = []
            for chunk_index_in_path, window_messages in enumerate(windows):
                window_only_messages = [message for _, message in window_messages]
                blocks = [
                    _format_email_block(message, message_order)
                    for message_order, message in window_messages
                ]
                window_message_orders = [message_order for message_order, _ in window_messages]
                window_start_order = min(window_message_orders)
                window_end_order = max(window_message_orders)
                participant_info = extract_participants(window_only_messages)
                participants_display = _compact_participants_display(
                    str(participant_info["display"])
                )
                participants = list(participant_info["tokens"])
                date_start = date_yyyy_mm_dd(window_only_messages[0].get("date"))
                date_end = date_yyyy_mm_dd(window_only_messages[-1].get("date"))
                date_range_display = (
                    f"{date_start} to {date_end}"
                    if date_start and date_end
                    else date_start or date_end or "unknown"
                )

                content = (
                    f"Subject: {subject or 'unknown'}\n"
                    f"Date: {date_range_display}\n"
                    f"Participants: {participants_display}\n\n" + "\n\n".join(blocks)
                )
                chunk_id = f"{chain_key}-{chunk_index_in_path + 1}"
                path_chunk_descriptors.append(
                    {
                        "chunk_id": chunk_id,
                        "window_start_order": window_start_order,
                        "window_end_order": window_end_order,
                        "content": content,
                        "subject": subject,
                        "date_start": date_start,
                        "date_end": date_end,
                        "participants": participants,
                        "thread_message_count": len(path_messages),
                    }
                )

            for descriptor_index, descriptor in enumerate(path_chunk_descriptors):
                parent_chunk_id = ""
                start_order = int(descriptor["window_start_order"])
                if start_order > 1:
                    parent_message_order = start_order - 1
                    for previous_descriptor in reversed(path_chunk_descriptors[:descriptor_index]):
                        previous_start = int(previous_descriptor["window_start_order"])
                        previous_end = int(previous_descriptor["window_end_order"])
                        if previous_start <= parent_message_order <= previous_end:
                            parent_chunk_id = str(previous_descriptor["chunk_id"])
                            break
                descriptor["parent_chunk_id"] = parent_chunk_id

            child_chunk_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
            for descriptor in path_chunk_descriptors:
                parent_chunk_id = str(descriptor.get("parent_chunk_id") or "")
                if parent_chunk_id:
                    child_chunk_ids_by_parent_id[parent_chunk_id].append(
                        str(descriptor["chunk_id"])
                    )

            for descriptor in path_chunk_descriptors:
                chunk_id = str(descriptor["chunk_id"])
                child_chunk_ids = list(child_chunk_ids_by_parent_id.get(chunk_id, []))
                metadata: tuple[tuple[str, object], ...] = (
                    ("file", thread_id),
                    ("index", len(chunks)),
                    ("thread_id", thread_id),
                    ("chunk_id", chunk_id),
                    ("parent_chunk_id", descriptor.get("parent_chunk_id") or ""),
                    ("child_chunk_ids", child_chunk_ids),
                    ("subject", descriptor["subject"]),
                    ("date_start", descriptor["date_start"]),
                    ("date_end", descriptor["date_end"]),
                    ("participants", descriptor["participants"]),
                    ("thread_message_count", descriptor["thread_message_count"]),
                )
                chunks.append(Chunk(content=str(descriptor["content"]), metadata=metadata))

        return chunks

    def chunk_file(self, file_path: str | Path) -> list[Chunk]:
        """Chunk all email threads in a JSON or JSONL file."""
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")

        if file_path.suffix == ".jsonl":
            raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raw_rows = json.loads(text)
            if not isinstance(raw_rows, list):
                raise ValueError(
                    f"Expected a JSON array in {file_path}, got {type(raw_rows).__name__}"
                )

        schema_warnings = validate_rows(raw_rows)
        if schema_warnings:
            print(f"[email chunker] {file_path.name}: {len(schema_warnings)} schema warnings")

        threads_by_id: dict[str, list[dict]] = defaultdict(list)
        for email_message in raw_rows:
            thread_id = email_message.get("thread_id") or email_message.get("id") or ""
            threads_by_id[thread_id].append(email_message)

        all_chunks: list[Chunk] = []
        for thread_id, email_messages in threads_by_id.items():
            all_chunks.extend(self.chunk_thread(email_messages, thread_id))

        return all_chunks

    def chunk_folder(self, folder_path: str | Path) -> ChunkCollection:
        """Chunk all email JSON/JSONL files in a folder."""
        folder_path = Path(folder_path).resolve()
        all_chunks: list[Chunk] = []

        json_files = sorted(
            [
                *folder_path.rglob("*.json"),
                *folder_path.rglob("*.jsonl"),
            ]
        )
        if not json_files:
            print(f"No .json or .jsonl files found in {folder_path}")
            return ChunkCollection([])

        for file_path in json_files:
            try:
                chunks = self.chunk_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        print(
            f"Chunked {len(json_files)} file(s) -> "
            f"{len(all_chunks)} chunks ({len(set(c.get_metadata('thread_id') for c in all_chunks))} threads)"
        )
        return ChunkCollection(all_chunks)
