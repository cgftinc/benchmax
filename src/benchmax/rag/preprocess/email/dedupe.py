#!/usr/bin/env python3
"""Thread-first MinHash dedup for parsed email JSONL

See preprocess/email/schema.py for expected input schema and validation helpers.

Note:
- Participant identity cleanup (redactions/placeholders/formatting variants)
  should happen during parsing; deduper consumes canonicalized participant fields.

Outputs:
- Message-level deduped JSONL: same row schema as input.
- Report JSON with counts, cluster samples, and consolidation_map.

High-level dedupe flow:
1. Construct deterministic ordered threads from parsed message rows.
2. Generate MinHash/LSH candidate thread pairs.
3. Verify candidate pairs with exact Jaccard and apply guards:
   overall text, body-only text, and participant similarity.
4. Collapse merge-eligible pairs into clusters and keep one representative per cluster.
5. Run subset dedupe on thread subject+body text, with participant similarity guard.
6. Emit message-level output and audit report.

Public entrypoints:
- `dedupe_email_jsonl(...)`: read parsed message JSONL, dedupe, write outputs + report.
- `dedupe_parsed_messages(...)`: run dedupe in-memory on already loaded rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from benchmax.rag.preprocess.email.schema import validate_rows

# -----------------------------------------------------------------------------
# Defaults / tuning constants
# -----------------------------------------------------------------------------

DEFAULT_INPUT_PATH = "emails_parsed.jsonl"
DEFAULT_OUTPUT_PATH = "emails_deduped.jsonl"
DEFAULT_REPORT_PATH = "dedup_report.json"

# MinHash + Jaccard settings (paper-inspired defaults requested by user)
MINHASH_BANDS = 9
MINHASH_ROWS = 27
SHINGLE_SIZE = 5
MINHASH_SEED = 17
JACCARD_THRESHOLD = 0.9
OVERALL_BODY_CHECK_THRESHOLD = 0.9
BODY_JACCARD_THRESHOLD = 0.9
PARTICIPANT_JACCARD_THRESHOLD = 0.9

# Reporting limits
MAX_REPORT_CLUSTERS = 60
MAX_REPORT_MEMBERS = 25
MAX_REPORT_MESSAGES_PER_THREAD = 25
MAX_REPLY_TO_DIAGNOSTIC_SAMPLES = 50
SUBSET_MAX_SUPERSET_LEN_RATIO = 6.0
SUBSET_MAX_PARTICIPANT_COUNT_DIFF = 1

# Hash/minhash internals
P61 = (1 << 61) - 1
MAX_HASH = (1 << 63) - 1

# Text normalization helpers
TOKEN_RE = re.compile(r"[a-z0-9']+")
WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupeConfig:
    """Runtime config for dedupe behavior.

    Kept as a dataclass so callers can override only what they need while
    preserving default behavior everywhere else.
    """

    bands: int = MINHASH_BANDS
    rows: int = MINHASH_ROWS
    shingle_size: int = SHINGLE_SIZE
    seed: int = MINHASH_SEED
    jaccard_threshold: float = JACCARD_THRESHOLD
    overall_body_check_threshold: float = OVERALL_BODY_CHECK_THRESHOLD
    body_jaccard_threshold: float = BODY_JACCARD_THRESHOLD
    participant_jaccard_threshold: float = PARTICIPANT_JACCARD_THRESHOLD
    max_report_clusters: int = MAX_REPORT_CLUSTERS
    max_report_members: int = MAX_REPORT_MEMBERS
    max_report_messages_per_thread: int = MAX_REPORT_MESSAGES_PER_THREAD


@dataclass
class ThreadDoc:
    idx: int
    thread_id: str
    thread_signature_id: str
    messages: list[dict[str, Any]]
    message_indices: list[int]
    participants_norm: list[str]
    full_text_norm: str
    body_text_norm: str
    thread_text_subject_body: str
    shingles: set[int]
    body_shingles: set[int]
    participant_shingles: set[int]
    signature: list[int]


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _as_int(v: Any, fallback: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return fallback


def _norm_ws(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _stable_hash64(text: str) -> int:
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big") & MAX_HASH


def _build_shingles(tokens: list[str], shingle_size: int) -> set[int]:
    if not tokens:
        return {0}
    if len(tokens) < shingle_size:
        return {_stable_hash64(" ".join(tokens))}
    out: set[int] = set()
    for i in range(0, len(tokens) - shingle_size + 1):
        out.add(_stable_hash64(" ".join(tokens[i : i + shingle_size])))
    return out or {0}


def _exact_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _minhash_signature(
    shingle_hashes: set[int], a_coeffs: list[int], b_coeffs: list[int]
) -> list[int]:
    if not shingle_hashes:
        return [MAX_HASH] * len(a_coeffs)
    sig = [MAX_HASH] * len(a_coeffs)
    for x in shingle_hashes:
        for i, (a, b) in enumerate(zip(a_coeffs, b_coeffs)):
            v = (a * x + b) % P61
            if v < sig[i]:
                sig[i] = v
    return sig


def _extract_participants_from_message(row: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def add_addr(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            t = _norm_ws(v).lower()
            if t:
                values.append(t)
            return
        if isinstance(v, dict):
            name = _as_str(v.get("name"))
            email = _as_str(v.get("email"))
            if name:
                add_addr(name)
            if email:
                add_addr(email)
            return
        if isinstance(v, list):
            for item in v:
                add_addr(item)

    add_addr(row.get("from"))
    add_addr(row.get("to"))
    add_addr(row.get("cc"))
    return sorted(set(values))


def _thread_message_sort_key(item: tuple[int, dict[str, Any]]) -> tuple[str, str, int]:
    i, row = item
    return (
        _as_str(row.get("date")),
        _as_str(row.get("id")),
        i,
    )


def _make_thread_signature_id(participants_norm: list[str], thread_text_subject_body: str) -> str:
    raw = "\n".join(["participants=" + "|".join(participants_norm), thread_text_subject_body])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _order_thread_items_by_reply_to(
    items: list[tuple[int, dict[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any]]], int, list[dict[str, Any]]]:
    """Order one thread's messages using reply_to linkage.

    Returns:
    - ordered items
    - orphan reply count (reply_to points to unknown id)
    - sample orphan diagnostics
    """

    rows_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for i, row in items:
        mid = _as_str(row.get("id"))
        if mid and mid not in rows_by_id:
            rows_by_id[mid] = (i, row)

    children: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    roots: list[tuple[int, dict[str, Any]]] = []
    orphan_reply_count = 0
    orphan_samples: list[dict[str, Any]] = []

    for i, row in items:
        rid = _as_str(row.get("reply_to"))
        if not rid:
            roots.append((i, row))
            continue
        if rid not in rows_by_id:
            orphan_reply_count += 1
            if len(orphan_samples) < MAX_REPLY_TO_DIAGNOSTIC_SAMPLES:
                orphan_samples.append(
                    {
                        "message_id": _as_str(row.get("id")),
                        "reply_to": rid,
                    }
                )
            roots.append((i, row))
            continue
        children.setdefault(rid, []).append((i, row))

    roots.sort(key=_thread_message_sort_key)
    for key in list(children.keys()):
        children[key].sort(key=_thread_message_sort_key)

    ordered: list[tuple[int, dict[str, Any]]] = []
    visited_ids: set[str] = set()

    def visit(item: tuple[int, dict[str, Any]]) -> None:
        _, row = item
        mid = _as_str(row.get("id"))
        if mid in visited_ids:
            return
        visited_ids.add(mid)
        ordered.append(item)
        for child in children.get(mid, []):
            visit(child)

    for root in roots:
        visit(root)

    # Include any disconnected/cyclic leftovers deterministically.
    leftovers = sorted(items, key=_thread_message_sort_key)
    for item in leftovers:
        _, row = item
        mid = _as_str(row.get("id"))
        if mid not in visited_ids:
            visit(item)

    return ordered, orphan_reply_count, orphan_samples


def _choose_cluster_representative(members: list[ThreadDoc]) -> ThreadDoc:
    # Deterministic representative selection for stable output diffs.
    return sorted(members, key=lambda t: (-len(t.full_text_norm), -len(t.messages), t.idx))[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _construct_threads(
    rows: list[dict[str, Any]],
    config: DedupeConfig,
    a_coeffs: list[int],
    b_coeffs: list[int],
) -> tuple[list[ThreadDoc], dict[str, Any]]:
    """Build deterministic thread documents from message rows.

    Each `ThreadDoc` carries all normalized text views required by the dedupe
    algorithm: full text, body-only text, subject+body text, and participants.
    """

    by_thread: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, row in enumerate(rows):
        tid = _as_str(row.get("thread_id")) or f"thread_{i}"
        by_thread.setdefault(tid, []).append((i, row))

    thread_docs: list[ThreadDoc] = []
    orphan_reply_to_count = 0
    orphan_reply_to_samples: list[dict[str, Any]] = []

    for t_idx, tid in enumerate(sorted(by_thread.keys())):
        items_sorted, orphan_count, orphan_samples = _order_thread_items_by_reply_to(by_thread[tid])
        orphan_reply_to_count += orphan_count
        for sample in orphan_samples:
            if len(orphan_reply_to_samples) >= MAX_REPLY_TO_DIAGNOSTIC_SAMPLES:
                break
            orphan_reply_to_samples.append({"thread_id": tid, **sample})

        msg_indices = [i for i, _ in items_sorted]
        messages = [row for _, row in items_sorted]

        # Build normalized thread-level representations.
        thread_participants: list[str] = []
        combined_lines: list[str] = []
        body_lines: list[str] = []
        subject_body_lines: list[str] = []

        for row in messages:
            msg_participants = _extract_participants_from_message(row)
            thread_participants.extend(msg_participants)
            subject = _norm_ws(_as_str(row.get("subject"))).lower()
            body = _norm_ws(_as_str(row.get("body"))).lower()
            participants_blob = " | ".join(msg_participants)

            combined_lines.append(
                f"participants: {participants_blob} | subject: {subject} | body: {body}"
            )
            body_lines.append(body)
            subject_body_lines.append(f"subject: {subject} | body: {body}")

        participants_norm = sorted(set(thread_participants))
        combined_text_norm = _norm_ws("\n".join(combined_lines))
        body_text_norm = _norm_ws("\n".join(body_lines))
        thread_text_subject_body = _norm_ws("\n".join(subject_body_lines))

        shingles = _build_shingles(_tokenize(combined_text_norm), config.shingle_size)
        body_shingles = _build_shingles(_tokenize(body_text_norm), config.shingle_size)
        participant_shingles = {_stable_hash64(p) for p in participants_norm} or {0}
        signature = _minhash_signature(shingles, a_coeffs, b_coeffs)

        thread_docs.append(
            ThreadDoc(
                idx=t_idx,
                thread_id=tid,
                thread_signature_id=_make_thread_signature_id(
                    participants_norm=participants_norm,
                    thread_text_subject_body=thread_text_subject_body,
                ),
                messages=messages,
                message_indices=msg_indices,
                participants_norm=participants_norm,
                full_text_norm=combined_text_norm,
                body_text_norm=body_text_norm,
                thread_text_subject_body=thread_text_subject_body,
                shingles=shingles,
                body_shingles=body_shingles,
                participant_shingles=participant_shingles,
                signature=signature,
            )
        )

    diagnostics = {
        "orphan_reply_to_count": orphan_reply_to_count,
        "orphan_reply_to_samples": orphan_reply_to_samples,
    }
    return thread_docs, diagnostics


def _build_minhash_coefficients(config: DedupeConfig) -> tuple[list[int], list[int]]:
    num_perm = config.bands * config.rows
    rnd = random.Random(config.seed)
    a_coeffs = [rnd.randrange(1, P61 - 1) for _ in range(num_perm)]
    b_coeffs = [rnd.randrange(0, P61 - 1) for _ in range(num_perm)]
    return a_coeffs, b_coeffs


def dedupe_parsed_messages(
    rows: list[dict[str, Any]],
    *,
    config: DedupeConfig = DedupeConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run thread-first dedupe on in-memory parsed rows.

    Returns:
    - kept_messages: message-level rows (same schema as input)
    - report: audit metadata and consolidation map
    """

    if config.bands <= 0 or config.rows <= 0:
        raise ValueError("bands and rows must be > 0")

    schema_warnings = validate_rows(rows)

    a_coeffs, b_coeffs = _build_minhash_coefficients(config)
    num_perm = config.bands * config.rows

    threads, diagnostics = _construct_threads(rows, config, a_coeffs, b_coeffs)

    # Step 1: LSH candidate generation from MinHash signatures.
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for t in threads:
        for b in range(config.bands):
            s = b * config.rows
            e = s + config.rows
            key = (b, tuple(t.signature[s:e]))
            buckets.setdefault(key, []).append(t.idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, j in combinations(sorted(members), 2):
            candidate_pairs.add((i, j))

    # Step 2: exact Jaccard verification on full thread text + guards.
    verified_pairs: list[tuple[int, int, float, float, float]] = []
    for i, j in sorted(candidate_pairs):
        overall_jac = _exact_jaccard(threads[i].shingles, threads[j].shingles)
        if overall_jac < config.jaccard_threshold:
            continue
        body_jac = _exact_jaccard(threads[i].body_shingles, threads[j].body_shingles)
        participant_jac = _exact_jaccard(
            threads[i].participant_shingles, threads[j].participant_shingles
        )
        verified_pairs.append((i, j, overall_jac, body_jac, participant_jac))

    dsu = DSU(len(threads))
    merge_eligible_pairs: list[tuple[int, int, float, float, float]] = []
    body_guard_blocked_pairs: list[tuple[int, int, float, float, float]] = []
    participant_guard_blocked_pairs: list[tuple[int, int, float, float, float]] = []

    for i, j, overall_jac, body_jac, participant_jac in verified_pairs:
        if (
            overall_jac <= config.overall_body_check_threshold
            or body_jac <= config.body_jaccard_threshold
        ):
            body_guard_blocked_pairs.append((i, j, overall_jac, body_jac, participant_jac))
            continue
        if participant_jac <= config.participant_jaccard_threshold:
            participant_guard_blocked_pairs.append((i, j, overall_jac, body_jac, participant_jac))
            continue

        dsu.union(i, j)
        merge_eligible_pairs.append((i, j, overall_jac, body_jac, participant_jac))

    pair_lookup: dict[tuple[int, int], tuple[float, float, float]] = {}
    for i, j, overall_jac, body_jac, participant_jac in verified_pairs:
        pair_lookup[(min(i, j), max(i, j))] = (overall_jac, body_jac, participant_jac)

    # Step 3: cluster collapse (thread-level) after merge-eligible unions.
    clusters_by_root: dict[int, list[ThreadDoc]] = {}
    for t in threads:
        root = dsu.find(t.idx)
        clusters_by_root.setdefault(root, []).append(t)

    kept_after_cluster: set[int] = set()
    removed_by_minhash: dict[int, dict[str, Any]] = {}
    cluster_samples: list[dict[str, Any]] = []

    for root, members in sorted(clusters_by_root.items()):
        rep = _choose_cluster_representative(members)
        kept_after_cluster.add(rep.idx)

        if len(members) <= 1:
            continue

        members_sorted = sorted(members, key=lambda x: x.idx)
        cluster_samples.append(
            {
                "cluster_root_idx": root,
                "cluster_size": len(members_sorted),
                "kept_idx": rep.idx,
                "kept_thread_id": rep.thread_id,
                "sample_members": [
                    {
                        "idx": m.idx,
                        "thread_id": m.thread_id,
                        "message_count": len(m.messages),
                    }
                    for m in members_sorted[: config.max_report_members]
                ],
            }
        )

        for m in members_sorted:
            if m.idx == rep.idx:
                continue
            max_overall = 0.0
            max_body = 0.0
            max_participant = 0.0
            for other in members_sorted:
                if other.idx == m.idx:
                    continue
                key = (min(m.idx, other.idx), max(m.idx, other.idx))
                if key in pair_lookup:
                    o, b, p = pair_lookup[key]
                    max_overall = max(max_overall, o)
                    max_body = max(max_body, b)
                    max_participant = max(max_participant, p)

            removed_by_minhash[m.idx] = {
                "reason": "minhash_thread_cluster",
                "dropped_idx": m.idx,
                "dropped_thread_id": m.thread_id,
                "dropped_message_count": len(m.messages),
                "kept_idx": rep.idx,
                "kept_thread_id": rep.thread_id,
                "kept_message_count": len(rep.messages),
                "max_cluster_jaccard": round(max_overall, 6),
                "max_cluster_body_jaccard": round(max_body, 6),
                "max_cluster_participant_jaccard": round(max_participant, 6),
                "dropped_thread_rows": m.messages[: config.max_report_messages_per_thread],
                "kept_thread_rows": rep.messages[: config.max_report_messages_per_thread],
            }

    # Step 4: subset dedupe on thread-level subject+body text.
    # Additional guard: participants must also be sufficiently similar.
    kept_candidates = sorted(kept_after_cluster)
    removed_by_subset: dict[int, dict[str, Any]] = {}
    subset_participant_guard_blocked_pairs = 0

    order = sorted(
        kept_candidates, key=lambda idx: (len(threads[idx].thread_text_subject_body), idx)
    )

    for pos, i in enumerate(order):
        if i in removed_by_subset:
            continue
        smaller = threads[i].thread_text_subject_body
        if not smaller:
            continue
        len_i = len(smaller)
        max_len = int(len_i * SUBSET_MAX_SUPERSET_LEN_RATIO)
        pcount_i = len(threads[i].participants_norm)

        for j in order[pos + 1 :]:
            if j in removed_by_subset:
                continue
            larger = threads[j].thread_text_subject_body
            len_j = len(larger)
            if not larger or len_j < len_i:
                continue
            if len_j > max_len:
                break
            pcount_j = len(threads[j].participants_norm)
            if abs(pcount_i - pcount_j) > SUBSET_MAX_PARTICIPANT_COUNT_DIFF:
                continue

            if smaller not in larger:
                continue

            subset_participant_jac = _exact_jaccard(
                threads[i].participant_shingles, threads[j].participant_shingles
            )
            if subset_participant_jac <= config.participant_jaccard_threshold:
                subset_participant_guard_blocked_pairs += 1
                continue

            removed_by_subset[i] = {
                "reason": "subset_thread_subject_body",
                "dropped_idx": i,
                "dropped_thread_id": threads[i].thread_id,
                "dropped_message_count": len(threads[i].messages),
                "kept_idx": j,
                "kept_thread_id": threads[j].thread_id,
                "kept_message_count": len(threads[j].messages),
                "subset_len": len(smaller),
                "superset_len": len(larger),
                "subset_participant_jaccard": round(subset_participant_jac, 6),
                "dropped_thread_rows": threads[i].messages[: config.max_report_messages_per_thread],
                "kept_thread_rows": threads[j].messages[: config.max_report_messages_per_thread],
            }
            break

    all_removed: dict[int, dict[str, Any]] = {}
    all_removed.update(removed_by_minhash)
    all_removed.update(removed_by_subset)

    kept_threads = [t for t in threads if t.idx not in all_removed]
    kept_threads.sort(key=lambda t: (t.thread_id, t.idx))

    kept_messages: list[dict[str, Any]] = []
    for t in kept_threads:
        kept_messages.extend(t.messages)

    consolidation_map = [dict(all_removed[k]) for k in sorted(all_removed)]

    removed_by_reason: dict[str, int] = {}
    for rec in consolidation_map:
        reason = _as_str(rec.get("reason"))
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1

    report: dict[str, Any] = {
        "params": {
            "jaccard_threshold": config.jaccard_threshold,
            "bands": config.bands,
            "rows": config.rows,
            "num_perm": num_perm,
            "shingle_size": config.shingle_size,
            "seed": config.seed,
            "overall_body_check_threshold": config.overall_body_check_threshold,
            "body_jaccard_threshold": config.body_jaccard_threshold,
            "participant_jaccard_threshold": config.participant_jaccard_threshold,
            "subset_max_superset_len_ratio": SUBSET_MAX_SUPERSET_LEN_RATIO,
            "subset_max_participant_count_diff": SUBSET_MAX_PARTICIPANT_COUNT_DIFF,
        },
        "counts": {
            "input_messages": len(rows),
            "input_threads": len(threads),
            "lsh_buckets": len(buckets),
            "candidate_pairs": len(candidate_pairs),
            "verified_pairs": len(verified_pairs),
            "merge_eligible_pairs": len(merge_eligible_pairs),
            "body_guard_blocked_pairs": len(body_guard_blocked_pairs),
            "participant_guard_blocked_pairs": len(participant_guard_blocked_pairs),
            "minhash_clusters": sum(1 for members in clusters_by_root.values() if len(members) > 1),
            "removed_minhash": len(removed_by_minhash),
            "removed_subset": len(removed_by_subset),
            "subset_participant_guard_blocked_pairs": subset_participant_guard_blocked_pairs,
            "removed_total": len(all_removed),
            "kept_threads_total": len(kept_threads),
            "output_messages_total": len(kept_messages),
            "orphan_reply_to_count": diagnostics["orphan_reply_to_count"],
            "schema_warning_count": len(schema_warnings),
        },
        "schema_warnings_sample": schema_warnings[:50],
        "removed_by_reason": removed_by_reason,
        "orphan_reply_to_samples": diagnostics["orphan_reply_to_samples"],
        "minhash_clusters_sample": sorted(
            cluster_samples,
            key=lambda x: x["cluster_size"],
            reverse=True,
        )[: config.max_report_clusters],
        "consolidation_map": consolidation_map,
    }

    return kept_messages, report


def dedupe_email_jsonl(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    dry_run: bool = False,
    config: DedupeConfig = DedupeConfig(),
) -> dict[str, Any]:
    """High-level reusable API to dedupe parsed email JSONL files."""

    in_path = Path(input_path)
    out_path = Path(output_path)
    rep_path = Path(report_path)

    rows = _read_jsonl(in_path)
    kept_messages, report = dedupe_parsed_messages(rows, config=config)

    # Attach file metadata after algorithm execution.
    report["input_path"] = str(in_path)
    report["output_path"] = str(out_path)
    report["dry_run"] = bool(dry_run)

    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not dry_run:
        _write_jsonl(out_path, kept_messages)

    return report


def dedupe_email_folder(
    folder: str | Path,
    output_name: str = "_deduped.jsonl",
    report_name: str = "_dedup_report.json",
    *,
    config: DedupeConfig = DedupeConfig(),
) -> dict[str, Any]:
    """Combine all .jsonl files in a folder, dedupe, and write a single output.

    Skips files whose name starts with '_' (preprocessor outputs).
    Writes <folder>/<output_name> and <folder>/<report_name>.

    Returns the dedupe report dict.
    """
    folder = Path(folder)
    jsonl_files = sorted(f for f in folder.rglob("*.jsonl") if not f.name.startswith("_"))
    if not jsonl_files:
        print(f"No .jsonl files found in {folder}")
        return {"counts": {"input_messages": 0, "output_messages_total": 0}}

    all_rows: list[dict[str, Any]] = []
    for f in jsonl_files:
        rows = _read_jsonl(f)
        print(f"  Read {f.name}: {len(rows)} messages")
        all_rows.extend(rows)
    print(f"Combined: {len(all_rows)} messages from {len(jsonl_files)} files")

    kept_messages, report = dedupe_parsed_messages(all_rows, config=config)

    out_path = folder / output_name
    _write_jsonl(out_path, kept_messages)

    rep_path = folder / report_name
    report["input_path"] = str(folder)
    report["output_path"] = str(out_path)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    c = report["counts"]
    print(
        f"Dedupe: {c['input_messages']} msgs / {c['input_threads']} threads "
        f"→ {c['output_messages_total']} msgs / {c['kept_threads_total']} threads "
        f"(removed {c['removed_total']} threads)"
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thread-first MinHash dedup for parsed emails.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single parsed-email JSONL file")
    group.add_argument("--folder", help="Folder to combine and dedupe all .jsonl files in")
    p.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--report", default=DEFAULT_REPORT_PATH)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.folder:
        dedupe_email_folder(args.folder)
    else:
        report = dedupe_email_jsonl(
            input_path=args.input,
            output_path=args.output,
            report_path=args.report,
            dry_run=args.dry_run,
        )
        c = report["counts"]
        print(
            "Dedup complete: "
            f"threads_in={c['input_threads']} threads_kept={c['kept_threads_total']} "
            f"threads_removed={c['removed_total']} messages_out={c['output_messages_total']}"
        )
        print(f"Report: {args.report}")
        if not args.dry_run:
            print(f"Message output: {args.output}")


if __name__ == "__main__":
    main()
