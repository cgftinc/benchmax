"""Load + detect + normalize a dataset/traces JSONL into a uniform view model.

The ``castform data view`` command sniffs the first non-empty line of a JSONL
file, classifies it into one of four known on-disk shapes (two RAG, two trace
variants), and normalizes every row into a single uniform shape per ``kind`` so
the HTML renderer (slice 2) stays simple. Anything unrecognized degrades to a
``generic`` raw-record view — this loader **never hard-errors** on input it does
not understand.

On-disk shapes (derived from the writer code, NOT the web-app viewer's
post-transform model — they differ):
  - rag/qa-gen          ``rag/qa_generation/formatters/train_eval.py:_to_row``
  - rag/scaffold        ``{prompt, ground_truth}`` plain strings (simple env)
  - traces/trace-example ``traces/processing.py:TrainingExample.to_jsonl_dict``
  - traces/normalized-trace ``traces/adapter.py:NormalizedTrace.from_dict``

The traces models live behind the ``[traces]`` extra; they are imported lazily
inside the trace path so RAG/generic viewing works without it, and a missing
extra falls back to the generic raw view rather than raising.
"""

from __future__ import annotations

import importlib.resources
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Top-level keys that map to dedicated columns in a qa-gen row; the rest roll
# into ``extra``.
_RAG_QAGEN_FIELDS = ("question", "answer", "qa_type", "reference_chunks")


def _classify(record: Any, type_override: str | None) -> tuple[str, str]:
    """Return ``(kind, variant)`` for the file from its first record.

    ``type_override`` (one of ``rag``/``traces``/``generic``) forces the kind;
    the variant is still sub-detected within rag/traces. Auto-detection uses a
    collision-free precedence (trace rows carry ``prompt_messages``, scaffold
    carries ``prompt``) and falls back to ``generic`` — never an error.
    """
    if type_override == "generic":
        return ("generic", "unknown")
    if type_override == "rag":
        return ("rag", _rag_variant(record))
    if type_override == "traces":
        return ("traces", _trace_variant(record))
    if type_override == "chunks":
        return ("chunks", "corpus-chunk")

    if not isinstance(record, dict):
        return ("generic", "unknown")
    if "prompt_messages" in record:
        return ("traces", "trace-example")
    if "messages" in record:
        return ("traces", "normalized-trace")
    if "question" in record and "answer" in record:
        return ("rag", "qa-gen")
    if "prompt" in record and "ground_truth" in record:
        return ("rag", "scaffold")
    # CorpusChunk shape: {id, content, metadata, score?} — checked after the
    # rag/trace shapes (none of which carry a top-level `content`).
    if "content" in record and "id" in record and "metadata" in record:
        return ("chunks", "corpus-chunk")
    return ("generic", "unknown")


def _rag_variant(record: Any) -> str:
    if isinstance(record, dict) and "question" not in record and "prompt" in record:
        return "scaffold"
    return "qa-gen"


def _trace_variant(record: Any) -> str:
    if isinstance(record, dict) and "prompt_messages" not in record and "messages" in record:
        return "normalized-trace"
    return "trace-example"


def _normalize_rag(record: Any, variant: str) -> dict[str, Any]:
    """Collapse either RAG variant into one row shape.

    qa-gen keeps ``question``/``answer``/``qa_type``/``reference_chunks`` and
    rolls remaining fields into ``extra``; scaffold maps ``prompt``→``question``
    and ``ground_truth``→``answer`` (no qa_type/reference_chunks).
    """
    record = record if isinstance(record, dict) else {}
    if variant == "scaffold":
        extra = {k: v for k, v in record.items() if k not in ("prompt", "ground_truth")}
        return {
            "question": record.get("prompt", ""),
            "answer": record.get("ground_truth", ""),
            "qa_type": None,
            "reference_chunks": None,
            "extra": extra or None,
        }
    extra = {k: v for k, v in record.items() if k not in _RAG_QAGEN_FIELDS}
    return {
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "qa_type": record.get("qa_type"),
        "reference_chunks": record.get("reference_chunks"),
        "extra": extra or None,
    }


_CHUNK_FIELDS = ("id", "content", "metadata", "score")


def _normalize_chunk(record: Any) -> dict[str, Any]:
    """Normalize a corpus chunk (`{id, content, metadata, score?}`) row; any
    other top-level fields roll into ``extra``.
    """
    record = record if isinstance(record, dict) else {}
    extra = {k: v for k, v in record.items() if k not in _CHUNK_FIELDS}
    return {
        "id": record.get("id", ""),
        "content": record.get("content", ""),
        "score": record.get("score"),
        "metadata": record.get("metadata") or {},
        "extra": extra or None,
    }


def _normalize_traces(records: list[Any], variant: str) -> list[dict[str, Any]]:
    """Collapse either trace variant into one row shape (one msg shape via
    ``TraceMessage.to_dict``). Raises ``ImportError`` if the ``[traces]`` extra
    is absent — the caller degrades to a generic view in that case.
    """
    from benchmax.traces.adapter import NormalizedTrace, normalize_message

    out: list[dict[str, Any]] = []
    for raw in records:
        record = raw if isinstance(raw, dict) else {}
        if variant == "normalized-trace":
            # from_dict requires "id"; default it so a snapshot missing the
            # field still renders rather than raising.
            nt = NormalizedTrace.from_dict({**record, "id": record.get("id", "")})
            metadata = dict(nt.metadata)
            if nt.timestamp is not None:
                metadata.setdefault("timestamp", nt.timestamp)
            if nt.errors:
                metadata.setdefault("errors", nt.errors)
            out.append({
                "id": nt.id,
                "turn_index": None,
                "scores": nt.scores,
                "metadata": metadata,
                "prompt_messages": [m.to_dict() for m in nt.messages],
                "completion_messages": [],
            })
            continue

        prompt_messages = [
            normalize_message(m).to_dict()
            for m in record.get("prompt_messages", [])
            if isinstance(m, dict)
        ]
        gt = record.get("ground_truth")
        completion_messages = (
            [normalize_message(gt).to_dict()] if isinstance(gt, dict) and gt else []
        )
        init = record.get("init_rollout_args")
        init = init if isinstance(init, dict) else {}
        metadata = {k: v for k, v in init.items() if k not in ("trace_id", "turn_index", "scores")}
        out.append({
            "id": init.get("trace_id", ""),
            "turn_index": init.get("turn_index"),
            "scores": init.get("scores", {}),
            "metadata": metadata,
            "prompt_messages": prompt_messages,
            "completion_messages": completion_messages,
        })
    return out


def _key_union(records: list[Any]) -> list[str]:
    """Top-level keys across all dict records, ordered by frequency desc then
    first appearance — the column order for the generic fallback table.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    seq = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in record:
            counts[key] += 1
            if key not in first_seen:
                first_seen[key] = seq
                seq += 1
    return sorted(counts, key=lambda k: (-counts[k], first_seen[k]))


def build_view_model(
    records: list[Any],
    *,
    source: str,
    type_override: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Classify + normalize parsed records into the embeddable view model.

    ``limit`` caps embedded rows (``0``/negative = all); ``total`` always
    reflects the full record count so the caller can report what was capped.
    """
    total = len(records)
    apply_cap = bool(limit and limit > 0)
    capped = apply_cap and total > limit
    shown = records[:limit] if apply_cap else list(records)

    first = next((r for r in shown if isinstance(r, dict)), None)
    kind, variant = _classify(first, type_override)

    if kind == "rag":
        rows: list[Any] = [_normalize_rag(r, variant) for r in shown]
    elif kind == "chunks":
        rows = [_normalize_chunk(r) for r in shown]
    elif kind == "traces":
        try:
            rows = _normalize_traces(shown, variant)
        except ImportError:
            kind, variant = "generic", "unknown"  # [traces] extra absent → raw view
            rows = list(shown)
    else:
        rows = list(shown)

    model: dict[str, Any] = {
        "kind": kind,
        "variant": variant,
        "source": source,
        "total": total,
        "shown": len(shown),
        "capped": capped,
        "rows": rows,
    }
    if kind == "generic":
        model["columns"] = _key_union(shown)
    return model


def _read_records(path: Path) -> list[Any]:
    """Parse a JSONL file into records, skipping blank and unparseable lines."""
    records: list[Any] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_view_model(
    path: str | Path,
    *,
    type_override: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Read a JSONL file and return its view model (see ``build_view_model``)."""
    path = Path(path)
    records = _read_records(path)
    return build_view_model(records, source=path.name, type_override=type_override, limit=limit)


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

_TEMPLATE_PACKAGE = "benchmax.cli"
_TEMPLATE_RESOURCE = "templates/viewer.html"
_VIEW_MODES = ("auto", "table", "messages", "raw")


def _safe_json(model: dict[str, Any]) -> str:
    """Serialize *model* for embedding in a ``<script>`` block.

    HTML entities are NOT decoded inside ``<script>`` raw text, so the data
    must stay valid JSON. We unicode-escape ``<``/``>``/``&`` (all valid JSON
    escapes) which both keeps ``JSON.parse`` working and prevents a ``</script>``
    (or ``<!--``) breakout from attacker-controlled row content. Load-bearing —
    the page renders user strings into its own ``file://`` origin.
    """
    raw = json.dumps(model, ensure_ascii=False)
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _load_template() -> str:
    return (
        importlib.resources.files(_TEMPLATE_PACKAGE)
        .joinpath(_TEMPLATE_RESOURCE)
        .read_text(encoding="utf-8")
    )


def render_html(model: dict[str, Any], *, view: str = "auto") -> str:
    """Render the self-contained viewer HTML with *model* embedded as JSON."""
    if view not in _VIEW_MODES:
        view = "auto"
    template = _load_template()
    # Substitute the view marker first so an injected payload that happens to
    # contain the marker token can never be re-scanned.
    html = template.replace("__INITIAL_VIEW__", view)
    return html.replace("__VIEW_MODEL_JSON__", _safe_json(model))


def write_html(model: dict[str, Any], out_path: str | Path, *, view: str = "auto") -> Path:
    """Render and write the viewer to *out_path*; return the path written."""
    out = Path(out_path)
    out.write_text(render_html(model, view=view), encoding="utf-8")
    return out
