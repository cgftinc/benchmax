"""Apply the pass-2 (codex) overturns to verdicts_v2.jsonl (reproducible).

Genuine defects and the reinstated Cloud Flare KEEP are deterministic; the templated
pairs are re-decided LIVE against the (fixed) R3 equivalence set — KEEP iff the
equivalence set now covers >= 1 duplicate (size >= 2), else DROP; the five citation
fixes re-quote the answering gold line and mark the verdict ``fix`` (retained). Writes
the updated verdicts + an audit table (``overturns_applied.json``) and prints the
per-pair disposition.
"""

from __future__ import annotations

import json
from pathlib import Path

from overturns import DROP_REASON, FIX_CITATION, resolve
from equivalence import build_equivalence_sets
from build_golden import _current_spec
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.credentials import resolve_read_dsn_provider

_DATASETS = Path(__file__).resolve().parent / "datasets"
_VERDICTS = _DATASETS / "verdicts_v2.jsonl"
_MIRROR = Path(__file__).resolve().parent / "scratchpad" / "verdicts_v2.jsonl"


def main() -> int:
    ov = resolve(str(_DATASETS))
    rows = [json.loads(l) for l in _VERDICTS.read_text().splitlines() if l.strip()]
    by_key = {(r["query"]): r for r in rows}

    # live equiv-set sizes for the templated pairs
    templated = [o for o in ov if o["action"] == "templated"]
    ro = NeonClient(resolve_read_dsn_provider(None))
    spec = _current_spec()
    eq = build_equivalence_sets(ro, spec, [o["gold_hash"] for o in templated])

    audit = []
    for o in ov:
        r = by_key.get(o["query"])
        assert r is not None and o["gold_hash"] in r["gold_chunk_hashes"], o["id"]
        action = o["action"]
        if action == "drop":
            r["verdict"], r["reason"], r["citation"] = "drop", DROP_REASON[o["id"]], o["note"]
            disp = "drop"
        elif action == "restore":
            r["verdict"], r["reason"] = "keep", "valid"
            r["citation"] = "two-alternative answer is complete: 'at Cloud Connector LB (Cloud Flare) or AI Gateway middleware'"
            disp = "keep(restored)"
        elif action == "fix":
            r["verdict"], r["reason"], r["citation"] = "fix", "citation-corrected", FIX_CITATION[o["id"]]
            disp = "fix(citation)"
        elif action == "templated":
            size = len(eq.get(o["gold_hash"], [o["gold_hash"]]))
            if size >= 2:
                r["verdict"], r["reason"] = "keep", "valid"
                r["citation"] = f"templated but equivalence set covers {size - 1} duplicate(s); metric credits any"
                disp = f"keep(equiv={size})"
            else:
                r["verdict"], r["reason"], r["citation"] = "drop", "non-unique-templated-answer", (
                    f"templated / non-unique gold and equivalence set found no duplicate (size {size}); no unique gold"
                )
                disp = f"drop(equiv={size})"
        audit.append({"id": o["id"], "disposition": disp, "reason": r["reason"], "note": o["note"]})

    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    _VERDICTS.write_text(text, encoding="utf-8")
    if _MIRROR.parent.exists():
        _MIRROR.write_text(text, encoding="utf-8")
    (_DATASETS / "overturns_applied.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    from collections import Counter

    tally = Counter(r["verdict"] for r in rows)
    print("verdict tally:", dict(tally))
    print("retained (keep+fix):", tally["keep"] + tally["fix"], "dropped:", tally["drop"])
    for a in audit:
        print(f"  {a['id']:6} -> {a['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
