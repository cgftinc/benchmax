"""Pass-2 (codex) overturns applied to the answerability verdicts.

Each entry is ``(id, qa_file, line, gold_prefix, action, note)``. Actions:

* ``drop`` — a genuine intrinsic defect; dropped regardless of the metric.
* ``templated`` — templated / non-unique gold; KEPT iff the R3 equivalence set now
  covers the duplicate chunks (equiv-set size >= 2), else dropped.
* ``fix`` — gold answers but the recorded citation failed the contract; the citation
  is re-derived (or the pair dropped if the gold does not actually answer).
* ``restore`` — a prior over-eager drop reinstated as a valid KEEP.

Line numbers are 1-based into the reused qa jsonl; the gold prefix disambiguates.
"""

from __future__ import annotations

QA = {
    "LE": "qa_lexical/eval_dataset.jsonl",
    "LT": "qa_lexical/train_dataset.jsonl",
    "VE": "qa_vector/eval_dataset.jsonl",
    "VT": "qa_vector/train_dataset.jsonl",
}

# (id, line, gold_prefix, action, note)
OVERTURNS = [
    # genuine defects — drop regardless of equivalence
    ("VT:53", 53, "28708463", "drop", "answer-leak: query supplies the 32,767 limit then asks for the maximum"),
    ("LT:32", 32, "29e0c45f", "drop", "wrong-gold: gold only says 'See the Motivation section', no answer"),
    ("VT:7", 7, "af21454b", "drop", "stripped-anchor: 'this stricter approach' drops the Category Maturity Scorecard anchor"),
    ("VT:40", 40, "0e1cad4b", "drop", "stripped-anchor: 'in this workflow' drops the Compliance Frameworks anchor"),
    ("VT:20", 20, "fe279826", "drop", "under-spec: Canada benefits rule omits 'Canada'"),
    # templated / non-unique gold — keep iff equivalence set covers the duplicates
    ("LE:5", 5, "8eabbfe2", "templated", "AI Model Validation dup ai-core/data-science"),
    ("LT:7", 7, "b6645974", "templated", "new-hire seller omits segment; mid-market vs enterprise"),
    ("LT:8", 8, "355e0f51", "templated", "async daily updates omits team; stand-up template many pages"),
    ("LT:31", 31, "84b722a5", "templated", "Strategic Field Org dup; citation omits team"),
    ("LT:33", 33, "34eb62f4", "templated", "k8s support policy dup build//distribution/"),
    ("LT:39", 39, "c768add1", "templated", "solution-validation confidence copied UX Research/Product Dev"),
    ("LT:46", 46, "061178aa", "templated", "RED-data/Tier-1 recurs Critical Systems/Logging"),
    ("VE:7", 7, "d6e38c49", "templated", "AI indemnification dup current+versioned terms"),
    ("VT:11", 11, "8264a512", "templated", "senior competency triad across matrices"),
    ("VT:29", 29, "ccc77f95", "templated", "Distribution MR guidance byte-for-byte build//distribution/"),
    ("VT:31", 31, "41fd94d2", "templated", "customer-attribution consent across agreements"),
    ("VT:45", 45, "b0e04cd6", "templated", "confidentiality clause current+v5 testing agreements"),
    ("VT:56", 56, "44aa2f69", "templated", "community triage dup two Distribution paths"),
    ("VT:59", 59, "28f471c4", "templated", "FedRAMP guidance across Dynamic/Composition Analysis; query names neither"),
    ("VT:64", 64, "e3e20d1f", "templated", "add-on invoicing across agreements"),
    ("VT:65", 65, "c262d76f", "templated", "Siphon index_granularity=512 in two design docs"),
    # gold answers but citation failed the contract — re-derive citation or drop
    ("LT:2", 2, "abdf3f7b", "fix", "citation quoted Trust Center bullet, not the sub-processor req"),
    ("LT:5", 5, "dea95d0c", "fix", "citation only the first Zoom recovery step"),
    ("LT:16", 16, "24e6a692", "fix", "citation is just the 'Ending calls gracefully' heading"),
    ("LT:27", 27, "06a52681", "fix", "citation is just 'How to assign Agents to an account?'"),
    ("LT:41", 41, "5a27a35a", "fix", "citation only 1 of 3 CTP requirements"),
    # prior over-drop reinstated
    ("VT:51", 51, "f0eaf0e5", "restore", "Cloud Flare two-alternative answer is complete; restore KEEP"),
]


# Drop reason per genuine-defect id (used when action == "drop").
DROP_REASON = {
    "VT:53": "answer-leak",
    "LT:32": "wrong-gold",
    "VT:7": "stripped-anchor",
    "VT:40": "stripped-anchor",
    "VT:20": "stripped-anchor",
}

# Corrected citation per fix id — the gold line that actually answers the query
# (verified against gold content; all five golds do answer, only the citation failed).
FIX_CITATION = {
    "LT:2": "Sub-processors will implement and maintain security measures substantively similar to those listed in this Exhibit.",
    "LT:5": "you may need to first select your `user@gitlab.com` account in step 4 above ... return to step 3 but this time select GitLab / GitLab Unfiltered at step 4.",
    "LT:16": "Setting context and expectations before you start the call is the best way to a graceful exit.",
    "LT:27": "Open the Collections Window and click on the `Accounts` tab ... Click the agent name or Unassigned. Select an agent from the dropdown list. Click Update to save the settings.",
    "LT:41": "At least 2 x GitLab-certified trainers as full-time employees; the training team must collectively be certified to deliver at least 4 standard GitLab training courses, with CI/CD and Security Essentials mandatory.",
}


def resolve(datasets_dir: str) -> list[dict]:
    """Resolve each overturn to its (query, gold_hash); assert the prefix matches."""
    import json
    from pathlib import Path

    out = []
    cache: dict[str, list[str]] = {}
    for oid, line, prefix, action, note in OVERTURNS:
        code = oid.split(":")[0]
        path = Path(datasets_dir) / QA[code]
        if str(path) not in cache:
            cache[str(path)] = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(cache[str(path)][line - 1])
        golds = [rc["id"] for rc in row.get("reference_chunks", [])]
        match = [g for g in golds if g.startswith(prefix)]
        if not match:
            raise AssertionError(f"{oid}: prefix {prefix} not in line {line} golds {golds}")
        out.append(
            {
                "id": oid,
                "action": action,
                "note": note,
                "query": row["question"],
                "gold_hash": match[0],
                "all_golds": golds,
            }
        )
    return out
