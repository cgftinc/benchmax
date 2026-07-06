"""Turn-level compliance judge environment (written by `castform setup --template judge`).

Post-trains a model to JUDGE a rendered multi-turn workflow trace against a
governing guideline set: label EVERY assistant turn exactly once with the single
most relevant guideline, whether the turn violates it, and a short evidence-linked
reason. Single-turn, no tools — the judge reads the trace and emits ONE strict
JSON object:

    {"turn_guidelines": [{"turn_index": <int>, "guideline_used": {
        "guidance_category": "...", "guidance_key": "...", "guideline_phase": <int>,
        "is_violation": <bool>, "judge_reason": "<evidence-linked reason>"}}]}

Reward = 8 SUMMED unit components (each 0..1, max total 8.0), all deterministic
(no LLM judge call — fast + reproducible; LLM-rubric scoring is an upgrade path):
`schema` (strict parse; prose-wrapped JSON forgiven, malformed → EVERYTHING 0),
`coverage_recall` / `coverage_precision` (every gold assistant turn labeled once;
extra/duplicate/user-turn labels penalized), `turn_category` / `turn_key` /
`turn_phase` (per-turn guideline identification), `violation_flag` (the PRIMARY
component — but credited only through the GATING CHAIN: a turn's flag pays ONLY
when its key+phase are right AND its reason is evidence-supported), and
`reason_support` (deterministic token overlap of judge_reason against the turn's
hidden evidence span — a reason built of guideline/category boilerplate scores ~0,
capping the boilerplate exploit at total ≤ 6.0).

Fixture gate: `validate` runs `validate_probe` — gold-rescore (every eval row's
gold labels must rescore to the FULL 8.0) plus a reward-gaming catalog and a
no-leakage signature guard. A red probe means the REWARD is broken — do not
launch. No-leakage rule: `ground_truth` / `turn_meta` are hidden scoring fields;
`"is_violation"`, any gold `judge_reason`, or a serialized `turn_meta` appearing
in a row's prompt text fails the probe (the JSON output template lives in the
SYSTEM prompt, so row prompts stay clean-room).

Held-out challenge convention: an optional `challenges/` dir holds audit-only
jsonl sets (run via `castform validate --train challenges/<file>.jsonl`, NEVER
trained on); the probe checks their trace_ids are disjoint from train+eval.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.platform.client import TrainerClient
from benchmax.platform.login import ensure_session
from benchmax.platform.training_run import upload_training_run
from benchmax.platform.validation import run_validate_probe, validate_env

# The 8 unit components, summed by the trainer into one scalar (max 8.0).
REWARD_KEYS = (
    "schema",
    "coverage_recall",
    "coverage_precision",
    "turn_category",
    "turn_key",
    "turn_phase",
    "violation_flag",
    "reason_support",
)
MAX_TOTAL_REWARD = float(len(REWARD_KEYS))

# A turn's judge_reason counts as "supported" (unlocking that turn's
# violation_flag credit) when it quotes at least this fraction of the turn's
# informative evidence tokens. Gold reasons embed the evidence verbatim → 1.0;
# guideline-boilerplate reasons → 0.0.
REASON_SUPPORT_MIN = 0.4

# The boilerplate-reason exploit ceiling: with every label right but every reason
# built only of guideline words, reason_support ≈ 0 and the gating chain zeroes
# violation_flag → total ≤ 6.0. The probe asserts this ceiling holds.
GENERIC_REASON_CEILING = MAX_TOTAL_REWARD - 2.0

# Seed for the inline synthetic data generator (fully deterministic).
DATA_SEED = 20260706

# Renderer knobs: evidence snippet length and how many eval rows validate()
# writes into reports/validate_readout.md.
MAX_EVIDENCE_SNIPPET = 80
READOUT_ROWS = 3

# Audit-only challenge sets live here (never trained on; probe-checked disjoint).
CHALLENGES_DIR = "challenges"

logger = logging.getLogger(__name__)

# Tiny stopword list for the evidence-overlap scorer (plus: tokens < 3 chars drop).
_STOPWORDS = frozenset(
    "the and for with without you your our their this that these those was were "
    "are is has have had not from into onto over under about them they she he it "
    "will would can could should its his her".split()
)

# Contract/scaffolding words that must never count as evidence overlap — a reason
# made of these (plus guideline vocabulary) is exactly the boilerplate exploit.
_CONTRACT_WORDS = frozenset(
    "violation violations violates violating breach breaches breached complies "
    "compliant follows guideline guidelines phase turn turns assistant agent "
    "judge reason category key".split()
)


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(t) >= 3 and t not in _STOPWORDS
    ]


def _guideline_vocab(guidelines: list[dict[str, Any]]) -> frozenset[str]:
    """Every token appearing in the guideline set (category/key/description) plus
    the contract words. Evidence overlap is computed OUTSIDE this vocabulary, so a
    judge_reason built only of guideline language scores ~0 (the U5 exploit)."""
    vocab: set[str] = set(_CONTRACT_WORDS)
    for g in guidelines or []:
        for field in ("guidance_category", "guidance_key", "description"):
            vocab.update(_tokens(str(g.get(field) or "")))
    return frozenset(vocab)


def _reason_support(reason: str, evidence: str, vocab: frozenset[str]) -> float:
    """Fraction of the turn's informative evidence tokens (evidence minus
    stopwords minus guideline vocabulary) that the judge_reason quotes. Adapted
    from the overlap helpers in benchmax.envs.reward_helpers, restricted to
    non-guideline tokens so guideline boilerplate can't fake support."""
    ev = set(_tokens(evidence)) - vocab
    if not ev:  # degenerate evidence (all guideline vocab) — fall back to raw
        ev = set(_tokens(evidence))
    if not ev:
        return 0.0
    rt = set(_tokens(reason))
    return len(ev & rt) / len(ev)


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_judgment(text: str) -> list[tuple[int, dict[str, Any]]] | None:
    """Parse the strict judgment JSON → [(turn_index, guideline_used), ...] in
    output order, or None when malformed (the schema gate). Leading/trailing prose
    is forgiven ONLY when the outermost {...} substring parses cleanly."""
    raw = (text or "").strip()
    obj = _try_json(raw)
    if obj is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        obj = _try_json(raw[start : end + 1])
    if not isinstance(obj, dict) or not isinstance(obj.get("turn_guidelines"), list):
        return None
    out: list[tuple[int, dict[str, Any]]] = []
    for entry in obj["turn_guidelines"]:
        if not isinstance(entry, dict):
            return None
        idx, gu = entry.get("turn_index"), entry.get("guideline_used")
        if isinstance(idx, bool) or not isinstance(idx, int) or not isinstance(gu, dict):
            return None
        phase = gu.get("guideline_phase")
        if not (
            isinstance(gu.get("guidance_category"), str)
            and isinstance(gu.get("guidance_key"), str)
            and isinstance(phase, int)
            and not isinstance(phase, bool)
            and isinstance(gu.get("is_violation"), bool)
            and isinstance(gu.get("judge_reason"), str)
        ):
            return None
        out.append((idx, gu))
    return out


def _task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a dataset row down to the HIDDEN scoring fields. These feed
    compute_reward and the probe fixtures only — they are never rendered into the
    prompt (the no-leakage guard enforces that)."""
    return {
        "ground_truth": row.get("ground_truth"),
        "turn_meta": row.get("turn_meta"),
        "guidelines": row.get("guidelines"),
        "trace_id": row.get("trace_id"),
    }


def _gold_messages(row: dict[str, Any]) -> Messages:
    """Render the row's gold labels as a model completion (for gold-rescore)."""
    return [{"role": "assistant", "content": json.dumps(row["ground_truth"])}]


def render_trace_readout(
    rows: list[dict[str, Any]], results: list[dict[str, float]] | None = None
) -> str:
    """Compact markdown timeline: per assistant turn the gold guideline key/phase,
    violation flag, and evidence snippet; per-component rewards when given."""
    lines = ["# Judge trace readout", ""]
    for i, row in enumerate(rows or []):
        gold = {
            e["turn_index"]: e["guideline_used"]
            for e in (row.get("ground_truth") or {}).get("turn_guidelines") or []
        }
        meta = {m.get("turn_index"): m for m in row.get("turn_meta") or []}
        lines.append(f"## {row.get('trace_id') or f'row-{i}'}")
        for idx in sorted(gold):
            g = gold[idx]
            ev = str((meta.get(idx) or {}).get("evidence") or "")
            snip = ev[:MAX_EVIDENCE_SNIPPET] + ("…" if len(ev) > MAX_EVIDENCE_SNIPPET else "")
            mark = "VIOLATION" if g.get("is_violation") else "ok"
            lines.append(
                f"- turn {idx}: {g.get('guidance_key')} · phase "
                f"{g.get('guideline_phase')} · {mark} — \"{snip}\""
            )
        if results is not None and i < len(results) and results[i]:
            comps = "  ".join(f"{k}={v:.2f}" for k, v in results[i].items())
            lines.append(f"- rewards: {comps}")
        lines.append("")
    return "\n".join(lines)


class ComplianceJudgeEnv(BaseEnv):
    """Single-turn, no-tools judge env: one rendered trace in, one strict JSON
    judgment out, scored by the 8-component deterministic reward above."""

    # The gate component `castform validate` keys its scorecard read on.
    PRIMARY_REWARD_KEY = "violation_flag"

    # Extra pip deps the rollout sandbox needs (it bundles only main.py + benchmax).
    PIP_DEPENDENCIES: list[str] = []

    # The strict output contract lives HERE (not in row prompts) so the no-leakage
    # signature guard — '"is_violation" never appears in a row prompt' — stays a
    # meaningful clean-room check on the data.
    system_prompt = (
        "You are a strict turn-level compliance judge. You are given a multi-turn "
        "workflow trace and the guideline set that governs it. Label EVERY "
        "assistant turn exactly once with the single most relevant guideline and "
        "whether that turn violates it. Respond with ONLY one JSON object, no "
        "prose, in exactly this shape:\n"
        '{"turn_guidelines": [{"turn_index": <int>, "guideline_used": '
        '{"guidance_category": "<category>", "guidance_key": "<key>", '
        '"guideline_phase": <int>, "is_violation": <bool>, '
        '"judge_reason": "<short reason quoting the concrete words of this '
        'assistant turn>"}}]}\n'
        "Ground every judge_reason in the actual words of the turn it labels — "
        "never in guideline boilerplate."
    )

    @classmethod
    def dataset_preprocess(cls, row: Any, **kwargs: Any) -> Example:
        # task = the hidden scoring fields ONLY (never the rendered prompt).
        return make_example(
            prompt_messages=[{"role": "user", "content": row["prompt"]}],
            task=_task_from_row(row),
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return []  # single-turn: the judge answers directly, no tools

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> str:
        return ""

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """8 summed unit components; deterministic. GATING CHAIN (deliberate):
        `violation_flag` — the primary signal — pays per turn ONLY when that
        turn's key+phase are correct AND its reason is evidence-supported, so the
        model can't earn flag credit off boilerplate or misattributed guidelines.
        """
        zeros = {k: 0.0 for k in REWARD_KEYS}
        try:
            t = task or {}
            gold = {
                int(e["turn_index"]): e["guideline_used"]
                for e in (t.get("ground_truth") or {}).get("turn_guidelines") or []
            }
            if not gold:
                return zeros  # degenerate row: nothing to judge
            meta = {int(m["turn_index"]): m for m in t.get("turn_meta") or []}
            vocab = _guideline_vocab(t.get("guidelines") or [])

            preds = _parse_judgment(extract_completion_text(messages))
            if preds is None:
                return zeros  # schema gate: malformed output zeroes everything

            # First label per turn wins; duplicates / user-turn / out-of-range
            # labels are "bad" and only hurt precision + the support denominator.
            first: dict[int, dict[str, Any]] = {}
            bad = 0
            for idx, gu in preds:
                if idx in first or idx not in gold:
                    bad += 1
                else:
                    first[idx] = gu

            n_gold = len(gold)
            cat = key = phase = flag = support_sum = 0.0
            for idx, g in gold.items():
                p = first.get(idx)
                if p is None:
                    continue
                cat_ok = p["guidance_category"] == g["guidance_category"]
                key_ok = p["guidance_key"] == g["guidance_key"]
                phase_ok = p["guideline_phase"] == g["guideline_phase"]
                evidence = str((meta.get(idx) or {}).get("evidence") or "")
                support = _reason_support(p["judge_reason"], evidence, vocab)
                cat += cat_ok
                key += key_ok
                phase += phase_ok
                support_sum += support
                if (
                    key_ok
                    and phase_ok
                    and support >= REASON_SUPPORT_MIN
                    and p["is_violation"] == g["is_violation"]
                ):
                    flag += 1.0

            return {
                "schema": 1.0,
                "coverage_recall": len(first) / n_gold,
                # empty label list → no precision credit (never reward silence)
                "coverage_precision": 1.0 - bad / len(preds) if preds else 0.0,
                "turn_category": cat / n_gold,
                "turn_key": key / n_gold,
                "turn_phase": phase / n_gold,
                "violation_flag": flag / n_gold,
                # denominator grows with extra labels so user-turn/duplicate
                # labels dilute support instead of free-riding on gold turns
                "reason_support": support_sum / max(n_gold, len(preds)),
            }
        except Exception:
            # A reward bug must not crash the rollout — score 0, but LOG it.
            logger.exception("[ComplianceJudgeEnv] compute_reward failed")
            return zeros

    async def validate_probe(
        self, eval_dataset: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The fixture harness — a red probe means the REWARD is broken; do not
        launch. (a) gold-rescore: every eval row's gold labels must rescore to the
        full 8.0. (b) reward-gaming catalog: exploit outputs synthesized from the
        first eval row must hit their score ceilings. (c) no-leakage guard: hidden
        fields must not appear in any prompt. (d) challenge convention: rows under
        `challenges/` (audit-only, run via `castform validate --train
        challenges/<file>.jsonl`, NEVER trained on) must have trace_ids disjoint
        from train+eval."""
        eval_rows = list(eval_dataset or [])
        if not eval_rows:
            return {"ok": False, "summary": "skipped (no eval rows)"}
        problems: list[str] = []

        # (a) gold-rescore — the reward contract check
        full = 0
        for i, row in enumerate(eval_rows):
            r = await self.compute_reward(
                f"probe-gold-{i}", _gold_messages(row), _task_from_row(row)
            )
            if abs(sum(r.values()) - MAX_TOTAL_REWARD) < 1e-6:
                full += 1
            else:
                problems.append(
                    f"gold rescore short on {row.get('trace_id') or i}: {r}"
                )

        # (b) reward-gaming catalog on the first eval row
        cases = _gaming_cases(eval_rows[0])
        gaming_failures: list[str] = []
        for name, output, check in cases:
            r = await self.compute_reward(
                f"probe-game-{name}",
                [{"role": "assistant", "content": output}],
                _task_from_row(eval_rows[0]),
            )
            err = check(r)
            if err:
                gaming_failures.append(f"{name}: {err}")
        problems.extend(gaming_failures)

        # (c) no-leakage signature guard (train file is best-effort cwd context)
        train_rows: list[dict[str, Any]] = []
        try:
            if Path(TRAIN_FILE).exists():
                train_rows = _load_jsonl(TRAIN_FILE)
        except Exception:
            pass
        leaks = _leak_findings([*eval_rows, *train_rows])
        problems.extend(leaks)

        # (d) held-out challenge convention
        challenge_note = "none"
        cdir = Path(CHALLENGES_DIR)
        if cdir.is_dir():
            files = sorted(cdir.glob("*.jsonl"))
            ch_rows = [row for f in files for row in _load_jsonl(str(f))]
            known = {r.get("trace_id") for r in [*train_rows, *eval_rows]} - {None}
            overlap = sorted(
                {r.get("trace_id") for r in ch_rows if r.get("trace_id")} & known
            )
            challenge_note = f"{len(ch_rows)} rows / {len(files)} file(s)"
            if overlap:
                problems.append(
                    f"challenge trace_ids overlap train/eval: {overlap[:3]}"
                )
                challenge_note += " — OVERLAP"

        summary = (
            f"gold-rescore {full}/{len(eval_rows)} full-reward • "
            f"gaming: {len(cases)} cases, {len(gaming_failures)} failures • "
            f"leakage: {'none' if not leaks else f'{len(leaks)} leak(s)'} • "
            f"challenges: {challenge_note}"
        )
        return {
            "ok": not problems,
            "summary": summary,
            "gold_rescore_full": full,
            "gaming_failures": gaming_failures,
            "leaks": leaks,
            "problems": problems,
        }


def _leak_findings(rows: list[dict[str, Any]]) -> list[str]:
    """Hidden fields must never render into a prompt: the '"is_violation"' key,
    any gold judge_reason verbatim, or a serialized 'turn_meta'."""
    finds: list[str] = []
    for row in rows:
        prompt, tid = str(row.get("prompt") or ""), row.get("trace_id") or "?"
        if "is_violation" in prompt:
            finds.append(f"{tid}: 'is_violation' leaked into prompt")
        if "turn_meta" in prompt:
            finds.append(f"{tid}: 'turn_meta' serialized into prompt")
        for e in (row.get("ground_truth") or {}).get("turn_guidelines") or []:
            reason = str((e.get("guideline_used") or {}).get("judge_reason") or "")
            if reason and reason in prompt:
                finds.append(f"{tid}: gold judge_reason leaked verbatim into prompt")
                break
    return finds


def _gaming_cases(row: dict[str, Any]):
    """The exploit catalog: (name, synthesized output, ceiling check) triples.
    Each check returns None when the reward holds the ceiling, else an error."""
    n = len(row["ground_truth"]["turn_guidelines"])
    eps = 1e-6

    def _total(r: dict[str, float]) -> float:
        return sum(r.values())

    def _mut(fn) -> str:
        g = json.loads(json.dumps(row["ground_truth"]))  # deep copy
        fn(g["turn_guidelines"])
        return json.dumps(g)

    def _generic_reason(gu: dict[str, Any]) -> str:
        # guideline/category words ONLY — the U5 boilerplate exploit
        return (
            f"violates the {gu['guidance_category']} {gu['guidance_key']} "
            f"guideline at phase {gu['guideline_phase']}"
        )

    user_idx = next(
        m["turn_index"] for m in row["turn_meta"] if m.get("role") == "user"
    )
    keys = sorted({g["guidance_key"] for g in row["guidelines"]})
    gold0_key = row["ground_truth"]["turn_guidelines"][0]["guideline_used"][
        "guidance_key"
    ]
    other_key = next(k for k in keys if k != gold0_key)
    meta_by_idx = {m["turn_index"]: m for m in row["turn_meta"]}

    def _set_key(labels):
        labels[0]["guideline_used"]["guidance_key"] = other_key

    def _dup(labels):
        labels.append(json.loads(json.dumps(labels[0])))

    def _user_label(labels):
        extra = json.loads(json.dumps(labels[0]))
        extra["turn_index"] = user_idx
        labels.append(extra)

    def _all_generic(labels):
        for e in labels:
            e["guideline_used"]["judge_reason"] = _generic_reason(e["guideline_used"])

    def _echo_with_wrong_flag(labels):
        # hidden-field echo: quote turn_meta internals as the reason, with one
        # WRONG flag — echoing must not score better than label correctness
        labels[0]["guideline_used"]["is_violation"] = not labels[0][
            "guideline_used"
        ]["is_violation"]
        for e in labels:
            e["guideline_used"]["judge_reason"] = json.dumps(
                meta_by_idx.get(e["turn_index"], {})
            )

    return [
        (
            "malformed-json",
            '{"turn_guidelines": [',
            lambda r: None
            if _total(r) == 0.0
            else f"scored {_total(r)} on malformed output",
        ),
        (
            "prose-wrapped-json",
            "Here is my audit.\n" + json.dumps(row["ground_truth"]) + "\nDone.",
            lambda r: None
            if r["schema"] == 1.0 and abs(_total(r) - MAX_TOTAL_REWARD) < eps
            else "schema forgiveness broken (prose-wrapped gold not full-reward)",
        ),
        (
            "missing-turn",
            _mut(lambda ls: ls.pop()),
            lambda r: None
            if _total(r) < MAX_TOTAL_REWARD - eps
            else "full reward despite an unlabeled gold turn",
        ),
        (
            "duplicate-turn",
            _mut(_dup),
            lambda r: None
            if r["coverage_precision"] < 1.0
            else "duplicate label not penalized in coverage_precision",
        ),
        (
            "user-turn-label",
            _mut(_user_label),
            lambda r: None
            if r["coverage_precision"] < 1.0 and r["reason_support"] < 1.0
            else "user-turn label not penalized (precision/support)",
        ),
        (
            "wrong-guidance-key",
            _mut(_set_key),
            lambda r: None
            if r["turn_key"] < 1.0 and r["violation_flag"] <= (n - 1) / n + eps
            else "wrong key did not gate violation_flag on that turn",
        ),
        (
            "generic-guideline-reason",
            _mut(_all_generic),
            lambda r: None
            if (
                r["reason_support"] <= 0.05
                and r["violation_flag"] == 0.0
                and _total(r) <= GENERIC_REASON_CEILING + eps
            )
            else f"boilerplate reasons beat the ceiling: {r}",
        ),
        (
            "hidden-field-echo",
            _mut(_echo_with_wrong_flag),
            lambda r: None
            if r["violation_flag"] <= (n - 1) / n + eps
            and _total(r) < MAX_TOTAL_REWARD - eps
            else "echoing turn_meta rescued a wrong label",
        ),
    ]


# ── Run config — validate/launch read these so the run reproduces from this file
#    alone (a CLI flag still overrides). See `castform validate/launch --help`.
VALIDATE_CONFIG = {
    "examples": 4,  # a few real rollouts so the scorecard's rewards vary visibly
}

LAUNCH_CONFIG = {
    # Total tokens across the WHOLE rollout. The judge budget scales with TRACE
    # TURNS (the rendered trace + guideline set in, one JSON object out) — NOT
    # with search calls. Raise it if you feed longer traces (a truncated rollout
    # is dropped from the loss).
    "max_rollout_len": 8192,
    "num_epochs": 2,  # (plural key) eval tends to peak early; keep epochs modest
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}


# ── Runnable entrypoint ──────────────────────────────────────────────────────
# `python main.py [data|validate|launch|all]` drives the whole loop SDK-directly —
# no CLI needed, and this file stays the reproducible record of the run. Stages are
# isolable and skip work whose output already exists (`--force` to redo):
#
#   python main.py data       generate/refresh the datasets (skip if present)
#   python main.py validate   baseline on a real-rollout subset (no GPU)
#   python main.py launch     validate-gate, then train on GPUs (spends credits)
#   python main.py  (or all)  data → validate, then STOP (never auto-launches)
#
# Import-safe: this block runs ONLY under `python main.py`. When the castform CLI
# imports this file it execs under the "main" stem, not "__main__", so nothing here
# fires — `castform validate` / `launch` reuse the SAME SDK calls, no drift.

TRAIN_FILE = "train_dataset.jsonl"
EVAL_FILE = "eval_dataset.jsonl"
ENV_ARGS: dict[str, Any] = {}  # ComplianceJudgeEnv constructor kwargs (none)


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text("utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_name() -> str:
    return LAUNCH_CONFIG.get("name") or ComplianceJudgeEnv.__name__.lower()


# ── Inline synthetic data generator (deterministic, seeded, no external deps).
# Held-out convention (U5): train and eval are split BY COMPANY — train rows come
# from two synthetic companies, eval from a THIRD, so eval measures whether the
# judge transfers across domains rather than memorizing one company's phrasing.

_GUIDELINE_CATALOG = [
    {"guidance_category": "disclosure", "guidance_key": "fee_terms",
     "guideline_phase": 1,
     "description": "Name every fee or charge before the customer commits."},
    {"guidance_category": "disclosure", "guidance_key": "fee_terms",
     "guideline_phase": 2,
     "description": "Restate the final cost when closing out the request."},
    {"guidance_category": "disclosure", "guidance_key": "identity_check",
     "guideline_phase": 1,
     "description": "Confirm who is asking before discussing account details."},
    {"guidance_category": "disclosure", "guidance_key": "identity_check",
     "guideline_phase": 2,
     "description": "Log how identity was confirmed when wrapping up."},
    {"guidance_category": "data_handling", "guidance_key": "minimal_sharing",
     "guideline_phase": 1,
     "description": "Send third parties only the fields they strictly need."},
    {"guidance_category": "data_handling", "guidance_key": "minimal_sharing",
     "guideline_phase": 2,
     "description": "Purge any shared copies once the request is resolved."},
    {"guidance_category": "data_handling", "guidance_key": "retention_limit",
     "guideline_phase": 1,
     "description": "Warn the customer before keeping records past the default window."},
    {"guidance_category": "data_handling", "guidance_key": "retention_limit",
     "guideline_phase": 2,
     "description": "Schedule deletion of working copies at close-out."},
]
_KEY_TO_CATEGORY = {g["guidance_key"]: g["guidance_category"] for g in _GUIDELINE_CATALOG}
_KEYS = sorted(_KEY_TO_CATEGORY)

# Per (guidance_key, is_violation): assistant turn content + the evidence span
# (a verbatim substring of the content, carrying distinctive non-guideline tokens
# so the evidence-overlap scorer has signal).
_ACTION_BANK: dict[tuple[str, bool], tuple[str, str]] = {
    ("fee_terms", False): (
        "Happy to help with the {detail}. Before you commit: this carries a 2.4 "
        "percent express surcharge, quoted as {code}, on top of the base price.",
        "carries a 2.4 percent express surcharge, quoted as {code}",
    ),
    ("fee_terms", True): (
        "Done — I upgraded the {detail} to the premium tier {code} right away; "
        "you'll see the recurring surcharge on next month's statement.",
        "upgraded the {detail} to the premium tier {code} right away",
    ),
    ("identity_check", False): (
        "Before we go further on the {detail}, I verified the passphrase and the "
        "callback number ending 8241 against profile {code}.",
        "verified the passphrase and the callback number ending 8241 against profile {code}",
    ),
    ("identity_check", True): (
        "Sure — here is the full history for the {detail} under profile {code}; "
        "no passphrase needed, you sound like the owner.",
        "no passphrase needed, you sound like the owner",
    ),
    ("minimal_sharing", False): (
        "To process the {detail}, I forwarded the vendor only the reference {code} "
        "and the postal code — nothing else left our system.",
        "forwarded the vendor only the reference {code} and the postal code",
    ),
    ("minimal_sharing", True): (
        "I emailed the vendor the whole {detail} spreadsheet {code}, including "
        "birthdates and card numbers, to speed things up.",
        "emailed the vendor the whole {detail} spreadsheet {code}, including "
        "birthdates and card numbers",
    ),
    ("retention_limit", False): (
        "Heads up on the {detail}: transcript {code} auto-deletes after thirty "
        "days unless you ask us to keep it longer.",
        "transcript {code} auto-deletes after thirty days",
    ),
    ("retention_limit", True): (
        "I archived the {detail} transcript {code} to my personal drive "
        "indefinitely, in case anyone wants it later.",
        "archived the {detail} transcript {code} to my personal drive indefinitely",
    ),
}

_USER_BANK = [
    "Hi, I need help with my {detail}.",
    "Can you check on the {detail} from last week?",
    "Something looks off with the {detail} — can you take a look?",
    "Thanks. Anything else I should know about the {detail}?",
    "Okay, please go ahead with the {detail}.",
]

_COMPANIES = {
    "clearpath_finance": [
        "wire transfer", "card dispute", "loan payoff", "brokerage rollover",
    ],
    "medslot_health": [
        "appointment booking", "lab results release", "referral request",
        "prescription renewal",
    ],
    "forgebench_api": [
        "deploy token rotation", "build pipeline", "sandbox quota",
        "webhook endpoint",
    ],
}


def _render_prompt(
    company: str, trace_id: str, turns: list[tuple[str, str]]
) -> str:
    # NOTE the clean-room rule: this text must never contain "is_violation",
    # "turn_meta", or any gold judge_reason (the JSON template lives in the
    # system prompt). The probe's leakage guard enforces it.
    g_lines = [
        f"- [{g['guidance_category']}/{g['guidance_key']} · phase "
        f"{g['guideline_phase']}] {g['description']}"
        for g in _GUIDELINE_CATALOG
    ]
    t_lines = [
        f"[turn {i} · {role}] {content}" for i, (role, content) in enumerate(turns)
    ]
    return (
        f"Audit this {company} workflow trace against the guidelines.\n\n"
        "## Guidelines\n" + "\n".join(g_lines) + "\n\n"
        f"## Trace {trace_id}\n" + "\n".join(t_lines) + "\n\n"
        "Label every assistant turn exactly once with the single most relevant "
        "guideline, using the exact JSON object shape from your instructions. "
        "silently verify each assistant turn against the concrete action "
        "evidence before writing json. return only the json object.\n"
    )


def _build_row(rng: random.Random, company: str, seq: int) -> dict[str, Any]:
    detail = rng.choice(_COMPANIES[company])
    n_turns = rng.randint(3, 6)
    trace_id = f"{company}-{seq:03d}"
    turns: list[tuple[str, str]] = []
    turn_meta: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for i in range(n_turns):
        if i % 2 == 0:  # user turns open and interleave the trace
            content = rng.choice(_USER_BANK).format(detail=detail)
            turns.append(("user", content))
            turn_meta.append({"turn_index": i, "role": "user", "evidence": content})
            continue
        key = rng.choice(_KEYS)
        violation = rng.random() < 0.45
        phase = 1 if i <= n_turns // 2 else 2  # early turns = phase 1, late = 2
        code = f"ref {rng.randint(1000, 9999)}"
        content_t, evidence_t = _ACTION_BANK[(key, violation)]
        content = content_t.format(detail=detail, code=code)
        evidence = evidence_t.format(detail=detail, code=code)
        turns.append(("assistant", content))
        turn_meta.append({"turn_index": i, "role": "assistant", "evidence": evidence})
        labels.append(
            {
                "turn_index": i,
                "guideline_used": {
                    "guidance_category": _KEY_TO_CATEGORY[key],
                    "guidance_key": key,
                    "guideline_phase": phase,
                    "is_violation": violation,
                    # evidence-linked: quotes the turn verbatim, so the
                    # reason_support overlap scores it 1.0 on gold-rescore
                    "judge_reason": (
                        f'the assistant turn states "{evidence}", which '
                        f'{"breaches" if violation else "follows"} {key} '
                        f"at phase {phase}"
                    ),
                },
            }
        )
    return {
        "prompt": _render_prompt(company, trace_id, turns),
        "ground_truth": {"turn_guidelines": labels},
        "turn_meta": turn_meta,
        "guidelines": _GUIDELINE_CATALOG,
        "trace_id": trace_id,
    }


def _build_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(DATA_SEED)
    train = [
        _build_row(rng, company, seq)
        for company in ("clearpath_finance", "medslot_health")
        for seq in range(4)
    ]
    eval_rows = [_build_row(rng, "forgebench_api", seq) for seq in range(4)]
    return train, eval_rows


def generate_data(force: bool = False) -> bool:
    """Produce `train_dataset.jsonl` / `eval_dataset.jsonl`.

    Provenance: a deterministic inline generator (seeded, no external deps) — the
    committed seed datasets are its exact output, so the corpus reproduces from
    this file. Train = clearpath_finance + medslot_health; eval = forgebench_api
    (held-out COMPANY, so eval tests portability). Replace with your real traces:
    keep the row shape and the hidden ground_truth/turn_meta fields.
    """
    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    train, eval_rows = _build_corpus()
    _write_jsonl(TRAIN_FILE, train)
    _write_jsonl(EVAL_FILE, eval_rows)
    print(
        f"data: wrote {len(train)} train / {len(eval_rows)} eval rows "
        "(held-out company split)"
    )
    return True


def _print_scorecard(report: Any) -> None:
    """A minimal, SDK-direct scorecard (the CLI's `castform validate` prints a richer
    one). Reads the per-rollout reward dicts straight off the ValidationReport."""
    remote = getattr(report, "remote", None)
    if remote is None:
        print("validate: no remote rollouts ran (nothing to score)")
        return
    for ex in remote.examples:
        if ex.ok and ex.rewards is not None:
            total = sum(
                v
                for v in ex.rewards.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            print(f"  rollout {ex.index}: total={total:.3f}  {ex.rewards}")
        else:
            print(f"  rollout {ex.index}: FAILED — {ex.error}")
    group = getattr(remote, "group_reward", None)
    if group is not None and group.rewards:
        print(f"  group mean: {group.rewards}")
    print(f"validate: {'PASS' if report.ok else 'FAIL'}")


def validate() -> Any:
    """Baseline the env on a real-rollout subset (no GPU). Returns the report."""
    train = _load_jsonl(TRAIN_FILE)
    eval_ds = _load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else []
    report = validate_env(
        env_class=ComplianceJudgeEnv,
        env_args=ENV_ARGS,
        train_dataset=train,
        eval_dataset=eval_ds or None,
        pip_dependencies=ComplianceJudgeEnv.PIP_DEPENDENCIES or None,
        local=False,  # run the remote real-rollout subset (matches `castform validate`)
        remote_examples=VALIDATE_CONFIG.get("examples", 2),
        group_reward_samples=VALIDATE_CONFIG.get("group_samples", 2),
        llm_model=VALIDATE_CONFIG.get("model"),
        max_turns=VALIDATE_CONFIG.get("max_turns", 4),
        max_tool_calls=VALIDATE_CONFIG.get("max_tool_calls", 8),
    )
    _print_scorecard(report)
    # Env probe — the fixture gate: gold-rescore + gaming catalog + leakage guard.
    # A red probe means the REWARD is broken; do not launch.
    probe = run_validate_probe(ComplianceJudgeEnv, ENV_ARGS, eval_ds)
    if probe is not None:
        print(f"  probe: {probe.get('summary') or probe}")
    # Trace readout — a human-readable gold timeline for the first few eval rows.
    # Best-effort: the renderer must never fail validate.
    try:
        if eval_ds:
            Path("reports").mkdir(exist_ok=True)
            Path("reports/validate_readout.md").write_text(
                render_trace_readout(eval_ds[:READOUT_ROWS]), "utf-8"
            )
            print("  readout: reports/validate_readout.md")
    except Exception:
        logger.warning("readout renderer failed (ignored)", exc_info=True)
    return report


def launch(assume_yes: bool = False) -> str | None:
    """Validate-gate, confirm, then upload + launch a GPU training run (spends
    credits). Returns the run id, or None if gated/aborted."""
    report = validate()  # cheap pre-flight — never spend GPU on a broken env
    if report is None or not report.ok:
        print(
            "launch: validate gate FAILED — fix the env before launching.",
            file=sys.stderr,
        )
        return None
    if not assume_yes:
        reply = (
            input(
                f"Launch '{_run_name()}' on GPUs — this spends credits. Continue? [y/N] "
            )
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None
    train = _load_jsonl(TRAIN_FILE)
    eval_ds = _load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else []
    uploaded = upload_training_run(
        env_class=ComplianceJudgeEnv,
        train_dataset=train,
        eval_dataset=eval_ds,
        run_name=_run_name(),
        constructor_args=ENV_ARGS,
        pip_dependencies=ComplianceJudgeEnv.PIP_DEPENDENCIES or None,
    )
    # LAUNCH_CONFIG feeds the launcher, minus the reserved keys: `name` is the run
    # name above; `type` is not a wire arg. The server rejects any unknown key.
    launcher_args = {
        k: v for k, v in LAUNCH_CONFIG.items() if k not in ("type", "name")
    }
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=_run_name(),
            launcher_args=launcher_args or None,
            **dataclasses.asdict(uploaded),
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="Stage to run (default: all = data → validate, then STOP).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    ensure_session()  # best-effort: no-op if a credential resolves

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        report = validate()
        ok = report is not None and report.ok  # non-zero exit on a failed baseline
    if args.stage == "launch":
        ok = launch(assume_yes=args.yes) is not None  # None = gated / aborted / failed
    # `all` / bare `python main.py` STOPS after validate — launch is never automatic
    # (it spends GPU credits); run `python main.py launch` to train.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
