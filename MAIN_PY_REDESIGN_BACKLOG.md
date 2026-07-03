# `main.py` redesign — out-of-scope backlog

Tracked follow-up for the "`main.py` as the primary runnable artifact" redesign
(the shipped work: `main.py` seed + runner, the env-agnostic 3-slot framework —
validate-probe hook, env-supplied reward-audit gate, launch token-budget guard +
manifest — and the RAG reference implementation). Everything below was deliberately
**out of scope** for that change; this doc is the durable record so it isn't lost.

Grouped by owner surface. Each item: what, why deferred, where it lives.

## 1. Library defaults (benchmax env lib)

- **Lib-default `SearchEnv.compute_reward`.** The audited reward shape (strict
  `<answer>` extraction, gate secondaries × correctness, an UNGATED `retrieval_hit`,
  citations by id-hash OR title-path, deterministic length term, no
  `search_efficiency`) currently lives in the **seed template**
  (`scaffold/rag_main.py`), not the library default
  (`envs/postgres_search/search_env.py::SearchEnv.compute_reward`). Fold the audited
  shape into the lib default in a separate pass so envs that *don't* override it
  inherit the good reward. Deferred to keep this change centered on the seed +
  framework; the lib change touches every existing SearchEnv user.

## 2. First-party env templates (fast-follow)

- **Judge template.** A non-RAG reference env for LLM-judge / trace-labeling tasks.
  The framework was built to generalize to it (arbitrary reward components render;
  the reward-audit gate is env-supplied via `PRIMARY_REWARD_KEY`; the validate-probe
  hook is env-agnostic). Requirements to preserve (from design brief §4 / feedback
  U5): arbitrary reward components (a judge returns ~8), pluggable validate fixtures
  (gold-rescore + reward-gaming fixtures), a no-leakage signature guard,
  held-out-challenge registration, and a trace renderer. Its input-side experiment
  lineage ask is partially answered by the launch manifest (§4 run-record extends it).

- **Traces template — lib-assisted, NOT from scratch.** The data-gen half already
  exists: `benchmax/traces/` (`TracesPipeline.run()` one-call entry in
  `traces/pipeline.py`; `build_training_examples` / `apply_filters` / `split_dataset`
  / adapters) wired via `castform data traces`. Missing is only the **env + reward**:
  author a `CustomTraceEnv` (single-turn `BaseEnv` + `dataset_preprocess` + one
  concrete comparative reward) in the `main.py` seed shape, using the deprecated
  wizard `core/platform-service/.../codegen/traces.ts` (`CustomTraceEnv`) as a
  reference. ⚠ **Reconcile the row-shape drift**: the lib now writes `to_jsonl_dict`
  (`traces/processing.py` — `ground_truth` is a **message dict**; `trace_id` /
  `turn_index` / `scores` under `init_rollout_args`), but the wizard's
  `dataset_preprocess` reads the *older* shape (`completion_messages`, `ground_truth`
  as a **string**). The env must read the new `to_jsonl_dict` shape. Deferred so the
  framework is proven with RAG before a second env-type's reward surface lands
  (regresses nothing — `castform setup --template` is `generic`/`rag` only).

## 3. CLI ephemeral-ops feature builds (surface defined; build later)

- **Rollout / transcript inspection** — first-class read of
  `/v1/train/runs/{id}/rollouts/*`. ⚠ **page is 1-indexed.**
- **Checkpoint / run `compare` verbs** — diff two runs' scalars/rollouts.
- **`doctor` + dev/prod profiles** — echo the resolved URLs + credential source per
  environment.
- **`why-failed`** — surface the scheduler `log_tail` for a failed run.
- **`trainer_ref` exposure** — surface/pin the trainer image ref.
- **Widen the `--set` allowlist** — `max_tokens_per_gpu`, `context_parallel_size`,
  `log_probs_chunk_size`, entropy coef, learning rate… (driven off the live
  `--list-args` schema; the server gates the real set).
- **`max_tool_calls` as a `--set` knob** — today it stays the server default 8 at
  launch (only `--max-tool-calls` on `validate`). A tool-heavy env is capped at 8 in
  training until this ships.
- **`validate --save-rollouts` / sampling flags** — persist the rollout transcripts a
  `validate` produced for offline audit; sample-size / seed controls.

## 4. Platform / trainer / image fixes

- **gemma-26B OOM.** A per-model YAML default for `log_probs_chunk_size`. ⚠ **The
  diagnosis conflicts between sources** — forward-pass logprob over the 262k vocab
  (U4) vs backward activation-recompute (U2). The launch budget-guard's model-aware
  warning deliberately does **not** overclaim a cause (see the plan's Risk 3); the
  platform owner must reconcile the two diagnoses before picking the fix.
- **CP > 1 (context parallel) on gemma is an untested subsystem** — warn if it's
  offered as an OOM lever until it's validated.
- **Non-prod URL propagation seam.** The judge / embedder can still misroute to prod
  in non-prod runs; the run's own environment URLs must reach every in-env client.
- **Ship `benchmax.rag.corpus.embed` to the sandbox image** — so provider-embedding
  RAG envs don't `ImportError` at rollout.
- **Trainer-image `benchmax` must be ≥ this PR before the rag template rolls out
  (deploy-ordering).** Verified live via the RAG e2e: the rag seed's audited reward
  references benchmax symbols added in #58+ (`score_citations`, plus this PR's
  `evaluate_single_rubric` / `CORRECTNESS_RUBRIC`), which cloudpickle bundles **by
  reference**. The deployed trainer image's `benchmax` predates #58, so the sandbox
  fails at unpickle (`Can't get attribute 'score_citations' …` → `BundlingError`,
  retried as a transient worker error) — every rag rollout. NOT a code bug (the
  #58 scaffold `rag_run.py` on `main` has the identical dependency); the
  authoring path (setup → ingest → qa-gen → the local gold-hit@k probe) all works.
  Fix: publish this PR's `benchmax` and rebuild/redeploy the trainer image, then
  re-run `python main.py validate` on the rag env to confirm the rollout side.
  Sibling to the `corpus.embed` item above.
- **qa-gen `GroundingLLMFilter` never-thresholded bug**
  (`rag/qa_generation/filters/grounding_llm.py`) — the grounding score is computed
  but not actually gated on a threshold, so ungrounded QA pairs pass.
- **Experiment run-record (OUTPUT-side lineage).** The launch manifest
  (`.castform/runs/<run_id>.json`) is the **input**-side record (env/dataset hashes,
  args, commit). The output side — run → checkpoint → metric history — is still
  missing; this is the judge team's #1 ask, only half-answered today.
- **Run-recovery semantics** — distinguish preemption (retryable) from terminal
  failure; warn on a duplicate launch of the same env+data+args.
- **Cost / ETA telemetry** — surface projected credit spend + time-to-finish.

## 5. Deferred from the shipped slices

- **Truncation surfacing from `report` token counts** (Slice 6). `castform launch`
  now warns *pre-launch* when the estimated rollout exceeds `max_rollout_len`, but
  surfacing *actual* truncation in `validate` / `runs` output needs per-rollout token
  counts, which `ValidationReport` / `ExampleValidation`
  (`platform/validation.py` / `platform/client.py`) do **not** carry. Add a token-
  count field to the report (platform change) before this can render.

---

*Companion to the untracked `MAIN_PY_REDESIGN_PLAN.md` / `MAIN_PY_REDESIGN_DESIGN_BRIEF.md`
(design brief §8 is the source for §1–4 above).*
