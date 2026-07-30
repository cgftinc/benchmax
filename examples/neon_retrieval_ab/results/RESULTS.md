# neon retrieval a/b — bm25 vs vector vs hybrid

paired offline retrieval comparison on one query set against the live `gitlab_handbook_neon` corpus. no gpu, no training, no re-ingest, no re-embed of the corpus.

## run

- corpus: `gitlab_handbook_neon` (31,665 chunks, active version, read-only)
- datasets: `examples/neon_rag_smoke/datasets/train_large.jsonl`, `examples/neon_rag_smoke/datasets/eval_large.jsonl`
- rows loaded: **430**
- rows with usable (non-empty) gold: **430**
- rows excluded for missing gold: **0** (none — every row carries exactly one gold file)
- identical settings across all three arms: `top_k=10`, no metadata filter, `text_search_config=pg_catalog.english`, schema `benchmax_corpus`. only `mode` differs. nothing was tuned per mode.
- embeddings: `text-embedding-3-large` computed once per question (7 batched calls for 430 questions) and reused by both the vector and the hybrid arm.
- retrieval calls: 1290 (concurrency 4, 92.7s wall).
- failed queries: **0** — every query returned in every mode.

hit@k is measured over the RANKED FILE list: the top-10 chunks are mapped to their `metadata.file` (the same field the gold uses) and deduped preserving first-occurrence rank. a row counts as a hit when ANY of its gold files appears.

## per-mode metrics

| mode | n | hit@1 | hit@5 | hit@10 | MRR@10 | mean gold rank (when found) | gold not in top-10 |
|---|---|---|---|---|---|---|---|
| bm25 | 430 | 0.4814 | 0.6953 | 0.7419 | 0.5737 | 1.91 | 111 |
| vector | 430 | 0.4628 | 0.7163 | 0.7512 | 0.5629 | 1.97 | 107 |
| hybrid | 430 | 0.5628 | 0.7581 | 0.8233 | 0.6464 | 1.95 | 76 |

## paired analysis: hybrid vs bm25

mcnemar's test run as the **exact two-sided binomial test** on the discordant pairs (`scipy.stats.binomtest(b, b+c, 0.5)`), which is the correct choice at these small discordant counts. the continuity-corrected chi-square statistic is shown for reference only and is not the test being used.

| k | both hit | hybrid only (b) | bm25 only (c) | both miss | b+c | net (b-c) | exact binomial p | chi2 (cc) |
|---|---|---|---|---|---|---|---|---|
| 1 | 173 | 69 | 34 | 154 | 103 | +35 | 0.0007 | 11.223 |
| 5 | 283 | 43 | 16 | 88 | 59 | +27 | 0.0006 | 11.458 |
| 10 | 309 | 45 | 10 | 66 | 55 | +35 | 2.06e-06 | 21.018 |

## paired analysis: vector vs bm25

| k | both hit | vector only (b) | bm25 only (c) | both miss | b+c | net (b-c) | exact binomial p | chi2 (cc) |
|---|---|---|---|---|---|---|---|---|
| 1 | 136 | 63 | 71 | 160 | 134 | -8 | 0.5455 | 0.366 |
| 5 | 251 | 57 | 48 | 74 | 105 | +9 | 0.4351 | 0.610 |
| 10 | 273 | 50 | 46 | 61 | 96 | +4 | 0.7596 | 0.094 |

## fusion lift (hybrid over bm25)

| k | rescued (bm25 miss -> hybrid hit) | broken (bm25 hit -> hybrid miss) | net |
|---|---|---|---|
| 1 | 69 | 34 | +35 |
| 5 | 43 | 16 | +27 |
| 10 | 45 | 10 | +35 |

## query-style breakdown (derived proxy — not ground truth)

the 430 rows carry no style field (their only keys are `question`, `answer`, `reference_chunks`), so this bucketing is **derived from the question text alone** by `style_proxy.classify`. it is a proxy, not a label: a question is `keyword` when it is shorter than 12 tokens **and** does not lead with an interrogative or auxiliary **and** contains no first/second-person pronoun; otherwise it is `paraphrase`. the rule never looks at gold, at retrieval output, or at any per-row search mode, so it cannot leak the outcome. the 156-row `gitlab_handbook_bm25_neon` golden was deliberately NOT used as a style source — it bakes a favourable `search_mode` per row, which is the confound this a/b removes.

| bucket | n | bm25 hit@5 | vector hit@5 | hybrid hit@5 |
|---|---|---|---|---|
| keyword | 204 | 0.9216 | 0.7843 | 0.8775 |
| paraphrase | 226 | 0.4912 | 0.6549 | 0.6504 |

paired counts at hit@5 within the `keyword` bucket (n=204):

| comparison | both hit | arm only (b) | bm25 only (c) | both miss | net | exact binomial p |
|---|---|---|---|---|---|---|
| hybrid vs bm25 | 177 | 2 | 11 | 14 | -9 | 0.0225 |
| vector vs bm25 | 156 | 4 | 32 | 12 | -28 | 1.94e-06 |

paired counts at hit@5 within the `paraphrase` bucket (n=226):

| comparison | both hit | arm only (b) | bm25 only (c) | both miss | net | exact binomial p |
|---|---|---|---|---|---|---|
| hybrid vs bm25 | 106 | 41 | 5 | 74 | +36 | 4.41e-08 |
| vector vs bm25 | 95 | 53 | 16 | 62 | +37 | 9.10e-06 |

## worked discordant examples

discordant queries at hit@5 for hybrid vs bm25, both directions, with each mode's top-3 files so the result can be checked by hand.

**row 4 (train, style=paraphrase) — hybrid wins**

- query: `On GitLab.com, what kinds of issues is this work supposed to catch only when someone is dealing with the huge live information set instead of small trial examples?`
- gold: `engineering/architecture/design-documents/database_testing/_index.md`
- bm25 top-3 (gold not in top-10):
  - `solutions-architects/center-of-excellence/demo-architecture.md`
  - `product/groups/fulfillment/_index.md`
  - `product/product-processes/continuous-interviewing.md`
- vector top-3 (gold rank 1):
  - `engineering/architecture/design-documents/database_testing/_index.md` **<- gold**
  - `engineering/architecture/design-documents/gitlab_ci_events/proposal-4-creating-events-via-ci-files.md`
  - `engineering/data-engineering/database-excellence/database-frameworks/doc/partitioning.md`
- hybrid top-3 (gold rank 1):
  - `engineering/architecture/design-documents/database_testing/_index.md` **<- gold**
  - `solutions-architects/center-of-excellence/demo-architecture.md`
  - `product/groups/fulfillment/_index.md`

**row 5 (train, style=keyword) — hybrid wins**

- query: `customer-facing interface copy change reviewer`
- gold: `engineering/infrastructure-platforms/gitlab-dedicated/switchboard.md`
- bm25 top-3 (gold rank 8):
  - `customer-success/csm/escalations/infrastructure.md`
  - `sales/insidesales.md`
  - `security/product-security/psirt/runbooks/psirt-case-lifecycle.md`
- vector top-3 (gold rank 1):
  - `engineering/infrastructure-platforms/gitlab-dedicated/switchboard.md` **<- gold**
  - `marketing/developer-relations/workflows-tools/content-review.md`
  - `marketing/localization/translation_mr_review_workflow.md`
- hybrid top-3 (gold rank 1):
  - `engineering/infrastructure-platforms/gitlab-dedicated/switchboard.md` **<- gold**
  - `customer-success/csm/escalations/infrastructure.md`
  - `marketing/developer-relations/workflows-tools/content-review.md`

**row 17 (train, style=paraphrase) — hybrid wins**

- query: `At GitLab, if someone is moving to a different job inside the company, what timeline should the two bosses usually aim for so the handoff happens quickly but still sensibly?`
- gold: `people-group/promotions-transfers.md`
- bm25 top-3 (gold not in top-10):
  - `engineering/development/sec/software-supply-chain-security/oncall/handoff-and-continuity.md`
  - `engineering/devops/oncall/handoff-and-continuity.md`
  - `engineering/readmes/alexives.md`
- vector top-3 (gold rank 1):
  - `people-group/promotions-transfers.md` **<- gold**
  - `acquisitions/acquisition-process/integration.md`
  - `hiring/talent-acquisition-framework/internal-hiring-process.md`
- hybrid top-3 (gold rank 1):
  - `people-group/promotions-transfers.md` **<- gold**
  - `engineering/development/sec/software-supply-chain-security/oncall/handoff-and-continuity.md`
  - `acquisitions/acquisition-process/integration.md`

**row 31 (train, style=paraphrase) — hybrid wins**

- query: `If I need Duncan Harris to weigh in on something, what’s the preferred way to ask so he has time to think before replying?`
- gold: `engineering/readmes/duncan-harris.md`
- bm25 top-3 (gold not in top-10):
  - `support/support-pods/ci-cd/_index.md`
  - `support/support-pods/sec/_index.md`
  - `support/support-pods/ai/_index.md`
- vector top-3 (gold rank 5):
  - `customer-success/csm/readmes/ofalk/_index.md`
  - `sales/readmes/ian-steward.md`
  - `engineering/readmes/david-wainaina.md`
- hybrid top-3 (gold rank 2):
  - `engineering/readmes/matt-kirkevold/index.md`
  - `engineering/readmes/duncan-harris.md` **<- gold**
  - `support/support-pods/ci-cd/_index.md`

**row 27 (train, style=keyword) — bm25 wins**

- query: `support ticket help customer first`
- gold: `support/workflows/how-to-get-help.md`
- bm25 top-3 (gold rank 3):
  - `support/support-engineer-responsibilities.md`
  - `support/training/_index.md`
  - `support/workflows/how-to-get-help.md` **<- gold**
- vector top-3 (gold not in top-10):
  - `support/internal-support/_index.md`
  - `customer-success/csm/support.md`
  - `support/workflows/working-on-tickets.md`
- hybrid top-3 (gold rank 7):
  - `customer-success/csm/support.md`
  - `support/internal-support/_index.md`
  - `support/support-engineer-responsibilities.md`

**row 29 (train, style=keyword) — bm25 wins**

- query: `post-sales ownership after deal closes by customer tier`
- gold: `customer-success/roles-overview/_index.md`
- bm25 top-3 (gold rank 1):
  - `customer-success/roles-overview/_index.md` **<- gold**
  - `sales/field-operations/channel-operations/sales-faq.md`
  - `customer-success/pre-sales-post-sales-transition.md`
- vector top-3 (gold not in top-10):
  - `sales/commercial/comm-sales-opp-stages/_index.md`
  - `sales/sales-operating-procedures/deal-closure.md`
  - `sales/build-value-with-customers.md`
- hybrid top-3 (gold rank 6):
  - `sales/field-operations/channel-operations/partner-faq.md`
  - `sales/field-operations/channel-operations/sales-faq.md`
  - `customer-success/account-team.md`

**row 40 (train, style=keyword) — bm25 wins**

- query: `public repositories access any shard location`
- gold: `engineering/architecture/design-documents/cells/decisions/017_container_registry.md`
- bm25 top-3 (gold rank 5):
  - `engineering/architecture/design-documents/cells/container_registry_routing_service.md`
  - `engineering/architecture/design-documents/artifact_registry/decisions/021_authorization.md`
  - `engineering/architecture/design-documents/artifact_registry/_index.md`
- vector top-3 (gold not in top-10):
  - `engineering/architecture/design-documents/artifact_registry/decisions/021_authorization.md`
  - `engineering/architecture/design-documents/artifact_registry/decisions/007_database_schema.md`
  - `engineering/data-engineering/database-excellence/database-frameworks/doc/root-namespace-sharding/index.md`
- hybrid top-3 (gold not in top-10):
  - `engineering/architecture/design-documents/cells/container_registry_routing_service.md`
  - `engineering/data-engineering/database-excellence/database-frameworks/doc/root-namespace-sharding/index.md`
  - `engineering/data-engineering/database-excellence/database-frameworks/doc/fdw-sharding.md`

**row 59 (train, style=paraphrase) — bm25 wins**

- query: `If I need an easy-to-edit place to review the current list of app-issued credentials and browser session values, where is that information shared in the Google Spreadsheet?`
- gold: `engineering/architecture/design-documents/cells/routable_tokens.md`
- bm25 top-3 (gold rank 5):
  - `enterprise-data/platform/rstudio/index.md`
  - `tools-and-tips/_index.md`
  - `security/customer-support-operations/resources/coding-standards.md`
- vector top-3 (gold not in top-10):
  - `product/ux/experience-research/research-panel-management.md`
  - `enterprise-data/platform/_index.md`
  - `people-group/general-onboarding/tanewki-tips.md`
- hybrid top-3 (gold not in top-10):
  - `tools-and-tips/other-apps.md`
  - `product/ux/experience-research/research-panel-management.md`
  - `enterprise-data/platform/rstudio/index.md`

## verdict

at hit@5, hybrid vs bm25 is b=43 / c=16 (net +27, exact binomial p=0.0006); vector vs bm25 is b=57 / c=48 (net +9, exact binomial p=0.4351).

### is the difference real?

**hybrid vs bm25: yes, and it is not marginal.** hybrid beats bm25 on every headline
metric (hit@1 0.5628 vs 0.4814, hit@5 0.7581 vs 0.6953, hit@10 0.8233 vs 0.7419, MRR@10
0.6464 vs 0.5737) and the paired test rejects the null at every k (exact two-sided
binomial p = 7.3e-04 / 5.8e-04 / 2.1e-06 at k = 1 / 5 / 10). 35 queries that bm25 never
surfaces at all inside the top-10 are recovered by hybrid.

**vector vs bm25: no — statistically indistinguishable in aggregate.** net +9 at hit@5 on
105 discordant pairs, p = 0.44; hit@1 is net -8, p = 0.55; hit@10 net +4, p = 0.76. taken
on its own the aggregate vector-vs-bm25 comparison is a **null result** and should be
reported as one. swapping bm25 for pure vector retrieval buys nothing measurable here.

### the aggregate hybrid win hides a large, opposite-signed style split

the aggregate number is not the whole story, and the honest reading is that hybrid is a
**trade, not a free lunch**:

* on the `keyword` half (n=204), bm25 is the **best** mode (hit@5 0.9216 vs hybrid 0.8775
  vs vector 0.7843) and hybrid is significantly WORSE than bm25 (b=2, c=11, net -9,
  p = 0.0225). vector is far worse still (net -28, p = 1.9e-06).
* on the `paraphrase` half (n=226), bm25 collapses (hit@5 0.4912) and both dense arms win
  decisively (hybrid 0.6504, b=41, c=5, net +36, p = 4.4e-08; vector 0.6549, net +37,
  p = 9.1e-06).

so hybrid's aggregate advantage is bought entirely on the paraphrase half and paid for on
the keyword half. because the two buckets are close to balanced here (204 / 226), the
paraphrase gain dominates. **a different query mix flips the conclusion**: a
keyword-dominated workload would prefer plain bm25.

for reference, a hypothetical style-routed oracle (bm25 on `keyword`, dense on
`paraphrase`) reaches hit@5 0.7814 (vector) / 0.7791 (hybrid) versus hybrid-everywhere at
0.7581 — a further +2.3 points, but it needs a router that does not exist and is out of
scope here.

### recommendation on the ~13h x 8xA100 gpu training a/b: **GO — with a narrowed question**

the retrieval-level effect is large enough and clean enough to be worth spending gpu time
on, but the run should be scoped to the comparison that actually has signal:

* **run hybrid vs bm25.** the effect is significant at every k and the +8.1-point hit@10
  gain (35 fewer queries with no reachable gold at all) is the kind of gap that can plausibly
  move end-to-end reward — the policy cannot cite what retrieval never returns.
* **do NOT spend gpu hours on vector vs bm25.** that arm is a measured null at the
  retrieval layer; there is no offline effect for training to amplify.

caveats that should be stated in whatever the gpu run reports:

1. this harness issues the DATASET question verbatim. in the rl env the policy writes its
   own search queries, and policy-authored queries may sit in a different place on the
   keyword/paraphrase axis than these do. the offline delta is an upper-bound-ish proxy for
   the retrieval quality the policy will actually experience, not a prediction of reward.
2. retrieval hit@k is an input to reward, not reward. the reward path also depends on
   whether the policy cites what it retrieved (`retrieval_hit` scores CITED, not returned),
   so a +6-point hit@5 does not translate one-for-one.
3. the style split is a DERIVED proxy (see above), not a labelled attribute. it is stable
   and mechanistically sensible — bm25 wins terse term queries, dense wins paraphrases —
   but individual rows can be mislabelled.
4. these 430 questions were llm-generated against this same corpus, so their phrasing
   distribution is an artifact of that generator, not of real user traffic.

