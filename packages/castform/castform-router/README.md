# Castform router

One inference-time flow:

```text
task -> Qwen scorer -> cheapest adequate policy -> LiteLLM backend -> response
```

Clients send a normal OpenAI-compatible request to LiteLLM using the public model
`castform-auto`. The LiteLLM pre-call hook extracts the task, asks Qwen for one
success probability per configured backend, and selects the least expensive backend
at or above `CASTFORM_ROUTER_QUALITY_THRESHOLD`. It rewrites the request model to
that backend alias; LiteLLM dispatches the original request and returns the backend
response unchanged.

The scorer never sees backend costs. If no backend reaches the threshold, the policy
uses the highest-scored backend. If scoring fails, it uses
`CASTFORM_AUTO_FALLBACK_MODEL`.

## Layout

```text
castform-router/
├── castform_router/        # scorer contract, policy, and LiteLLM callback
├── litellm/
│   ├── config.yaml         # public alias, scorer alias, and backend aliases
│   ├── Dockerfile          # LiteLLM plus the local callback package
│   ├── compose.yaml        # one-service local deployment
│   └── .env.example
└── tests/
```

The offline repository and fine-tuning pipeline lives separately in
`../castform-router-training`.

## Run with Docker Compose

```bash
cp litellm/.env.example litellm/.env
# Edit model endpoints and credentials in litellm/.env.
docker compose --env-file litellm/.env -f litellm/compose.yaml up --build
```

The image installs this Python package, allowing LiteLLM to import
`castform_router.litellm_callback.castform_auto_router`. The callback intercepts
only `castform-auto`; its request to the `castform-router-qwen` alias passes through
without being routed again.

The default catalog contains `small-route`, `medium-route`, and `large-route`.
Every `name` in `CASTFORM_AUTO_BACKENDS_JSON` must have a matching `model_name` in
`litellm/config.yaml`. Add or remove aliases in both places when changing the
catalog.

Send a request after the health check passes:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"castform-auto","messages":[{"role":"user","content":"Fix the parser"}]}'
```

Router details are attached to the internal LiteLLM request metadata under
`castform_router`. There is no UI, workflow generator, harness-specific routing,
session pinning, or tracing subsystem.

## AI router market guide

An AI router decides where an AI request should go. The name covers four different
decisions:

1. **Model routing:** Which model or agent is most likely to complete this task?
2. **Provider routing:** Which inference provider should host the selected model?
3. **Operational routing:** Which deployment is available, fast, compliant, and
   within budget?
4. **Ensemble routing:** Is the task important enough to call several models and
   combine their answers?

Most products focus on provider and operational routing. Castform focuses on model
and agent selection:

> Bring your GitHub. Castform learns which approved models and coding agents
> succeed on your engineering work, then chooses the cheapest adequate route.

### Market map

| Product | What it primarily routes | Learned task selection | Customer-specific training | Self-hostable |
| --- | --- | ---: | ---: | ---: |
| OpenRouter Auto | Prompt to model, then model to provider | Yes | No end-to-end customer workflow | No |
| OpenRouter Pareto | Coding tier to an economical model | Partly | No | No |
| OpenRouter Fusion | Task to a multi-model deliberation panel | Yes | No | No |
| Not Diamond | Prompt to model or custom route | Yes | Yes, from supplied evaluations | No |
| RouteLLM | Weak versus strong model | Yes | Trainable | Yes |
| Microsoft Foundry Model Router | Prompt to an Azure-hosted model | Yes | No repository learning loop | No |
| AWS Bedrock Intelligent Prompt Routing | Prompt between two related Bedrock models | Yes | No application-specific learning | No |
| LiteLLM | Request to provider deployment | No, not by default | No | Yes |
| Portkey | Rules to providers, models, and fallback chains | Rules | Configuration rather than learning | Partly |
| Vercel AI Gateway | Model to provider and ordered fallbacks | No | No | No |
| Cloudflare AI Gateway | Rules to models and providers | Rules | Configuration rather than learning | No |
| Requesty | Request to a fast provider route | Operational | No | No |
| TensorZero | Requests among configured inference variants | Configurable | Feedback-driven optimization | Yes |

### Task-aware routers

**OpenRouter Auto** analyzes the prompt and selects from a curated model pool. It is
powered by Not Diamond and supports a cost-versus-quality preference. OpenRouter's
main advantage is distribution: one API key, one balance, many models, and provider
failover. It is generic rather than trained on one company's repositories.
[Documentation](https://openrouter.ai/docs/guides/routing/routers/auto-router)

**OpenRouter Pareto** is coding-focused. It groups models using general coding
performance and selects an economical model in the requested tier. Castform's
equivalent definition of “adequate” should come from measured outcomes on the
customer's work rather than a public coding leaderboard.
[Documentation](https://openrouter.ai/docs/guides/routing/routers/pareto-router)

**OpenRouter Fusion** decides when a task warrants multi-model deliberation. It asks
a panel in parallel, has a judge compare the answers, and produces a final response.
This can improve hard answers but commonly costs four to five times a single
completion. It is an ensemble, not normally a cost-saving route.
[Documentation](https://openrouter.ai/docs/guides/routing/routers/fusion-router)

**Not Diamond** is Castform's closest intelligence-layer competitor. It offers
pretrained chat and coding routers and custom routers trained from customer
evaluation data. It can select commercial models, fine-tuned endpoints, or complete
agents. The important distinction is the data pipeline: Not Diamond can train from
an evaluation set; Castform should construct reproducible coding evaluations from
GitHub and execute the candidate routes.
[Overview](https://docs.notdiamond.ai/docs/what-is-model-routing) ·
[Custom training](https://docs.notdiamond.ai/docs/router-training-quickstart)

**RouteLLM** is an open-source research baseline that commonly chooses between a
weak, inexpensive model and a strong, expensive model. It is self-hostable and
trainable, but it does not reconstruct repository tasks or evaluate coding agents
by terminal outcomes. Castform should include it in comparative benchmarks.
[Repository](https://github.com/lm-sys/RouteLLM)

**Microsoft Foundry Model Router** is a trained router offered as an Azure model. It
has Quality, Balanced, and Cost modes, model allowlists, governance, and automatic
failover. Its strength is enterprise Azure distribution; its limitation for
Castform's use case is generic, cloud-bound routing without a GitHub-derived private
training loop.
[Documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router)

**AWS Bedrock Intelligent Prompt Routing** predicts response quality and switches
between two models in the same family. AWS states that it cannot adjust routing
using application-specific performance data, making specialized enterprise
workloads a natural opening for Castform.
[Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html)

### Gateways and operational routers

**LiteLLM** normalizes provider APIs and handles credentials, load balancing,
budgets, retries, fallbacks, and deployment selection. It normally expects the
logical model to have already been chosen. That makes it Castform's execution plane,
not its task-intelligence layer.
[Documentation](https://docs.litellm.ai/docs/proxy/load_balancing)

**Portkey** provides an enterprise gateway with conditional rules, load balancing,
fallbacks, guardrails, caching, budgets, and observability. Its documented
conditional routing follows operator-authored rules rather than predicting task
success from customer outcomes.
[Documentation](https://portkey.ai/docs/product/ai-gateway/conditional-routing)

**Vercel AI Gateway** offers a unified model API, provider selection, ordered model
fallbacks, budgets, and monitoring. It primarily optimizes delivery and resilience,
not semantic task-to-model selection.
[Documentation](https://vercel.com/docs/ai-gateway)

**Cloudflare AI Gateway** lets customers build versioned routing flows with visual
or JSON conditions for geography, quotas, user segments, experiments, models, and
fallbacks. It is an edge policy engine rather than a customer-trained scorer.
[Documentation](https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/)

**Requesty** includes latency-based routing that learns recent provider performance.
It predicts which route will be fast, not which model will correctly complete a
particular engineering task.
[Documentation](https://docs.requesty.ai/features/latency-routing)

**TensorZero** combines a self-hosted gateway, evaluation, observability, feedback,
experimentation, and optimization. Its feedback flywheel is strategically relevant,
but it is a broad LLMOps platform rather than a GitHub-native coding router.
[Repository](https://github.com/tensorzero/tensorzero)

### Castform's wedge

Existing routers generally learn or route from prompts, public benchmarks, generic
preference data, static tiers, manual rules, or aggregate provider health. Castform
should use private software-development evidence:

```text
GitHub repositories
  -> reconstructed issues and pull requests
  -> repository snapshots before each solution
  -> tests and acceptance criteria
  -> candidate model and agent rollouts
  -> measured success, cost, and latency
  -> customer-specific Qwen scorer
```

At runtime the scorer estimates:

```text
P(success | task, repository context, customer, route)
```

A route should eventually describe the entire execution configuration—not only a
model ID—including its agent, prompt, tools, context strategy, reasoning level, and
token budget. The deterministic policy then chooses the cheapest approved route
whose predicted success clears the customer's threshold.

GitHub history alone does not reveal the best model. It supplies representative
tasks and often their verifiers; candidate routes still need to attempt those tasks
to create the counterfactual success labels. This evaluation pipeline, rather than
the threshold formula or provider proxy, is the potential moat.

### What Castform must prove

Evaluate on held-out, time-separated repository tasks and compare against:

- always using the strongest model;
- always using the cheapest model;
- static rules;
- OpenRouter Auto and Pareto;
- Not Diamond's pretrained and custom routers;
- RouteLLM trained on the same data; and
- an oracle that selects the cheapest successful route.

Report task success, cost per successful task, latency, escalation rate,
calibration error, savings at matched quality, and quality at matched spend. Until
those measurements exist, “much better” is the product hypothesis. The defensible
claim is:

> Castform is designed to become materially better for a specific customer because
> it evaluates and learns from that customer's engineering work.

### Positioning

- **Against OpenRouter:** OpenRouter makes many models easy to access. Castform
  learns which one completes your company's engineering work.
- **Against Not Diamond:** Not Diamond trains from evaluation data. Castform builds
  coding evaluations from GitHub and measures complete agent outcomes.
- **Against cloud routers:** Cloud routers use general training inside one
  ecosystem. Castform learns privately and can route across clouds, direct
  providers, and self-hosted models.
- **Against gateways:** Gateways reliably deliver a request. Castform decides which
  model or agent deserves it.

Castform should not become another model marketplace, billing aggregator, workflow
editor, or generic observability suite. Its focused proposition is:

> Castform turns GitHub history into verified coding tasks, learns which approved
> models and agents succeed on them, and routes new work to the cheapest option
> likely to pass.
