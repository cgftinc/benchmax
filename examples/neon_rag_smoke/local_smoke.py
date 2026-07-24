"""Local no-GPU integration smoke: neon corpus → SearchEnv → rollout → reward,
plus baked-DSN-through-cloudpickle credential transport against the ACTUAL bundle
transport shape.

Proves, in order and with a hard assert at each step:
  a. a SearchEnv is built over the live gitlab_handbook bm25 neon corpus with the
     RO DSN baked as a `str` into NeonSearch.
  b. `(type(env), constructor_args)` survives `cloudpickle.dumps`/`.loads` — the
     exact `(env_class, constructor_args)` shape benchmax.bundle pickles — and
     the reconstructed env retrieves ranked results (baked DSN survived).
  c. with `NEON_CORPUS_DSN_RO` UNSET in the process env, real rollouts driven
     through `SearchEnv.run_rollout` return ranked retrieval AND a computed
     reward — proving the BAKED string, not the env var, is read at runtime.

No GPU, no sky launch. Policy + judge run against https://llm.castform.dev.
Exits non-zero on any failure; a genuine failure is reported faithfully, never
papered over.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import cloudpickle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_env import (  # noqa: E402
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    GITLAB_SMOKE_ROWS,
    build_env,
    neon_search_constructor_args,
)

from benchmax.auth import StaticBearerAuth, bind_model_auth  # noqa: E402
from benchmax.envs.shared_types import RolloutRequest  # noqa: E402

DSN_ENV_VAR = "NEON_CORPUS_DSN_RO"
CREDS_FILE = Path.home() / ".config" / "neon-benchmax.env"
PROBE_QUERY = "GitLab Security dashboard enablement link"


def _fail(msg: str) -> None:
    print(f"\nSMOKE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_creds() -> dict[str, str]:
    """Parse KEY=VALUE lines from the creds file (no secret values logged)."""
    creds: dict[str, str] = {}
    for line in CREDS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        creds[key.strip()] = val.strip().strip('"').strip("'")
    return creds


def _search_evidence(messages: list[dict]) -> list[str]:
    """Pull tool-result payloads (the search output fed back to the model)."""
    return [
        str(m.get("content", ""))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]


async def _run_rollout(env, row: dict, idx: int, key: str):
    example = env._example_from_row(row)
    request = RolloutRequest(
        rollout_id=f"neon-smoke-{idx}",
        example=example,
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        model_auth=StaticBearerAuth(key),
    )
    # InjectedAuth("judge") inside compute_reward resolves through this binding.
    with bind_model_auth({"judge": StaticBearerAuth(key)}):
        return await env.run_rollout(request)


def main() -> None:
    creds = _load_creds()
    dsn = creds.get(DSN_ENV_VAR)
    key = creds.get("PLATFORM_API_KEY")
    if not dsn:
        _fail(f"{DSN_ENV_VAR} not found in {CREDS_FILE}")
    if not key:
        _fail(f"PLATFORM_API_KEY not found in {CREDS_FILE}")
    # Ensure the env var is NOT set at build time either — the baked str is the
    # only credential source under test.
    os.environ.pop(DSN_ENV_VAR, None)

    print("=" * 72)
    print("STEP a: build SearchEnv over neon bm25 corpus with baked RO DSN str")
    print("=" * 72)
    env = build_env(dsn)
    constructor_args = neon_search_constructor_args(dsn)
    print(f"  env class          : {type(env).__name__}")
    print(f"  search backend     : {env._search.get_params()}")
    print(f"  judge              : {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print(f"  reward keys        : {env.reward_keys}")
    print(f"  max_turns/tool     : {env.max_turns}/{env.max_tool_calls}")

    print("\n" + "=" * 72)
    print("STEP b: cloudpickle round-trip of (env_class, constructor_args)")
    print("        (mirrors benchmax.bundle: cloudpickle.dumps((env_class, args)))")
    print("=" * 72)
    blob = cloudpickle.dumps((type(env), constructor_args))
    print(f"  pickled bytes      : {len(blob)}")
    env_cls, args = cloudpickle.loads(blob)
    if env_cls is not type(env):
        _fail(f"unpickled env class {env_cls!r} != {type(env)!r}")
    if not isinstance(args, dict):
        _fail(f"unpickled constructor_args is {type(args).__name__}, expected dict")
    env2 = env_cls(**args)
    print(f"  reconstructed      : {type(env2).__name__} (baked dsn in closure)")
    hits = env2._search.search(PROBE_QUERY, mode="lexical", top_k=3)
    if not hits:
        _fail("reconstructed env returned zero retrieval results")
    print(f"  retrieval OK       : {len(hits)} ranked results via reconstructed env")
    for i, h in enumerate(hits, 1):
        print(
            f"    {i}. src={h['source']}  score={h['score']:.3f}  "
            f"| {h['content'][:70].replace(chr(10), ' ')}"
        )

    print("\n" + "=" * 72)
    print(f"STEP c: UNSET {DSN_ENV_VAR}, drive real rollouts (retrieval + reward)")
    print("=" * 72)
    os.environ.pop(DSN_ENV_VAR, None)
    if DSN_ENV_VAR in os.environ:
        _fail(f"{DSN_ENV_VAR} still present in process env")
    print(f"  proof: {DSN_ENV_VAR} in os.environ = {DSN_ENV_VAR in os.environ}")

    # Deterministic retrieval proof with the env var gone (not LLM-dependent).
    unset_hits = env2._search.search(PROBE_QUERY, mode="lexical", top_k=2)
    if not unset_hits:
        _fail("retrieval returned nothing with env var unset (baked dsn not read)")
    print(f"  direct search with env unset -> {len(unset_hits)} results (baked dsn read)")

    finished = 0
    searched = 0
    for idx, row in enumerate(GITLAB_SMOKE_ROWS):
        print(f"\n  --- rollout {idx} : {row['question'][:60]}...")
        rollout = asyncio.run(_run_rollout(env2, row, idx, key))
        evidence = _search_evidence(rollout.messages)
        if evidence:
            searched += 1
        print(f"      termination_reason : {rollout.termination_reason}")
        print(f"      search tool calls  : {len(evidence)}")
        if evidence:
            print(f"      retrieval sample   : {evidence[0][:180].replace(chr(10), ' ')}")
        print(f"      rewards            : {rollout.rewards}")
        if rollout.termination_reason == "finished":
            if not rollout.rewards or set(rollout.rewards) != set(env2.reward_keys):
                _fail(
                    f"rollout {idx} finished but reward shape wrong: {rollout.rewards}"
                )
            finished += 1

    print("\n" + "=" * 72)
    print("GATE")
    print("=" * 72)
    if searched == 0:
        _fail("no rollout invoked the search tool — retrieval path not exercised")
    if finished == 0:
        _fail(
            "no rollout reached 'finished' with a computed reward — "
            "compute_reward (judge) was never exercised"
        )
    print(f"  rollouts that retrieved   : {searched}/{len(GITLAB_SMOKE_ROWS)}")
    print(f"  rollouts with computed rwd: {finished}/{len(GITLAB_SMOKE_ROWS)}")
    print(f"  {DSN_ENV_VAR} unset during rollouts: True")
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
