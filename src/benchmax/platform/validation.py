"""Local environment validation — catches training failures before job submission.

Simulates one mini training step locally (no GPU, no network) to verify
the env class contract matches what the trainer expects.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle

_TOOL_TIMEOUT = 30.0


def _run_async(coro: Any, timeout: float = _TOOL_TIMEOUT) -> Any:
    """Run a coroutine with a timeout."""
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def _build_dummy_args(
    schema: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """Build dummy tool args from a JSON schema, respecting enums."""
    args: dict[str, Any] = {}
    for pname, pschema in schema.get("properties", {}).items():
        ptype = pschema.get("type", "string")
        if "enum" in pschema:
            args[pname] = pschema["enum"][0]
        elif pname in ("query", "text", "search_query"):
            args[pname] = query
        elif ptype == "string":
            args[pname] = "test"
        elif ptype == "integer":
            args[pname] = 10
        elif ptype == "number":
            args[pname] = 1.0
        elif ptype == "boolean":
            args[pname] = True
        elif ptype == "array":
            args[pname] = []
        elif ptype == "object":
            args[pname] = {}
    return args


def _ensure_nest_asyncio() -> None:
    """Patch asyncio for Jupyter notebooks (running event loop)."""
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass


def validate_env(
    env_class: type,
    env_args: dict[str, Any],
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]] | None = None,
) -> bool:
    """Validate an environment class against the trainer's calling conventions.

    Runs a comprehensive set of checks that mirror how the trainer actually
    calls env methods, including a simulated rollout with real tool calls
    and reward computation.

    Can be called standalone before ``train()`` or is called automatically
    by ``train(validate_env=True)`` (the default).

    Warning:
        This function calls your env's tools with dummy arguments
        against real backends. If your tools have side effects
        (writes, deletes, sends), use a test backend.

    Args:
        env_class: The environment class (e.g., SearchEnv).
        env_args: Constructor kwargs for the env (same as train(env_args=...)).
        train_dataset: Training examples (list of dicts with question/answer).
        eval_dataset: Optional eval examples. Uses train_dataset[:2] if not given.

    Returns:
        True if all checks pass, False otherwise.
    """
    _ensure_nest_asyncio()

    if not train_dataset:
        print("  \u2717 train_dataset is empty")
        return False

    examples = train_dataset[:5]
    passed = 0
    failed = 0

    print(
        "  \u26a0 Tools will be called with dummy args against"
        " real backends. Use a test backend if tools have"
        " side effects."
    )

    print("Environment Validation")

    # ── 1. dataset_preprocess ────────────────────────────────────
    preprocessed = None
    try:
        preprocessed = env_class.dataset_preprocess(examples[0])
        required = {"id", "seed_messages"}
        if not isinstance(preprocessed, dict) or not required.issubset(preprocessed):
            missing = (
                required - set(preprocessed)
                if isinstance(preprocessed, dict)
                else required
            )
            print(
                f"  \u2717 dataset_preprocess did not return Example "
                f"(missing {sorted(missing)})"
            )
            print(
                "    Fix: return benchmax.envs.example_id.make_example("
                "seed_messages=[...], task=...)."
            )
            failed += 1
        else:
            print("  \u2713 dataset_preprocess returns Example with id + seed_messages")
            passed += 1
    except Exception as exc:
        print(f"  \u2717 dataset_preprocess raised {type(exc).__name__}: {exc}")
        failed += 1

    # ── 2. seed_messages shape ───────────────────────────────────
    if preprocessed and isinstance(preprocessed, dict) and "seed_messages" in preprocessed:
        seed = preprocessed["seed_messages"]
        if not isinstance(seed, list) or not all(
            isinstance(mm, dict) and "role" in mm and "content" in mm for mm in seed
        ):
            print("  \u2717 seed_messages is not a list of {role,content} dicts")
            failed += 1
        elif not seed:
            print("  \u2717 seed_messages is empty")
            failed += 1
        else:
            print(f"  \u2713 seed_messages is a {len(seed)}-message chat list")
            passed += 1
    else:
        print("  - seed_messages shape check: skipped (no preprocessed result)")

    # ── 3. load_dataset ──────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            tmp_path = f.name

        result = env_class.load_dataset("json", data_files=tmp_path, split="train")
        if isinstance(result, tuple) and len(result) == 2:
            ds, _ = result
            if len(ds) > 0:
                print(
                    f'  \u2713 load_dataset accepts ("json", data_files=...,'
                    f' split="train") — {len(ds)} rows'
                )
                passed += 1
            else:
                print("  \u2717 load_dataset returned empty dataset")
                failed += 1
        else:
            print(
                f"  \u2717 load_dataset returned {type(result).__name__},"
                " expected (Dataset, str | None)"
            )
            failed += 1

        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"  \u2717 load_dataset raised {type(exc).__name__}: {exc}")
        print('    Fix: load_dataset must accept ("json", data_files=path, split="train").')
        failed += 1

    # ── 4. Instantiate env + list_tools + run_tool ───────────────
    env = None
    try:
        env = env_class(**env_args)
    except Exception as exc:
        print(f"  \u2717 env instantiation failed: {type(exc).__name__}: {exc}")
        failed += 1

    if env is not None:
        try:
            tools = _run_async(env.list_tools())
            print(f"  \u2713 list_tools returns {len(tools)} tool(s)")
            passed += 1

            if tools:
                tool = tools[0]
                dummy_args = _build_dummy_args(tool.input_schema, "test query")

                try:
                    result = _run_async(
                        env.run_tool(rollout_id="test", tool_name=tool.name, **dummy_args)
                    )
                    if isinstance(result, str):
                        print(f"  \u2713 run_tool returns string (tested: {tool.name})")
                        passed += 1
                    else:
                        print(
                            f"  \u2717 run_tool returned {type(result).__name__}, expected string"
                        )
                        failed += 1
                except Exception as exc:
                    print(f"  \u2717 run_tool raised {type(exc).__name__}: {exc}")
                    print("    Fix: run_tool must return a string. If tools need a real backend,")
                    print(
                        "    the training loop calls run_tool when the model generates tool_calls."
                    )
                    failed += 1
            else:
                print("  - run_tool: skipped (no tools defined)")
        except Exception as exc:
            print(f"  \u2717 list_tools raised {type(exc).__name__}: {exc}")
            failed += 1

    # ── 5. compute_reward with trainer-style args ────────────────
    if env is not None and isinstance(preprocessed, dict) and "seed_messages" in preprocessed:
        try:
            task = preprocessed.get("task")
            init_args = preprocessed.get("init_rollout_args") or {}

            # Simulate the trainer's call: a fake one-turn transcript echoing
            # the seed plus a stub assistant turn. compute_reward receives
            # task verbatim and runtime kwargs (init_rollout_args fields).
            messages = list(preprocessed["seed_messages"]) + [
                {"role": "assistant", "content": "I found the answer based on the search results."}
            ]

            reward = _run_async(
                env.compute_reward(
                    rollout_id="test-rollout",
                    messages=messages,
                    task=task,
                    **init_args,
                )
            )

            if not isinstance(reward, dict):
                print(
                    f"  \u2717 compute_reward returned"
                    f" {type(reward).__name__}, expected dict[str, float]"
                )
                failed += 1
            else:
                bad_values = {
                    k: type(v).__name__
                    for k, v in reward.items()
                    if not isinstance(v, (int, float))
                }
                if bad_values:
                    print(f"  \u2717 compute_reward has non-float values: {bad_values}")
                    failed += 1
                else:
                    non_finite = {k: v for k, v in reward.items() if not math.isfinite(v)}
                    if non_finite:
                        print(f"  \u2717 compute_reward has NaN/Inf values: {non_finite}")
                        print(
                            "    Fix: reward values must be finite."
                            " NaN/Inf break training gradients."
                        )
                        failed += 1
                    else:
                        print(f"  \u2713 compute_reward returns dict[str, float]: {reward}")
                        passed += 1
        except Exception as exc:
            print(f"  \u2717 compute_reward raised {type(exc).__name__}: {exc}")
            print("    Fix: compute_reward signature is (rollout_id, messages, task, **kwargs).")
            print("    Read example data from `task`, runtime fields from `kwargs`.")
            failed += 1

    # ── 5b. Simulated rollout (E2E) ──────────────────────────────
    if env is not None and isinstance(preprocessed, dict) and "seed_messages" in preprocessed:
        try:
            seed_messages = preprocessed["seed_messages"]
            task = preprocessed.get("task")
            init_args = preprocessed.get("init_rollout_args") or {}

            tools = _run_async(env.list_tools())

            # Seed text used for dummy tool-arg generation.
            first_user = next(
                (m["content"] for m in seed_messages if m.get("role") == "user"),
                "",
            )
            query_text = first_user[:200] if isinstance(first_user, str) else "test"

            # ── Call each tool twice (catch stateful bugs) ──────
            transcript: list[dict[str, Any]] = list(seed_messages)
            tool_call_count = 0
            for tool in tools:
                tool_args = _build_dummy_args(tool.input_schema, query_text)
                for _ in range(2):
                    result = _run_async(
                        env.run_tool(
                            rollout_id="sim-rollout",
                            tool_name=tool.name,
                            **tool_args,
                        )
                    )
                    transcript.append({"role": "assistant", "content": "Calling tool."})
                    transcript.append({"role": "tool", "content": str(result)[:500]})
                    tool_call_count += 1

            # Final assistant message echoing ground truth if available.
            gt = (task or {}).get("ground_truth")
            transcript.append({"role": "assistant", "content": str(gt or "test answer")})

            reward = _run_async(
                env.compute_reward(
                    rollout_id="sim-rollout",
                    messages=transcript,
                    task=task,
                    **init_args,
                )
            )

            if not isinstance(reward, dict):
                print(
                    f"  \u2717 simulated rollout: compute_reward returned {type(reward).__name__}"
                )
                failed += 1
            else:
                bad = {
                    k: v
                    for k, v in reward.items()
                    if not isinstance(v, (int, float)) or not math.isfinite(v)
                }
                if bad:
                    print(f"  \u2717 simulated rollout: bad reward values: {bad}")
                    failed += 1
                else:
                    tools_desc = f"{tool_call_count} tool calls" if tool_call_count else "no tools"
                    print(f"  \u2713 simulated rollout OK ({tools_desc}, reward={reward})")
                    passed += 1

        except Exception as exc:
            print(f"  \u2717 simulated rollout failed: {type(exc).__name__}: {exc}")
            failed += 1

    # ── 6. Pickle round-trip ─────────────────────────────────────
    # Use cloudpickle on both sides so envs that import from local modules
    # (registered via local_modules in bundle_env) round-trip the same way as
    # they will on the trainer. Plain pickle.loads can read simple cloudpickle
    # output but breaks on by-value module pickling — silently mismatching
    # local validation vs trainer behavior.
    try:
        data = cloudpickle.dumps(env_class)
        restored_cls = cloudpickle.loads(data)
        restored_env = restored_cls(**env_args)
        tools = _run_async(restored_env.list_tools())
        print(f"  \u2713 pickle round-trip OK ({len(data)} bytes, {len(tools)} tools)")
        passed += 1
    except Exception as exc:
        print(f"  \u2717 pickle round-trip failed: {type(exc).__name__}: {exc}")
        failed += 1

    # \u2500\u2500 6a. Local-modules guard \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Same-process round-trip above succeeds even when local_modules are
    # forgotten, because the user's working module is already in sys.modules
    # so cloudpickle's by-reference resolves via cache. On a fresh worker
    # process there's no cache and the import fails. Inspect the pickle's
    # find_class refs to catch this pre-upload.
    try:
        from benchmax.bundle import unregistered_local_refs

        risky = unregistered_local_refs(cloudpickle.dumps(env_class))
        if risky:
            print(
                f"  \u2717 {env_class.__name__}: missing "
                f"local_modules=[{', '.join(risky)}]"
            )
            print(
                "    (round-trip above passed because sys.modules cache "
                "hides this in-process; trainer will fail to import)"
            )
            failed += 1
        else:
            print("  \u2713 no unregistered local-module references")
            passed += 1
    except Exception as exc:
        print(f"  \u2717 local-modules check failed: {type(exc).__name__}: {exc}")
        failed += 1

    # ── 6b. env_args pickle ────────────────────────────────────────
    try:
        args_data = cloudpickle.dumps(env_args)
        restored_args = cloudpickle.loads(args_data)
        assert isinstance(restored_args, dict)
        print(f"  \u2713 env_args pickle round-trip OK ({len(args_data)} bytes)")
        passed += 1
    except Exception as exc:
        print(f"  \u2717 env_args pickle failed: {type(exc).__name__}: {exc}")
        print(
            "    Fix: env_args must be serializable. Lambdas, SDK"
            " clients, and Pydantic models may not pickle."
        )
        failed += 1

    # ── 7. System prompt ─────────────────────────────────────────
    if env is not None:
        sp = getattr(env, "system_prompt", None)
        if not sp or not isinstance(sp, str):
            print("  \u2717 system_prompt is missing or not a string")
            failed += 1
        else:
            msg = f"  \u2713 system_prompt: {len(sp)} chars"
            if len(sp) > 10000:
                msg += " (warning: very long — consider shortening)"
            print(msg)
            passed += 1

    # ── Summary ──────────────────────────────────────────────────
    print()
    if failed == 0:
        print(f"All {passed} checks passed. Safe to call train().")
    else:
        print(f"{failed} check(s) failed. Fix before calling train().")

    return failed == 0
