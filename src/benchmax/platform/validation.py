"""Local environment validation — catches training failures before job submission.

Simulates one mini training step locally (no GPU, no network) to verify
the env class contract matches what the trainer expects.
"""

from __future__ import annotations

import asyncio
import json
import math
import pickle
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
        if not isinstance(preprocessed, dict) or "prompt" not in preprocessed:
            print("  \u2717 dataset_preprocess did not return StandardizedExample")
            print(
                "    Fix: Must return StandardizedExample with prompt,"
                " ground_truth, init_rollout_args."
            )
            failed += 1
        else:
            print("  \u2713 dataset_preprocess returns StandardizedExample")
            passed += 1
    except Exception as exc:
        print(f"  \u2717 dataset_preprocess raised {type(exc).__name__}: {exc}")
        failed += 1

    # ── 2. Prompt hashability ────────────────────────────────────
    if preprocessed and isinstance(preprocessed, dict) and "prompt" in preprocessed:
        prompt = preprocessed["prompt"]
        try:
            hash(prompt)
            {prompt: True}
            if not isinstance(prompt, str):
                print(f"  \u2717 prompt is hashable but not a string — got {type(prompt).__name__}")
                print("    Fix: dataset_preprocess should return prompt as a string.")
                failed += 1
            else:
                print("  \u2713 prompt is hashable (string)")
                passed += 1
        except TypeError:
            print(f"  \u2717 prompt is NOT hashable — got {type(prompt).__name__}")
            print(
                "    Fix: dataset_preprocess must return prompt as a"
                " string, not a list of messages."
            )
            print("    The reward worker uses prompts as dict keys for reroll tracking.")
            failed += 1
    else:
        print("  - prompt hashability: skipped (no preprocessed result)")

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

    # ── 5. compute_reward with trainer-style kwargs ──────────────
    if env is not None and isinstance(preprocessed, dict) and "prompt" in preprocessed:
        try:
            sample_gt = preprocessed.get("ground_truth", "")
            init_args = preprocessed.get("init_rollout_args", {})
            if not isinstance(init_args, dict):
                init_args = {}

            # Build the sample dict the same way reward_worker.py does:
            # {**data_record, **completion_dict, **data_record["init_rollout_args"]}
            flattened = {
                "prompt": preprocessed["prompt"],
                "ground_truth": sample_gt,
                "init_rollout_args": init_args,
                "rollout_ids": "test-rollout",
                "completions": "I found the answer based on the search results.",
                **init_args,
            }

            reward = _run_async(
                env.compute_reward(
                    rollout_id=flattened["rollout_ids"],
                    completion=flattened["completions"],
                    **flattened,
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
            print("    Fix: compute_reward must accept (rollout_id, completion, **sample).")
            print("    The trainer flattens init_rollout_args into the sample dict.")
            failed += 1

    # ── 5b. Simulated rollout (E2E) ────────────────────────────────
    if env is not None and isinstance(preprocessed, dict) and "prompt" in preprocessed:
        try:
            example = examples[0]
            prompt_text = preprocessed["prompt"]
            gt = preprocessed.get("ground_truth", "")
            init_args = preprocessed.get("init_rollout_args", {})
            if not isinstance(init_args, dict):
                init_args = {}

            tools = _run_async(env.list_tools())

            # ── Call each tool twice (catch stateful bugs) ──────
            completion_msgs: list[dict[str, Any]] = []
            tool_call_count = 0
            for tool in tools:
                query = prompt_text[:200] if prompt_text else "test"
                tool_args = _build_dummy_args(tool.input_schema, query)
                for _ in range(2):
                    result = _run_async(
                        env.run_tool(
                            rollout_id="sim-rollout",
                            tool_name=tool.name,
                            **tool_args,
                        )
                    )
                    completion_msgs.append({"role": "assistant", "content": "Calling tool."})
                    completion_msgs.append({"role": "tool", "content": str(result)[:500]})
                    tool_call_count += 1

            # Final assistant message with ground truth as answer
            answer_text = str(gt or "test answer")
            completion_msgs.append({"role": "assistant", "content": answer_text})

            # ── Call compute_reward with full context ───────────
            # Same flattening as reward_worker.py line 190.
            # Pass through all fields from the original example.
            sim_sample: dict[str, Any] = {
                **example,
                "prompt": prompt_text,
                "ground_truth": gt,
                "init_rollout_args": init_args,
                "rollout_ids": "sim-rollout",
                "completions": completion_msgs,
                **init_args,
            }

            reward = _run_async(
                env.compute_reward(
                    rollout_id=sim_sample["rollout_ids"],
                    completion=sim_sample["completions"],
                    **sim_sample,
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
    try:
        data = cloudpickle.dumps(env_class)
        restored_cls = pickle.loads(data)
        restored_env = restored_cls(**env_args)
        tools = _run_async(restored_env.list_tools())
        print(f"  \u2713 pickle round-trip OK ({len(data)} bytes, {len(tools)} tools)")
        passed += 1
    except Exception as exc:
        print(f"  \u2717 pickle round-trip failed: {type(exc).__name__}: {exc}")
        failed += 1

    # ── 6b. env_args pickle ────────────────────────────────────────
    try:
        args_data = cloudpickle.dumps(env_args)
        restored_args = pickle.loads(args_data)
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
