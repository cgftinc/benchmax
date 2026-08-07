"""Model-free local contract test for the live BenchMAX environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmax.envs import (
    BaseRollout,
    Example,
    JsonRow,
    RolloutRequest,
    StaticBearerAuth,
)

from order_resolution import benchmark_spec as spec
from order_resolution.fixtures import render_jsonl
from order_resolution.grading import capture_world_snapshot
from order_resolution.order_env import OrderResolutionEnv, world_id_for_rollout
from order_resolution.policy import (
    DISPOSITION_BY_CODE,
    INTENTS,
    ITEM_ID_RULES,
    CodeClass,
    ItemIdRule,
    classify,
    required_missing_fields,
)


@dataclass(frozen=True, slots=True)
class ScriptCall:
    name: str
    arguments: dict[str, Any]


class ScriptedOrderResolutionEnv(OrderResolutionEnv):
    """Run fixed tool scripts through inherited concurrent group execution."""

    def __init__(
        self,
        runtime_database_url: str,
        *,
        scripts: dict[str, list[ScriptCall]],
        seeded_barrier: asyncio.Barrier | None = None,
        mutated_barrier: asyncio.Barrier | None = None,
    ) -> None:
        super().__init__(runtime_database_url, retain_demo_worlds=True)
        self._scripts = scripts
        self._seeded_barrier = seeded_barrier
        self._mutated_barrier = mutated_barrier

    async def run_rollout(self, request: RolloutRequest[JsonRow]) -> BaseRollout:
        async with self.rollout_context(request.rollout_id, request.example):
            if self._seeded_barrier is not None:
                await self._seeded_barrier.wait()
            messages = [dict(message) for message in request.example.payload["prompt_messages"]]
            for call in self._scripts[request.rollout_id]:
                result = await self.run_tool(request.rollout_id, call.name, **call.arguments)
                messages.append({"role": "tool", "content": str(result)})
            if self._mutated_barrier is not None:
                await self._mutated_barrier.wait()
            rollout = _rollout(request.rollout_id, request.example, messages=messages)
            rewards = await self.compute_reward(rollout)
            return replace(rollout, rewards=rewards)


async def run_contract_test(database_url: str, data_dir: Path) -> dict[str, int]:
    """Exercise every oracle plus isolation, zero-reward, and terminal guards."""

    env = OrderResolutionEnv(database_url)
    correct_count = 0
    incorrect_count = 0
    try:
        dataset = await env.create_dataset("train", data_dir)
        if len(dataset) != 180:
            raise AssertionError(f"expected 180 training examples, got {len(dataset)}")
        if any(len(example.id) != 64 or example.id.lower() != example.id for example in dataset):
            raise AssertionError("dataset contains a noncanonical example ID")

        semaphore = asyncio.Semaphore(16)

        async def correct(index: int, example: Example[JsonRow]) -> None:
            nonlocal correct_count
            async with semaphore:
                rewards, _ = await _direct_attempt(
                    env,
                    example,
                    rollout_id=f"oracle-correct-{index:03d}",
                    script=_correct_script(example.payload),
                )
            if rewards["task_success"] != 1.0:
                raise AssertionError(
                    f"correct trace failed for {example.payload['task_id']}: {rewards}"
                )
            correct_count += 1

        await asyncio.gather(*(correct(index, example) for index, example in enumerate(dataset)))

        representatives: dict[str, Example[JsonRow]] = {}
        for example in dataset:
            representatives.setdefault(example.payload["outcome_class"], example)
        for outcome, example in sorted(representatives.items()):
            rewards, _ = await _direct_attempt(
                env,
                example,
                rollout_id=f"incorrect-{outcome}",
                script=_incorrect_script(example.payload),
            )
            if rewards["task_success"] != 0.0:
                raise AssertionError(f"incorrect {outcome} trace unexpectedly passed")
            incorrect_count += 1

        missing_example = representatives["clarify"]
        missing_rewards, _ = await _direct_attempt(
            env,
            missing_example,
            rollout_id="missing-reply",
            script=[
                ScriptCall(
                    "get_order",
                    {"order_number": missing_example.payload["fixture"]["ids"]["order_number"]},
                )
            ],
        )
        if missing_rewards["task_success"] != 0.0:
            raise AssertionError("missing reply unexpectedly passed")
    finally:
        await env.aclose()

    await _assert_terminal_guard(database_url, data_dir)
    await _assert_concurrent_sibling_isolation(database_url, data_dir)
    return {
        "correct_oracles": correct_count,
        "incorrect_representatives": incorrect_count,
        "terminal_guard": 1,
        "concurrent_siblings": 2,
    }


async def _assert_terminal_guard(database_url: str, data_dir: Path) -> None:
    env = OrderResolutionEnv(database_url, retain_demo_worlds=True)
    rollout_id = "terminal-guard"
    try:
        dataset = await env.create_dataset("train", data_dir)
        example = next(
            example
            for example in dataset
            if example.payload["action_family"] == "change_address"
            and example.payload["outcome_class"] == "execute"
        )
        async with env.rollout_context(rollout_id, example):
            for call in _correct_script(example.payload):
                await env.run_tool(rollout_id, call.name, **call.arguments)
            before = await env._database.read(  # noqa: SLF001 - contract-only state proof
                lambda connection: capture_world_snapshot(
                    connection, world_id_for_rollout(rollout_id)
                )
            )
            ids = example.payload["fixture"]["ids"]
            rejected = await env.run_tool(
                rollout_id,
                "cancel_order_item",
                order_number=ids["order_number"],
                order_item_id=ids["target_item_id"],
                reason="must not run",
            )
            duplicate = await env.run_tool(
                rollout_id, "reply_to_customer", **example.payload["expected_reply"]
            )
            after = await env._database.read(  # noqa: SLF001 - contract-only state proof
                lambda connection: capture_world_snapshot(
                    connection, world_id_for_rollout(rollout_id)
                )
            )
            if rejected != {"ok": False, "code": "EPISODE_TERMINAL"}:
                raise AssertionError("post-reply mutation was not rejected")
            if duplicate != {"ok": False, "code": "EPISODE_TERMINAL"}:
                raise AssertionError("duplicate reply was not rejected")
            if before != after:
                raise AssertionError("post-reply tool changed database state")
            rewards = await env.compute_reward(_rollout(rollout_id, example))
            if rewards["task_success"] != 1.0:
                raise AssertionError("terminal guard corrupted a correct trace")
    finally:
        await env.aclose()


async def _assert_concurrent_sibling_isolation(database_url: str, data_dir: Path) -> None:
    loader = OrderResolutionEnv(database_url)
    try:
        dataset = await loader.create_dataset("train", data_dir)
        example = next(
            example
            for example in dataset
            if example.payload["action_family"] == "cancel_item"
            and example.payload["outcome_class"] == "execute"
        )
    finally:
        await loader.aclose()
    ids = example.payload["fixture"]["ids"]
    sibling_a = "isolation-sibling-a"
    sibling_b = "isolation-sibling-b"
    wrong_reply = {**example.payload["expected_reply"], "outcome_code": "WRONG_OUTCOME"}
    scripts = {
        sibling_a: _correct_script(example.payload),
        sibling_b: [
            ScriptCall(
                "change_shipping_address",
                {
                    "order_number": ids["order_number"],
                    "address": example.payload["fixture"]["requested_address"],
                },
            ),
            ScriptCall("reply_to_customer", wrong_reply),
        ],
    }
    env = ScriptedOrderResolutionEnv(
        database_url,
        scripts=scripts,
        seeded_barrier=asyncio.Barrier(2),
        mutated_barrier=asyncio.Barrier(2),
    )
    try:
        requests = [
            RolloutRequest(
                rollout_id=rollout_id,
                example=example,
                model="unused",
                base_url="http://unused.invalid/v1",
                model_auth=StaticBearerAuth("unused"),
                split="train",
            )
            for rollout_id in (sibling_a, sibling_b)
        ]
        outcomes = await env.run_group(requests)
        if outcomes[sibling_a].rewards["task_success"] != 1.0:
            raise AssertionError("correct isolation sibling failed")
        if outcomes[sibling_b].rewards["task_success"] != 0.0:
            raise AssertionError("incorrect isolation sibling passed")
        state_a = await env._database.read(  # noqa: SLF001 - contract-only state proof
            lambda connection: capture_world_snapshot(connection, world_id_for_rollout(sibling_a))
        )
        state_b = await env._database.read(  # noqa: SLF001 - contract-only state proof
            lambda connection: capture_world_snapshot(connection, world_id_for_rollout(sibling_b))
        )
        if state_a["order_items"][ids["target_item_id"]]["status"] != "cancelled":
            raise AssertionError("sibling A cancellation is missing")
        if state_b["order_items"][ids["target_item_id"]]["status"] != "unfulfilled":
            raise AssertionError("sibling A cancellation leaked into sibling B")
        if state_a["shipping_address"] == example.payload["fixture"]["requested_address"]:
            raise AssertionError("sibling B address change leaked into sibling A")
        if state_b["shipping_address"] != example.payload["fixture"]["requested_address"]:
            raise AssertionError("sibling B address change is missing")
    finally:
        await env.aclose()


async def _direct_attempt(
    env: OrderResolutionEnv,
    example: Example[JsonRow],
    *,
    rollout_id: str,
    script: list[ScriptCall],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    async with env.rollout_context(rollout_id, example):
        for call in script:
            results.append(await env.run_tool(rollout_id, call.name, **call.arguments))
        rewards = await env.compute_reward(_rollout(rollout_id, example))
    return rewards, results


def _rollout(
    rollout_id: str,
    example: Example[JsonRow],
    *,
    messages: list[dict[str, Any]] | None = None,
) -> BaseRollout:
    return BaseRollout(
        rollout_id=rollout_id,
        termination_reason="finished",
        messages=messages or [dict(message) for message in example.payload["prompt_messages"]],
        example_args={
            key: value for key, value in example.payload.items() if key != "prompt_messages"
        },
        split="train",
    )


def _correct_script(row: JsonRow) -> list[ScriptCall]:
    ids = row["fixture"]["ids"]
    script = [ScriptCall("get_order", {"order_number": ids["order_number"]})]
    if row["outcome_class"] != "clarify":
        if row["action_family"] == "cancel_item":
            script.append(
                ScriptCall(
                    "cancel_order_item",
                    {
                        "order_number": ids["order_number"],
                        "order_item_id": ids["target_item_id"],
                        "reason": "customer request",
                    },
                )
            )
        elif row["action_family"] == "change_address":
            script.append(
                ScriptCall(
                    "change_shipping_address",
                    {
                        "order_number": ids["order_number"],
                        "address": row["fixture"]["requested_address"],
                    },
                )
            )
        else:
            script.append(
                ScriptCall(
                    "replace_order_item_variant",
                    {
                        "order_number": ids["order_number"],
                        "order_item_id": ids["target_item_id"],
                        "new_variant_id": ids["new_variant_id"],
                        "reason": "customer requested variant",
                    },
                )
            )
    script.append(ScriptCall("reply_to_customer", dict(row["expected_reply"])))
    return script


def _incorrect_script(row: JsonRow) -> list[ScriptCall]:
    ids = row["fixture"]["ids"]
    wrong_reply = {**row["expected_reply"], "outcome_code": "WRONG_OUTCOME"}
    return [
        ScriptCall("get_order", {"order_number": ids["order_number"]}),
        ScriptCall("reply_to_customer", wrong_reply),
    ]


# ---------------------------------------------------------------------------
# order-resolution-v2 oracle compilation
# ---------------------------------------------------------------------------


class OracleError(RuntimeError):
    """An oracle policy could not be executed faithfully against the service."""


def oracle_rollout_id(task_id: str) -> str:
    """A stable task-derived rollout id, so oracle worlds are reproducible."""

    return f"oracle-{spec.BENCHMARK_ID}-{task_id}"


async def compile_v2_oracles(database_url: str, data_dir: Path) -> list[dict[str, Any]]:
    """Replay every training policy through the real service and record it.

    Each oracle uses only a customer fact or a value a previous tool returned,
    so the frozen demonstrations teach observation rather than hidden state.
    """

    env = OrderResolutionEnv(database_url)
    traces: list[dict[str, Any]] = []
    try:
        dataset = await env.create_dataset("train", data_dir)
        if len(dataset) != spec.TRAIN_ROW_COUNT:
            raise OracleError(f"expected {spec.TRAIN_ROW_COUNT} training rows, got {len(dataset)}")
        semaphore = asyncio.Semaphore(8)

        async def compile_one(example: Example[JsonRow]) -> dict[str, Any]:
            async with semaphore:
                return await _compile_one_oracle(env, example)

        traces = list(await asyncio.gather(*(compile_one(example) for example in dataset)))
    finally:
        await env.aclose()
    traces.sort(key=lambda trace: trace["task_id"])
    return traces


async def _compile_one_oracle(
    env: OrderResolutionEnv, example: Example[JsonRow]
) -> dict[str, Any]:
    row = example.payload
    rollout_id = oracle_rollout_id(row["task_id"])
    executed: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    async with env.rollout_context(rollout_id, example):
        order_number = _customer_order_number(row)
        observed = await _call(
            env, rollout_id, executed, "get_order", {"order_number": order_number}
        )
        if not observed.get("ok"):
            raise OracleError(f"{row['task_id']}: get_order failed with {observed.get('code')}")
        terminal_code, resolved_item_id = await _run_oracle_command(
            env, rollout_id, executed, row=row, order_number=order_number, order=observed
        )
        reply = _oracle_reply(
            row,
            order_number=order_number,
            terminal_code=terminal_code,
            resolved_item_id=resolved_item_id,
        )
        result = await _call(env, rollout_id, executed, "reply_to_customer", reply)
        if not result.get("ok") or not result.get("terminal"):
            raise OracleError(f"{row['task_id']}: terminal reply was rejected: {result}")
        after = await env._database.read(  # noqa: SLF001 - oracle-only state proof
            lambda connection: capture_world_snapshot(connection, world_id_for_rollout(rollout_id))
        )
        rewards = await env.compute_reward(_rollout(rollout_id, example))
    if rewards["task_success"] != 1.0:
        raise OracleError(f"{row['task_id']}: oracle scored {rewards['task_success']}")
    return {
        "benchmark_id": spec.BENCHMARK_ID,
        "task_id": row["task_id"],
        "prompt_messages": row["prompt_messages"],
        "completion_messages": _completion_messages(row["task_id"], executed),
        "final_snapshot_sha256": _sha256_json(after),
        "reward": rewards["task_success"],
    }


async def _run_oracle_command(
    env: OrderResolutionEnv,
    rollout_id: str,
    executed: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    row: JsonRow,
    order_number: str,
    order: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Run the one business command, returning its code and the resolved item.

    Both values come from tool results, never from the hidden fixture, so the
    frozen demonstrations contain no ungrounded identifier.
    """

    action = row["action_family"]
    clarify = row["outcome_class"] == "clarify"
    if action == "change_address":
        if clarify:
            return None, None
        result = await _call(
            env,
            rollout_id,
            executed,
            "change_shipping_address",
            {"order_number": order_number, "address": row["fixture"]["requested_address"]},
        )
        return str(result["code"]), None

    if action == "cancel_item":
        if clarify:
            # The request names no product, so no line is uniquely selectable.
            return None, None
        item = _resolve_item_from_order(row, order)
        result = await _call(
            env,
            rollout_id,
            executed,
            "cancel_order_item",
            {
                "order_number": order_number,
                "order_item_id": item["order_item_id"],
                "reason": "customer request",
            },
        )
        return str(result["code"]), item["order_item_id"]

    item = _resolve_item_from_order(row, order)
    if clarify:
        return None, item["order_item_id"]
    size = row["fixture"]["requested_size"]
    availability = await _call(
        env,
        rollout_id,
        executed,
        "check_variant_availability",
        {
            "order_number": order_number,
            "order_item_id": item["order_item_id"],
            "requested_options": {"size": size},
        },
    )
    candidates = [
        candidate
        for candidate in availability.get("candidates", [])
        if candidate["variant_id"] != item["variant_id"]
    ]
    if len(candidates) != 1:
        raise OracleError(
            f"{row['task_id']}: size {size!r} resolved {len(candidates)} candidate variants"
        )
    result = await _call(
        env,
        rollout_id,
        executed,
        "replace_order_item_variant",
        {
            "order_number": order_number,
            "order_item_id": item["order_item_id"],
            "new_variant_id": candidates[0]["variant_id"],
            "reason": "customer requested a different size",
        },
    )
    return str(result["code"]), item["order_item_id"]


def _resolve_item_from_order(row: JsonRow, order: dict[str, Any]) -> dict[str, Any]:
    """Pick the one line whose visible product name the customer named."""

    message = row["prompt_messages"][-1]["content"].lower()
    matches = [item for item in order["items"] if str(item["product_name"]).lower() in message]
    if len(matches) != 1:
        raise OracleError(
            f"{row['task_id']}: customer request matches {len(matches)} visible order lines"
        )
    return matches[0]


def _customer_order_number(row: JsonRow) -> str:
    order_number = row["fixture"]["ids"]["order_number"]
    if order_number not in row["prompt_messages"][-1]["content"]:
        raise OracleError(f"{row['task_id']}: the customer message omits its order number")
    return order_number


def _oracle_reply(
    row: JsonRow,
    *,
    order_number: str,
    terminal_code: str | None,
    resolved_item_id: str | None,
) -> dict[str, Any]:
    """Build the reply from the public policy plus the command's own code."""

    intent = {
        "cancel_item": INTENTS[0],
        "change_address": INTENTS[1],
        "replace_variant": INTENTS[2],
    }[row["action_family"]]
    if terminal_code is None:
        outcome_code = str(intent.needs_information)
    else:
        if classify(terminal_code) not in {
            CodeClass.TERMINAL_COMPLETED,
            CodeClass.TERMINAL_CANNOT_COMPLETE,
        }:
            raise OracleError(f"{row['task_id']}: command returned non-terminal {terminal_code}")
        outcome_code = terminal_code
    item_rule = ITEM_ID_RULES[outcome_code]
    if item_rule is ItemIdRule.FORBIDDEN:
        order_item_id = None
    else:
        order_item_id = resolved_item_id
        if item_rule is ItemIdRule.REQUIRED and order_item_id is None:
            raise OracleError(f"{row['task_id']}: {outcome_code} needs a tool-resolved item")
    return {
        "disposition": str(DISPOSITION_BY_CODE[outcome_code]),
        "outcome_code": outcome_code,
        "order_number": order_number,
        "order_item_id": order_item_id,
        "missing_fields": sorted(required_missing_fields(outcome_code)),
    }


async def _call(
    env: OrderResolutionEnv,
    rollout_id: str,
    executed: list[tuple[str, dict[str, Any], dict[str, Any]]],
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await env.run_tool(rollout_id, name, **arguments)
    serializable = json.loads(json.dumps(result))
    executed.append((name, arguments, serializable))
    return serializable


def _completion_messages(
    task_id: str, executed: Sequence[tuple[str, dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, (name, arguments, result) in enumerate(executed):
        call_id = f"call-{task_id}-{index:02d}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments, sort_keys=True, separators=(",", ":")
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, sort_keys=True, separators=(",", ":")),
            }
        )
    return messages


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


async def run_v2_contract_test(
    database_url: str, data_dir: Path, *, compile_oracles: bool
) -> dict[str, Any]:
    """Compile the live oracle traces and re-prove the environment guards."""

    summary: dict[str, Any] = {"compiled_oracles": 0}
    if compile_oracles:
        traces = await compile_v2_oracles(database_url, data_dir)
        (data_dir / "oracle_traces.jsonl").write_text(render_jsonl(traces), encoding="utf-8")
        summary["compiled_oracles"] = len(traces)

    env = OrderResolutionEnv(database_url)
    incorrect = 0
    try:
        dataset = await env.create_dataset("train", data_dir)
        representatives: dict[str, Example[JsonRow]] = {}
        for example in dataset:
            representatives.setdefault(example.payload["outcome_class"], example)
        for outcome, example in sorted(representatives.items()):
            rewards, _ = await _direct_attempt(
                env,
                example,
                rollout_id=f"v2-incorrect-{outcome}",
                script=_incorrect_script(example.payload),
            )
            if rewards["task_success"] != 0.0:
                raise AssertionError(f"incorrect {outcome} trace unexpectedly passed")
            incorrect += 1
    finally:
        await env.aclose()

    await _assert_terminal_guard(database_url, data_dir)
    await _assert_concurrent_sibling_isolation(database_url, data_dir)
    return {
        **summary,
        "incorrect_representatives": incorrect,
        "terminal_guard": 1,
        "concurrent_siblings": 2,
    }


__all__ = ["compile_v2_oracles", "oracle_rollout_id", "run_contract_test", "run_v2_contract_test"]
