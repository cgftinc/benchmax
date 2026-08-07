"""Pure contract tests for the BenchMAX environment adapter."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from benchmax.envs import canonical_example_id
from order_resolution.order_env import (
    OrderResolutionEnv,
    validate_tool_arguments,
    world_id_for_rollout,
)
from order_resolution.policy import (
    CORRECT_AND_RETRY_CODES,
    PROTOCOL_ONLY_CODES,
    REPLY_OUTCOME_CODES,
    render_system_contract,
    reply_tool_schema,
)

DATA_DIR = Path(__file__).parents[1] / "data"
LOCAL_DATABASE_URL = "postgresql://order_resolution:order_resolution@localhost/order_resolution"


@pytest.mark.asyncio
async def test_dataset_preserves_payload_and_uses_canonical_ids() -> None:
    env = OrderResolutionEnv(LOCAL_DATABASE_URL)
    try:
        dataset = await env.create_dataset("train", DATA_DIR, max_examples=2)
    finally:
        await env.aclose()

    assert len(dataset) == 2
    for example in dataset:
        assert re.fullmatch(r"[0-9a-f]{64}", example.id)
        assert example.id == canonical_example_id(example.payload)
        assert example.payload["task_id"].startswith("train-")
        assert example.payload["prompt_messages"][0]["role"] == "system"
        assert example.payload["prompt_messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_tool_contract_is_closed_and_has_one_terminal_reply() -> None:
    env = OrderResolutionEnv(LOCAL_DATABASE_URL)
    try:
        tools = await env.list_tools()
    finally:
        await env.aclose()

    names = [tool["function"]["name"] for tool in tools]
    assert names == [
        "lookup_orders",
        "get_order",
        "check_variant_availability",
        "cancel_order_item",
        "change_shipping_address",
        "replace_order_item_variant",
        "reply_to_customer",
    ]
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in tools)
    reply = tools[-1]["function"]
    parameters = reply["parameters"]
    assert parameters == reply_tool_schema()
    # order_item_id is required-but-nullable: the model always states whether it
    # resolved an item instead of silently omitting the key.
    assert parameters["required"] == [
        "disposition",
        "outcome_code",
        "order_number",
        "order_item_id",
        "missing_fields",
    ]
    assert parameters["properties"]["order_item_id"]["type"] == ["string", "null"]
    assert parameters["properties"]["disposition"]["enum"] == [
        "completed",
        "needs_information",
        "cannot_complete",
    ]
    assert parameters["properties"]["outcome_code"]["enum"] == list(REPLY_OUTCOME_CODES)
    assert parameters["properties"]["missing_fields"]["items"]["enum"] == [
        "order_item_id",
        "shipping_address.postal_code",
        "requested_options.size",
    ]


@pytest.mark.asyncio
async def test_reply_tool_rejects_retry_and_protocol_codes() -> None:
    """Domain error codes are visible to the model but are never reply outcomes."""

    env = OrderResolutionEnv(LOCAL_DATABASE_URL)
    try:
        tools = await env.list_tools()
    finally:
        await env.aclose()

    allowed = set(tools[-1]["function"]["parameters"]["properties"]["outcome_code"]["enum"])
    assert allowed.isdisjoint(str(code) for code in CORRECT_AND_RETRY_CODES)
    assert allowed.isdisjoint(str(code) for code in PROTOCOL_ONLY_CODES)
    for code in (*CORRECT_AND_RETRY_CODES, *PROTOCOL_ONLY_CODES):
        with pytest.raises(ValueError, match="not an allowed value"):
            validate_tool_arguments(
                "reply_to_customer",
                {
                    "disposition": "cannot_complete",
                    "outcome_code": str(code),
                    "order_number": "ORD-1",
                    "order_item_id": None,
                    "missing_fields": [],
                },
            )


def test_system_contract_publishes_the_whole_vocabulary_and_no_answers() -> None:
    contract = render_system_contract()
    for code in (*REPLY_OUTCOME_CODES, *CORRECT_AND_RETRY_CODES, *PROTOCOL_ONLY_CODES):
        assert str(code) in contract
    for fact in ("order_item_id", "shipping_address.postal_code", "requested_options.size"):
        assert fact in contract
    for disposition in ("completed", "needs_information", "cannot_complete"):
        assert disposition in contract
    assert "get_order" in contract
    assert "check_variant_availability" in contract
    # General policy only: no task target, expected outcome, or hidden assertion.
    lowered = contract.lower()
    for leak in ("expected", "required_state", "forbidden_state", "task_id", "the answer"):
        assert leak not in lowered
    assert render_system_contract() == contract


@pytest.mark.asyncio
async def test_environment_declares_disposable_runtime_dsn_constraint() -> None:
    env = OrderResolutionEnv(LOCAL_DATABASE_URL)
    try:
        diagnostics = env.validation_diagnostics()
    finally:
        await env.aclose()

    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "warning"
    assert diagnostics[0].code == "disposable_runtime_dsn_bundled"
    assert "admin URL" in diagnostics[0].message


def test_rollout_world_ids_are_stable_and_isolated() -> None:
    first = world_id_for_rollout("rollout-a")
    assert first == world_id_for_rollout("rollout-a")
    assert first != world_id_for_rollout("rollout-b")
    assert re.fullmatch(r"world-[0-9a-f]{64}", first)


def test_world_ttl_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="at least 60"):
        OrderResolutionEnv(LOCAL_DATABASE_URL, world_ttl_seconds=59)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        (
            "change_shipping_address",
            {"order_number": "ORD-1", "address": "not-an-object"},
            "arguments.address must be an object",
        ),
        (
            "check_variant_availability",
            {
                "order_number": "ORD-1",
                "order_item_id": "ITEM-1",
                "requested_options": {"size": 42},
            },
            "arguments.requested_options.size must be a string",
        ),
        (
            "reply_to_customer",
            {
                "disposition": "completed",
                "outcome_code": "ITEM_CANCELLED",
                "order_number": "ORD-1",
                "order_item_id": None,
                "missing_fields": "none",
            },
            "arguments.missing_fields must be an array",
        ),
    ],
)
def test_nested_schema_invalid_tool_arguments_are_rejected(
    tool_name: str, arguments: dict, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        validate_tool_arguments(tool_name, arguments)
