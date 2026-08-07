"""BenchMAX 0.2.x adapter, tools, rollout worlds, and sparse reward."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    Dataset,
    DatasetSplit,
    Example,
    JsonlDataset,
    JsonRow,
    Tool,
    canonical_example_id,
)
from benchmax.envs.base import resolve_dataset_path

from order_resolution.command_codes import EnvelopeCode
from order_resolution.database import Database
from order_resolution.domain import (
    DomainInvariantError,
    OrderResolutionService,
    assert_world_invariants,
)
from order_resolution.fixtures import delete_operational_world, seed_database_world
from order_resolution.grading import (
    capture_world_snapshot,
    grade_snapshots,
    store_episode_result,
)
from order_resolution.policy import reply_tool_schema
from order_resolution.schema import worlds

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnvironmentDiagnostic:
    """Static validation issue kept local so hosted bundles stay self-contained."""

    severity: str
    code: str
    message: str
    location: str | None = None


@dataclass(slots=True)
class RolloutState:
    rollout_id: str
    world_id: str
    case_id: str
    row: JsonRow
    before: dict[str, Any]
    terminal: bool = False
    tool_errors: int = 0
    tool_calls: int = 0
    tool_names: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.tool_names is None:
            self.tool_names = Counter()


class OrderResolutionEnv(BaseEnv):
    """Resolve post-purchase requests against isolated relational worlds."""

    max_turns = 8
    max_tool_calls = 16

    def __init__(
        self,
        runtime_database_url: str,
        *,
        retain_demo_worlds: bool = False,
        world_ttl_seconds: int = 3_600,
    ) -> None:
        super().__init__()
        if world_ttl_seconds < 60:
            raise ValueError("world_ttl_seconds must be at least 60")
        self._database = Database(runtime_database_url)
        self._service = OrderResolutionService(self._database)
        self._retain_demo_worlds = retain_demo_worlds
        self._world_ttl_seconds = world_ttl_seconds
        self._states: dict[str, RolloutState] = {}
        self._states_lock = asyncio.Lock()

    def validation_diagnostics(self) -> tuple[EnvironmentDiagnostic, ...]:
        return (
            EnvironmentDiagnostic(
                severity="warning",
                code="disposable_runtime_dsn_bundled",
                message=(
                    "hosted validation bundles an expiring child-branch runtime DSN; "
                    "never bundle the admin URL or Neon API key"
                ),
                location="constructor.runtime_database_url",
            ),
        )

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> Dataset[JsonRow]:
        return JsonlDataset(
            resolve_dataset_path(base_dir, f"{split}.jsonl"),
            row_to_example=_to_example,
            max_examples=max_examples,
        )

    async def list_tools(self) -> list[Tool]:
        return list(TOOLS)

    @asynccontextmanager
    async def rollout_context(
        self, rollout_id: str, example: Example[JsonRow]
    ) -> AsyncGenerator[None]:
        world_id = world_id_for_rollout(rollout_id)

        async def seed(connection) -> str:
            return await seed_database_world(
                connection,
                row=example.payload,
                world_id=world_id,
                ttl_seconds=self._world_ttl_seconds,
            )

        case_id = await self._database.transaction(seed)
        before = await self._database.read(
            lambda connection: capture_world_snapshot(connection, world_id)
        )
        state = RolloutState(
            rollout_id=rollout_id,
            world_id=world_id,
            case_id=case_id,
            row=example.payload,
            before=before,
        )
        async with self._states_lock:
            if rollout_id in self._states:
                raise RuntimeError(f"duplicate active rollout_id: {rollout_id}")
            self._states[rollout_id] = state
        try:
            yield
        finally:
            if not self._retain_demo_worlds:
                try:
                    await self._database.transaction(
                        lambda connection: delete_operational_world(connection, world_id)
                    )
                except Exception:
                    logger.exception("order_resolution.world_cleanup_failed world_id=%s", world_id)
            async with self._states_lock:
                self._states.pop(rollout_id, None)

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> dict[str, Any]:
        state = self._require_state(rollout_id)
        if state.terminal:
            return {"ok": False, "code": EnvelopeCode.EPISODE_TERMINAL}
        state.tool_calls += 1
        assert state.tool_names is not None
        state.tool_names[tool_name] += 1
        try:
            validate_tool_arguments(tool_name, tool_args)
            result = await self._dispatch_tool(state, tool_name, tool_args)
        except (KeyError, TypeError, ValueError) as error:
            state.tool_errors += 1
            return {"ok": False, "code": EnvelopeCode.INVALID_ARGUMENT, "message": str(error)}
        if tool_name == "reply_to_customer" and result.get("ok") and result.get("terminal"):
            state.terminal = True
        return result

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        state = self._require_state(rollout.rollout_id)
        after = await self._database.read(
            lambda connection: capture_world_snapshot(connection, state.world_id)
        )
        invariant_failure = 0.0
        try:
            await self._database.read(
                lambda connection: assert_world_invariants(connection, state.world_id)
            )
        except DomainInvariantError:
            invariant_failure = 1.0
        grade = grade_snapshots(
            before=state.before,
            after=after,
            required=state.row["required_state"],
            forbidden=state.row["forbidden_state"],
            expected_disposition=state.row["expected_disposition"],
            expected_reply=state.row["expected_reply"],
        )
        if invariant_failure:
            grade = replace(
                grade,
                task_success=0.0,
                failures=(*grade.failures, "world invariant failure"),
            )
        repeated_tools = sum(max(0, count - 1) for count in (state.tool_names or {}).values())
        diagnostics = {
            "required_state_fraction": grade.required_state_fraction,
            "forbidden_mutation": grade.forbidden_mutation,
            "invariant_failure": invariant_failure,
            "correct_disposition": grade.correct_disposition,
            "structured_reply_correct": grade.structured_reply_correct,
            "tool_error": float(state.tool_errors > 0),
            "unnecessary_tool_calls": float(repeated_tools),
            "terminal_reply": float(state.terminal),
        }
        await self._database.transaction(
            lambda connection: store_episode_result(
                connection,
                world_id=state.world_id,
                scenario_id=state.row["task_id"],
                before=state.before,
                after=after,
                grade=grade,
                diagnostics=diagnostics,
            )
        )
        return {
            "task_success": grade.task_success,
            "_required_state_fraction": grade.required_state_fraction,
            "_forbidden_mutation": grade.forbidden_mutation,
            "_invariant_failure": invariant_failure,
            "_correct_disposition": grade.correct_disposition,
            "_structured_reply_correct": grade.structured_reply_correct,
            "_tool_error": float(state.tool_errors > 0),
            "_unnecessary_tool_calls": float(repeated_tools),
        }

    async def cleanup_expired_worlds(self, *, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)

        async def cleanup(connection) -> int:
            result = await connection.execute(
                sa.delete(worlds).where(worlds.c.expires_at <= cutoff)
            )
            return result.rowcount

        return int(await self._database.transaction(cleanup))

    async def aclose(self) -> None:
        await self._database.aclose()

    async def _dispatch_tool(
        self, state: RolloutState, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        world_id = state.world_id
        if tool_name == "lookup_orders":
            return await self._service.lookup_orders(
                world_id=world_id,
                customer_email=arguments["customer_email"],
                order_number=arguments.get("order_number"),
            )
        if tool_name == "get_order":
            return await self._service.get_order(
                world_id=world_id, order_number=arguments["order_number"]
            )
        if tool_name == "check_variant_availability":
            return await self._service.check_variant_availability(
                world_id=world_id,
                order_number=arguments["order_number"],
                order_item_id=arguments["order_item_id"],
                requested_options=arguments["requested_options"],
            )
        if tool_name == "cancel_order_item":
            return await self._service.cancel_order_item(
                world_id=world_id,
                order_number=arguments["order_number"],
                order_item_id=arguments["order_item_id"],
                reason=arguments["reason"],
            )
        if tool_name == "change_shipping_address":
            return await self._service.change_shipping_address(
                world_id=world_id,
                order_number=arguments["order_number"],
                address=arguments["address"],
            )
        if tool_name == "replace_order_item_variant":
            return await self._service.replace_order_item_variant(
                world_id=world_id,
                order_number=arguments["order_number"],
                order_item_id=arguments["order_item_id"],
                new_variant_id=arguments["new_variant_id"],
                reason=arguments["reason"],
            )
        if tool_name == "reply_to_customer":
            return await self._service.reply_to_customer(
                world_id=world_id,
                case_id=state.case_id,
                disposition=arguments["disposition"],
                outcome_code=arguments["outcome_code"],
                order_number=arguments["order_number"],
                order_item_id=arguments.get("order_item_id"),
                missing_fields=arguments["missing_fields"],
            )
        state.tool_errors += 1
        return {"ok": False, "code": EnvelopeCode.UNKNOWN_TOOL}

    def _require_state(self, rollout_id: str) -> RolloutState:
        try:
            return self._states[rollout_id]
        except KeyError as error:
            raise RuntimeError(f"rollout {rollout_id!r} has no active world") from error


def _to_example(row: JsonRow) -> Example[JsonRow]:
    required = {
        "task_id",
        "prompt_messages",
        "required_state",
        "forbidden_state",
        "expected_disposition",
        "expected_reply",
        "fixture",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"order-resolution row is missing: {', '.join(missing)}")
    return Example(id=canonical_example_id(row), payload=row)


def world_id_for_rollout(rollout_id: str) -> str:
    return f"world-{hashlib.sha256(rollout_id.encode()).hexdigest()}"


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TEXT = {"type": "string", "minLength": 1}
ORDER_NUMBER = {"type": "string", "minLength": 1}
ORDER_ITEM_ID = {"type": "string", "minLength": 1}
ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "line1": TEXT,
        "line2": {"type": ["string", "null"]},
        "city": TEXT,
        "region": TEXT,
        "postal_code": TEXT,
        "country": TEXT,
    },
    "required": ["line1", "city", "region", "postal_code", "country"],
    "additionalProperties": False,
}
TOOLS: tuple[Tool, ...] = (
    _tool(
        "lookup_orders",
        "find this customer's orders",
        {"customer_email": TEXT, "order_number": ORDER_NUMBER},
        ["customer_email"],
    ),
    _tool(
        "get_order",
        "inspect one order and its fulfillment state",
        {"order_number": ORDER_NUMBER},
        ["order_number"],
    ),
    _tool(
        "check_variant_availability",
        "find in-stock variants matching requested options",
        {
            "order_number": ORDER_NUMBER,
            "order_item_id": ORDER_ITEM_ID,
            "requested_options": {
                "type": "object",
                "additionalProperties": {"type": "string", "minLength": 1},
            },
        },
        ["order_number", "order_item_id", "requested_options"],
    ),
    _tool(
        "cancel_order_item",
        "atomically cancel one eligible order item",
        {"order_number": ORDER_NUMBER, "order_item_id": ORDER_ITEM_ID, "reason": TEXT},
        ["order_number", "order_item_id", "reason"],
    ),
    _tool(
        "change_shipping_address",
        "atomically change an eligible order's shipping address",
        {"order_number": ORDER_NUMBER, "address": ADDRESS_SCHEMA},
        ["order_number", "address"],
    ),
    _tool(
        "replace_order_item_variant",
        "atomically replace one eligible item with a same-price in-stock variant",
        {
            "order_number": ORDER_NUMBER,
            "order_item_id": ORDER_ITEM_ID,
            "new_variant_id": TEXT,
            "reason": TEXT,
        },
        ["order_number", "order_item_id", "new_variant_id", "reason"],
    ),
    {
        "type": "function",
        "function": {
            "name": "reply_to_customer",
            "description": (
                "send the one terminal structured customer reply; dispositions, outcome codes, "
                "and missing fields are closed enums owned by the published reply policy"
            ),
            "parameters": reply_tool_schema(),
        },
    },
)

_TOOL_ARGUMENT_SCHEMAS = {
    tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS
}


def validate_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> None:
    """Validate model arguments against the advertised closed tool schema."""

    schema = _TOOL_ARGUMENT_SCHEMAS.get(tool_name)
    if schema is None:
        return
    _validate_schema_value(schema, arguments, path="arguments")


def _validate_schema_value(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    expected_type = schema.get("type")
    allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if value is None and "null" in allowed_types:
        return
    if "string" in allowed_types:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path} is too short")
        allowed = schema.get("enum")
        if allowed is not None and value not in allowed:
            raise ValueError(f"{path} is not an allowed value")
        return
    if "object" in allowed_types:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
        additional = schema.get("additionalProperties", True)
        if additional is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} has unknown fields: {', '.join(extra)}")
        for key, item in value.items():
            property_schema = properties.get(key)
            if property_schema is None and isinstance(additional, Mapping):
                property_schema = additional
            if property_schema is not None:
                _validate_schema_value(property_schema, item, path=f"{path}.{key}")
        return
    if "array" in allowed_types:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be an array")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_schema_value(item_schema, item, path=f"{path}[{index}]")
        if schema.get("uniqueItems") and len({_canonical_argument(item) for item in value}) != len(
            value
        ):
            raise ValueError(f"{path} must contain unique items")
        return
    raise TypeError(f"{path} has an unsupported schema type")


def _canonical_argument(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


__all__ = [
    "OrderResolutionEnv",
    "TOOLS",
    "validate_tool_arguments",
    "world_id_for_rollout",
]
