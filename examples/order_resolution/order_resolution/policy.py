"""The public fulfillment reply policy shown to every v2 episode.

This module is the single owner of the business reply contract: which
dispositions exist, which outcome codes a model may report, which facts a
clarification may ask for, which cross-field combinations are valid, and how
every :class:`~order_resolution.command_codes.CommandCode` maps onto those
choices.

The rendered system contract and reply-tool schema are derived from the same
tables, so the prompt, the advertised schema, and the runtime validator cannot
disagree. Everything here is general API documentation given identically to
every task; the target item, database state, expected disposition, expected
code, and grader assertions stay hidden.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NamedTuple

from order_resolution.command_codes import CommandCode


class Disposition(StrEnum):
    """The three terminal ways a support case can close."""

    COMPLETED = "completed"
    NEEDS_INFORMATION = "needs_information"
    CANNOT_COMPLETE = "cannot_complete"


class ClarificationCode(StrEnum):
    """Reply-only outcome codes; no business command emits these."""

    NEEDS_ORDER_ITEM = "NEEDS_ORDER_ITEM"
    NEEDS_POSTAL_CODE = "NEEDS_POSTAL_CODE"
    NEEDS_VARIANT_OPTIONS = "NEEDS_VARIANT_OPTIONS"


class MissingFact(StrEnum):
    """The only facts a clarification may request."""

    ORDER_ITEM_ID = "order_item_id"
    POSTAL_CODE = "shipping_address.postal_code"
    VARIANT_SIZE = "requested_options.size"


class CodeClass(StrEnum):
    """How a model may act on a command result."""

    TERMINAL_COMPLETED = "terminal_completed"
    TERMINAL_CANNOT_COMPLETE = "terminal_cannot_complete"
    CORRECT_AND_RETRY = "correct_and_retry"
    PROTOCOL_ONLY = "protocol_only"


class ItemIdRule(StrEnum):
    """Whether a reply's ``order_item_id`` must name a resolved item."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


#: Exhaustive classification of every command code. The test suite fails when a
#: new :class:`CommandCode` member is not listed here.
CODE_CLASSES: Mapping[CommandCode, CodeClass] = MappingProxyType(
    {
        CommandCode.ITEM_CANCELLED: CodeClass.TERMINAL_COMPLETED,
        CommandCode.SHIPPING_ADDRESS_CHANGED: CodeClass.TERMINAL_COMPLETED,
        CommandCode.ITEM_VARIANT_REPLACED: CodeClass.TERMINAL_COMPLETED,
        CommandCode.ALREADY_HANDED_TO_CARRIER: CodeClass.TERMINAL_CANNOT_COMPLETE,
        CommandCode.VARIANT_OUT_OF_STOCK: CodeClass.TERMINAL_CANNOT_COMPLETE,
        CommandCode.PRICE_OR_PRODUCT_MISMATCH: CodeClass.TERMINAL_CANNOT_COMPLETE,
        CommandCode.ORDER_NOT_FOUND: CodeClass.CORRECT_AND_RETRY,
        CommandCode.ORDER_ITEM_NOT_FOUND: CodeClass.CORRECT_AND_RETRY,
        CommandCode.VARIANT_NOT_FOUND: CodeClass.CORRECT_AND_RETRY,
        CommandCode.ALREADY_CANCELLED: CodeClass.CORRECT_AND_RETRY,
        CommandCode.ALREADY_REQUESTED_VARIANT: CodeClass.CORRECT_AND_RETRY,
        CommandCode.REPLY_ALREADY_SENT: CodeClass.PROTOCOL_ONLY,
    }
)

TERMINAL_COMPLETED_CODES = tuple(
    code for code, kind in CODE_CLASSES.items() if kind is CodeClass.TERMINAL_COMPLETED
)
TERMINAL_CANNOT_COMPLETE_CODES = tuple(
    code for code, kind in CODE_CLASSES.items() if kind is CodeClass.TERMINAL_CANNOT_COMPLETE
)
CORRECT_AND_RETRY_CODES = tuple(
    code for code, kind in CODE_CLASSES.items() if kind is CodeClass.CORRECT_AND_RETRY
)
PROTOCOL_ONLY_CODES = tuple(
    code for code, kind in CODE_CLASSES.items() if kind is CodeClass.PROTOCOL_ONLY
)

#: Every outcome code the reply tool accepts, in the order the schema advertises.
REPLY_OUTCOME_CODES: tuple[str, ...] = (
    *(str(code) for code in TERMINAL_COMPLETED_CODES),
    *(str(code) for code in ClarificationCode),
    *(str(code) for code in TERMINAL_CANNOT_COMPLETE_CODES),
)


class IntentPolicy(NamedTuple):
    """One customer intent and the outcomes it can reach."""

    intent: str
    completed: CommandCode
    needs_information: ClarificationCode
    missing_fact: MissingFact
    cannot_complete: tuple[CommandCode, ...]


INTENTS: tuple[IntentPolicy, ...] = (
    IntentPolicy(
        intent="cancel item",
        completed=CommandCode.ITEM_CANCELLED,
        needs_information=ClarificationCode.NEEDS_ORDER_ITEM,
        missing_fact=MissingFact.ORDER_ITEM_ID,
        cannot_complete=(CommandCode.ALREADY_HANDED_TO_CARRIER,),
    ),
    IntentPolicy(
        intent="change address",
        completed=CommandCode.SHIPPING_ADDRESS_CHANGED,
        needs_information=ClarificationCode.NEEDS_POSTAL_CODE,
        missing_fact=MissingFact.POSTAL_CODE,
        cannot_complete=(CommandCode.ALREADY_HANDED_TO_CARRIER,),
    ),
    IntentPolicy(
        intent="replace variant",
        completed=CommandCode.ITEM_VARIANT_REPLACED,
        needs_information=ClarificationCode.NEEDS_VARIANT_OPTIONS,
        missing_fact=MissingFact.VARIANT_SIZE,
        cannot_complete=(
            CommandCode.ALREADY_HANDED_TO_CARRIER,
            CommandCode.VARIANT_OUT_OF_STOCK,
            CommandCode.PRICE_OR_PRODUCT_MISMATCH,
        ),
    ),
)

#: The exact ``missing_fields`` list each clarification code requires.
CLARIFICATION_FACTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {str(intent.needs_information): (str(intent.missing_fact),) for intent in INTENTS}
)

#: The disposition each outcome code belongs to.
DISPOSITION_BY_CODE: Mapping[str, Disposition] = MappingProxyType(
    {
        **{str(code): Disposition.COMPLETED for code in TERMINAL_COMPLETED_CODES},
        **{str(code): Disposition.NEEDS_INFORMATION for code in ClarificationCode},
        **{str(code): Disposition.CANNOT_COMPLETE for code in TERMINAL_CANNOT_COMPLETE_CODES},
    }
)

#: Whether the reply must name a uniquely resolved item. ``ALREADY_HANDED_TO_CARRIER``
#: is optional because it denies both item and whole-order requests.
ITEM_ID_RULES: Mapping[str, ItemIdRule] = MappingProxyType(
    {
        str(CommandCode.ITEM_CANCELLED): ItemIdRule.REQUIRED,
        str(CommandCode.ITEM_VARIANT_REPLACED): ItemIdRule.REQUIRED,
        str(CommandCode.VARIANT_OUT_OF_STOCK): ItemIdRule.REQUIRED,
        str(CommandCode.PRICE_OR_PRODUCT_MISMATCH): ItemIdRule.REQUIRED,
        str(ClarificationCode.NEEDS_VARIANT_OPTIONS): ItemIdRule.REQUIRED,
        str(CommandCode.SHIPPING_ADDRESS_CHANGED): ItemIdRule.FORBIDDEN,
        str(ClarificationCode.NEEDS_POSTAL_CODE): ItemIdRule.FORBIDDEN,
        str(ClarificationCode.NEEDS_ORDER_ITEM): ItemIdRule.FORBIDDEN,
        str(CommandCode.ALREADY_HANDED_TO_CARRIER): ItemIdRule.OPTIONAL,
    }
)


class ReplyContractError(ValueError):
    """A reply that the public policy does not permit."""


def classify(code: str) -> CodeClass | None:
    """Classify a command code, or ``None`` when it is not a command code."""

    try:
        return CODE_CLASSES[CommandCode(code)]
    except ValueError:
        return None


def is_reply_code(code: str) -> bool:
    return code in DISPOSITION_BY_CODE


def required_missing_fields(outcome_code: str) -> tuple[str, ...]:
    return CLARIFICATION_FACTS.get(outcome_code, ())


def validate_reply(
    *,
    disposition: str,
    outcome_code: str,
    order_item_id: str | None,
    missing_fields: Sequence[str],
) -> None:
    """Reject any cross-field combination the public policy does not allow.

    Raises :class:`ReplyContractError` with a model-readable message. Callers
    surface that as a non-terminal tool result so the episode keeps its one
    valid reply.
    """

    if disposition not in set(Disposition):
        raise ReplyContractError(
            f"disposition {disposition!r} is not one of "
            f"{', '.join(str(value) for value in Disposition)}"
        )
    if not is_reply_code(outcome_code):
        kind = classify(outcome_code)
        if kind is CodeClass.CORRECT_AND_RETRY:
            raise ReplyContractError(
                f"{outcome_code} means the call arguments were stale or wrong; re-inspect the "
                "order and retry the command instead of replying with this code"
            )
        if kind is CodeClass.PROTOCOL_ONLY:
            raise ReplyContractError(
                f"{outcome_code} is a protocol result and can never be a customer outcome"
            )
        raise ReplyContractError(
            f"outcome_code {outcome_code!r} is not one of {', '.join(REPLY_OUTCOME_CODES)}"
        )

    expected_disposition = DISPOSITION_BY_CODE[outcome_code]
    if disposition != expected_disposition:
        raise ReplyContractError(
            f"outcome_code {outcome_code} requires disposition {expected_disposition}, "
            f"not {disposition}"
        )

    expected_fields = list(required_missing_fields(outcome_code))
    if sorted(set(missing_fields)) != sorted(expected_fields):
        rendered = ", ".join(expected_fields) if expected_fields else "an empty list"
        raise ReplyContractError(f"outcome_code {outcome_code} requires missing_fields {rendered}")

    rule = ITEM_ID_RULES[outcome_code]
    resolved = isinstance(order_item_id, str) and bool(order_item_id.strip())
    if rule is ItemIdRule.REQUIRED and not resolved:
        raise ReplyContractError(
            f"outcome_code {outcome_code} applies to one resolved item; order_item_id is required"
        )
    if rule is ItemIdRule.FORBIDDEN and resolved:
        raise ReplyContractError(
            f"outcome_code {outcome_code} does not identify a single item; "
            "order_item_id must be null"
        )


def reply_tool_schema() -> dict[str, Any]:
    """The closed parameter schema advertised for ``reply_to_customer``."""

    return {
        "type": "object",
        "properties": {
            "disposition": {"type": "string", "enum": [str(value) for value in Disposition]},
            "outcome_code": {"type": "string", "enum": list(REPLY_OUTCOME_CODES)},
            "order_number": {"type": "string", "minLength": 1},
            "order_item_id": {"type": ["string", "null"]},
            "missing_fields": {
                "type": "array",
                "items": {"type": "string", "enum": [str(value) for value in MissingFact]},
                "uniqueItems": True,
            },
        },
        # order_item_id is required-but-nullable so the model always states
        # whether it resolved an item rather than silently omitting the key.
        "required": [
            "disposition",
            "outcome_code",
            "order_number",
            "order_item_id",
            "missing_fields",
        ],
        "additionalProperties": False,
    }


def outcome_table_lines() -> list[str]:
    """One rendered policy line per intent, in declaration order."""

    lines = []
    for intent in INTENTS:
        denials = ", ".join(str(code) for code in intent.cannot_complete)
        lines.append(
            f"- {intent.intent}: completed -> {intent.completed}; "
            f"needs_information -> {intent.needs_information} "
            f"(missing_fields: {intent.missing_fact}); "
            f"cannot_complete -> {denials}"
        )
    return lines


def render_system_contract() -> str:
    """Render the complete public policy given to every v2 episode."""

    retry = ", ".join(str(code) for code in CORRECT_AND_RETRY_CODES)
    protocol = ", ".join(str(code) for code in PROTOCOL_ONLY_CODES)
    sections = [
        "you are a post-purchase support agent. resolve the customer's latest request using "
        "only the typed business tools, then send exactly one structured customer reply.",
        "",
        "fulfillment policy:",
        "- inspect the order with get_order before any mutation. never choose arbitrarily "
        "among plausible order lines.",
        "- when a required customer fact or a unique item selector is missing, do not mutate "
        "anything: reply with needs_information.",
        "- otherwise call the one relevant command, then copy its returned canonical code "
        "verbatim into your reply.",
        "- check_variant_availability before replacing a variant.",
        "",
        "outcome codes by intent:",
        *outcome_table_lines(),
        "",
        "dispositions:",
        "- completed: a mutation succeeded.",
        "- cannot_complete: a known policy boundary blocks the request.",
        "- needs_information: exactly one of the three clarification cases above applies.",
        "",
        "command results you must not reply with:",
        f"- correct and retry within your remaining budget: {retry}. these mean your arguments "
        "were stale or wrong; re-inspect the order and call again.",
        f"- protocol result, never a new reply: {protocol}.",
        "the reply tool rejects these codes and the case stays open.",
        "",
        "reply fields:",
        "- order_item_id: the resolved order item id, or null for address actions and for an "
        "unresolved cancellation target. always pass the key explicitly.",
        "- missing_fields: exactly the fact listed above for a clarification, and an empty "
        "list otherwise.",
        "- send exactly one reply for the latest customer request, then stop.",
    ]
    return "\n".join(sections)


__all__ = [
    "CLARIFICATION_FACTS",
    "CODE_CLASSES",
    "CORRECT_AND_RETRY_CODES",
    "ClarificationCode",
    "CodeClass",
    "DISPOSITION_BY_CODE",
    "Disposition",
    "INTENTS",
    "ITEM_ID_RULES",
    "ItemIdRule",
    "MissingFact",
    "PROTOCOL_ONLY_CODES",
    "REPLY_OUTCOME_CODES",
    "ReplyContractError",
    "TERMINAL_CANNOT_COMPLETE_CODES",
    "TERMINAL_COMPLETED_CODES",
    "classify",
    "is_reply_code",
    "render_system_contract",
    "reply_tool_schema",
    "required_missing_fields",
    "validate_reply",
]
