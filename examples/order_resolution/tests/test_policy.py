"""The published reply policy: classification, combinations, and rendering."""

from __future__ import annotations

import json

import pytest
from order_resolution.command_codes import CODES_BY_COMMAND, CommandCode, EnvelopeCode
from order_resolution.policy import (
    CLARIFICATION_FACTS,
    CODE_CLASSES,
    CORRECT_AND_RETRY_CODES,
    DISPOSITION_BY_CODE,
    INTENTS,
    ITEM_ID_RULES,
    PROTOCOL_ONLY_CODES,
    REPLY_OUTCOME_CODES,
    TERMINAL_CANNOT_COMPLETE_CODES,
    TERMINAL_COMPLETED_CODES,
    ClarificationCode,
    CodeClass,
    Disposition,
    MissingFact,
    ReplyContractError,
    classify,
    render_system_contract,
    reply_tool_schema,
    validate_reply,
)


def test_every_command_code_is_classified() -> None:
    """Adding a domain code without deciding how a model may report it fails here."""

    assert set(CODE_CLASSES) == set(CommandCode)
    assert len(CommandCode) == 12
    assert set(CODE_CLASSES.values()) == set(CodeClass)


def test_classification_matches_the_published_contract() -> None:
    assert set(TERMINAL_COMPLETED_CODES) == {
        CommandCode.ITEM_CANCELLED,
        CommandCode.SHIPPING_ADDRESS_CHANGED,
        CommandCode.ITEM_VARIANT_REPLACED,
    }
    assert set(TERMINAL_CANNOT_COMPLETE_CODES) == {
        CommandCode.ALREADY_HANDED_TO_CARRIER,
        CommandCode.VARIANT_OUT_OF_STOCK,
        CommandCode.PRICE_OR_PRODUCT_MISMATCH,
    }
    assert set(CORRECT_AND_RETRY_CODES) == {
        CommandCode.ORDER_NOT_FOUND,
        CommandCode.ORDER_ITEM_NOT_FOUND,
        CommandCode.VARIANT_NOT_FOUND,
        CommandCode.ALREADY_CANCELLED,
        CommandCode.ALREADY_REQUESTED_VARIANT,
    }
    assert set(PROTOCOL_ONLY_CODES) == {CommandCode.REPLY_ALREADY_SENT}


def test_every_emitted_command_code_is_a_known_member() -> None:
    emitted = {code for codes in CODES_BY_COMMAND.values() for code in codes}
    assert emitted == set(CommandCode)
    assert set(CODES_BY_COMMAND) == {
        "lookup_orders",
        "get_order",
        "check_variant_availability",
        "cancel_order_item",
        "change_shipping_address",
        "replace_order_item_variant",
        "reply_to_customer",
    }


def test_envelope_codes_are_disjoint_from_command_codes() -> None:
    assert {str(code) for code in EnvelopeCode}.isdisjoint(str(code) for code in CommandCode)


def test_nine_cell_intent_mapping_is_complete() -> None:
    assert [intent.intent for intent in INTENTS] == [
        "cancel item",
        "change address",
        "replace variant",
    ]
    replacement = INTENTS[-1]
    assert replacement.cannot_complete == (
        CommandCode.ALREADY_HANDED_TO_CARRIER,
        CommandCode.VARIANT_OUT_OF_STOCK,
        CommandCode.PRICE_OR_PRODUCT_MISMATCH,
    )
    for intent in INTENTS:
        assert DISPOSITION_BY_CODE[str(intent.completed)] is Disposition.COMPLETED
        assert DISPOSITION_BY_CODE[str(intent.needs_information)] is Disposition.NEEDS_INFORMATION
        for code in intent.cannot_complete:
            assert DISPOSITION_BY_CODE[str(code)] is Disposition.CANNOT_COMPLETE


def test_reply_vocabulary_is_narrower_than_the_command_vocabulary() -> None:
    assert len(REPLY_OUTCOME_CODES) == 9
    assert set(REPLY_OUTCOME_CODES) == {
        *(str(code) for code in TERMINAL_COMPLETED_CODES),
        *(str(code) for code in ClarificationCode),
        *(str(code) for code in TERMINAL_CANNOT_COMPLETE_CODES),
    }
    assert set(REPLY_OUTCOME_CODES).isdisjoint(str(code) for code in CORRECT_AND_RETRY_CODES)
    assert set(REPLY_OUTCOME_CODES).isdisjoint(str(code) for code in PROTOCOL_ONLY_CODES)


def test_three_declared_clarification_facts() -> None:
    assert {str(fact) for fact in MissingFact} == {
        "order_item_id",
        "shipping_address.postal_code",
        "requested_options.size",
    }
    assert CLARIFICATION_FACTS == {
        "NEEDS_ORDER_ITEM": ("order_item_id",),
        "NEEDS_POSTAL_CODE": ("shipping_address.postal_code",),
        "NEEDS_VARIANT_OPTIONS": ("requested_options.size",),
    }


@pytest.mark.parametrize(
    ("outcome_code", "order_item_id"),
    [
        ("ITEM_CANCELLED", "item-1"),
        ("SHIPPING_ADDRESS_CHANGED", None),
        ("ITEM_VARIANT_REPLACED", "item-1"),
        ("NEEDS_ORDER_ITEM", None),
        ("NEEDS_POSTAL_CODE", None),
        ("NEEDS_VARIANT_OPTIONS", "item-1"),
        ("ALREADY_HANDED_TO_CARRIER", "item-1"),
        ("ALREADY_HANDED_TO_CARRIER", None),
        ("VARIANT_OUT_OF_STOCK", "item-1"),
        ("PRICE_OR_PRODUCT_MISMATCH", "item-1"),
    ],
)
def test_accepts_every_valid_combination(outcome_code: str, order_item_id: str | None) -> None:
    validate_reply(
        disposition=str(DISPOSITION_BY_CODE[outcome_code]),
        outcome_code=outcome_code,
        order_item_id=order_item_id,
        missing_fields=list(CLARIFICATION_FACTS.get(outcome_code, ())),
    )


@pytest.mark.parametrize("code", sorted(str(code) for code in CORRECT_AND_RETRY_CODES))
def test_retry_codes_are_rejected_with_retry_guidance(code: str) -> None:
    with pytest.raises(ReplyContractError, match="re-inspect the order and call again|retry"):
        validate_reply(
            disposition="cannot_complete",
            outcome_code=code,
            order_item_id=None,
            missing_fields=[],
        )


def test_protocol_code_is_never_a_customer_outcome() -> None:
    with pytest.raises(ReplyContractError, match="protocol result"):
        validate_reply(
            disposition="cannot_complete",
            outcome_code="REPLY_ALREADY_SENT",
            order_item_id=None,
            missing_fields=[],
        )


def test_rejects_a_mismatched_disposition() -> None:
    with pytest.raises(ReplyContractError, match="requires disposition completed"):
        validate_reply(
            disposition="cannot_complete",
            outcome_code="ITEM_CANCELLED",
            order_item_id="item-1",
            missing_fields=[],
        )


def test_rejects_an_unknown_disposition() -> None:
    with pytest.raises(ReplyContractError, match="is not one of"):
        validate_reply(
            disposition="escalated",
            outcome_code="ITEM_CANCELLED",
            order_item_id="item-1",
            missing_fields=[],
        )


def test_rejects_an_invented_outcome_code() -> None:
    with pytest.raises(ReplyContractError, match="is not one of"):
        validate_reply(
            disposition="needs_information",
            outcome_code="multiple_matching_items_specify_item",
            order_item_id=None,
            missing_fields=["order_item_id"],
        )


@pytest.mark.parametrize(
    ("outcome_code", "missing_fields"),
    [
        ("ITEM_CANCELLED", ["order_item_id"]),
        ("NEEDS_ORDER_ITEM", []),
        ("NEEDS_POSTAL_CODE", ["order_item_id"]),
        ("NEEDS_VARIANT_OPTIONS", ["shipping_address.postal_code", "requested_options.size"]),
        ("ALREADY_HANDED_TO_CARRIER", ["order_item_id"]),
    ],
)
def test_rejects_wrong_missing_fields(outcome_code: str, missing_fields: list[str]) -> None:
    with pytest.raises(ReplyContractError, match="requires missing_fields"):
        validate_reply(
            disposition=str(DISPOSITION_BY_CODE[outcome_code]),
            outcome_code=outcome_code,
            order_item_id="item-1" if outcome_code != "NEEDS_POSTAL_CODE" else None,
            missing_fields=missing_fields,
        )


@pytest.mark.parametrize("outcome_code", ["ITEM_CANCELLED", "NEEDS_VARIANT_OPTIONS"])
def test_requires_an_item_id_when_the_item_is_resolved(outcome_code: str) -> None:
    for empty in (None, "", "   "):
        with pytest.raises(ReplyContractError, match="order_item_id is required"):
            validate_reply(
                disposition=str(DISPOSITION_BY_CODE[outcome_code]),
                outcome_code=outcome_code,
                order_item_id=empty,
                missing_fields=list(CLARIFICATION_FACTS.get(outcome_code, ())),
            )


@pytest.mark.parametrize(
    "outcome_code", ["SHIPPING_ADDRESS_CHANGED", "NEEDS_POSTAL_CODE", "NEEDS_ORDER_ITEM"]
)
def test_forbids_an_item_id_for_address_and_unresolved_cancellation(outcome_code: str) -> None:
    with pytest.raises(ReplyContractError, match="must be null"):
        validate_reply(
            disposition=str(DISPOSITION_BY_CODE[outcome_code]),
            outcome_code=outcome_code,
            order_item_id="item-1",
            missing_fields=list(CLARIFICATION_FACTS.get(outcome_code, ())),
        )


def test_every_reply_code_has_an_item_id_rule() -> None:
    assert set(ITEM_ID_RULES) == set(REPLY_OUTCOME_CODES)


def test_classify_ignores_non_command_codes() -> None:
    assert classify("ITEM_CANCELLED") is CodeClass.TERMINAL_COMPLETED
    assert classify("NEEDS_ORDER_ITEM") is None
    assert classify("INVALID_REPLY_CONTRACT") is None


def test_reply_schema_is_closed_and_json_serializable() -> None:
    schema = json.loads(json.dumps(reply_tool_schema()))
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["missing_fields"]["uniqueItems"] is True
    for field in ("disposition", "outcome_code"):
        assert schema["properties"][field]["enum"]
    assert schema["properties"]["missing_fields"]["items"]["enum"]


def test_system_contract_is_deterministic_and_names_every_tool_rule() -> None:
    contract = render_system_contract()
    assert contract == render_system_contract()
    assert "never choose arbitrarily" in contract
    assert "exactly one structured customer reply" in contract
    assert "the reply tool rejects these codes and the case stays open." in contract
