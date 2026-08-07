"""The complete typed vocabulary emitted by order-resolution commands.

``domain.py`` returns these members instead of independently owned strings, so
adding a business outcome is a single edit here. ``policy.py`` imports this
module and classifies every :class:`CommandCode` member; that classification
test fails whenever a new code is added without deciding how a model may
report it.
"""

from __future__ import annotations

from enum import StrEnum


class CommandCode(StrEnum):
    """Every ``code`` an atomic business command can return."""

    # Successful mutations.
    ITEM_CANCELLED = "ITEM_CANCELLED"
    SHIPPING_ADDRESS_CHANGED = "SHIPPING_ADDRESS_CHANGED"
    ITEM_VARIANT_REPLACED = "ITEM_VARIANT_REPLACED"

    # Policy boundaries: the request is understood and refused.
    ALREADY_HANDED_TO_CARRIER = "ALREADY_HANDED_TO_CARRIER"
    VARIANT_OUT_OF_STOCK = "VARIANT_OUT_OF_STOCK"
    PRICE_OR_PRODUCT_MISMATCH = "PRICE_OR_PRODUCT_MISMATCH"

    # Stale or incorrect arguments: re-inspect and retry within budget.
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    ORDER_ITEM_NOT_FOUND = "ORDER_ITEM_NOT_FOUND"
    VARIANT_NOT_FOUND = "VARIANT_NOT_FOUND"
    ALREADY_CANCELLED = "ALREADY_CANCELLED"
    ALREADY_REQUESTED_VARIANT = "ALREADY_REQUESTED_VARIANT"

    # Protocol result: the episode already produced its one terminal reply.
    REPLY_ALREADY_SENT = "REPLY_ALREADY_SENT"


class EnvelopeCode(StrEnum):
    """Environment-level protocol errors that never reach a business command."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    EPISODE_TERMINAL = "EPISODE_TERMINAL"
    INVALID_REPLY_CONTRACT = "INVALID_REPLY_CONTRACT"


#: Codes each command may emit. Used to prove the enum stays exhaustive as the
#: command surface changes.
CODES_BY_COMMAND = {
    "lookup_orders": frozenset(),
    "get_order": frozenset({CommandCode.ORDER_NOT_FOUND}),
    "check_variant_availability": frozenset({CommandCode.ORDER_ITEM_NOT_FOUND}),
    "cancel_order_item": frozenset(
        {
            CommandCode.ITEM_CANCELLED,
            CommandCode.ORDER_ITEM_NOT_FOUND,
            CommandCode.ALREADY_CANCELLED,
            CommandCode.ALREADY_HANDED_TO_CARRIER,
        }
    ),
    "change_shipping_address": frozenset(
        {
            CommandCode.SHIPPING_ADDRESS_CHANGED,
            CommandCode.ORDER_NOT_FOUND,
            CommandCode.ALREADY_HANDED_TO_CARRIER,
        }
    ),
    "replace_order_item_variant": frozenset(
        {
            CommandCode.ITEM_VARIANT_REPLACED,
            CommandCode.ORDER_ITEM_NOT_FOUND,
            CommandCode.ALREADY_REQUESTED_VARIANT,
            CommandCode.ALREADY_HANDED_TO_CARRIER,
            CommandCode.VARIANT_NOT_FOUND,
            CommandCode.VARIANT_OUT_OF_STOCK,
            CommandCode.PRICE_OR_PRODUCT_MISMATCH,
        }
    ),
    "reply_to_customer": frozenset({CommandCode.REPLY_ALREADY_SENT}),
}


__all__ = ["CODES_BY_COMMAND", "CommandCode", "EnvelopeCode"]
