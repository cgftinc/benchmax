"""Public `benchmax-sft-v1` supervised-finetuning dataset contract.

One row is a closed JSON object with required ``messages``, optional ``tools``,
and optional ``metadata``:

- ``system``/``user`` messages have exactly ``role`` and non-empty string
  ``content``.
- ``assistant`` messages have optional string-or-null ``content``, optional
  non-empty ``tool_calls``, and optional integer ``weight`` of ``0`` or ``1``
  (omitted means ``1``); each needs non-empty content or at least one tool
  call, and every row needs at least one assistant turn with effective
  weight ``1``.
- ``tool`` results have exactly ``role``, string ``content``, and non-empty
  ``tool_call_id``; every tool call receives exactly one result, in
  declaration order, before the next non-tool message.
- OpenAI-style function tool definitions/calls are validated deeply; called
  function names must match a definition.
- ``metadata`` is an open finite JSON object (``_castform_``-prefixed keys are
  reserved for the runtime).

Validation is strict and all-or-nothing: images/audio/multimodal content
parts, fractional weights, legacy prompt/completion shapes, duplicate JSON
keys, lone surrogates, non-finite numbers, and nesting deeper than 64 levels
are rejected with ordered, line-aware diagnostics. Canonical serialization is
deterministic byte-for-byte for equivalent inputs.
"""

from benchmax.sft.dataset import SFT_DATASET_FORMAT, SftDataset
from benchmax.sft.issues import SftDatasetError, SftIssue

__all__ = [
    "SFT_DATASET_FORMAT",
    "SftDataset",
    "SftDatasetError",
    "SftIssue",
]
