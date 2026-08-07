"""Frozen experiment identity for the order-resolution v2 benchmark.

This module owns experiment knowledge only: benchmark identity, the sealed v1
predecessor, generation seed, model arms, wave geometry, task selection rules,
decision gates, and the permitted data/artifact paths. Business semantics —
which outcome codes exist and how a reply is judged — belong to
``order_resolution.policy`` and ``order_resolution.command_codes``.

The maintainer freezing a new experiment edits this module. The maintainer
changing a fulfillment rule edits the policy owner instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

BENCHMARK_ID = "order-resolution-v2"
SPEC_SCHEMA_VERSION = 2
GENERATION_SEED = 20260806

# ---------------------------------------------------------------------------
# Sealed v1 predecessor
# ---------------------------------------------------------------------------

PREDECESSOR_BENCHMARK_ID = "order-resolution-v1"
PREDECESSOR_MANIFEST_PATH = "artifacts/baseline.json"
PREDECESSOR_DECISION = "repair"
PREDECESSOR_ROLLOUT_COUNT = 972

#: Byte hashes of every completed v1 artifact and dataset. These are immutable
#: evidence for the repair decision; no v2 command may rewrite them.
PREDECESSOR_SHA256 = MappingProxyType(
    {
        "artifacts/baseline.json": (
            "5664c88144a9119a08f33048ec54cbb215aaea8472779108ae67ecf325efed43"
        ),
        "artifacts/baseline.raw.jsonl": (
            "2e88051e950878dd0cdd11be0c4480d6bca8c2b54cfd7d198044dfb7b5112eb2"
        ),
        "artifacts/baseline.html": (
            "8eeaa5f6f8a8cc28c9c5c15ba08270f431efb2132a92cb8f537f71ff81b0221e"
        ),
        "data/train.jsonl": "9d61c2eace61b3e406cfcc06bfd39e8cc609b245d37d151d3651fbe065a8251d",
        "data/eval.jsonl": "75e40d0d2979b32789f47358b63f62ca438f21cdadd230a969e39b68edb48825",
        "data/oracle_traces.jsonl": (
            "7e64990d1ef51f2279204f16c1942a2730d89bfbff1971bfa466c80a01c8a8fe"
        ),
    }
)

# ---------------------------------------------------------------------------
# Permitted paths
# ---------------------------------------------------------------------------

DATA_ROOT = "data/v2"
ARTIFACT_ROOT = "artifacts/benchmark-v2"

TRAIN_DATA_PATH = f"{DATA_ROOT}/train.jsonl"
EVAL_DATA_PATH = f"{DATA_ROOT}/eval.jsonl"
EVAL_CHECKSUM_PATH = f"{DATA_ROOT}/eval.sha256"
ORACLE_DATA_PATH = f"{DATA_ROOT}/oracle_traces.jsonl"

SPEC_PATH = f"{ARTIFACT_ROOT}/spec.json"
HOSTED_VALIDATION_PATH = f"{ARTIFACT_ROOT}/hosted-validation.json"
CANARY_AUTHORIZATION_PATH = f"{ARTIFACT_ROOT}/canary/authorization.json"
DEMO_PATH = f"{ARTIFACT_ROOT}/demo.json"

#: Both canary attempt directories are frozen before the first model call so an
#: infrastructure retry can never reuse or overwrite a completed attempt.
CANARY_ATTEMPT_ROOTS = (
    f"{ARTIFACT_ROOT}/canary/attempt-01",
    f"{ARTIFACT_ROOT}/canary/attempt-02",
)
FULL_ROOT = f"{ARTIFACT_ROOT}/full"

WAVE_FILENAMES = ("manifest.json", "rollouts.raw.jsonl", "seal.json")
FULL_REPORT_FILENAME = "report.html"

ALLOWED_ARTIFACT_PATHS = (
    SPEC_PATH,
    HOSTED_VALIDATION_PATH,
    CANARY_AUTHORIZATION_PATH,
    *(f"{root}/{name}" for root in CANARY_ATTEMPT_ROOTS for name in WAVE_FILENAMES),
    *(f"{FULL_ROOT}/{name}" for name in WAVE_FILENAMES),
    f"{FULL_ROOT}/{FULL_REPORT_FILENAME}",
    DEMO_PATH,
)

ALLOWED_DATA_PATHS = (
    TRAIN_DATA_PATH,
    EVAL_DATA_PATH,
    EVAL_CHECKSUM_PATH,
    ORACLE_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Model arms
# ---------------------------------------------------------------------------

SMALL_MODEL = "qwen3.5-4b"
FRONTIER_GPT_MODEL = "gpt-5.6-sol"
FRONTIER_GROK_MODEL = "grok-4.3"

PRODUCTION_PROMPT = "production"
TWO_SHOT_PROMPT = "two_oracle_examples"


class ArmSpec(NamedTuple):
    """One model/prompt combination run across every wave."""

    id: str
    model: str
    prompt: str


ARMS = (
    ArmSpec("small_base", SMALL_MODEL, PRODUCTION_PROMPT),
    ArmSpec("small_two_shot", SMALL_MODEL, TWO_SHOT_PROMPT),
    ArmSpec("frontier_gpt", FRONTIER_GPT_MODEL, PRODUCTION_PROMPT),
    ArmSpec("frontier_grok", FRONTIER_GROK_MODEL, PRODUCTION_PROMPT),
)
FRONTIER_ARM_IDS = ("frontier_gpt", "frontier_grok")
BASE_ARM_ID = "small_base"
TWO_SHOT_ARM_ID = "small_two_shot"
REQUIRED_MODELS = (SMALL_MODEL, FRONTIER_GPT_MODEL, FRONTIER_GROK_MODEL)

# ---------------------------------------------------------------------------
# Task grid and selections
# ---------------------------------------------------------------------------

ACTION_FAMILIES = ("cancel_item", "change_address", "replace_variant")
OUTCOME_CLASSES = ("execute", "clarify", "deny")
CELLS = tuple(f"{family}-{outcome}" for family in ACTION_FAMILIES for outcome in OUTCOME_CLASSES)

TRAIN_ROWS_PER_CELL = 20
EVAL_ROWS_PER_CELL = 10
TRAIN_ROW_COUNT = TRAIN_ROWS_PER_CELL * len(CELLS)
EVAL_ROW_COUNT = EVAL_ROWS_PER_CELL * len(CELLS)

#: Training-cell indices reserved for the signal probe, the canary, and the two
#: frozen demonstrations. The three sets are disjoint so no wave reuses a row.
SIGNAL_PROBE_INDICES = (0, 1, 2, 3)
CANARY_INDICES = (18, 19)
ORACLE_DEMO_INDEX = 0
ORACLE_DEMO_CELLS = ("cancel_item-clarify", "replace_variant-deny")

#: Held-out cell indices replayed in the report and the frozen demo artifact.
REPORT_DEMO_CELLS = (
    "cancel_item-execute",
    "cancel_item-clarify",
    "change_address-execute",
    "change_address-deny",
    "replace_variant-execute",
    "replace_variant-clarify",
)
REPORT_DEMO_INDEX = 0
STRESS_INDICES = (0, 1, 2)

# ---------------------------------------------------------------------------
# Catalog namespace
# ---------------------------------------------------------------------------

#: Bumping this invalidates the content-addressed namespace, so a changed
#: generator can never reuse or partially overwrite existing catalog rows.
CATALOG_GENERATOR_VERSION = 2
CATALOG_PRODUCTS = 250
CATALOG_ADJECTIVES = (
    "Alpine",
    "Amber",
    "Aurora",
    "Basalt",
    "Cedar",
    "Coastal",
    "Cobalt",
    "Copper",
    "Driftwood",
    "Ember",
    "Fjord",
    "Garnet",
    "Harbor",
    "Indigo",
    "Juniper",
    "Lantern",
    "Meadow",
    "Nordic",
    "Onyx",
    "Prairie",
    "Quartz",
    "Riverstone",
    "Saffron",
    "Tundra",
    "Willow",
)
CATALOG_NOUNS = (
    "Jacket",
    "Backpack",
    "Kettle",
    "Lamp",
    "Notebook",
    "Blanket",
    "Mug",
    "Chair",
    "Rug",
    "Bottle",
)
CATALOG_CATEGORIES = ("apparel", "home", "outdoors", "office")
CATALOG_SIZES = ("small", "medium", "large")

#: Disjoint product ranges per split. No visible entity crosses the boundary.
TRAIN_PRODUCT_RANGE = (0, 160)
EVAL_PRODUCT_RANGE = (160, CATALOG_PRODUCTS)

#: Offset from the target product to its visibly different distractor. 13 shifts
#: both the adjective and the noun, so the two order lines never share a word.
DISTRACTOR_OFFSET = 13

# ---------------------------------------------------------------------------
# Prompt template banks
# ---------------------------------------------------------------------------

#: Style/length strata shared by both splits so paraphrase difficulty matches.
PROMPT_STRATA = ("direct_short", "polite_short", "polite_long", "narrative_long", "terse_short")

#: The six request shapes. A clarify shape omits exactly the one fact its
#: expected reply names; nothing else differs from its execute/deny sibling.
REQUEST_SHAPES = (
    "cancel_item.identified",
    "cancel_item.ambiguous",
    "change_address.full",
    "change_address.no_postal",
    "replace_variant.sized",
    "replace_variant.no_size",
)

_TRAIN_TEMPLATES = {
    "cancel_item.identified": (
        "Please cancel the {product} on order {order_number}.",
        "Could you cancel my {product} from order {order_number}? Thanks.",
        "I ordered a few things and have changed my mind about one of them. Please cancel the"
        " {product} on order {order_number} and refund that line.",
        "I placed order {order_number} last week and my partner already bought the same thing,"
        " so the {product} on it is no longer needed.",
        "cancel {product}, order {order_number}",
    ),
    "cancel_item.ambiguous": (
        "Please cancel the item on order {order_number}.",
        "Could you cancel the item I ordered on {order_number}? Thanks.",
        "I would like to cancel the item on order {order_number}; please process the refund"
        " once that is done.",
        "I was looking at order {order_number} again this morning and decided the item should"
        " go, so please cancel it.",
        "cancel the item on {order_number}",
    ),
    "change_address.full": (
        "Please ship order {order_number} to {line1}, {city}, {region} {postal_code},"
        " {country}.",
        "Could you send order {order_number} to {line1}, {city}, {region} {postal_code},"
        " {country}? Thanks.",
        "I have moved since placing order {order_number}. The new delivery address is {line1},"
        " {city}, {region} {postal_code}, {country}; please update it.",
        "Order {order_number} is heading to my old house because I typed the wrong details."
        " Use {line1}, {city}, {region} {postal_code}, {country} instead.",
        "ship {order_number} to {line1}, {city}, {region} {postal_code}, {country}",
    ),
    "change_address.no_postal": (
        "Please ship order {order_number} to {line1}, {city}, {region}, {country}.",
        "Could you send order {order_number} to {line1}, {city}, {region}, {country}? Thanks.",
        "I have moved since placing order {order_number}. The new delivery address is {line1},"
        " {city}, {region}, {country}; please update it.",
        "Order {order_number} is heading to my old house because I typed the wrong details."
        " Use {line1}, {city}, {region}, {country} instead.",
        "ship {order_number} to {line1}, {city}, {region}, {country}",
    ),
    "replace_variant.sized": (
        "Please swap the {product} on order {order_number} for the {size} one.",
        "Could you change the {product} on order {order_number} to {size}? Thanks.",
        "The {product} I ordered on {order_number} will not fit. Please exchange it for the"
        " {size} version at the same price.",
        "I measured again after placing order {order_number} and the {product} on it needs to"
        " be the {size} one.",
        "swap {product} on {order_number} to {size}",
    ),
    "replace_variant.no_size": (
        "Please swap the {product} on order {order_number} for a different size.",
        "Could you change the size of the {product} on order {order_number}? Thanks.",
        "The {product} I ordered on {order_number} will not fit. Please exchange it for another"
        " size at the same price.",
        "I measured again after placing order {order_number} and the {product} on it needs to"
        " be a different size.",
        "swap {product} on {order_number} to another size",
    ),
}

_EVAL_TEMPLATES = {
    "cancel_item.identified": (
        "Remove the {product} from order {order_number}.",
        "Would you mind taking the {product} off order {order_number}?",
        "Hello, I need to make one small change: the {product} on order {order_number} should"
        " be dropped and the money returned.",
        "My budget is tighter than expected this month, so on order {order_number} I want to"
        " drop the {product} while keeping the rest.",
        "drop {product} from {order_number}",
    ),
    "cancel_item.ambiguous": (
        "Remove the item from order {order_number}.",
        "Would you mind taking the item off order {order_number}?",
        "Hello, about order {order_number}: the item is not needed any more and should be"
        " dropped from it.",
        "My budget is tighter than expected this month, so on order {order_number} I want to"
        " drop the item.",
        "drop the item from {order_number}",
    ),
    "change_address.full": (
        "Change delivery for order {order_number} to {line1}, {city}, {region} {postal_code},"
        " {country}.",
        "Would you mind redirecting order {order_number} to {line1}, {city}, {region}"
        " {postal_code}, {country}?",
        "Hello, my delivery details have changed. Order {order_number} should now go to"
        " {line1}, {city}, {region} {postal_code}, {country}.",
        "I will be staying with family when order {order_number} arrives, so deliver it to"
        " {line1}, {city}, {region} {postal_code}, {country} rather than home.",
        "redirect {order_number}: {line1}, {city}, {region} {postal_code}, {country}",
    ),
    "change_address.no_postal": (
        "Change delivery for order {order_number} to {line1}, {city}, {region}, {country}.",
        "Would you mind redirecting order {order_number} to {line1}, {city}, {region},"
        " {country}?",
        "Hello, my delivery details have changed. Order {order_number} should now go to"
        " {line1}, {city}, {region}, {country}.",
        "I will be staying with family when order {order_number} arrives, so deliver it to"
        " {line1}, {city}, {region}, {country} rather than home.",
        "redirect {order_number}: {line1}, {city}, {region}, {country}",
    ),
    "replace_variant.sized": (
        "Exchange the {product} on order {order_number} for the {size} option.",
        "Would you mind switching the {product} on order {order_number} over to {size}?",
        "Hello, the {product} from order {order_number} is the wrong fit; I would like the"
        " {size} option instead, at the same price.",
        "When order {order_number} was placed I guessed the fit, and now I know the {product}"
        " should be {size}.",
        "switch {product}, {order_number}, {size}",
    ),
    "replace_variant.no_size": (
        "Exchange the {product} on order {order_number} for another option.",
        "Would you mind switching the {product} on order {order_number} to a different size?",
        "Hello, the {product} from order {order_number} is the wrong fit; I would like a"
        " different size instead, at the same price.",
        "When order {order_number} was placed I guessed the fit, and now I know the {product}"
        " is the wrong size.",
        "switch {product}, {order_number}, different size",
    ),
}

#: Split-partitioned paraphrase banks. Families are partitioned before row
#: expansion, so no eval row can share a wording family with a training row.
PROMPT_TEMPLATES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "train": MappingProxyType({shape: _TRAIN_TEMPLATES[shape] for shape in REQUEST_SHAPES}),
        "eval": MappingProxyType({shape: _EVAL_TEMPLATES[shape] for shape in REQUEST_SHAPES}),
    }
)


def oracle_demo_task_ids() -> tuple[str, ...]:
    """The two training rows frozen as demonstrations before any model call."""

    return tuple(f"train-{cell}-{ORACLE_DEMO_INDEX:02d}" for cell in ORACLE_DEMO_CELLS)


def request_shape(action_family: str, outcome_class: str) -> str:
    """Map a task cell onto the customer-request shape it uses."""

    clarify = outcome_class == "clarify"
    if action_family == "cancel_item":
        return "cancel_item.ambiguous" if clarify else "cancel_item.identified"
    if action_family == "change_address":
        return "change_address.no_postal" if clarify else "change_address.full"
    if action_family == "replace_variant":
        return "replace_variant.no_size" if clarify else "replace_variant.sized"
    raise SpecError(f"unknown action family {action_family!r}")


def prompt_template(split: str, action_family: str, outcome_class: str, index: int) -> tuple[
    str, str, str
]:
    """Return the ``(template, shape, stratum)`` for one row of a cell."""

    shape = request_shape(action_family, outcome_class)
    try:
        bank = PROMPT_TEMPLATES[split][shape]
    except KeyError as error:
        raise SpecError(f"no {split!r} template bank for {shape!r}") from error
    position = index % len(bank)
    return bank[position], shape, PROMPT_STRATA[position]


# ---------------------------------------------------------------------------
# Wave geometry
# ---------------------------------------------------------------------------

CANARY_CONCURRENCY = 6
CANARY_GROUP_SIZE = 1
CANARY_TASKS_PER_ARM = len(CELLS) * len(CANARY_INDICES)
CANARY_ROLLOUTS = CANARY_TASKS_PER_ARM * len(ARMS)
MAX_CANARY_ATTEMPTS = len(CANARY_ATTEMPT_ROOTS)

FULL_CONCURRENCY = 16
FULL_GROUP_SIZE = 1
STRESS_REPEATS = 3
TRAINING_GROUP_SIZE = 8

FULL_ROLLOUTS = EVAL_ROW_COUNT * len(ARMS)
STRESS_ROLLOUTS = len(CELLS) * len(STRESS_INDICES) * len(ARMS) * STRESS_REPEATS
SIGNAL_PROBE_ROLLOUTS = len(CELLS) * len(SIGNAL_PROBE_INDICES) * TRAINING_GROUP_SIZE
TOTAL_FULL_WAVE_ROLLOUTS = FULL_ROLLOUTS + STRESS_ROLLOUTS + SIGNAL_PROBE_ROLLOUTS

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

#: Predeclared canary gates. A failure here is a binding product result; only a
#: sealed ``infrastructure_failure`` permits the one frozen second attempt.
CANARY_GATES = MappingProxyType(
    {
        "expected_records": CANARY_ROLLOUTS,
        "max_infrastructure_failures": 0,
        "max_non_scorable": 0,
        "max_invariant_failures": 0,
        "frontier_min_successes": 13,
        "frontier_min_successes_per_cell": 1,
        "base_min_successes": 3,
        "base_max_successes": 14,
        "min_frontier_base_gap": 2,
        "two_shot_min_successes_per_outcome_class": 1,
    }
)

#: The original v1 decision thresholds, deliberately unchanged for v2.
FULL_GATES = MappingProxyType(
    {
        "max_infrastructure_failure_rate": 0.02,
        "min_frontier_success_rate": 0.70,
        "min_base_success_rate": 0.15,
        "max_base_success_rate": 0.80,
        "min_frontier_base_gap": 0.10,
        "min_model_attributable_base_failures": 10,
        "min_mixed_signal_groups": 9,
        "min_sibling_success_rate": 0.10,
        "max_sibling_success_rate": 0.90,
    }
)

CANARY_STATUSES = ("proceed", "repair_again", "harden", "no_headroom", "infrastructure_failure")
FULL_DECISIONS = ("go", "harden", "repair_again", "no_headroom")

USAGE_ACCOUNTING = MappingProxyType(
    {
        "mode": "omitted",
        "reason": "user-approved waiver; the proxy does not expose exact provider usage",
    }
)


class SpecError(RuntimeError):
    """The requested benchmark identity or frozen selection is not valid."""


class PredecessorSealError(RuntimeError):
    """The sealed v1 predecessor no longer matches its recorded bytes or decision."""


def assert_benchmark_id(benchmark_id: str) -> str:
    """Reject any identity other than the one this spec module owns."""

    if benchmark_id != BENCHMARK_ID:
        raise SpecError(f"unsupported benchmark id {benchmark_id!r}; expected {BENCHMARK_ID!r}")
    return benchmark_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_predecessor_bytes(example_root: Path) -> dict[str, str]:
    """Recompute every sealed v1 hash and fail on the first drift."""

    missing: list[str] = []
    mismatched: list[str] = []
    verified: dict[str, str] = {}
    for relative, expected in sorted(PREDECESSOR_SHA256.items()):
        path = example_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatched.append(relative)
            continue
        verified[relative] = actual
    if missing:
        raise PredecessorSealError(f"sealed v1 files are missing: {', '.join(sorted(missing))}")
    if mismatched:
        raise PredecessorSealError(
            f"sealed v1 files no longer match their recorded bytes: {', '.join(sorted(mismatched))}"
        )
    return verified


def verify_predecessor_decision(manifest_path: Path) -> dict[str, Any]:
    """Confirm the v1 manifest still records the binding ``repair`` outcome."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredecessorSealError(f"cannot read v1 manifest {manifest_path.name}") from error
    if not isinstance(manifest, dict):
        raise PredecessorSealError("v1 manifest must contain a JSON object")

    # Schema v1 predates versioned identities and is implicitly the predecessor.
    recorded_id = manifest.get("benchmark_id")
    if recorded_id is None:
        if manifest.get("schema_version") != 1:
            raise PredecessorSealError("v1 manifest has no benchmark id and is not schema v1")
        recorded_id = PREDECESSOR_BENCHMARK_ID
    if recorded_id != PREDECESSOR_BENCHMARK_ID:
        raise PredecessorSealError(
            f"v1 manifest records benchmark {recorded_id!r}, expected {PREDECESSOR_BENCHMARK_ID!r}"
        )
    if manifest.get("status") != "complete":
        raise PredecessorSealError("v1 manifest is not a completed result")
    if manifest.get("rollout_count") != PREDECESSOR_ROLLOUT_COUNT:
        raise PredecessorSealError(
            f"v1 manifest records {manifest.get('rollout_count')!r} rollouts, "
            f"expected {PREDECESSOR_ROLLOUT_COUNT}"
        )
    decision = manifest.get("report", {}).get("decision", {}).get("status")
    if decision != PREDECESSOR_DECISION:
        raise PredecessorSealError(
            f"v1 decision is {decision!r}, expected the binding {PREDECESSOR_DECISION!r}"
        )
    return {
        "benchmark_id": recorded_id,
        "decision": decision,
        "rollout_count": manifest["rollout_count"],
    }


def verify_predecessor(example_root: Path, *, benchmark_id: str = BENCHMARK_ID) -> dict[str, Any]:
    """Verify the sealed v1 result before any v2 state is read or written."""

    assert_benchmark_id(benchmark_id)
    verified = verify_predecessor_bytes(example_root)
    decision = verify_predecessor_decision(example_root / PREDECESSOR_MANIFEST_PATH)
    return {
        "benchmark_id": benchmark_id,
        "predecessor": decision,
        "files_verified": len(verified),
    }


def spec_identity() -> dict[str, Any]:
    """The frozen experiment fields recorded in ``spec.json``."""

    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "generation_seed": GENERATION_SEED,
        "predecessor": {
            "benchmark_id": PREDECESSOR_BENCHMARK_ID,
            "decision": PREDECESSOR_DECISION,
            "rollout_count": PREDECESSOR_ROLLOUT_COUNT,
            "sha256": dict(PREDECESSOR_SHA256),
        },
        "arms": [arm._asdict() for arm in ARMS],
        "geometry": {
            "canary": {
                "tasks_per_arm": CANARY_TASKS_PER_ARM,
                "rollouts": CANARY_ROLLOUTS,
                "group_size": CANARY_GROUP_SIZE,
                "concurrency": CANARY_CONCURRENCY,
                "indices": list(CANARY_INDICES),
                "attempt_roots": list(CANARY_ATTEMPT_ROOTS),
            },
            "full": {
                "rollouts": FULL_ROLLOUTS,
                "stress_rollouts": STRESS_ROLLOUTS,
                "signal_probe_rollouts": SIGNAL_PROBE_ROLLOUTS,
                "total_rollouts": TOTAL_FULL_WAVE_ROLLOUTS,
                "group_size": FULL_GROUP_SIZE,
                "stress_repeats": STRESS_REPEATS,
                "training_group_size": TRAINING_GROUP_SIZE,
                "concurrency": FULL_CONCURRENCY,
            },
        },
        "gates": {"canary": dict(CANARY_GATES), "full": dict(FULL_GATES)},
        "paths": {
            "data": list(ALLOWED_DATA_PATHS),
            "artifacts": list(ALLOWED_ARTIFACT_PATHS),
        },
        "usage_accounting": dict(USAGE_ACCOUNTING),
    }


__all__ = [
    "ALLOWED_ARTIFACT_PATHS",
    "ALLOWED_DATA_PATHS",
    "ARMS",
    "BENCHMARK_ID",
    "CANARY_ATTEMPT_ROOTS",
    "CANARY_GATES",
    "CANARY_INDICES",
    "CANARY_ROLLOUTS",
    "FULL_GATES",
    "GENERATION_SEED",
    "PREDECESSOR_DECISION",
    "PREDECESSOR_SHA256",
    "PredecessorSealError",
    "SPEC_SCHEMA_VERSION",
    "SpecError",
    "TOTAL_FULL_WAVE_ROLLOUTS",
    "assert_benchmark_id",
    "spec_identity",
    "verify_predecessor",
    "verify_predecessor_bytes",
    "verify_predecessor_decision",
]
