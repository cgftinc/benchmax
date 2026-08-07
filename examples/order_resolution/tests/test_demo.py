from __future__ import annotations

import pytest
from order_resolution.demo import assert_redacted_artifact, demo_arm, redact_value


def test_v1_demo_arm_stays_the_two_shot_arm() -> None:
    assert demo_arm({"schema_version": 1}) == "small_two_shot"


def test_v2_demo_arm_is_the_best_full_arm() -> None:
    manifest = {
        "schema_version": 2,
        "report": {
            "arms": {
                "small_base": {"success_rate": 0.2},
                "small_two_shot": {"success_rate": 0.4},
                "frontier_gpt": {"success_rate": 0.8},
                "frontier_grok": {"success_rate": 0.7},
            }
        },
    }
    assert demo_arm(manifest) == "frontier_gpt"


def test_v2_demo_arm_breaks_ties_deterministically() -> None:
    manifest = {
        "schema_version": 2,
        "report": {
            "arms": {
                "frontier_grok": {"success_rate": 0.8},
                "frontier_gpt": {"success_rate": 0.8},
            }
        },
    }
    assert demo_arm(manifest) == demo_arm(manifest) == "frontier_gpt"


def test_redaction_removes_pii_and_pseudonymizes_identifiers() -> None:
    redacted = redact_value(
        {
            "email": "customer@example.test",
            "line1": "10 Main Street",
            "order_number": "OR-E00001",
            "nested": {"item-eval-001-a": {"customer_id": "customer-1"}},
            "path": "orders.OR-E00001.status",
        }
    )

    assert redacted["email"] == "[redacted]"
    assert redacted["line1"] == "[redacted]"
    assert redacted["order_number"].startswith("ref-")
    assert list(redacted["nested"])[0].startswith("ref-")
    assert "OR-E00001" not in redacted["path"]


def test_final_artifact_scan_rejects_email_and_database_credentials() -> None:
    assert_redacted_artifact('{"safe":"value"}')
    with pytest.raises(RuntimeError, match="email-like"):
        assert_redacted_artifact('{"email":"customer@example.test"}')
    with pytest.raises(RuntimeError, match="credential-like"):
        assert_redacted_artifact('{"url":"postgresql://user:password@db.example/test"}')
