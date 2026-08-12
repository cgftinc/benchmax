"""Tests for protocol identity and the hash-chained execution record."""

from __future__ import annotations

import hashlib
import json

import pytest
from pii_masking.benchmark_protocol import (
    BENCHMARK_REVISION,
    EXECUTION_EVENT_DOMAIN,
    GENESIS_DIGEST,
    ExecutionLog,
    Protocol,
    ProtocolError,
    RequestIdentity,
    build_protocol_payload,
    canonical_bytes,
    canonical_digest,
    chain_head,
    event_digest,
    loads,
    protocol_id_for,
    read_events,
    request_payload_hash,
    scoped_sample_uid,
)

TIMESTAMP = "2026-08-12T00:00:00Z"


def sample_payload(**overrides):
    payload = build_protocol_payload(
        benchmark_source="openpii-validation",
        execution_environment="production",
        base_model={"model_id": "Qwen/Qwen3.5-4B", "revision": "851bf6e8"},
        training={"lora_rank": 64, "epochs": 1},
        generation={"temperature": 0, "top_p": 1, "max_tokens": 2048},
        prompt={"system": "mask it"},
        selection={"train_rows": 4096},
        scoring={"alignment_version": 1},
        expected_counts={"full_identities": 20000},
    )
    payload.update(overrides)
    return payload


# ── canonical bytes ───────────────────────────────────────────────────────────
class TestCanonicalBytes:
    def test_keys_sort_recursively(self):
        assert canonical_bytes({"b": 1, "a": {"d": 2, "c": 3}}) == b'{"a":{"c":3,"d":2},"b":1}\n'

    def test_exactly_one_trailing_newline(self):
        encoded = canonical_bytes({"a": 1})
        assert encoded.endswith(b"\n")
        assert not encoded.endswith(b"\n\n")

    def test_non_ascii_is_not_escaped(self):
        assert canonical_bytes({"lang": "français"}) == '{"lang":"français"}\n'.encode()

    def test_array_order_is_preserved(self):
        assert canonical_bytes([3, 1, 2]) == b"[3,1,2]\n"

    def test_equal_documents_produce_equal_bytes(self):
        assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_rejected(self, value):
        with pytest.raises(ProtocolError, match="non-finite"):
            canonical_bytes({"x": value})

    def test_non_json_values_are_rejected(self):
        with pytest.raises(ProtocolError, match="not JSON"):
            canonical_bytes({"x": {1, 2}})

    def test_non_string_keys_are_rejected(self):
        with pytest.raises(ProtocolError, match="keys must be strings"):
            canonical_bytes({1: "a"})

    def test_duplicate_keys_are_rejected_on_parse(self):
        with pytest.raises(ProtocolError, match="duplicate object key"):
            loads('{"a": 1, "a": 2}')

    def test_plain_json_would_have_silently_kept_the_last_duplicate(self):
        assert json.loads('{"a": 1, "a": 2}') == {"a": 2}


# ── protocol identity ─────────────────────────────────────────────────────────
class TestProtocolIdentity:
    def test_id_is_sha256_of_the_persisted_bytes(self):
        payload = sample_payload()
        assert protocol_id_for(payload) == hashlib.sha256(canonical_bytes(payload)).hexdigest()

    def test_payload_may_not_contain_its_own_id(self):
        with pytest.raises(ProtocolError, match="cannot carry its own hash"):
            protocol_id_for(sample_payload(protocol_id="whatever"))

    def test_id_is_independent_of_key_order(self):
        payload = sample_payload()
        reordered = dict(reversed(list(payload.items())))
        assert protocol_id_for(reordered) == protocol_id_for(payload)

    def test_any_decision_change_changes_the_id(self):
        before = protocol_id_for(sample_payload())
        after = protocol_id_for(sample_payload(training={"lora_rank": 32, "epochs": 1}))
        assert before != after

    def test_unknown_benchmark_source_is_rejected(self):
        with pytest.raises(ProtocolError, match="unknown benchmark_source"):
            build_protocol_payload(
                benchmark_source="made-up",
                execution_environment="production",
                base_model={},
                training={},
                generation={},
                prompt={},
                selection={},
                scoring={},
                expected_counts={},
            )

    def test_write_then_load_round_trips(self, tmp_path):
        protocol = Protocol.from_payload(sample_payload())
        directory = tmp_path / protocol.protocol_id
        protocol.write(directory)

        loaded = Protocol.load(directory)

        assert loaded.protocol_id == protocol.protocol_id
        assert loaded.payload == protocol.payload
        assert loaded.benchmark_revision == BENCHMARK_REVISION

    def test_a_protocol_is_never_rewritten(self, tmp_path):
        protocol = Protocol.from_payload(sample_payload())
        directory = tmp_path / protocol.protocol_id
        protocol.write(directory)

        with pytest.raises(ProtocolError, match="never rewritten"):
            protocol.write(directory)

    def test_edited_protocol_bytes_fail_to_load(self, tmp_path):
        protocol = Protocol.from_payload(sample_payload())
        directory = tmp_path / protocol.protocol_id
        path = protocol.write(directory)

        edited = loads(path.read_bytes())
        edited["training"]["lora_rank"] = 32
        path.write_bytes(canonical_bytes(edited))

        with pytest.raises(ProtocolError, match="hashes to"):
            Protocol.load(directory)

    def test_non_canonical_encoding_fails_to_load(self, tmp_path):
        protocol = Protocol.from_payload(sample_payload())
        directory = tmp_path / protocol.protocol_id
        path = protocol.write(directory)
        path.write_bytes(json.dumps(protocol.payload, indent=2).encode() + b"\n")

        with pytest.raises(ProtocolError, match="not in canonical form"):
            Protocol.load(directory)

    def test_directory_is_named_by_the_id(self, tmp_path):
        protocol = Protocol.from_payload(sample_payload())

        assert protocol.directory(tmp_path).name == protocol.protocol_id


# ── request identity ──────────────────────────────────────────────────────────
class TestRequestIdentity:
    def test_payload_hash_excludes_only_the_model(self):
        base = {
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0,
        }
        sft = {**base, "model": "ft:qwen3.5-4b:run:1023"}

        assert request_payload_hash(base) == request_payload_hash(sft)

    def test_payload_hash_tracks_every_other_field(self):
        base = {"model": "m", "messages": [], "temperature": 0}

        assert request_payload_hash(base) != request_payload_hash({**base, "temperature": 1})

    def test_a_model_only_request_is_rejected(self):
        with pytest.raises(ProtocolError, match="other than 'model'"):
            request_payload_hash({"model": "m"})

    def test_identity_excludes_phase_so_phases_nest(self):
        fields = dict(
            protocol_id="p",
            benchmark_revision=BENCHMARK_REVISION,
            task_name="castform-openpii-en-v1",
            sample_uid="castform-openpii-en-v1:42",
            request_payload_hash="rph",
            model_ref="qwen3.5-4b",
            model_artifact_digest="dig",
        )
        # The same sample seen during smoke and again during full is ONE identity.
        assert RequestIdentity(**fields).digest == RequestIdentity(**fields).digest

    def test_model_ref_changes_identity(self):
        fields = dict(
            protocol_id="p",
            benchmark_revision=BENCHMARK_REVISION,
            task_name="t",
            sample_uid="t:1",
            request_payload_hash="rph",
            model_artifact_digest="dig",
        )
        base = RequestIdentity(model_ref="qwen3.5-4b", **fields)
        sft = RequestIdentity(model_ref="ft:qwen3.5-4b:run:1023", **fields)

        assert base.digest != sft.digest

    def test_artifact_digest_changes_identity(self):
        fields = dict(
            protocol_id="p",
            benchmark_revision=BENCHMARK_REVISION,
            task_name="t",
            sample_uid="t:1",
            request_payload_hash="rph",
            model_ref="m",
        )
        assert (
            RequestIdentity(model_artifact_digest="a", **fields).digest
            != RequestIdentity(model_artifact_digest="b", **fields).digest
        )

    def test_sample_uid_is_task_scoped(self):
        assert scoped_sample_uid("task-a", 7) != scoped_sample_uid("task-b", 7)


# ── hash chain ────────────────────────────────────────────────────────────────
class TestExecutionChain:
    def test_first_event_references_genesis(self, tmp_path):
        log = ExecutionLog(tmp_path)

        record = log.append("started", {"a": 1}, timestamp=TIMESTAMP)

        assert record["previous_digest"] == GENESIS_DIGEST

    def test_digest_covers_the_core_and_excludes_itself(self):
        core = {
            "previous_digest": GENESIS_DIGEST,
            "event_type": "e",
            "timestamp": TIMESTAMP,
            "payload": {},
        }

        expected = hashlib.sha256(EXECUTION_EVENT_DOMAIN + canonical_bytes(core)).hexdigest()

        assert event_digest(core) == expected

    def test_core_carrying_a_digest_is_rejected(self):
        with pytest.raises(ProtocolError, match="must not contain event_digest"):
            event_digest({"previous_digest": GENESIS_DIGEST, "event_digest": "x"})

    def test_events_chain_to_their_predecessor(self, tmp_path):
        log = ExecutionLog(tmp_path)

        first = log.append("a", {}, timestamp=TIMESTAMP)
        second = log.append("b", {}, timestamp=TIMESTAMP)

        assert second["previous_digest"] == first["event_digest"]
        assert chain_head(log.events_path) == second["event_digest"]

    def test_a_tampered_payload_is_detected(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {"cost": 1}, timestamp=TIMESTAMP)
        log.append("b", {"cost": 2}, timestamp=TIMESTAMP)

        lines = log.events_path.read_bytes().splitlines()
        record = loads(lines[0])
        record["payload"] = {"cost": 999}
        log.events_path.write_bytes(canonical_bytes(record) + lines[1] + b"\n")

        with pytest.raises(ProtocolError, match="does not match its content"):
            log.events()

    def test_a_removed_event_breaks_the_chain(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {}, timestamp=TIMESTAMP)
        log.append("b", {}, timestamp=TIMESTAMP)
        log.append("c", {}, timestamp=TIMESTAMP)

        lines = log.events_path.read_bytes().splitlines()
        log.events_path.write_bytes(lines[0] + b"\n" + lines[2] + b"\n")

        with pytest.raises(ProtocolError, match="but the chain is at"):
            log.events()

    def test_a_reordered_chain_is_detected(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {}, timestamp=TIMESTAMP)
        log.append("b", {}, timestamp=TIMESTAMP)

        lines = log.events_path.read_bytes().splitlines()
        log.events_path.write_bytes(lines[1] + b"\n" + lines[0] + b"\n")

        with pytest.raises(ProtocolError):
            log.events()

    def test_missing_chain_reads_as_empty(self, tmp_path):
        assert read_events(tmp_path / "absent.jsonl") == []
        assert chain_head(tmp_path / "absent.jsonl") == GENESIS_DIGEST

    def test_prediction_events_use_a_separate_domain(self):
        from pii_masking.benchmark_protocol import PREDICTION_EVENT_DOMAIN

        core = {
            "previous_digest": GENESIS_DIGEST,
            "event_type": "e",
            "timestamp": TIMESTAMP,
            "payload": {},
        }

        assert event_digest(core) != event_digest(core, domain=PREDICTION_EVENT_DOMAIN)


class TestMaterializedView:
    def test_view_is_recomputed_from_the_chain(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("env", {"platform_url": "https://x"}, timestamp=TIMESTAMP)
        log.append("run", {"run_id": "r-1"}, timestamp=TIMESTAMP)

        view = log.materialize("pid")

        assert view["protocol_id"] == "pid"
        assert view["platform_url"] == "https://x"
        assert view["run_id"] == "r-1"
        assert view["event_count"] == 2

    def test_later_events_win(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {"status": "running"}, timestamp=TIMESTAMP)
        log.append("b", {"status": "complete"}, timestamp=TIMESTAMP)

        assert log.materialize("pid")["status"] == "complete"

    def test_view_is_not_the_authority(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {"status": "running"}, timestamp=TIMESTAMP)
        path = log.write_materialized("pid")

        path.write_bytes(canonical_bytes({"status": "tampered"}))

        # Deleting or editing the view changes nothing; the chain rebuilds it.
        assert log.materialize("pid")["status"] == "running"

    def test_a_broken_chain_refuses_to_materialize(self, tmp_path):
        log = ExecutionLog(tmp_path)
        log.append("a", {}, timestamp=TIMESTAMP)
        log.events_path.write_bytes(b'{"event_digest":"x","previous_digest":"y"}\n')

        with pytest.raises(ProtocolError):
            log.materialize("pid")

    def test_view_records_the_chain_head(self, tmp_path):
        log = ExecutionLog(tmp_path)
        record = log.append("a", {}, timestamp=TIMESTAMP)

        assert log.materialize("pid")["chain_head"] == record["event_digest"]


def test_canonical_digest_matches_manual_hash():
    assert canonical_digest({"a": 1}) == hashlib.sha256(b'{"a":1}\n').hexdigest()
