"""Inference, retry, pause, and journal tests against an offline fake endpoint.

No network, no credentials, no real OpenAI client. The fake endpoint is scripted
per test so every branch of the retry and durability contract is reachable.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pii_masking.benchmark_inference import (
    BASE_MODEL_KEY,
    MAX_ATTEMPTS,
    MAX_CONCURRENCY,
    OUTCOME_SUCCESS,
    SFT_MODEL_KEY,
    UNKNOWN_REMOTE_OUTCOME,
    InferenceError,
    JournalState,
    ModelArm,
    PairedRunner,
    PauseController,
    PredictionJournal,
    Sample,
    SendResult,
    assert_request_parity,
    assert_resumable,
    attestation_fingerprint,
    build_request,
    classify_response,
    load_journal,
    nearest_rank_p95,
    retry_delay,
    sft_artifact_digest,
    validate_completeness,
    validate_sft_model_ref,
)
from pii_masking.benchmark_protocol import Protocol, build_protocol_payload

BASE_REF = "qwen3.5-4b"
SFT_REF = "ft:qwen3.5-4b:run-1:1023"


def make_protocol():
    return Protocol.from_payload(
        build_protocol_payload(
            benchmark_source="openpii-validation",
            execution_environment="production",
            base_model={"model_id": "Qwen/Qwen3.5-4B"},
            training={"lora_rank": 64},
            generation={"temperature": 0, "top_p": 1, "n": 1, "max_tokens": 2048},
            prompt={"system": "mask it"},
            selection={},
            scoring={},
            expected_counts={},
        )
    )


def arms(base_digest="base-dig", adapter_digest="adapter-dig"):
    return [
        ModelArm(BASE_MODEL_KEY, BASE_REF, base_digest),
        ModelArm(SFT_MODEL_KEY, SFT_REF, sft_artifact_digest(base_digest, adapter_digest)),
    ]


def samples(count=3):
    return [Sample("task-a", f"u{i}", f"text {i}") for i in range(count)]


class FakeEndpoint:
    """Scripted responses, recording every request body it receives."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default or SendResult(status=200, content="ok")
        self.requests: list[dict] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(self, request):
        self.requests.append(request)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
            result = self.script.pop(0) if self.script else self.default
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.concurrent -= 1


def run(runner, rows):
    return asyncio.run(runner.run(rows))


def make_runner(tmp_path, endpoint, **kwargs):
    return PairedRunner(
        make_protocol(),
        PredictionJournal(tmp_path / "predictions.jsonl"),
        kwargs.pop("model_arms", arms()),
        endpoint,
        sleep=_no_sleep,
        **kwargs,
    )


async def _no_sleep(_seconds):
    return None


# ── model refs ────────────────────────────────────────────────────────────────
class TestModelRefs:
    def test_exact_numeric_ref_is_accepted(self):
        assert validate_sft_model_ref("ft:qwen3.5-4b:run-1:1023") == 1023

    @pytest.mark.parametrize(
        "ref",
        ["ft:qwen3.5-4b:run-1:latest", "latest", "qwen3.5-4b", "ft:a:b", "ft:a:b:c:d", ""],
    )
    def test_moving_or_malformed_refs_are_rejected(self, ref):
        with pytest.raises(InferenceError, match="exact numeric checkpoint"):
            validate_sft_model_ref(ref)

    def test_artifact_digest_binds_adapter_to_base(self):
        assert sft_artifact_digest("base-1", "ad-1") != sft_artifact_digest("base-2", "ad-1")
        assert sft_artifact_digest("base-1", "ad-1") != sft_artifact_digest("base-1", "ad-2")


# ── request parity ────────────────────────────────────────────────────────────
class TestRequestParity:
    def test_requests_differ_only_by_model(self):
        protocol = make_protocol()
        base = build_request(protocol, "hello", BASE_REF)
        sft = build_request(protocol, "hello", SFT_REF)

        assert base["model"] != sft["model"]
        assert assert_request_parity(base, sft)

    def test_any_other_difference_is_fatal(self):
        protocol = make_protocol()
        base = build_request(protocol, "hello", BASE_REF)
        sft = build_request(protocol, "hello", SFT_REF)
        sft["temperature"] = 0.7

        with pytest.raises(InferenceError, match="more than the model field"):
            assert_request_parity(base, sft)

    def test_generation_settings_come_from_the_protocol(self):
        request = build_request(make_protocol(), "hello", BASE_REF)

        assert request["temperature"] == 0
        assert request["max_tokens"] == 2048
        assert request["messages"][0]["content"] == "mask it"


# ── retry classification ──────────────────────────────────────────────────────
class TestRetryPolicy:
    @pytest.mark.parametrize("status", [200, 201])
    def test_2xx_is_success(self, status):
        assert classify_response(status, None) == OUTCOME_SUCCESS

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
    def test_retryable_statuses(self, status):
        assert classify_response(status, None) == "retryable"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_fatal_statuses_are_not_retried(self, status):
        assert classify_response(status, None) == "fatal"

    def test_transport_errors_are_retryable(self):
        assert classify_response(None, OSError("boom")) == "retryable"

    def test_retry_after_is_honored_and_capped(self):
        import random

        assert retry_delay(1, 5.0, random.Random(0)) == 5.0
        assert retry_delay(1, 999.0, random.Random(0)) == 60.0

    def test_backoff_is_jittered_and_capped(self):
        import random

        delays = [retry_delay(10, None, random.Random(seed)) for seed in range(20)]

        assert all(0.0 <= d <= 30.0 for d in delays)
        assert len(set(delays)) > 1  # jitter, not a fixed schedule


# ── runner behavior ───────────────────────────────────────────────────────────
class TestRunner:
    def test_both_arms_are_issued_per_sample(self, tmp_path):
        endpoint = FakeEndpoint()
        runner = make_runner(tmp_path, endpoint)

        summary = run(runner, samples(3))

        assert summary["completed"] == 6
        assert len(endpoint.requests) == 6

    def test_arm_order_alternates_on_the_sample_hash(self, tmp_path):
        endpoint = FakeEndpoint()
        runner = make_runner(tmp_path, endpoint)

        run(runner, samples(12))

        firsts = [arm for index, (_, arm) in enumerate(runner.issued_order) if index % 2 == 0]
        # Deterministic but balanced: not every sample leads with the same arm.
        assert len(set(firsts)) == 2

    def test_arm_order_is_deterministic(self, tmp_path):
        first = make_runner(tmp_path / "a", FakeEndpoint())
        second = make_runner(tmp_path / "b", FakeEndpoint())

        run(first, samples(8))
        run(second, samples(8))

        assert first.issued_order == second.issued_order

    def test_concurrency_never_exceeds_the_cap(self, tmp_path):
        endpoint = FakeEndpoint()
        runner = make_runner(tmp_path, endpoint, concurrency=MAX_CONCURRENCY)

        run(runner, samples(20))

        assert endpoint.max_concurrent <= MAX_CONCURRENCY

    def test_concurrency_above_the_cap_is_refused(self, tmp_path):
        with pytest.raises(InferenceError, match="exceeds the cap"):
            make_runner(tmp_path, FakeEndpoint(), concurrency=MAX_CONCURRENCY + 1)

    def test_a_retryable_status_is_retried_then_succeeds(self, tmp_path):
        endpoint = FakeEndpoint(
            script=[SendResult(status=503), SendResult(status=200, content="ok")]
        )
        runner = make_runner(tmp_path, endpoint)

        run(runner, samples(1))

        records = PredictionJournal(tmp_path / "predictions.jsonl").records()
        attempts = [r for r in records if r["record"] == "attempt_end"]
        assert attempts[0]["outcome"] == "retryable"
        assert attempts[1]["outcome"] == OUTCOME_SUCCESS

    def test_a_fatal_status_is_not_retried(self, tmp_path):
        endpoint = FakeEndpoint(default=SendResult(status=400))
        runner = make_runner(tmp_path, endpoint)

        run(runner, samples(1))

        # One attempt per arm, no retries.
        assert len(endpoint.requests) == 2

    def test_attempts_are_capped_at_five(self, tmp_path):
        endpoint = FakeEndpoint(default=SendResult(status=503))
        runner = make_runner(tmp_path, endpoint, model_arms=[arms()[0]])

        summary = run(runner, samples(1))

        assert len(endpoint.requests) == MAX_ATTEMPTS
        assert summary["exhausted"] == 1

    def test_an_already_canonical_identity_is_not_reissued(self, tmp_path):
        endpoint = FakeEndpoint()
        runner = make_runner(tmp_path, endpoint)
        rows = samples(2)
        run(runner, rows)
        issued = len(endpoint.requests)

        # Second pass over the SAME rows: smoke inside pilot inside full.
        run(runner, rows)

        assert len(endpoint.requests) == issued

    def test_invalid_content_is_a_completed_prediction_not_a_retry(self, tmp_path):
        endpoint = FakeEndpoint(default=SendResult(status=200, content="Sure! here you go"))
        runner = make_runner(tmp_path, endpoint)

        summary = run(runner, samples(1))

        assert summary["completed"] == 2
        assert len(endpoint.requests) == 2


# ── journal durability ────────────────────────────────────────────────────────
class TestJournal:
    def test_attempt_start_is_written_before_the_response(self, tmp_path):
        endpoint = FakeEndpoint()
        runner = make_runner(tmp_path, endpoint)
        run(runner, samples(1))

        records = PredictionJournal(tmp_path / "predictions.jsonl").records()

        assert records[0]["record"] == "attempt_start"
        assert records[1]["record"] == "attempt_end"

    def test_a_start_without_an_end_is_an_unknown_remote_outcome(self, tmp_path):
        journal = PredictionJournal(tmp_path / "predictions.jsonl")
        journal.start_attempt("id-1", 1, timestamp="t")

        state = load_journal(tmp_path / "predictions.jsonl")

        assert state.unknown_outcomes == ["id-1"]
        assert state.canonical == {}

    def test_only_the_first_success_is_canonical(self, tmp_path):
        journal = PredictionJournal(tmp_path / "predictions.jsonl")
        journal.start_attempt("id-1", 1, timestamp="t")
        journal.end_attempt("id-1", 1, outcome=OUTCOME_SUCCESS, timestamp="t", content="first")
        journal.start_attempt("id-1", 2, timestamp="t")
        journal.end_attempt("id-1", 2, outcome=OUTCOME_SUCCESS, timestamp="t", content="second")

        state = load_journal(tmp_path / "predictions.jsonl")

        assert state.canonical["id-1"]["content"] == "first"
        assert state.duplicates == ["id-1"]

    def test_duplicate_successes_block_a_final_metric(self, tmp_path):
        state = JournalState(canonical={"a": {}}, duplicates=["a"])

        assert validate_completeness(state, ["a"])["final_eligible"] is False

    def test_exhausted_identities_make_the_report_incomplete(self, tmp_path):
        state = JournalState(canonical={}, exhausted=["a"])

        summary = validate_completeness(state, ["a"])

        assert summary["final_eligible"] is False
        assert summary["exhausted"] == 1

    def test_a_complete_journal_is_final_eligible(self):
        state = JournalState(canonical={"a": {}, "b": {}})

        assert validate_completeness(state, ["a", "b"])["final_eligible"] is True

    def test_unknown_outcomes_are_disclosed(self):
        state = JournalState(canonical={"a": {}}, unknown_outcomes=["a"])

        assert validate_completeness(state, ["a"])["unknown_remote_outcomes"] == 1

    def test_content_digest_is_recorded(self, tmp_path):
        journal = PredictionJournal(tmp_path / "predictions.jsonl")
        journal.start_attempt("id-1", 1, timestamp="t")
        journal.end_attempt("id-1", 1, outcome=OUTCOME_SUCCESS, timestamp="t", content="masked")

        record = journal.records()[-1]

        assert len(record["content_digest"]) == 64

    def test_resume_reuses_prior_canonical_responses(self, tmp_path):
        endpoint = FakeEndpoint()
        rows = samples(2)
        run(make_runner(tmp_path, endpoint), rows)
        issued = len(endpoint.requests)

        # A fresh runner, resuming from the journal on disk.
        resumed = make_runner(
            tmp_path, endpoint, state=load_journal(tmp_path / "predictions.jsonl")
        )
        run(resumed, rows)

        assert len(endpoint.requests) == issued


# ── attestation gating ────────────────────────────────────────────────────────
class TestResumeAttestation:
    def fingerprint(self, **overrides):
        fields = dict(
            model_listing=["qwen3.5-4b"],
            source_revision="rev-1",
            base_attestation={"manifest_digest": "d1"},
            model_refs=[BASE_REF, SFT_REF],
        )
        fields.update(overrides)
        return attestation_fingerprint(**fields)

    def test_an_unchanged_world_resumes(self):
        assert_resumable(self.fingerprint(), self.fingerprint())

    def test_model_listing_order_does_not_matter(self):
        assert self.fingerprint(model_listing=["a", "b"]) == self.fingerprint(
            model_listing=["b", "a"]
        )

    @pytest.mark.parametrize(
        "override",
        [
            {"source_revision": "rev-2"},
            {"base_attestation": {"manifest_digest": "d2"}},
            {"model_refs": [BASE_REF, "ft:qwen3.5-4b:run-1:511"]},
            {"model_listing": ["something-else"]},
        ],
    )
    def test_any_drift_refuses_to_resume(self, override):
        with pytest.raises(InferenceError, match="changed since the last run"):
            assert_resumable(self.fingerprint(), self.fingerprint(**override))


# ── pause control ─────────────────────────────────────────────────────────────
class TestPauseController:
    def test_retry_rate_below_threshold_does_not_pause(self):
        controller = PauseController()
        # Placed late: the window is ROLLING with a 50-attempt minimum, so two
        # retryables inside the first 50 attempts would be 4%, not 1%.
        for index in range(200):
            controller.record_attempt(index in (150, 199))

        assert controller.should_pause() is None

    def test_an_early_burst_pauses_even_if_the_overall_rate_is_low(self):
        controller = PauseController()
        for index in range(200):
            controller.record_attempt(index < 2)

        # 1% overall, but 4% across the first rolling window — a burst against a
        # shared endpoint is exactly what the trigger is for.
        assert controller.should_pause() == "retry_rate"

    def test_retry_rate_above_threshold_pauses(self):
        controller = PauseController()
        for index in range(200):
            controller.record_attempt(index < 20)  # 10%

        assert controller.should_pause() == "retry_rate"

    def test_retry_rate_needs_a_minimum_sample(self):
        controller = PauseController()
        controller.record_attempt(True)  # 100% of one attempt

        assert controller.should_pause() is None

    def test_latency_needs_three_calibration_windows(self):
        controller = PauseController()
        for _ in range(200):
            controller.record_latency(1.0)

        assert controller.calibrated is False

    def test_baseline_is_the_median_of_three_window_p95s(self):
        controller = PauseController()
        for _ in range(300):
            controller.record_latency(1.0)

        assert controller.calibrated is True
        assert controller.baseline == 1.0

    def test_latency_pause_requires_three_consecutive_breaches(self):
        controller = PauseController()
        for _ in range(300):
            controller.record_latency(1.0)

        for _ in range(200):  # two breaching windows
            controller.record_latency(10.0)
        assert controller.should_pause() is None

        for _ in range(100):  # third
            controller.record_latency(10.0)
        assert controller.should_pause() == "latency"

    def test_a_good_window_resets_the_breach_streak(self):
        controller = PauseController()
        for _ in range(300):
            controller.record_latency(1.0)
        for _ in range(200):
            controller.record_latency(10.0)
        for _ in range(100):
            controller.record_latency(1.0)  # recovers
        for _ in range(100):
            controller.record_latency(10.0)

        assert controller.should_pause() is None

    def test_nearest_rank_p95_returns_an_observed_value(self):
        assert nearest_rank_p95([1.0, 2.0, 3.0, 100.0]) == 100.0
        assert nearest_rank_p95(list(range(100))) == 94
        assert nearest_rank_p95([]) == 0.0

    def test_a_paused_runner_stops_scheduling(self, tmp_path):
        endpoint = FakeEndpoint(default=SendResult(status=503))
        runner = make_runner(tmp_path, endpoint, model_arms=[arms()[0]])

        summary = run(runner, samples(50))

        assert summary["paused"] == "retry_rate"
        # It stopped well before issuing 50 samples x 5 attempts.
        assert len(endpoint.requests) < 250


# ── scoring hand-off ──────────────────────────────────────────────────────────
class TestScoreJournal:
    def test_score_reports_completeness(self, tmp_path):
        from pii_masking.benchmark_inference import score_journal

        report = json.loads(score_journal(make_protocol(), JournalState(canonical={"a": {}})))

        assert report["completeness"]["complete"] == 0

    def test_final_refuses_an_incomplete_journal(self):
        from pii_masking.benchmark_inference import score_journal

        protocol = make_protocol()
        state = JournalState(canonical={}, exhausted=["a"])

        with pytest.raises(InferenceError, match="cannot emit a final metric"):
            score_journal(protocol, state, final=True)


def test_unknown_remote_outcome_constant_is_journalable():
    assert UNKNOWN_REMOTE_OUTCOME == "unknown_remote_outcome"


# ── client construction ───────────────────────────────────────────────────────
class TestClientConstruction:
    def test_client_disables_sdk_retries(self, monkeypatch):
        """The SDK retries silently by default; this module must be the sole owner."""
        from pii_masking import benchmark_inference

        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module = type("M", (), {"AsyncOpenAI": FakeAsyncOpenAI})
        monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

        benchmark_inference.create_client("https://example.invalid", "key")

        assert captured["max_retries"] == 0


# ── load-only adapter preflight ───────────────────────────────────────────────
class TestAdapterPreflight:
    def call_with(self, response, recorder=None):
        async def _call(body):
            if recorder is not None:
                recorder.append(body)
            return response

        return _call

    def preflight(self, response, recorder=None, **overrides):
        from pii_masking.benchmark_inference import run_adapter_preflight

        kwargs = dict(
            model_ref=SFT_REF,
            lora_path="c/loras/run-1/lora_peft_iter_0001023",
            expected_base_digest="base-dig",
            expected_adapter_digest="adapter-dig",
        )
        kwargs.update(overrides)
        return asyncio.run(run_adapter_preflight(self.call_with(response, recorder), **kwargs))

    def good_response(self, **overrides):
        response = {
            "cache_digest": "adapter-dig",
            "blob_digest": "adapter-dig",
            "normalized": False,
            "base": {"manifest_digest": "base-dig"},
        }
        response.update(overrides)
        return response

    def test_a_matching_adapter_attests_successfully(self):
        assert self.preflight(self.good_response())["cache_digest"] == "adapter-dig"

    def test_no_completion_request_is_issued(self):
        recorded: list[dict] = []

        self.preflight(self.good_response(), recorded)

        assert len(recorded) == 1
        assert "messages" not in recorded[0]
        assert recorded[0]["model_ref"] == SFT_REF

    def test_a_moving_alias_fails_before_any_call(self):
        recorded: list[dict] = []

        with pytest.raises(InferenceError, match="exact numeric checkpoint"):
            self.preflight(self.good_response(), recorded, model_ref="ft:q:run-1:latest")

        assert recorded == []

    def test_a_normalized_adapter_is_rejected(self):
        with pytest.raises(InferenceError, match="normalized"):
            self.preflight(self.good_response(normalized=True))

    def test_an_adapter_digest_mismatch_is_rejected(self):
        with pytest.raises(InferenceError, match="does not match the identity digest"):
            self.preflight(self.good_response(cache_digest="other"))

    def test_blob_and_cache_disagreement_is_rejected(self):
        with pytest.raises(InferenceError, match="disagree"):
            self.preflight(self.good_response(blob_digest="different"))

    def test_base_digest_drift_is_rejected(self):
        with pytest.raises(InferenceError, match="live base digest"):
            self.preflight(self.good_response(base={"manifest_digest": "drifted"}))


class TestUsageSummary:
    def test_usage_and_latency_roll_up_per_arm(self, tmp_path):
        from pii_masking.benchmark_inference import usage_summary

        endpoint = FakeEndpoint(
            default=SendResult(
                status=200, content="ok", usage={"prompt_tokens": 10, "completion_tokens": 5}
            )
        )
        run(make_runner(tmp_path, endpoint), samples(3))

        summary = usage_summary(tmp_path / "predictions.jsonl")

        assert set(summary) == {BASE_MODEL_KEY, SFT_MODEL_KEY}
        assert summary[BASE_MODEL_KEY]["requests"] == 3
        assert summary[BASE_MODEL_KEY]["total_tokens"] == 45
        assert "p95_latency" in summary[BASE_MODEL_KEY]

    def test_failed_attempts_are_excluded_from_usage(self, tmp_path):
        from pii_masking.benchmark_inference import usage_summary

        endpoint = FakeEndpoint(default=SendResult(status=400))
        run(make_runner(tmp_path, endpoint), samples(2))

        assert usage_summary(tmp_path / "predictions.jsonl") == {}


class TestJournalJoinability:
    def test_records_carry_the_sample_uid_and_arm(self, tmp_path):
        endpoint = FakeEndpoint()
        run(make_runner(tmp_path, endpoint), samples(2))

        ends = [
            r
            for r in PredictionJournal(tmp_path / "predictions.jsonl").records()
            if r["record"] == "attempt_end"
        ]

        assert ends, "expected terminal records"
        for record in ends:
            # A journal of opaque digests cannot be scored: both keys are needed
            # to join a response back to its row and its model.
            assert record["sample_uid"].startswith("task-a:")
            assert record["model_key"] in {BASE_MODEL_KEY, SFT_MODEL_KEY}

    def test_every_row_and_arm_pair_appears_exactly_once(self, tmp_path):
        endpoint = FakeEndpoint()
        run(make_runner(tmp_path, endpoint), samples(3))

        ends = [
            r
            for r in PredictionJournal(tmp_path / "predictions.jsonl").records()
            if r["record"] == "attempt_end" and r["outcome"] == OUTCOME_SUCCESS
        ]
        pairs = [(r["sample_uid"], r["model_key"]) for r in ends]

        assert len(pairs) == 6
        assert len(set(pairs)) == 6
