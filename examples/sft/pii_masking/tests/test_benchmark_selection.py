"""Tests for deterministic selection: quotas, duplicates, order-independence.

The fixtures are small so the arithmetic is checkable by hand: the frozen quota
rule and the nesting relationships are asserted against explicit numbers, not
against whatever the implementation happens to produce.
"""

from __future__ import annotations

import pytest
from pii_masking.benchmark_selection import (
    ENGLISH,
    NON_ENGLISH_LANGUAGES,
    TASK_OPENPII_EN,
    TASK_OPENPII_NONEN,
    TASK_PIIMB_EN,
    TASK_PIIMB_MULTI,
    DuplicateLedger,
    SelectionError,
    assert_disjoint,
    assert_label_coverage,
    build_evaluation_universe,
    duplicate_diagnostics,
    language_quotas,
    normalize_openpii_row,
    observed_labels,
    reject_placeholder_collisions,
    select_openpii_evaluation,
    select_train_and_development,
    sort_selection,
    take_language_balanced,
    to_sft_bytes,
)

LANGUAGE_COUNT = len(NON_ENGLISH_LANGUAGES)


def row(uid, language=ENGLISH, text=None, masked=None):
    return normalize_openpii_row(
        {
            "uid": uid,
            "language": language,
            "source_text": text if text is not None else f"text for {uid}",
            "masked_text": masked if masked is not None else f"masked [NAME_1] {uid}",
        }
    )


def population(per_language: int, english: int):
    """Synthetic rows: ``english`` English plus ``per_language`` of each other."""
    rows = [row(f"en-{i}") for i in range(english)]
    for code in NON_ENGLISH_LANGUAGES:
        rows.extend(row(f"{code}-{i}", language=code) for i in range(per_language))
    return rows


# ── quota arithmetic ──────────────────────────────────────────────────────────
class TestLanguageQuotas:
    def test_exact_multiple_splits_evenly(self):
        quotas = language_quotas(LANGUAGE_COUNT * 3)

        assert set(quotas.values()) == {3}
        assert sum(quotas.values()) == LANGUAGE_COUNT * 3

    def test_remainder_goes_to_the_first_codes_in_frozen_order(self):
        quotas = language_quotas(LANGUAGE_COUNT + 3)

        assert quotas["bg"] == 2 and quotas["cs"] == 2 and quotas["da"] == 2
        assert quotas["de"] == 1
        assert sum(quotas.values()) == LANGUAGE_COUNT + 3

    def test_frozen_counts_match_the_contract(self):
        # 2048 train / 128 development / 5000 task / 1000 pilot.
        assert sum(language_quotas(2048).values()) == 2048
        assert sum(language_quotas(128).values()) == 128
        assert sum(language_quotas(5000).values()) == 5000
        assert sum(language_quotas(1000).values()) == 1000

    def test_2048_across_22_languages(self):
        quotas = language_quotas(2048)

        assert quotas["bg"] == 94  # 2048 // 22 == 93, remainder 2048 % 22 == 2
        assert quotas["cs"] == 94
        assert quotas["da"] == 93

    def test_a_negative_total_is_rejected(self):
        with pytest.raises(SelectionError):
            language_quotas(-1)

    def test_insufficient_rows_for_one_language_is_fatal(self):
        rows = population(per_language=1, english=0)
        rows = [r for r in rows if r.language != "sv"]

        with pytest.raises(SelectionError, match="'sv'"):
            take_language_balanced(rows, LANGUAGE_COUNT, "d")

    def test_shortfall_is_not_silently_reallocated(self):
        rows = population(per_language=2, english=0)
        rows = [r for r in rows if not (r.language == "sv" and r.uid.endswith("-1"))]

        with pytest.raises(SelectionError):
            take_language_balanced(rows, LANGUAGE_COUNT * 2, "d")


# ── duplicate ledger ──────────────────────────────────────────────────────────
class TestDuplicateLedger:
    def test_same_uid_with_a_different_row_is_fatal(self):
        ledger = DuplicateLedger()
        ledger.add(row("u-1", text="one"), "d")

        with pytest.raises(SelectionError, match="two different rows"):
            ledger.add(row("u-1", text="two"), "d")

    def test_the_identical_row_twice_is_tolerated(self):
        ledger = DuplicateLedger()
        ledger.add(row("u-1"), "d")
        ledger.add(row("u-1"), "d")

        assert len(ledger) == 1

    def test_one_representative_per_exact_text(self):
        ledger = DuplicateLedger()
        ledger.extend([row("a", text="same"), row("b", text="same"), row("c", text="other")], "d")

        assert len(ledger.canonical_uids("d")) == 2

    def test_duplicate_choice_is_independent_of_arrival_order(self):
        rows = [row("a", text="same"), row("b", text="same"), row("c", text="same")]

        forward = DuplicateLedger()
        forward.extend(rows, "d")
        reverse = DuplicateLedger()
        reverse.extend(list(reversed(rows)), "d")

        assert forward.canonical_uids("d") == reverse.canonical_uids("d")

    def test_same_text_in_different_languages_still_deduplicates(self):
        ledger = DuplicateLedger()
        ledger.extend(
            [row("a", language="de", text="same"), row("b", language="fr", text="same")], "d"
        )

        assert len(ledger.canonical_uids("d")) == 1


# ── evaluation universe ───────────────────────────────────────────────────────
def small_universe(**kwargs):
    english = [row(f"en-{i}") for i in range(60)]
    multi = [row(f"{code}-{i}", language=code) for code in NON_ENGLISH_LANGUAGES for i in range(3)]
    return build_evaluation_universe(
        {TASK_OPENPII_EN: english, TASK_OPENPII_NONEN: multi},
        pilot_rows=kwargs.get("pilot_rows", 22),
        smoke_rows=kwargs.get("smoke_rows", 5),
        audit_rows=kwargs.get("audit_rows", 10),
    )


class TestEvaluationUniverse:
    def test_smoke_is_nested_in_pilot_and_pilot_in_the_task(self):
        universe = small_universe()

        for task in (TASK_OPENPII_EN, TASK_OPENPII_NONEN):
            task_uids = [r.uid for r in universe.tasks[task]]
            pilot_uids = [r.uid for r in universe.pilot[task]]
            smoke_uids = [r.uid for r in universe.smoke[task]]

            assert set(smoke_uids) <= set(pilot_uids) <= set(task_uids)

    def test_audit_is_nested_in_pilot(self):
        universe = small_universe()

        for task in universe.tasks:
            assert set(r.uid for r in universe.audit[task]) <= set(
                r.uid for r in universe.pilot[task]
            )

    def test_multilingual_pilot_preserves_language_quotas(self):
        universe = small_universe(pilot_rows=LANGUAGE_COUNT)

        languages = {r.language for r in universe.pilot[TASK_OPENPII_NONEN]}

        assert languages == set(NON_ENGLISH_LANGUAGES)

    def test_selection_is_stable_across_runs(self):
        first = small_universe()
        second = small_universe()

        assert [r.uid for r in first.pilot[TASK_OPENPII_EN]] == [
            r.uid for r in second.pilot[TASK_OPENPII_EN]
        ]

    def test_manifest_contains_no_source_text(self):
        universe = small_universe()

        import json

        encoded = json.dumps(universe.manifest())

        assert "text for" not in encoded
        assert "selection_hash" in encoded

    def test_source_text_containing_a_placeholder_is_rejected(self):
        rows = [row("a"), row("b", text="already [NAME_1] masked")]

        with pytest.raises(SelectionError, match="placeholder grammar"):
            reject_placeholder_collisions(rows, TASK_OPENPII_EN)

    def test_ordinary_brackets_are_not_placeholders(self):
        reject_placeholder_collisions([row("a", text="see [1] and [note]")], TASK_OPENPII_EN)


class TestOrderIndependence:
    def test_forward_and_reversed_streams_agree(self):
        rows = population(per_language=3, english=60)

        forward = select_openpii_evaluation(
            rows, task_rows=22, pilot_rows=22, smoke_rows=5, audit_rows=10
        )
        reverse = select_openpii_evaluation(
            list(reversed(rows)), task_rows=22, pilot_rows=22, smoke_rows=5, audit_rows=10
        )

        assert forward.manifest() == reverse.manifest()

    def test_sft_bytes_are_reproducible_across_stream_order(self):
        rows = population(per_language=3, english=60)

        forward = select_openpii_evaluation(
            rows, task_rows=22, pilot_rows=22, smoke_rows=5, audit_rows=10
        )
        reverse = select_openpii_evaluation(
            list(reversed(rows)), task_rows=22, pilot_rows=22, smoke_rows=5, audit_rows=10
        )

        assert to_sft_bytes(forward.tasks[TASK_OPENPII_EN]) == to_sft_bytes(
            reverse.tasks[TASK_OPENPII_EN]
        )

    def test_sorting_is_total_and_stable(self):
        rows = population(per_language=2, english=10)

        assert [r.uid for r in sort_selection(rows, "d")] == [
            r.uid for r in sort_selection(list(reversed(rows)), "d")
        ]


# ── train / development ───────────────────────────────────────────────────────
def training_population():
    rows = [row(f"tr-en-{i}") for i in range(40)]
    for code in NON_ENGLISH_LANGUAGES:
        rows.extend(row(f"tr-{code}-{i}", language=code) for i in range(4))
    return rows


def tiny_evaluation():
    return build_evaluation_universe(
        {TASK_OPENPII_EN: [row(f"ev-{i}") for i in range(30)]},
        pilot_rows=10,
        smoke_rows=2,
        audit_rows=3,
    )


class TestTrainDevelopmentSelection:
    def test_counts_and_language_mix_are_exact(self):
        splits = select_train_and_development(
            training_population(), tiny_evaluation(), train_rows=44, development_rows=22
        )

        assert len(splits.train) == 44
        assert len(splits.development) == 22
        assert sum(1 for r in splits.train if r.language == ENGLISH) == 22
        assert sum(1 for r in splits.development if r.language == ENGLISH) == 11

    def test_train_and_development_are_disjoint(self):
        splits = select_train_and_development(
            training_population(), tiny_evaluation(), train_rows=44, development_rows=22
        )

        assert not {r.uid for r in splits.train} & {r.uid for r in splits.development}

    def test_evaluation_lineage_is_excluded(self):
        evaluation = tiny_evaluation()
        leaking = training_population() + [row("ev-0")]

        splits = select_train_and_development(
            leaking, evaluation, train_rows=44, development_rows=22
        )

        assert "ev-0" not in {r.uid for r in splits.train}

    def test_evaluation_text_is_excluded_even_under_a_new_uid(self):
        evaluation = tiny_evaluation()
        shared_text = evaluation.tasks[TASK_OPENPII_EN][0].source_text
        leaking = training_population() + [row("fresh-uid", text=shared_text)]

        splits = select_train_and_development(
            leaking, evaluation, train_rows=44, development_rows=22
        )

        assert "fresh-uid" not in {r.uid for r in splits.train}
        assert "fresh-uid" not in {r.uid for r in splits.development}

    def test_a_leaked_row_is_caught_by_the_disjointness_assertion(self):
        evaluation = tiny_evaluation()

        with pytest.raises(SelectionError, match="shares source lineage"):
            assert_disjoint(evaluation.all_rows(), [row("ev-0")], label="train")

    def test_identical_text_is_caught_by_the_disjointness_assertion(self):
        evaluation = tiny_evaluation()
        shared = evaluation.tasks[TASK_OPENPII_EN][0].source_text

        with pytest.raises(SelectionError, match="identical to an evaluation row"):
            assert_disjoint(evaluation.all_rows(), [row("other", text=shared)], label="train")

    def test_insufficient_candidates_is_fatal(self):
        with pytest.raises(SelectionError):
            select_train_and_development(
                training_population(), tiny_evaluation(), train_rows=4096, development_rows=256
            )

    def test_manifest_records_language_counts_without_text(self):
        splits = select_train_and_development(
            training_population(), tiny_evaluation(), train_rows=44, development_rows=22
        )

        import json

        encoded = json.dumps(splits.manifest())

        assert "text for" not in encoded
        assert splits.manifest()["train"]["languages"][ENGLISH] == 22


class TestLabelCoverage:
    def test_labels_are_read_from_masked_text(self):
        rows = [row("a", masked="[GIVENNAME_1] and [PHONE_2]")]

        assert observed_labels(rows) == {"GIVENNAME", "PHONE"}

    def test_a_label_missing_from_train_is_fatal(self):
        train = [row("a", masked="[NAME_1]")]
        pop = train + [row("b", masked="[IBAN_1]")]

        with pytest.raises(SelectionError, match="IBAN"):
            assert_label_coverage(train, pop)

    def test_full_coverage_passes(self):
        train = [row("a", masked="[NAME_1]"), row("b", masked="[IBAN_1]")]

        assert_label_coverage(train, train)


# ── PIIMB task semantics ──────────────────────────────────────────────────────
class TestPiimbTaskSemantics:
    def test_cross_task_duplicates_are_retained_not_removed(self):
        shared = "the same sentence"
        english = [row(f"{TASK_PIIMB_EN}:{i}", text=f"{shared} {i}") for i in range(30)]
        multi = [
            row(f"{TASK_PIIMB_MULTI}:{i}", language=code, text=f"{shared} {i}")
            for i, code in enumerate(NON_ENGLISH_LANGUAGES)
        ]

        universe = build_evaluation_universe(
            {TASK_PIIMB_EN: english, TASK_PIIMB_MULTI: multi},
            pilot_rows=LANGUAGE_COUNT,
            smoke_rows=5,
            audit_rows=10,
        )

        # Every pinned row survives in its own task.
        assert len(universe.tasks[TASK_PIIMB_EN]) == 30
        assert len(universe.tasks[TASK_PIIMB_MULTI]) == LANGUAGE_COUNT

    def test_duplicates_are_reported_as_diagnostics(self):
        shared_text = "identical across tasks"
        # Both tasks need at least pilot_rows rows; the multilingual one is
        # additionally quota-balanced, so every frozen language must be present.
        english = [row(f"{TASK_PIIMB_EN}:0", text=shared_text)] + [
            row(f"{TASK_PIIMB_EN}:{i}") for i in range(1, LANGUAGE_COUNT + 5)
        ]
        multi = [
            row(f"{TASK_PIIMB_MULTI}:{code}", language=code, text=f"{shared_text} {code}")
            for code in NON_ENGLISH_LANGUAGES
        ]
        multi[0] = row(f"{TASK_PIIMB_MULTI}:bg", language="bg", text=shared_text)

        universe = build_evaluation_universe(
            {TASK_PIIMB_EN: english, TASK_PIIMB_MULTI: multi},
            pilot_rows=LANGUAGE_COUNT,
            smoke_rows=1,
            audit_rows=1,
        )

        assert duplicate_diagnostics(universe)["cross_task_text_overlaps"] == 1


class TestLedgerIsDiskBacked:
    """The ledger's reason to exist is not holding ~1M rows in memory."""

    def test_rows_are_rehydrated_from_disk_not_a_cache(self, tmp_path):
        ledger = DuplicateLedger(tmp_path / "ledger.sqlite")
        original = row("u-1", text="payload stays on disk")
        ledger.extend([original], "d")

        rehydrated = ledger.row("u-1")

        assert rehydrated.uid == original.uid
        assert rehydrated.source_text == original.source_text
        assert rehydrated.row_digest == original.row_digest

    def test_the_ledger_keeps_no_in_memory_row_payloads(self, tmp_path):
        ledger = DuplicateLedger(tmp_path / "ledger.sqlite")
        ledger.extend([row(f"u-{i}", text=f"body {i}") for i in range(50)], "d")

        # Any attribute holding the rows would defeat the disk-backed design.
        cached = [
            name
            for name, value in vars(ledger).items()
            if isinstance(value, (dict, list)) and len(value) >= 50
        ]

        assert cached == []
        assert len(ledger) == 50

    def test_an_unknown_uid_is_an_error_not_a_key_error(self, tmp_path):
        ledger = DuplicateLedger(tmp_path / "ledger.sqlite")

        with pytest.raises(SelectionError, match="not in the ledger"):
            ledger.row("absent")

    def test_canonical_choice_is_unchanged_by_the_disk_backing(self, tmp_path):
        rows = [row("a", text="same"), row("b", text="same"), row("c", text="other")]
        memory = DuplicateLedger()
        memory.extend(rows, "d")
        disk = DuplicateLedger(tmp_path / "ledger.sqlite")
        disk.extend(rows, "d")

        assert memory.canonical_uids("d") == disk.canonical_uids("d")
