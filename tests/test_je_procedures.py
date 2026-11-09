"""Check JE business rules, thresholds and source references."""

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal as D
import unicodedata

import pandas as pd
import pytest

from auditpy.je_tests import (
    JournalEntryTest, UnbalancedVoucherTest, DuplicateEntryTest, VoucherSequenceGapTest,
    WeekendHolidayTest, RoundAmountTest, NearApprovalLimitTest, LargeYearEndEntryTest,
    UnusualCounterAccountTest, MissingOrVagueDescriptionTest, make_threshold_test,
)
from auditpy.models import Finding, RiskLevel

KEY = ["source_file", "source_sheet", "posting_date", "voucher_no"]


def test_immutable_findings_risk_order_and_abstract_contract():
    findings = [Finding(
        "JE04", "V001", date(2024, 1, 6), ("1111",), D("100"), "sample_line",
        "Synthetic review example", risk, "sample.csv", None, ((2, "debit"),),
    ) for risk in (RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.MEDIUM)]
    assert [f.risk_level for f in sorted(findings, reverse=True)] == [
        RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW,
    ]
    with pytest.raises(FrozenInstanceError):
        findings[0].amount = D("999")
    with pytest.raises(TypeError):
        JournalEntryTest()


@pytest.mark.parametrize("threshold", [None, D("0"), D("NaN")])
def test_monetary_test_rejects_unusable_materiality(ledger_factory, threshold):
    with pytest.raises(ValueError):
        UnbalancedVoucherTest(KEY).run(ledger_factory([{"debit": "100"}]), threshold)


def test_je01_groups_dates_separately_and_retains_unassigned(ledger_factory):
    frame = ledger_factory([
        {"voucher_no": "SAME", "debit": "100"},
        {"voucher_no": "SAME", "credit": "100", "posting_date": date(2024, 1, 4)},
        {"voucher_no": "SMALL_GAP", "debit": "1000", "credit": "999"},
        {"voucher_no": None, "debit": "10"},
    ])
    procedure = UnbalancedVoucherTest(KEY)
    findings = procedure.run(frame, D("500"))
    assert len(findings) == 3
    high = next(f for f in findings if f.voucher_no == "SMALL_GAP")
    assert (high.amount, high.risk_amount, high.risk_level) == (D("1"), D("1000"), RiskLevel.HIGH)
    assert procedure.coverage["assigned_records"] == 3
    assert procedure.coverage["unassigned_records"] == 1
    assert {f.source_refs for f in findings} == {((2, None),), ((3, None),), ((4, None),)}


def test_je02_details_do_not_hide_candidate_duplicates(ledger_factory):
    frame = ledger_factory([
        {"debit": "1000", "description": "Allocation A"},
        {"debit": "1000", "description": "Allocation B"},
        {"account": "3311", "credit": "2000"},
        {"voucher_no": None, "debit": "1000"},
    ])
    before = frame.copy(deep=True)
    procedure = DuplicateEntryTest()
    findings = procedure.run(frame, D("1500"))
    assert len(findings) == 1
    assert findings[0].amount == D("1000")
    assert findings[0].risk_amount == D("2000")
    assert findings[0].risk_level == RiskLevel.HIGH
    assert findings[0].source_refs == ((2, None), (3, None))
    assert procedure.coverage["groups_with_detail_variations"] == 1
    assert procedure.coverage["possible_excess_records"] == 1
    assert procedure.coverage["excluded_records"] == 1
    pd.testing.assert_frame_equal(frame, before)


def test_je03_keeps_series_widths_and_unparsed_evidence(ledger_factory, audit_settings):
    frame = ledger_factory([{"voucher_no": value} for value in (
        "PC24010001", "PC24010001", "PC24010004", "PC24020001",
        "PC240100005", "MANUAL", None,
    )])
    policy = audit_settings["je_tests"]
    procedure = VoucherSequenceGapTest(policy["voucher_sequence_patterns"], policy["voucher_sequence_assumption"])
    findings = procedure.run(frame, D("500"))
    assert len(findings) == 1
    assert findings[0].source_refs == ((2, None), (4, None))
    assert findings[0].amount is None and findings[0].voucher_no is None
    assert procedure.coverage["missing_sequence_numbers"] == 2
    assert procedure.coverage["missing_key_records"] == 1
    assert procedure.coverage["unparsed_identifiers"] == (("sample.csv", None, 7, "MANUAL", "unmatched"),)


def test_je03_ambiguous_patterns_are_not_guessed(ledger_factory):
    pattern = r"(?P<series>PC[0-9]{4})(?P<sequence>[0-9]{4})"
    procedure = VoucherSequenceGapTest([pattern, pattern], "Synthetic sequence register.")
    assert procedure.run(ledger_factory([{"voucher_no": "PC24010001"}]), D("1")) == []
    assert procedure.coverage["ambiguous_identifiers"] == 1


def test_je04_overlap_zero_unknown_and_outside_year(ledger_factory, calendar_reference):
    frame = ledger_factory([
        {"posting_date": date(2024, 1, 6), "debit": "1000"},
        {"posting_date": date(2024, 2, 10), "debit": "-600"},
        {"posting_date": date(2024, 4, 30)},
        {"posting_date": date(2024, 1, 6), "debit": None, "voucher_no": None},
        {"posting_date": None}, {"posting_date": date(2025, 1, 1)},
    ])
    procedure = WeekendHolidayTest(calendar_reference)
    findings = procedure.run(frame, D("500"))
    assert len(findings) == 4
    assert [f.risk_level for f in findings].count(RiskLevel.HIGH) == 2
    assert [f.risk_level for f in findings].count(RiskLevel.LOW) == 1
    assert any(f.amount is None and f.voucher_no is None for f in findings)
    assert procedure.coverage["overlap_records"] == 1
    assert procedure.coverage["unavailable_date_records"] == 1
    assert procedure.coverage["outside_calendar_year_records"] == 1


def test_je05_partial_exact_multiples_and_threshold_equality(ledger_factory):
    frame = ledger_factory([
        {"debit": "-2000"}, {"debit": "100"}, {"debit": "100.01"},
        {"debit": "50", "credit": "1000"}, {"debit": None, "credit": "1000"}, {},
    ])
    procedure = RoundAmountTest([D("100"), D("1000")], "Synthetic screening multiples.")
    findings = procedure.run(frame, D("100"))
    assert len(findings) == 3
    assert procedure.coverage["multiple_base_records"] == 2
    assert procedure.coverage["negative_round_records"] == 1
    assert next(f for f in findings if f.source_refs == ((3, None),)).risk_level == RiskLevel.MEDIUM
    assert frame.iloc[0]["debit"] == D("-2000")


def test_je06_closure_and_positive_cash_credits_without_reversal_netting(ledger_factory):
    window = make_threshold_test(D("100"), D("0.95"))
    assert window(D("95")) and window(D("99.99"))
    assert not window(D("94.99")) and not window(D("100"))
    frame = ledger_factory([
        {"voucher_no": "A", "credit": "50"}, {"voucher_no": "A", "credit": "45"},
        {"voucher_no": "A", "credit": "-20"}, {"voucher_no": "B", "credit": "100"},
        {"voucher_no": "C", "credit": "97"},
        {"voucher_no": "D", "account": "642", "credit": "95"},
        {"voucher_no": None, "credit": "99"},
    ])
    procedure = NearApprovalLimitTest(KEY, D("100"), D("0.95"), ["111", "112"], "Synthetic unconfirmed limit.")
    findings = procedure.run(frame, D("96"))
    assert {f.voucher_no: f.amount for f in findings} == {"A": D("95"), "C": D("97")}
    assert next(f for f in findings if f.voucher_no == "A").source_refs == ((2, None), (3, None))
    assert procedure.coverage["unassigned_records"] == 1


def test_je07_partial_window_reversals_and_unassigned_fallback(ledger_factory):
    frame = ledger_factory([
        {"voucher_no": "A", "posting_date": date(2024, 12, 26), "debit": "1000"},
        {"voucher_no": "B", "posting_date": date(2024, 12, 25), "debit": "1000"},
        {"voucher_no": "C", "posting_date": date(2025, 1, 5), "debit": "1000"},
        {"voucher_no": "D", "posting_date": date(2024, 12, 31), "debit": "-600"},
        {"voucher_no": None, "posting_date": date(2024, 12, 31), "debit": "800"},
        {"voucher_no": "E", "posting_date": date(2024, 12, 31), "debit": None},
    ])
    procedure = LargeYearEndEntryTest(KEY, date(2024, 12, 31), 5, date(2024, 1, 1), date(2024, 12, 31))
    findings = procedure.run(frame, D("500"))
    assert {f.voucher_no for f in findings} == {"A", "D", None}
    assert next(f for f in findings if f.voucher_no == "D").amount == D("600")
    assert procedure.coverage["declared_window_coverage"] == "partial"
    assert len(procedure.coverage["uncovered_window_dates"]) == 5
    assert procedure.coverage["unavailable_amount_units"] == 1
    complete = LargeYearEndEntryTest(KEY, date(2024, 12, 31), 5, date(2024, 1, 1), date(2025, 1, 5))
    assert {f.voucher_no for f in complete.run(frame, D("500"))} == {"A", "C", "D", None}


def test_je08_effective_sign_and_missing_or_compound_counter_codes(ledger_factory, audit_settings):
    frame = ledger_factory([
        {"account": "2111", "counter_account": "5111", "debit": "100"},
        {"account": "2111", "counter_account": "5111", "debit": "-100"},
        {"account": "6421", "counter_account": "1111", "credit": "-700"},
        {"counter_account": None, "debit": "900"},
        {"counter_account": "511/3331", "debit": "900"},
        {"account": "2111", "counter_account": "5111", "debit": None},
    ])
    procedure = UnusualCounterAccountTest(audit_settings["je_tests"]["counter_account_rules"])
    findings = procedure.run(frame, D("500"))
    assert {f.source_refs for f in findings} == {((2, None),), ((4, None),)}
    assert procedure.coverage["non_single_code_records"] == 1
    assert procedure.coverage["missing_counter_account_records"] == 1
    assert frame.iloc[2]["credit"] == D("-700")


def test_je09_unicode_exact_phrases_and_other_lines_in_same_voucher(ledger_factory, audit_settings):
    generic = unicodedata.normalize("NFD", " ĐIỀU   CHỈNH!!! ")
    frame = ledger_factory([
        {"description": generic, "debit": "100"},
        {"description": "Điều chỉnh chi phí theo biên bản DC-01", "debit": "1000"},
        {"description": None}, {"description": "...", "debit": "-700"},
        {"description": 123, "debit": None},
    ])
    procedure = MissingOrVagueDescriptionTest(audit_settings["je_tests"]["vague_description_phrases"])
    findings = procedure.run(frame, D("500"))
    assert len(findings) == 4
    assert procedure.coverage["generic_phrase_records"] == 1
    assert procedure.coverage["affected_groups_with_other_descriptions"] == 1
    assert procedure.coverage["unavailable_amount_findings"] == 1
    assert {f.source_refs for f in findings} == {((2, None),), ((4, None),), ((5, None),), ((6, None),)}
    assert frame.iloc[0]["description"] == generic
