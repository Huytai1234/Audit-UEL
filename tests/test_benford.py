"""Verify Benford selection, hand calculations, statistical boundaries and evidence."""

from decimal import Decimal as D

import pandas as pd
import pytest

from auditpy.benford import (
    first_significant_digit, select_benford_candidates, benford_frequency,
    classify_mad, benford_statistics,
)
from auditpy.je_tests import BenfordTest
from auditpy.models import RiskLevel
from auditpy.utils import SchemaError


@pytest.mark.parametrize("amount,digit", [("100", 1), ("0.0031", 3), ("9E-100000", 9)])
def test_first_significant_digit_uses_decimal_representation(amount, digit):
    assert first_significant_digit(D(amount)) == digit


@pytest.mark.parametrize("amount", [D("0"), D("-1"), D("NaN"), D("Infinity"), None, True, 100.0])
def test_frequency_rejects_ineligible_amounts(amount):
    assert first_significant_digit(amount) is None
    with pytest.raises(ValueError):
        benford_frequency([amount])


def test_selection_preserves_repeated_values_and_distinct_pair_sides(ledger_factory):
    frame = ledger_factory([
        {"debit": "100", "source_row": 2, "source_side": "debit"},
        {"debit": "100", "source_row": 2, "source_side": "credit"},
        {"debit": "200", "exclude_from_income_statement": True},
        {"debit": "-50"}, {}, {"debit": None}, {"account": None, "debit": "100"},
    ])
    frame["exclude_from_income_statement"] = frame["exclude_from_income_statement"].eq(True)
    frame["closing_review_required"] = False
    before = frame.copy(deep=True)
    selected, coverage = select_benford_candidates(frame, "debit")
    assert selected["benford_amount"].tolist() == [D("100"), D("100")]
    assert coverage["selected"] == 2
    assert all(coverage[key] == 1 for key in (
        "identified_closing", "negative_amount", "zero_amount", "unavailable_amount", "missing_account",
    ))
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(SchemaError, match="uniquely"):
        select_benford_candidates(pd.concat([frame, frame.iloc[:1]]), "debit")


def test_hand_calculation_empty_population_and_sample_size_effect():
    amounts = [D("100"), D("200"), D("300"), D("0.0031")]
    frequency = benford_frequency(amounts)
    assert frequency["observed_count"].tolist() == [1, 1, 2, 0, 0, 0, 0, 0, 0]
    assert frequency["expected_rate"].sum() == pytest.approx(1)
    statistics = benford_statistics(amounts)
    assert statistics["chi_square"] == pytest.approx(6.254123200189831)
    assert statistics["mad"] == pytest.approx(0.09977111207467085)
    assert statistics["chi_square_screen"] == "small_expected_counts"
    repeated = benford_statistics(amounts * 100)
    assert repeated["mad"] == pytest.approx(statistics["mad"])
    assert repeated["chi_square"] == pytest.approx(100 * statistics["chi_square"])
    empty = benford_statistics([])
    assert empty["chi_square"] is None and empty["mad"] is None
    assert empty["chi_square_screen"] == "empty_population"
    assert len(benford_frequency([])) == 9
    assert benford_statistics([D("100")] * 109)["chi_square_screen"] == "small_expected_counts"
    assert benford_statistics([D("100")] * 110)["chi_square_screen"] == "above_critical"


@pytest.mark.parametrize("mad,band", [
    (None, "unavailable"), (D("0.005999"), "close_conformity"),
    (D("0.006"), "acceptable_conformity"), (D("0.012"), "marginal_conformity"),
    (D("0.015"), "marginal_conformity"), (D("0.015001"), "nonconformity"),
])
def test_mad_boundaries_do_not_overlap(mad, band):
    assert classify_mad(mad) == band


@pytest.mark.parametrize("mad", [True, D("NaN"), D("-0.01"), D("1.01"), "0.006"])
def test_mad_rejects_invalid_statistics(mad):
    with pytest.raises(ValueError):
        classify_mad(mad)


def test_benford_class_keeps_population_evidence_without_inventing_loss(ledger_factory, audit_settings, account_map):
    frame = ledger_factory([{"debit": "100"} for _ in range(110)])
    before = frame.copy(deep=True)
    procedure = BenfordTest(audit_settings["je_tests"]["benford_populations"], audit_settings["voucher_key"], account_map)
    findings = procedure.run(frame, D("500000"))
    assert len(findings) == 1 and findings[0].risk_level == RiskLevel.HIGH
    assert findings[0].amount is None and findings[0].risk_amount is None
    assert findings[0].voucher_no is None and findings[0].posting_date is None
    assert findings[0].source_refs == ((2, None), (3, None), (4, None))
    snapshot = procedure.results
    key = ("sample.csv", None, "all_ledger_debit")
    assert len(snapshot[key]["observations"]) == 110
    snapshot[key]["statistics"]["mad"] = 0
    snapshot[key]["observations"].iloc[0, 2] = 999
    assert procedure.results[key]["statistics"]["mad"] > 0.015
    assert procedure.results[key]["observations"].iloc[0, 2] == 2
    pd.testing.assert_frame_equal(frame, before)
    assert procedure.run(frame.iloc[:4], D("500000")) == []
    assert len(procedure.results) == 4


def test_benford_reuses_closing_exclusions_and_retains_uncertain_lines(ledger_factory, audit_settings, account_map):
    frame = ledger_factory([
        {"account": "5111", "counter_account": "911", "debit": "100"},
        {"account": "911", "counter_account": "5111", "credit": "100"},
        {"voucher_no": "UNCERTAIN", "account": "635", "counter_account": None, "debit": "200"},
    ])
    procedure = BenfordTest(audit_settings["je_tests"]["benford_populations"], audit_settings["voucher_key"], account_map)
    assert procedure.run(frame, D("1")) == []
    result = procedure.results[("sample.csv", None, "all_ledger_debit")]
    assert result["coverage"]["identified_closing"] == 2
    assert result["observations"]["benford_amount"].tolist() == [D("200")]
    assert result["coverage"]["selected_closing_review_records"] == 1


def test_debit_and_credit_populations_remain_separate(ledger_factory, audit_settings, account_map):
    frame = ledger_factory([
        {"account": "641", "debit": "3000000"},
        {"account": "642", "debit": "7000000"},
        {"account": "334", "credit": "10000000"},
        {"voucher_no": "SALE", "account": "1121", "debit": "20000000"},
        {"voucher_no": "SALE", "account": "5111", "credit": "20000000"},
    ])
    before = frame.copy(deep=True)
    procedure = BenfordTest(
        audit_settings["je_tests"]["benford_populations"],
        audit_settings["voucher_key"], account_map,
    )
    procedure.run(frame, D("100"))
    results = procedure.results
    debit = results[("sample.csv", None, "all_ledger_debit")]
    credit = results[("sample.csv", None, "all_ledger_credit")]
    assert debit["observations"]["benford_amount"].tolist() == [D("3000000"), D("7000000"), D("20000000")]
    assert credit["observations"]["benford_amount"].tolist() == [D("10000000"), D("20000000")]
    assert debit["observations"]["first_digit"].tolist() == [3, 7, 2]
    assert credit["observations"]["first_digit"].tolist() == [1, 2]
    assert sum(len(result["frequency"]) for result in results.values()) == 36
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("problem", ["missing_credit", "wrong_side", "restricted_ledger"])
def test_benford_policy_must_describe_both_sides(problem, audit_settings, account_map):
    policies = audit_settings["je_tests"]["benford_populations"]
    if problem == "missing_credit":
        policies.pop("all_ledger_credit")
    elif problem == "wrong_side":
        policies["all_ledger_credit"]["amount_column"] = "debit"
    else:
        policies["all_ledger_credit"]["account_prefixes"] = ["511"]
    with pytest.raises(ValueError):
        BenfordTest(policies, audit_settings["voucher_key"], account_map)
