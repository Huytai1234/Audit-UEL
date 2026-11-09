"""Verify editable benchmark schemas, missing rules and assessment boundaries."""

from copy import deepcopy
from decimal import Decimal
from io import StringIO

import pandas as pd
import pytest

from auditpy.models import RatioResult
from auditpy.ratios import (
    assess_ratio, build_ratio_report_tables, current_ratio, debt_to_equity,
    net_margin, prepare_benchmark_policies,
)
from auditpy.utils import SchemaError

D = Decimal
COLUMNS = [
    "ratio_code", "unit", "direction", "good_cutoff", "risk_cutoff",
    "valid_minimum",
]


def benchmark_table():
    return pd.DataFrame([
        ("current_ratio", "times", "higher", "1.5", "1.0", "0"),
        ("debt_to_equity", "times", "lower", "2.5", "5", "0."),
    ], columns=COLUMNS)


def test_six_column_csv_with_only_one_rule_does_not_require_metadata_or_other_ratios():
    table = pd.read_csv(StringIO(
        ",".join(COLUMNS) + "\ndebt_to_equity,times,lower,2.5,5.0,0."
    ), dtype=str, keep_default_na=False)
    policies = prepare_benchmark_policies(table)
    assert set(policies) == {"debt_to_equity"}
    result = assess_ratio(debt_to_equity(D("50"), D("100")), policies["debt_to_equity"])
    assert result["assessment"] == "Good"
    assert result["assessment_status"] == "assessed_policy"
    assert result["benchmark_basis"] == "configured_thresholds"
    assert result["source_url"] is None
    assert assess_ratio(current_ratio(D("150"), D("100")), policies.get("current_ratio"))[
        "assessment_status"
    ] == "benchmark_unavailable"
    assert assess_ratio(current_ratio(None, D("100")), policies.get("current_ratio"))[
        "assessment_status"
    ] == "input_unavailable"


@pytest.mark.parametrize("code,value,label", [
    ("current_ratio", "1.5", "Good"),
    ("current_ratio", "1", "Needs attention"),
    ("current_ratio", "0.99", "Risk"),
    ("debt_to_equity", "2.5", "Good"),
    ("debt_to_equity", "2.50001", "Needs attention"),
    ("debt_to_equity", "5", "Risk"),
])
def test_reference_boundaries_and_units(code, value, label):
    policies = prepare_benchmark_policies(benchmark_table())
    result = RatioResult(code, D(value), "times", "calculated")
    assert assess_ratio(result, policies[code])["assessment"] == label


def test_changed_configuration_changes_assessment_without_fixed_sample_expectations():
    table = benchmark_table().iloc[[1]].copy()
    table.loc[table.index[0], "good_cutoff"] = "0.4"
    policies = prepare_benchmark_policies(table)
    result = assess_ratio(debt_to_equity(D("50"), D("100")), policies["debt_to_equity"])
    assert result["assessment"] == "Needs attention"


@pytest.mark.parametrize("problem", [
    "missing_column", "duplicate_code", "duplicate_column", "blank_code",
    "invalid_direction", "nonfinite_cutoff", "inverted_cutoffs", "invalid_minimum",
])
def test_invalid_configuration_is_rejected_before_assessment(problem):
    table = benchmark_table()
    if problem == "missing_column":
        table = table.drop(columns="good_cutoff")
    elif problem == "duplicate_code":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif problem == "duplicate_column":
        table.columns = ["ratio_code", "ratio_code", *COLUMNS[2:]]
    else:
        field, value = {
            "blank_code": ("ratio_code", " "),
            "invalid_direction": ("direction", "sideways"),
            "nonfinite_cutoff": ("good_cutoff", "NaN"),
            "inverted_cutoffs": ("good_cutoff", "0.5"),
            "invalid_minimum": ("valid_minimum", "2"),
        }[problem]
        table.loc[0, field] = value
    with pytest.raises(SchemaError):
        prepare_benchmark_policies(table)


def test_negative_values_and_unit_mismatches_do_not_receive_a_good_label():
    policies = prepare_benchmark_policies(benchmark_table())
    rule = policies["debt_to_equity"]
    result = assess_ratio(debt_to_equity(D("-50"), D("100")), rule)
    assert result["assessment_status"] == "input_review_required"
    assert result["assessment"] == "Needs attention"
    with pytest.raises(SchemaError, match="unit"):
        assess_ratio(RatioResult("debt_to_equity", D("0.5"), "fraction", "calculated"), rule)


def test_whitespace_optional_metadata_and_input_protection():
    table = benchmark_table().iloc[[1]].copy()
    table.columns = [" " + name + " " for name in table.columns]
    table.iloc[0, 0] = " debt_to_equity "
    table["source_url"] = "https://example.com/reference"
    before = table.copy(deep=True)
    policies = prepare_benchmark_policies(table)
    pd.testing.assert_frame_equal(table, before)
    original = deepcopy(policies)
    result = assess_ratio(debt_to_equity(D("50"), D("100")), policies["debt_to_equity"])
    assert result["source_url"] == "https://example.com/reference"
    assert policies == original


@pytest.mark.parametrize("profit,label,review_required", [
    ("-1", "Risk", True),
    ("2", "Risk", True),
    ("5", "Needs attention", True),
    ("10", "Good", False),
    (None, None, True),
])
def test_review_keeps_financial_alerts_and_missing_inputs(profit, label, review_required):
    """A completed assessment must not hide a low-margin financial alert."""
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("net_margin", "fraction", "higher", "0.10", "0.05", ""),
    ], columns=COLUMNS))
    result = net_margin(None if profit is None else D(profit), D("100"))
    tables = build_ratio_report_tables((result,), policies)
    row = tables["Ratios"].iloc[0]
    assert row["assessment"] == label
    assert bool(len(tables["Review_Items"])) == review_required
    if profit is not None:
        assert row["assessment_status"] == "assessed_policy"
        assert result.value == D(profit) / D("100")
    else:
        assert row["assessment_status"] == "input_unavailable"
        assert row["reason"] == "missing_numerator"


def test_missing_policy_stays_in_review_even_when_the_ratio_is_positive():
    result = net_margin(D("10"), D("100"))
    tables = build_ratio_report_tables((result,), {})
    assert tables["Review_Items"]["code"].tolist() == ["net_margin"]
    row = tables["Review_Items"].iloc[0]
    assert row["assessment"] is None
    assert row["assessment_status"] == "benchmark_unavailable"


@pytest.mark.parametrize("days,label", [
    ("40", "Good"), ("60", "Needs attention"),
    ("80", "Needs attention"), ("81", "Risk"),
])
def test_receivable_days_and_turnover_agree_at_boundaries(days, label):
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("receivables_turnover", "times", "higher", "9.125", "4.5625", "0"),
        ("days_sales_outstanding", "days", "lower_exclusive", "40", "80", "0"),
    ], columns=COLUMNS))
    results = (
        RatioResult("days_sales_outstanding", D(days), "days", "calculated"),
        RatioResult("receivables_turnover", D("365") / D(days), "times", "calculated"),
    )
    assert [assess_ratio(r, policies[r.code])["assessment"] for r in results] == [label, label]


@pytest.mark.parametrize("days,label", [
    ("73", "Good"), ("100", "Needs attention"),
    ("365", "Needs attention"), ("366", "Risk"),
])
def test_inventory_days_and_turnover_agree_at_boundaries(days, label):
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("inventory_turnover", "times", "higher", "5", "1", "0"),
        ("days_inventory_outstanding", "days", "lower_exclusive", "73", "365", "0"),
    ], columns=COLUMNS))
    results = (
        RatioResult("days_inventory_outstanding", D(days), "days", "calculated"),
        RatioResult("inventory_turnover", D("365") / D(days), "times", "calculated"),
    )
    assert [assess_ratio(r, policies[r.code])["assessment"] for r in results] == [label, label]


@pytest.mark.parametrize("value,label,status", [
    ("-0.01", "Risk", "assessed_policy"),
    ("0", "Needs attention", "context_review_required"),
    ("0.20", "Needs attention", "context_review_required"),
])
def test_loss_screen_does_not_invent_a_good_reference(value, label, status):
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("gross_margin", "fraction", "higher", "", "0", ""),
    ], columns=COLUMNS))
    result = RatioResult("gross_margin", D(value), "fraction", "calculated")
    row = assess_ratio(result, policies[result.code])
    assert (row["assessment"], row["assessment_status"]) == (label, status)
    tables = build_ratio_report_tables((result,), policies)
    assert len(tables["Review_Items"]) == 1
    summary = dict(tables["Summary"].itertuples(index=False, name=None))
    assert summary["context_review_ratios"] == (status == "context_review_required")
    assert summary["calculated_without_policy"] == 0
    assert summary["assessed_ratios"] == (status == "assessed_policy")


@pytest.mark.parametrize("value", ["-20", "0", "90", None])
def test_cash_cycle_context_preserves_negative_values_and_missing_inputs(value):
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("cash_conversion_cycle", "days", "context", "", "", ""),
    ], columns=COLUMNS))
    result = RatioResult(
        "cash_conversion_cycle", None if value is None else D(value), "days",
        "unavailable" if value is None else "calculated",
        "missing_dso" if value is None else None,
    )
    row = assess_ratio(result, policies[result.code])
    if value is None:
        assert row["assessment"] is None
        assert row["assessment_status"] == "input_unavailable"
    else:
        assert row["assessment"] == "Needs attention"
        assert row["assessment_status"] == "context_review_required"
        assert result.value == D(value)


@pytest.mark.parametrize("direction,good,risk", [
    ("context", "1", ""), ("context", "", "0"),
    ("higher", "", ""), ("lower_exclusive", "", "80"),
    ("higher", "NaN", "0"),
])
def test_incomplete_or_misleading_reference_rules_are_rejected(direction, good, risk):
    with pytest.raises(SchemaError):
        prepare_benchmark_policies(pd.DataFrame([
            ("sample", "times", direction, good, risk, ""),
        ], columns=COLUMNS))
