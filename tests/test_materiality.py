"""Verify hand-calculated materiality, configuration choices and missing evidence."""

from copy import deepcopy
from decimal import Decimal

import pandas as pd
import pytest
from openpyxl import load_workbook

from auditpy.materiality import Materiality, build_materiality_tables
from auditpy.utils import SchemaError, write_excel_report

D = Decimal


@pytest.fixture
def materiality_inputs():
    statements = {
        "Summary": pd.DataFrame([
            ("period_start", "2024-01-01"), ("period_end", "2024-12-31"),
            ("unit_status", "Synthetic source units"), ("report_status", "provisional"),
        ], columns=["metric", "value"], dtype=object),
        "Income_Statement": pd.DataFrame([
            ("net_revenue", D("30000000")), ("profit_before_tax", D("2000000")),
        ], columns=["item", "amount"], dtype=object),
        "Balance_Sheet": pd.DataFrame([
            ("total_assets", D("20000000")),
        ], columns=["item", "amount"], dtype=object),
    }
    settings = {
        "period_start": "2024-01-01", "period_end": "2024-12-31",
        "materiality": {
            "benchmark": "profit_before_tax", "overall_rate": "0.05",
            "performance_rate": "0.50", "clearly_trivial_rate": "0.03",
            "rounding_increment": "1",
            "comparison_rates": {
                "profit_before_tax": ["0.05"], "net_revenue": ["0.005", "0.01"],
                "total_assets": ["0.01", "0.02"],
            },
            "benchmark_rationale": "Synthetic profitability assumption.",
            "performance_rationale": "Synthetic cautious planning choice.",
            "clearly_trivial_rationale": "Synthetic threshold; review qualitative matters.",
        },
    }
    return statements, settings


def selection_from(tables):
    return dict(tables["Selected"].itertuples(index=False, name=None))


def test_hand_calculated_levels_keep_raw_values_and_round_down():
    result = Materiality("profit_before_tax", D("2012345"), D("0.05")).calculate_levels(D("0.5"), D("0.03"))
    assert result["unrounded_materiality"] == D("100617.25")
    assert result["overall_materiality"] == D("100617")
    assert result["unrounded_performance_materiality"] == D("50308.5")
    assert result["performance_materiality"] == D("50308")
    assert result["unrounded_clearly_trivial_threshold"] == D("3018.51")
    assert result["clearly_trivial_threshold"] == D("3018")
    assert result["calculation_status"] == "calculated"


@pytest.mark.parametrize("step,expected", [("1000", "5000"), ("250", "5500")])
def test_rounding_uses_the_increment_not_only_its_decimal_exponent(step, expected):
    assert Materiality.round_amount(D("5678.90"), D(step)) == D(expected)


@pytest.mark.parametrize("amount,status", [
    (None, "missing_benchmark"), (D("0"), "non_positive_benchmark"),
    (D("-2000000"), "non_positive_benchmark"),
])
def test_unusable_benchmarks_do_not_become_zero_or_absolute_profit(amount, status):
    result = Materiality("profit_before_tax", amount, D("0.05")).calculate_levels(D("0.5"), D("0.03"))
    assert result["calculation_status"] == status
    assert result["overall_materiality"] is None
    assert result["performance_materiality"] is None
    assert result["clearly_trivial_threshold"] is None


@pytest.mark.parametrize("rate", [D("0"), D("1"), D("NaN"), 0.05])
def test_invalid_overall_rates_are_rejected(rate):
    with pytest.raises(ValueError):
        Materiality("profit_before_tax", D("2000000"), rate)


@pytest.mark.parametrize("performance,trivial", [
    (D("0.03"), D("0.5")), (D("0.5"), D("0")), (D("0.5"), D("NaN")),
])
def test_invalid_secondary_rates_are_rejected(performance, trivial):
    with pytest.raises(ValueError):
        Materiality("profit_before_tax", D("2000000"), D("0.05")).calculate_levels(performance, trivial)


def test_rounding_collapse_is_flagged_instead_of_becoming_a_usable_threshold():
    overall_zero = Materiality("profit_before_tax", D("10"), D("0.05")).calculate_levels(D("0.5"), D("0.03"))
    assert overall_zero["calculation_status"] == "rounded_to_zero"
    assert overall_zero["performance_materiality"] is None
    secondary_zero = Materiality("profit_before_tax", D("30"), D("0.05")).calculate_levels(D("0.5"), D("0.03"))
    assert secondary_zero["calculation_status"] == "rounded_levels_invalid"


def test_report_keeps_scope_source_and_inputs_unchanged(materiality_inputs):
    statements, settings = materiality_inputs
    original_settings = deepcopy(settings)
    original_tables = {name: table.copy(deep=True) for name, table in statements.items()}
    result = build_materiality_tables(statements, settings)
    selected = selection_from(result)
    assert selected["overall_materiality"] == D("100000")
    assert selected["performance_materiality"] == D("50000")
    assert selected["clearly_trivial_threshold"] == D("3000")
    assert selected["source_sheet"] == "Income_Statement" and selected["source_row"] == 3
    assert selected["unit_status"] == "Synthetic source units"
    assert selected["report_status"] == "provisional"
    assert len(result["Options"]) == 5 and result["Options"]["is_selected"].sum() == 1
    assert result["Checks"]["check_status"].eq("matched").all()
    assert settings == original_settings
    for name, table in statements.items():
        pd.testing.assert_frame_equal(table, original_tables[name])


@pytest.mark.parametrize("benchmark,rate,expected,option_count", [
    ("net_revenue", "0.005", "150000", 5),
    ("total_assets", "0.02", "400000", 5),
    ("net_revenue", "0.0065", "195000", 6),
])
def test_configuration_changes_the_choice_and_includes_a_custom_rate(
    materiality_inputs, benchmark, rate, expected, option_count,
):
    statements, settings = materiality_inputs
    settings["materiality"].update(benchmark=benchmark, overall_rate=rate)
    result = build_materiality_tables(statements, settings)
    selected = selection_from(result)
    assert selected["benchmark"] == benchmark
    assert selected["overall_materiality"] == D(expected)
    assert len(result["Options"]) == option_count
    assert result["Options"]["is_selected"].sum() == 1


def test_missing_chosen_benchmark_does_not_select_an_available_alternative(materiality_inputs):
    statements, settings = materiality_inputs
    statements["Income_Statement"].loc[1, "amount"] = None
    result = build_materiality_tables(statements, settings)
    selected = selection_from(result)
    assert selected["benchmark"] == "profit_before_tax"
    assert selected["calculation_status"] == "missing_benchmark"
    assert selected["performance_materiality"] is None
    assert result["Options"].loc[result["Options"]["benchmark"].eq("net_revenue"), "overall_materiality"].notna().all()
    assert result["Checks"]["check_status"].value_counts().to_dict() == {"unavailable": 4, "matched": 2}


@pytest.mark.parametrize("problem", ["period", "rationale", "benchmark", "decimal", "comparisons", "duplicate_options", "duplicate_item"])
def test_invalid_scope_policy_and_source_items_are_rejected(materiality_inputs, problem):
    statements, settings = materiality_inputs
    policy = settings["materiality"]
    if problem == "period":
        settings["period_start"] = "2025-01-01"
    elif problem == "rationale":
        policy["benchmark_rationale"] = " "
    elif problem == "benchmark":
        policy["benchmark"] = "unknown"
    elif problem == "decimal":
        policy["overall_rate"] = 0.05
    elif problem == "comparisons":
        del policy["comparison_rates"]["total_assets"]
    elif problem == "duplicate_options":
        policy["comparison_rates"]["profit_before_tax"].append("0.050")
    else:
        statements["Income_Statement"] = pd.concat([
            statements["Income_Statement"], statements["Income_Statement"].iloc[[1]],
        ], ignore_index=True)
    with pytest.raises(SchemaError):
        build_materiality_tables(statements, settings)


def test_export_preserves_exact_materiality_values(materiality_inputs, tmp_path):
    statements, settings = materiality_inputs
    statements["Income_Statement"].loc[1, "amount"] = D("2012345")
    tables = build_materiality_tables(statements, settings)
    path = tmp_path / "04_materiality.xlsx"
    write_excel_report(path, tables)
    workbook = load_workbook(path, read_only=True)
    try:
        assert workbook.sheetnames == ["Options", "Selected", "Levels", "Checks"]
        selected = dict(workbook["Selected"].iter_rows(min_row=2, max_col=2, values_only=True))
        assert selected["unrounded_materiality"] == "100617.25"
        assert selected["clearly_trivial_threshold"] == "3018"
    finally:
        workbook.close()
