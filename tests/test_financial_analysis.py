"""Check hand calculations, signed cash-flow bridges and monthly evidence."""

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from openpyxl import load_workbook

from auditpy.models import RatioResult
from auditpy.ledger import review_cash_flows, summarize_cash_flow_estimate
from auditpy.ratios import (
    account_side_total, accrual_ratio, altman_z_double_prime, attach_source_classifications,
    build_financial_analysis_tables, build_ratio_report_tables,
    estimate_interest_expense, get_ratio_registry, prepare_benchmark_policies,
    review_finance_costs, screen_monthly_series, summarize_finance_review,
)
from auditpy.statements import build_opening_balance_sheet
from auditpy.utils import SchemaError, write_excel_report

D = Decimal


@pytest.mark.parametrize("code,inputs,expected", [
    ("current_ratio", ("150", "100"), "1.5"),
    ("gross_margin", ("60", "200"), "0.3"),
    ("net_margin", ("20", "200"), "0.1"),
    ("quick_ratio", ("150", "50", "100"), "1"),
    ("cash_ratio", ("20", "100"), "0.2"),
    ("debt_to_assets", ("120", "300"), "0.4"),
    ("debt_to_equity", ("120", "180"), "0.6666666666666666666666666667"),
    ("interest_coverage", ("60", "10"), "6"),
    ("asset_turnover", ("600", "200", "400"), "2"),
    ("return_on_assets", ("30", "200", "400"), "0.1"),
    ("return_on_equity", ("30", "100", "200"), "0.2"),
    ("receivables_turnover", ("730", "80", "120"), "7.3"),
    ("inventory_turnover", ("365", "20", "80"), "7.3"),
    ("days_sales_outstanding", ("7.3",), "50"),
    ("days_inventory_outstanding", ("7.3",), "50"),
    ("days_payables_outstanding", ("365", "20", "40"), "30"),
    ("cash_conversion_cycle", ("50", "50", "30"), "70"),
    ("altman_z_double_prime", ("200", "100", "80", "400", "1000", "600"), "2.8756"),
    ("accrual_ratio", ("100", "60", "300", "500"), "0.1"),
])
def test_registered_formulas_match_hand_calculations(code, inputs, expected):
    arguments = tuple(D(value) for value in inputs)
    function = get_ratio_registry()[code]
    result = function(*arguments)
    assert result.value == D(expected)
    assert result.calculation_status == "calculated"
    assert result == function(*arguments)
    with pytest.raises(FrozenInstanceError):
        result.value = D("999")


@pytest.mark.parametrize("code,inputs", [
    ("current_ratio", (None, D("1"))),
    ("current_ratio", (True, D("1"))),
    ("gross_margin", (D("1"), D("0"))),
    ("quick_ratio", (D("150"), None, D("100"))),
    ("cash_ratio", (D("20"), D("NaN"))),
    ("return_on_assets", (D("30"), None, D("400"))),
    ("return_on_equity", (D("30"), D("-100"), D("300"))),
    ("days_sales_outstanding", (D("0"),)),
    ("days_payables_outstanding", (D("0"), D("20"), D("40"))),
    ("cash_conversion_cycle", (D("50"), D("50"), None)),
    ("accrual_ratio", (D("100"), None, D("300"), D("500"))),
])
def test_unusable_inputs_propagate_without_becoming_zero(code, inputs):
    result = get_ratio_registry()[code](*inputs)
    assert result.value is None
    assert result.calculation_status == "unavailable"
    assert result.reason


def test_signed_results_and_known_zero_payables_remain_meaningful():
    registry = get_ratio_registry()
    assert registry["days_payables_outstanding"](D("365"), D("0"), D("0")).value == 0
    assert registry["cash_conversion_cycle"](D("10"), D("20"), D("40")).value == -10
    assert registry["return_on_assets"](D("-30"), D("200"), D("400")).value == D("-0.1")
    assert accrual_ratio(D("100"), D("150"), D("300"), D("500")).value == D("-0.125")
    assert accrual_ratio(D("100"), D("-20"), D("300"), D("500")).value == D("0.3")
    assert accrual_ratio(D("100"), D("0"), D("300"), D("500")).value == D("0.25")


def test_altman_preserves_negative_components_scale_and_all_missing_names():
    inputs = dict(working_capital=D("200"), retained_earnings=D("100"), ebit=D("80"),
                  book_equity=D("400"), total_assets=D("1000"), total_liabilities=D("600"))
    original = inputs.copy()
    score = altman_z_double_prime(**inputs)
    assert altman_z_double_prime(**{name: value * 1000 for name, value in inputs.items()}).value == score.value
    negative = dict(inputs, working_capital=D("-200"), retained_earnings=D("-100"), ebit=D("-80"))
    assert altman_z_double_prime(**negative).value == D("-1.4756")
    missing = altman_z_double_prime(**dict(inputs, working_capital=None, ebit=None))
    assert missing.reason == "missing_inputs:working_capital,ebit"
    assert altman_z_double_prime(**dict(inputs, total_liabilities=D("0"))).value is None
    assert inputs == original


def test_finance_cost_candidates_preserve_reversals_closing_and_unresolved_sources():
    table = pd.DataFrame([
        ("635", "Loan interest", D("12"), D("0"), False, False, 2),
        ("635", "Loan interest reversal", D("-2"), D("0"), False, False, 3),
        ("635", "Exchange loss", D("30"), D("0"), False, False, 4),
        ("635", "Closing", D("0"), D("45"), True, False, 5),
        ("635", "Unresolved allocation", D("5"), D("0"), False, True, 6),
    ], columns=["account", "description", "debit", "credit",
                "exclude_from_income_statement", "closing_review_required", "source_row"])
    table = table.assign(source_file="synthetic.csv", source_sheet=None, source_side=None)
    before = table.copy(deep=True)
    details = review_finance_costs(table, ("interest",))
    summary = summarize_finance_review(details).set_index("interest_review_status")
    assert summary.at["interest_candidate", "net_finance_cost"] == D("10")
    assert summary.at["identified_closing", "net_finance_cost"] == D("-45")
    assert summary.at["unresolved_classification", "net_finance_cost"] == D("5")
    assert sum(details.loc[~details["exclude_from_income_statement"], "net_finance_cost"], D("0")) == D("45")
    assert details["source_row"].tolist() == [2, 3, 4, 5, 6]
    pd.testing.assert_frame_equal(table, before)


def test_identified_exchange_loss_does_not_block_or_increase_interest_estimate():
    table = pd.DataFrame([
        ("635", "Loan interest", D("12"), D("0"), False, False, 2),
        ("635", "Interest reversal", D("-2"), D("0"), False, False, 3),
        ("635", "Exchange loss", D("30"), D("0"), False, False, 4),
        ("635", "Closing", D("0"), D("40"), True, False, 5),
    ], columns=["account", "description", "debit", "credit", "exclude_from_income_statement", "closing_review_required", "source_row"])
    table = table.assign(source_file="sample.csv", source_sheet=None, source_side=None)
    before = table.copy(deep=True)
    details = review_finance_costs(table, ("interest",), ("exchange loss",))
    assert details["interest_review_status"].tolist() == [
        "interest_candidate", "interest_candidate", "non_interest_candidate", "identified_closing",
    ]
    amount, basis = estimate_interest_expense(details)
    assert amount == D("10") and basis == "identified_description_estimate"
    assert sum(details.loc[~details["exclude_from_income_statement"], "net_finance_cost"], D("0")) == D("40")
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("description,amount", [
    ("Unclear adjustment", D("5")),
    ("Interest and exchange loss", D("5")),
    ("Loan interest", None),
])
def test_interest_estimate_keeps_real_classification_or_amount_gaps(description, amount):
    table = pd.DataFrame([
        ("635", description, amount, D("0"), False, False, 2),
    ], columns=["account", "description", "debit", "credit", "exclude_from_income_statement", "closing_review_required", "source_row"])
    table = table.assign(source_file="sample.csv", source_sheet=None, source_side=None)
    details = review_finance_costs(table, ("interest",), ("exchange loss",))
    assert estimate_interest_expense(details)[0] is None


def test_zero_net_unknown_finance_row_is_not_assumed_to_be_zero_interest():
    details = pd.DataFrame([
        ("other_finance_cost_review", D("5"), D("5"), D("0")),
    ], columns=["interest_review_status", "debit", "credit", "net_finance_cost"])
    assert estimate_interest_expense(details)[0] is None


def test_source_cost_item_recovers_interest_missing_from_description():
    table = pd.DataFrame([
        ("635", "Loan settlement", "B29", D("12"), D("0"), False, False, 2),
        ("635", "Currency entry", "B30", D("30"), D("0"), False, False, 3),
        ("635", "Opening adjustment", "B33", D("5"), D("0"), False, True, 4),
    ], columns=["account", "description", "cost_item_code", "debit", "credit", "exclude_from_income_statement", "closing_review_required", "source_row"])
    table = table.assign(source_file="sample.csv", source_sheet=None, source_side=None)
    details = review_finance_costs(table, ("interest",), (), {"B29": "interest", "B30": "non_interest", "B33": "non_interest"})
    assert details["interest_review_status"].tolist() == ["interest_candidate", "non_interest_candidate", "non_interest_candidate"]
    assert details["interest_review_basis"].eq("source_cost_item").all()
    assert bool(details.loc[2, "closing_review_required"])
    assert estimate_interest_expense(details) == (D("12"), "source_cost_item_estimate")


def test_source_metadata_join_uses_references_not_position_and_preserves_inputs():
    ledger = pd.DataFrame({"source_file": ["a.xlsx", "a.xlsx"], "source_sheet": ["Sheet", "Sheet"],
                           "source_row": [2, 3], "source_side": [None, None], "debit": [D("10"), D("20")]})
    source = pd.DataFrame({"source_file": ["a.xlsx", "a.xlsx"], "source_sheet": ["Sheet", "Sheet"],
                           "source_row": [3, 2], "cost_item_code": ["B30", "B29"]})
    before = source.copy(deep=True)
    result = attach_source_classifications(ledger, source)
    assert result["cost_item_code"].tolist() == ["B29", "B30"]
    assert result["debit"].tolist() == [D("10"), D("20")]
    pd.testing.assert_frame_equal(source, before)
    with pytest.raises(SchemaError):
        attach_source_classifications(ledger, pd.concat([source, source.iloc[[0]]]))


def test_pair_metadata_join_keeps_both_source_sides():
    source = pd.DataFrame({"source_file": ["pair.csv", "pair.csv"], "source_sheet": [None, None],
                           "source_row": [2, 2], "source_side": ["debit", "credit"], "receipt_payment_code": [101, 101]})
    left = source.drop(columns="receipt_payment_code").iloc[::-1]
    result = attach_source_classifications(left, source)
    assert result["source_side"].tolist() == ["credit", "debit"]
    assert result["receipt_payment_code"].tolist() == [101, 101]


def test_source_metadata_join_rejects_missing_provenance():
    ledger = pd.DataFrame({
        "source_file": ["sample.xlsx"], "source_sheet": ["Sheet"],
        "source_row": [2], "source_side": [None],
        "debit": [D("10")], "credit": [D("0")],
    })
    source = ledger.drop(columns=["debit", "credit"]).assign(
        source_row=3, cost_item_code="B29",
    )
    with pytest.raises(SchemaError, match="no matching source row"):
        attach_source_classifications(ledger, source)


def test_source_metadata_join_preserves_missing_and_negative_amounts():
    ledger = pd.DataFrame({
        "source_file": ["sample.csv", "sample.csv"],
        "source_sheet": [None, None], "source_row": [2, 3],
        "source_side": [None, None],
        "debit": [D("-10.10"), None], "credit": [D("0"), D("10.10")],
    }, dtype=object)
    source = ledger.drop(columns=["debit", "credit"]).assign(cost_item_code="B29")
    before = ledger.copy(deep=True)
    result = attach_source_classifications(ledger, source)
    pd.testing.assert_frame_equal(result[ledger.columns], before)
    assert "_source_match" not in result


def test_balance_scope_and_opening_mapping_preserve_separate_sides(account_map):
    trial = pd.DataFrame([
        ("1111", D("100"), D("0")), ("1311", D("50"), D("20")),
        ("2293", D("0"), D("10")), ("4111", D("0"), D("120")),
    ], columns=["account", "opening_debit", "opening_credit"])
    trial = trial.assign(opening_basis="provided_reference", opening_source_file="sample.xlsx",
                         opening_source_sheet="Sheet", opening_source_rows=[(4,), (5,), (6,), (7,)])
    before = trial.copy(deep=True)
    statement, details = build_opening_balance_sheet(trial, account_map)
    values = statement.set_index("item")["amount"]
    assert values["total_assets"] == D("140")
    assert values["total_liabilities"] == D("20")
    assert values["equity"] == D("120")
    assert values["balance_difference"] == 0
    assert values["current_assets"] == D("140")
    assert values["current_liabilities"] == D("20")
    assert statement.set_index("item").loc["noncurrent_liabilities", "calculation_basis"] == "includes_disclosed_maturity_assumptions"
    assert details.loc[details["account"].eq("2293"), "maturity"].eq("current").all()
    assert set(details["balance_side"]) == {"opening_debit", "opening_credit"}
    assert "closing_debit" not in details
    pd.testing.assert_frame_equal(trial, before)
    assert account_side_total(trial, "331", "opening_debit") is None
    overlap = pd.DataFrame({"account": ["131", "1311"], "amount": [D("50"), D("50")]})
    with pytest.raises(SchemaError):
        account_side_total(overlap, "131", "amount")


def monthly_sample():
    months = tuple(str(period) for period in pd.period_range("2024-01", "2024-06", freq="M"))
    table = pd.DataFrame({"month": months, "net_revenue": [D("100")] * 5 + [D("400")]})
    return table, months


def test_monthly_hand_calculation_strict_boundary_and_immutable_input():
    table, months = monthly_sample()
    before = table.copy(deep=True)
    details, baseline = screen_monthly_series(table, "net_revenue", months)
    assert baseline["mean"] == D("150")
    assert abs(baseline["standard_deviation"] ** 2 - D("12500")) < D("1E-20")
    assert baseline["flagged_months"] == ("2024-06",)
    pd.testing.assert_frame_equal(table, before)
    boundary = table.iloc[:5].copy()
    boundary["net_revenue"] = [D("100")] * 4 + [D("200")]
    _, result = screen_monthly_series(boundary, "net_revenue", months[:5])
    assert result["standard_deviation"] == D("40")
    assert result["upper_bound"] == D("200")
    assert result["flagged_months"] == ()


@pytest.mark.parametrize("case", ["missing_month", "missing_amount", "nonfinite", "no_activity"])
def test_monthly_incomplete_baseline_does_not_claim_no_exceptions(case):
    table, months = monthly_sample()
    if case == "missing_month":
        table = table.iloc[:-1]
    elif case in ("missing_amount", "nonfinite"):
        table.loc[0, "net_revenue"] = None if case == "missing_amount" else D("NaN")
    else:
        table["activity_status"] = "observed_records"
        table.loc[0, "activity_status"] = "no_observed_records"
    details, baseline = screen_monthly_series(table, "net_revenue", months)
    assert baseline["baseline_status"] == "incomplete_series"
    assert details["outside_band"].isna().all()
    assert baseline["mean"] is None


def test_constant_series_and_duplicate_months():
    table, months = monthly_sample()
    details, baseline = screen_monthly_series(table.assign(net_revenue=D("100")), "net_revenue", months)
    assert baseline["baseline_status"] == "constant_series"
    assert details["z_score"].isna().all()
    assert not any(details["outside_band"])
    with pytest.raises(SchemaError):
        screen_monthly_series(pd.concat([table, table.iloc[:1]]), "net_revenue", months)


def test_report_preserves_unknowns_policy_coverage_and_exact_excel_amounts(tmp_path):
    results = (
        RatioResult("debt_to_equity", D("0.5"), "times", "calculated"),
        RatioResult("gross_margin", D("0.1234567890123456789"), "fraction", "calculated"),
        RatioResult("accrual_ratio", None, "fraction", "unavailable", "missing_operating_cash_flow"),
    )
    policies = prepare_benchmark_policies(pd.DataFrame([
        ("debt_to_equity", "times", "lower", "2.5", "5", "0"),
    ], columns=["ratio_code", "unit", "direction", "good_cutoff", "risk_cutoff", "valid_minimum"]))
    tables = build_ratio_report_tables(results, policies)
    summary = dict(tables["Summary"].itertuples(index=False, name=None))
    assert summary["calculated_ratios"] == 2 and summary["unavailable_ratios"] == 1
    assert summary["assessed_ratios"] == 1 and summary["calculated_without_policy"] == 1
    assert summary["assessment_coverage"] == "partial" and len(tables["Review_Items"]) == 2
    path = tmp_path / "financial_analysis.xlsx"
    write_excel_report(path, tables)
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        rows = list(book["Ratios"].values)
        value_index = rows[0].index("value")
        assert rows[2][value_index] == "0.1234567890123456789"
        assert rows[3][value_index] is None
    finally:
        book.close()
    with pytest.raises(SchemaError):
        build_ratio_report_tables((results[0], results[0]), policies)


def test_financial_builder_rejects_stale_reporting_scope():
    tables = {name: pd.DataFrame() for name in (
        "Trial_Balance", "Balance_Sheet", "Income_Statement", "Monthly", "Checks",
    )}
    tables["Summary"] = pd.DataFrame([
        ("period_start", "2023-01-01"), ("period_end", "2023-12-31"),
    ], columns=["metric", "value"])
    with pytest.raises(SchemaError, match="periods"):
        build_financial_analysis_tables(pd.DataFrame(), tables,
            {"period_start": "2024-01-01", "period_end": "2024-12-31"}, {}, pd.DataFrame())


@pytest.fixture
def cash_ledger_factory(ledger_factory):
    def build(rows):
        return ledger_factory([
            {"source_side": None, "entity_code": None, **row} for row in rows
        ])
    return build


@pytest.mark.parametrize("counter,description,expected", [
    ("141", "Employee expense advance refund", "operating"),
    ("3383", "Insurance collected for staff", "operating"),
    ("2442", "Operating warehouse deposit refund", "operating"),
    ("51501", "Bank deposit interest payment", "investing"),
    ("1311", "Chuyển số dư sang mã khách hàng khác", "noncash_adjustment"),
    ("1388", "TAT TOAN HDTG deposit interest", "investing"),
    ("635", "USD = 0; currency valuation adjustment", "noncash_adjustment"),
    ("635", "Realized exchange difference on settlement", "operating"),
    ("9999", "Nature not identified", "unclassified"),
])
def test_cash_classification_uses_nature_and_keeps_unknowns(
    cash_ledger_factory, audit_settings, counter, description, expected,
):
    table = cash_ledger_factory([dict(counter_account=counter, description=description, debit="10")])
    original = table.copy(deep=True)
    result = review_cash_flows(table, audit_settings["financial_analysis"]["cash_flow_rules"])
    assert result.iloc[0]["cash_flow_category"] == expected
    pd.testing.assert_frame_equal(table, original)
    summary = summarize_cash_flow_estimate(result)
    if expected == "unclassified":
        assert summary["cfo_point_estimate"] is None
        assert summary["cfo_lower"] == 0 and summary["cfo_upper"] == 10


def test_cash_supplier_acquisition_is_separate_from_maintenance(cash_ledger_factory, audit_settings):
    table = cash_ledger_factory([
        dict(account="3311", counter_account="2111", entity_code="CAR", credit="100"),
        dict(account="1121", counter_account="3311", entity_code="CAR", credit="100",
             description="Payment KIA0278/2024/HDMB"),
        dict(account="1121", counter_account="3311", entity_code="CAR", credit="5",
             description="BAO DUONG SUA CHUA XE 51L-20963"),
        dict(account="1121", counter_account="3311", entity_code="CAR", credit="2",
             description="Unspecified supplier payment"),
    ])
    result = review_cash_flows(table, audit_settings["financial_analysis"]["cash_flow_rules"])
    assert result["cash_flow_category"].tolist() == ["investing", "operating", "unclassified"]
    assert result["source_row"].tolist() == [3, 4, 5]
    assert summarize_cash_flow_estimate(result)["cfo_point_estimate"] is None


def test_cash_opening_assumption_is_date_and_voucher_specific(cash_ledger_factory, audit_settings):
    table = cash_ledger_factory([
        dict(counter_account=None, voucher_no="OD23120102", posting_date=date(2024, 1, 1),
             description="đầu kỳ RMC tháng 1", debit="8"),
        dict(counter_account=None, voucher_no="OD23120102", posting_date=date(2024, 1, 2),
             description="đầu kỳ RMC tháng 1", debit="8"),
    ])
    result = review_cash_flows(table, audit_settings["financial_analysis"]["cash_flow_rules"])
    assert result["cash_flow_category"].tolist() == ["noncash_adjustment", "unclassified"]


def test_cash_point_estimate_retains_signs_and_reconciles_adjustments(cash_ledger_factory, audit_settings):
    table = cash_ledger_factory([
        dict(counter_account="1311", debit="100"),
        dict(counter_account="1311", credit="-2"),
        dict(counter_account="2111", credit="30"),
        dict(counter_account="34111", debit="50"),
        dict(counter_account="635", credit="4", description="USD = 0"),
        dict(counter_account="1121", debit="7"),
        dict(account="1121", counter_account="1111", credit="7"),
    ])
    cash = review_cash_flows(table, audit_settings["financial_analysis"]["cash_flow_rules"])
    summary = summarize_cash_flow_estimate(cash)
    assert summary["estimate_status"] == "point_estimate_provisional"
    assert summary["cfo_point_estimate"] == D("102")
    assert summary["noncash_adjustment_net"] == D("-4")
    assert summary["internal_transfer_net"] == 0
    assert summary["observed_cash_movement"] == D("118")
    assert sum((summary[name] for name in (
        "identified_operating_cash_flow", "investing_cash_flow_proxy",
        "financing_cash_flow_proxy", "internal_transfer_net", "noncash_adjustment_net",
        "unclassified_cash_net",
    )), D(0)) == summary["observed_cash_movement"]
    assert accrual_ratio(D("20"), summary["cfo_point_estimate"], D("300"), D("500")).value == D("-0.205")
    assert summarize_cash_flow_estimate(cash, population_complete=False)["cfo_point_estimate"] is None
