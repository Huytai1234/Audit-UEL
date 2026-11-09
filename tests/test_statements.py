"""Check closing selection, statement arithmetic and monthly stock/flow treatment."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from auditpy.ledger import GeneralLedger, TrialBalance
from auditpy.statements import (
    BalanceSheet, FinancialStatement, IncomeStatement, build_monthly_summary,
    classify_closing_lines, map_balance_sheet_items, prepare_balance_sheet_balances,
)
from auditpy.utils import SchemaError
from auditpy.validators import prepare_trial_balance_reference

KEY = ["source_file", "source_sheet", "posting_date", "voucher_no"]


def tagged(ledger_factory, account_map, rows):
    ledger = GeneralLedger.from_classified_table(ledger_factory(rows))
    entries, unassigned = ledger.group_entries(KEY)
    return classify_closing_lines(entries, unassigned, account_map)


def test_mixed_voucher_closes_only_the_identified_lines(ledger_factory, account_map):
    rows = [
        {"account": "1311", "counter_account": "511", "debit": "100"},
        {"account": "511", "counter_account": "1311", "credit": "100"},
        {"account": "632", "counter_account": "156", "debit": "60", "description": "Closing costs"},
        {"account": "156", "counter_account": "632", "credit": "60"},
        {"account": "511", "counter_account": "911", "debit": "100"},
        {"account": "911", "counter_account": "511", "credit": "100"},
        {"account": "911", "counter_account": "632", "debit": "60"},
        {"account": "632", "counter_account": "911", "credit": "60"},
        {"account": "911", "counter_account": "4212", "debit": "40"},
        {"account": "4212", "counter_account": "911", "credit": "40"},
    ]
    lines = tagged(ledger_factory, account_map, rows)
    assert lines["exclude_from_income_statement"].dtype == bool
    assert lines["exclude_from_income_statement"].sum() == 6
    assert lines["reciprocal_line_found"].eq(True).sum() == 6
    result = IncomeStatement(lines, account_map).build().set_index("item")
    assert result.loc["net_revenue", "amount"] == Decimal("100")
    assert result.loc["cost_of_sales", "amount"] == Decimal("60")
    assert result.loc["profit_after_tax", "amount"] == Decimal("40")


@pytest.mark.parametrize("case", ["missing_key", "direct_equity", "negative", "unmapped", "missing_counter"])
def test_uncertain_closing_evidence_is_retained(ledger_factory, account_map, case):
    row = {"account": "511", "counter_account": "911", "debit": "100"}
    if case == "missing_key":
        row["voucher_no"] = None
    elif case == "direct_equity":
        row["counter_account"] = "4211"
    elif case == "negative":
        row["debit"] = "-100"
    elif case == "unmapped":
        row["account"] = "621"
    else:
        row["counter_account"] = None
    result = tagged(ledger_factory, account_map, [row]).iloc[0]
    assert result["closing_review_required"]
    assert bool(result["exclude_from_income_statement"]) == (case in ("missing_key", "negative"))
    assert result["debit"] == Decimal(row["debit"])
    if case == "missing_key":
        assert result["grouping_status"] == "unassigned"
        assert result["reciprocal_line_found"] is None


def test_income_formulas_missing_values_and_net_deduction_transfers(ledger_factory, account_map):
    rows = [
        {"account": "511", "credit": "100"}, {"account": "632", "debit": "60"},
        {"account": "515", "credit": "2"}, {"account": "635", "debit": "1"},
        {"account": "641", "debit": "8"}, {"account": "642", "debit": "5"},
        {"account": "711", "credit": "3"}, {"account": "811", "debit": "1"},
        {"account": "821", "debit": "6"},
    ]
    lines = tagged(ledger_factory, account_map, rows)
    report = IncomeStatement(lines, account_map).build().set_index("item")
    assert report.loc["gross_profit", "amount"] == Decimal("40")
    assert report.loc["operating_profit", "amount"] == Decimal("28")
    assert report.loc["other_profit", "amount"] == Decimal("2")
    assert report.loc["profit_after_tax", "amount"] == Decimal("24")
    missing = lines.copy()
    missing.loc[missing["account"].eq("632"), "debit"] = None
    incomplete = IncomeStatement(missing, account_map).build().set_index("item")
    assert incomplete.loc["profit_after_tax", "amount"] is None
    assert incomplete.loc["net_revenue", "amount"] == Decimal("100")
    deduction_rows = rows + [
        {"account": "521", "debit": "10"},
        {"account": "511", "debit": "10"}, {"account": "521", "credit": "10"},
    ]
    net = IncomeStatement(tagged(ledger_factory, account_map, deduction_rows), account_map).build().set_index("item")
    assert net.loc["net_revenue", "amount"] == Decimal("90")
    with pytest.raises(TypeError):
        FinancialStatement()
    with pytest.raises(SchemaError):
        IncomeStatement(tagged(ledger_factory, account_map, [{"account": "699", "debit": "1"}]), account_map)


def sample_balances():
    rows = [
        ["1111", "100", "0"], ["1311", "100", "30"], ["1561", "80", "0"],
        ["2294", "0", "10"], ["2111", "200", "0"], ["2141", "0", "50"],
        ["41111", "0", "390"],
    ]
    return pd.DataFrame([
        {"account": code, "closing_debit": Decimal(debit), "closing_credit": Decimal(credit)}
        for code, debit, credit in rows
    ], dtype=object)


def test_balance_sheet_contra_assets_maturity_gaps_and_missing_money(account_map):
    account_map = {key: value for key, value in account_map.items() if key != "balance_sheet_maturity"}
    items = map_balance_sheet_items(sample_balances(), account_map)
    statement = BalanceSheet(items)
    report = statement.build().set_index("item")
    assert report.loc["inventories", "amount"] == Decimal("70")
    assert report.loc["net_fixed_assets", "amount"] == Decimal("150")
    assert report.loc["total_assets", "amount"] == Decimal("420")
    assert report.loc["total_funding", "amount"] == Decimal("420")
    assert report.loc["current_liabilities", "amount"] is None
    assert report.loc["noncurrent_liabilities", "amount"] is None
    assert report.loc["current_liabilities", "mapped_amount"] == Decimal("0")
    assert statement.balance_check["within_tolerance"] is True
    missing = items.copy()
    missing.loc[missing["account"].eq("1111") & missing["balance_side"].eq("closing_debit"), "amount"] = None
    result = BalanceSheet(missing).build().set_index("item")
    assert result.loc["total_assets", "amount"] is None
    assert result.loc["balance_difference", "amount"] is None
    with pytest.raises(SchemaError):
        BalanceSheet(pd.concat([items, items.iloc[[0]]]))


def test_specific_mapping_and_tolerance_do_not_change_source_values(account_map):
    balances = pd.DataFrame([
        {"account": "33319", "closing_debit": Decimal("4"), "closing_credit": Decimal("0")},
        {"account": "12811", "closing_debit": Decimal("7"), "closing_credit": Decimal("0")},
    ])
    items = map_balance_sheet_items(balances, account_map)
    assert set(items.loc[items["account"].eq("33319"), "mapping_prefix"]) == {"33319"}
    assert set(items.loc[items["account"].eq("12811"), "item"]) == {"other_current_assets"}
    tiny_gap = pd.DataFrame([
        {"item": "cash", "amount": Decimal("100")},
        {"item": "equity", "amount": Decimal("100.035")},
    ])
    strict = BalanceSheet(tiny_gap)
    rounded = BalanceSheet(tiny_gap, balance_tolerance=Decimal("1"))
    assert strict.balance_check["within_tolerance"] is False
    assert rounded.balance_check["within_tolerance"] is True
    assert strict.balance_check["difference"] == rounded.balance_check["difference"] == Decimal("-0.035")
    with pytest.raises(SchemaError):
        map_balance_sheet_items(balances.assign(account=["199", "12811"]), account_map)


def test_maturity_assumption_preserves_separate_sides_and_total_balances(account_map):
    balances = sample_balances()
    before = balances.copy(deep=True)
    strict_map = {key: value for key, value in account_map.items() if key != "balance_sheet_maturity"}
    strict = BalanceSheet(map_balance_sheet_items(balances, strict_map)).build().set_index("item")
    details = map_balance_sheet_items(balances, account_map)
    report = BalanceSheet(details).build().set_index("item")
    assert report.loc["current_assets", "amount"] == Decimal("270")
    assert report.loc["noncurrent_assets", "amount"] == Decimal("150")
    assert report.loc["current_liabilities", "amount"] == Decimal("30")
    assert report.loc["noncurrent_liabilities", "amount"] == Decimal("0")
    # The zero noncurrent side is also conditional on the allocation assumption.
    assert report.loc["noncurrent_liabilities", "calculation_basis"] == "includes_disclosed_maturity_assumptions"
    for item in ("total_assets", "total_liabilities", "equity", "total_funding", "balance_difference"):
        assert report.loc[item, "amount"] == strict.loc[item, "amount"]
    ar = details.loc[details["account"].eq("1311")]
    assert ar["amount"].tolist() == [Decimal("100"), Decimal("30")]
    assert ar["item"].tolist() == ["receivables", "current_liabilities"]
    pd.testing.assert_frame_equal(balances, before)


def test_maturity_longest_prefix_splits_receivables_and_related_allowance(account_map):
    mapping = dict(account_map, balance_sheet_maturity={
        "131": "current", "1312": "noncurrent", "2293": "noncurrent",
    })
    balances = pd.DataFrame([
        {"account": code, "closing_debit": Decimal(debit), "closing_credit": Decimal(credit)}
        for code, debit, credit in (("1311", "100", "0"), ("1312", "80", "20"), ("2293", "0", "10"))
    ])
    details = map_balance_sheet_items(balances, mapping)
    report = BalanceSheet(details).build().set_index("item")
    assert report.loc["receivables", "amount"] == Decimal("170")
    assert report.loc["current_assets", "amount"] == Decimal("100")
    assert report.loc["noncurrent_assets", "amount"] == Decimal("70")
    assert report.loc["noncurrent_liabilities", "amount"] == Decimal("20")
    assert set(details.loc[details["account"].eq("1312"), "maturity_prefix"]) == {"1312"}


def test_partial_maturity_mapping_does_not_fill_other_unknowns(account_map):
    mapping = dict(account_map, balance_sheet_maturity={"1311": "current"})
    balances = pd.DataFrame([
        {"account": "1312", "closing_debit": Decimal("1"), "closing_credit": Decimal("2")},
    ])
    report = BalanceSheet(map_balance_sheet_items(balances, mapping)).build().set_index("item")
    assert report.loc["current_assets", "amount"] is None
    assert report.loc["current_liabilities", "amount"] is None
    assert report.loc["total_assets", "amount"] == Decimal("1")
    assert report.loc["total_liabilities", "amount"] == Decimal("2")


def test_maturity_classification_never_replaces_missing_money(account_map):
    balances = sample_balances()
    balances.loc[balances["account"].eq("1311"), "closing_debit"] = None
    report = BalanceSheet(map_balance_sheet_items(balances, account_map)).build().set_index("item")
    assert report.loc["current_assets", "amount"] is None
    assert report.loc["total_assets", "amount"] is None
    assert report.loc["current_liabilities", "amount"] == Decimal("30")


@pytest.mark.parametrize("rules", [{"131": "short"}, {131: "current"}, {"cash": "current"}, {"111": "noncurrent"}])
def test_invalid_maturity_rules_are_rejected(account_map, rules):
    with pytest.raises(SchemaError):
        map_balance_sheet_items(sample_balances(), dict(account_map, balance_sheet_maturity=rules))


def test_balance_reference_does_not_invent_optional_source_fields(
    ledger_factory, trial_balance_factory,
):
    classified = ledger_factory([{"account": "1111"}]).drop(columns=[
        "balance_debit", "balance_credit", "balance_debit_status", "balance_credit_status",
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([
        {"account": "1111", "opening_debit": "100", "closing_debit": "100"},
    ]))
    trial = TrialBalance.from_general_ledger(GeneralLedger.from_classified_table(classified), reference)
    result = prepare_balance_sheet_balances(classified, reference, trial)
    assert result.iloc[0]["closing_debit"] == Decimal("100")
    assert result.iloc[0]["entity_count"] is None
    assert result.iloc[0]["entity_opening_rows"] is None
    conflict = reference.copy()
    conflict.loc[:, "reference_status"] = "conflict"
    assert prepare_balance_sheet_balances(classified, conflict, trial).iloc[0]["closing_debit"] is None


def test_monthly_balances_are_stocks_and_missing_values_propagate(
    ledger_factory, trial_balance_factory, account_map,
):
    classified = ledger_factory([
        {"account": "1111", "debit": "1000", "voucher_no": "1"},
        {"account": "1311", "credit": "1000", "voucher_no": "1"},
        {"account": "1121", "debit": "300", "voucher_no": "2"},
        {"account": "1111", "credit": "300", "voucher_no": "2"},
        {"account": "1121", "debit": "-50", "voucher_no": "3", "posting_date": date(2024, 2, 3)},
        {"account": "1111", "credit": "-50", "voucher_no": "3", "posting_date": date(2024, 2, 3)},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([
        {"account": "1111", "opening_debit": "200"}, {"account": "1121"},
        {"account": "1311", "opening_credit": "200"},
    ]))
    ledger = GeneralLedger.from_classified_table(classified)
    trial = TrialBalance.from_general_ledger(
        ledger, reference, period_start=date(2024, 1, 1), period_end=date(2024, 3, 31),
    )
    entries, unassigned = ledger.group_entries(KEY)
    lines = classify_closing_lines(entries, unassigned, account_map)
    monthly, checks = build_monthly_summary(lines, trial, account_map, cash_closing_reference=Decimal("1200"))
    assert monthly["closing_cash"].tolist() == [Decimal("1200")] * 3
    assert monthly.loc[1, "cash_debit"] == monthly.loc[1, "cash_credit"] == Decimal("-50")
    assert monthly.loc[2, "record_count"] == 0
    assert monthly.loc[2, "activity_status"] == "no_observed_records"
    assert checks["check_status"].eq("matched").all()
    missing = lines.copy()
    missing.loc[missing["account"].eq("1111") & missing["debit"].gt(0), "debit"] = None
    unknown, checks = build_monthly_summary(missing, trial, account_map, cash_closing_reference=Decimal("1200"))
    assert unknown["closing_cash"].isna().all()
    assert checks.loc[checks["check_name"].eq("period_end_cash"), "check_status"].iloc[0] == "unavailable"
    wrong_dates = lines.copy()
    wrong_dates.loc[0, "posting_date"] = date(2025, 1, 1)
    with pytest.raises(SchemaError):
        build_monthly_summary(wrong_dates, trial, account_map)


def test_monthly_income_does_not_require_noncash_opening_balances(
    ledger_factory, trial_balance_factory, account_map,
):
    classified = ledger_factory([
        {"account": "1111"},
        {"account": "511", "credit": "10"},
        {"account": "632", "debit": "3"},
        {"account": "632", "debit": "5", "posting_date": date(2024, 2, 3)},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([
        {"account": "1111", "opening_debit": "100", "closing_debit": "100"},
    ]))
    ledger = GeneralLedger.from_classified_table(classified)
    trial = TrialBalance.from_general_ledger(
        ledger, reference, period_start=date(2024, 1, 1), period_end=date(2024, 2, 29),
    )
    assert trial.table.set_index("account").loc["632", "opening_debit"] is None
    entries, unassigned = ledger.group_entries(KEY)
    lines = classify_closing_lines(entries, unassigned, account_map)
    monthly, _ = build_monthly_summary(lines, trial, account_map, cash_closing_reference=Decimal("100"))
    assert monthly["profit_after_tax"].tolist() == [Decimal("7"), Decimal("-5")]
    assert monthly["closing_cash"].tolist() == [Decimal("100"), Decimal("100")]
    lines.loc[lines["posting_date"].eq(date(2024, 2, 3)), "debit"] = None
    missing, checks = build_monthly_summary(lines, trial, account_map)
    assert missing.loc[0, "profit_after_tax"] == Decimal("7")
    assert missing.loc[1, "profit_after_tax"] is None
    assert missing["closing_cash"].notna().all()
    assert checks.loc[checks["check_name"].eq("profit_after_tax"), "check_status"].iloc[0] == "unavailable"
