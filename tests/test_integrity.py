"""Check accounting invariants and the limits of reconciliation evidence."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from auditpy.utils import SchemaError
from auditpy.validators import (
    check_account_catalogue, check_basic_integrity, check_source_controls,
    check_voucher_balances, classify_ledger_rows, inspect_trial_balance,
    prepare_trial_balance_reference, reconcile_account_movements,
    reconcile_control_totals,
)

KEYS = ["source_file", "source_sheet", "posting_date", "voucher_no"]


def test_classification_keeps_opening_adjustments_and_unidentified_journals(ledger_factory):
    no_dates = {"voucher_no": None, "posting_date": None, "document_date": None}
    table = ledger_factory([
        dict(no_dates, description="Opening balance"),
        {"voucher_no": None, "description": "Opening adjustment", "debit": "100"},
        dict(no_dates, description="Total", debit="100"),
        dict(no_dates, description="Closing balance"),
        dict(no_dates, account=None, description=None,
             debit="=SUBTOTAL(9,K2:K9)", credit="=SUBTOTAL(9,L2:L9)"),
        {"account": None, "description": "Unidentified account"},
    ])
    before = table.copy(deep=True)
    classified = classify_ledger_rows(table, {
        "opening_balance": ["Opening balance"], "subtotal": ["Total"],
        "closing_balance": ["Closing balance"],
    })
    assert list(classified["row_type"]) == [
        "opening_balance", "journal_candidate", "subtotal",
        "closing_balance", "formula_control_candidate", "unclassified",
    ]
    assert len(classified) == len(table)
    pd.testing.assert_frame_equal(table, before)


def test_basic_checks_do_not_replace_missing_amounts_with_zero(ledger_factory):
    table = ledger_factory([
        {"debit": "100.10"}, {"credit": "100.10"},
        {"voucher_no": None, "debit": "-5", "document_date": date(2023, 12, 1)},
        {"account": None, "row_type": "unclassified"},
        {"debit": "", "posting_date": "31/02/2024"},
        {"posting_date": date(2023, 12, 31)},
    ])
    checked, counts, summary = check_basic_integrity(table, date(2024, 1, 1), date(2024, 12, 31))
    assert counts["negative_amount"] == 1
    assert counts["missing_debit"] == 1
    assert counts["invalid_posting_date"] == 1
    assert counts["posting_outside_period"] == 1
    assert counts["date_gap_exceeded"] == 1
    assert counts["zero_amount"] == 2
    assert summary["amounts_complete"] is False and summary["difference"] is None
    assert len(summary["missing_months"]) == 11
    assert int(checked["has_issue"].sum()) < sum(counts.values())
    _, _, balanced = check_basic_integrity(table.iloc[:2], date(2024, 1, 1), date(2024, 12, 31))
    assert balanced["total_debit"] == Decimal("100.10")
    assert balanced["difference"] == Decimal("0.00")


def test_date_gap_threshold_uses_strictly_more_than_configured_days(ledger_factory):
    table = ledger_factory([
        {"posting_date": date(2024, 2, 1), "document_date": date(2024, 1, 2)},
        {"posting_date": date(2024, 2, 2), "document_date": date(2024, 1, 2)},
    ])
    checked, _, _ = check_basic_integrity(table, date(2024, 1, 1), date(2024, 12, 31))
    assert list(checked["date_gap_exceeded"]) == [False, True]


def test_voucher_offsets_do_not_hide_group_differences_or_missing_keys(ledger_factory):
    table = ledger_factory([
        {"voucher_no": "A", "debit": "100"},
        {"voucher_no": "A", "credit": "90"},
        {"voucher_no": "B", "debit": "90"},
        {"voucher_no": "B", "credit": "100"},
        {"voucher_no": "A", "posting_date": date(2024, 1, 4), "debit": "50", "credit": "50"},
        {"voucher_no": None, "debit": "20", "credit": "20"},
    ])
    groups, unassigned = check_voucher_balances(table, KEYS)
    assert list(groups["difference"]) == [Decimal("10"), Decimal("-10"), Decimal("0")]
    assert len(unassigned) == 1 and unassigned.iloc[0]["source_row"] == 7
    assert int(groups["line_count"].sum()) + len(unassigned) == len(table)
    for side in ("debit", "credit"):
        assert sum(groups["total_" + side], Decimal("0")) + sum(unassigned[side], Decimal("0")) == sum(table[side], Decimal("0"))


def test_voucher_missing_amounts_and_missing_accounts_have_distinct_statuses(ledger_factory):
    table = ledger_factory([
        {"voucher_no": "A", "debit": None},
        {"voucher_no": "B", "account": None, "row_type": "unclassified"},
    ])
    groups, _ = check_voucher_balances(table, KEYS)
    by_voucher = groups.set_index("voucher_no")
    assert by_voucher.loc["A", "balance_status"] == "unavailable_amounts"
    assert by_voucher.loc["A", "total_credit"] is None
    assert by_voucher.loc["B", "balance_status"] == "balanced_requires_review"


def test_trial_balance_repeats_are_not_added_and_conflicts_are_not_selected(trial_balance_factory):
    table = trial_balance_factory([
        {"account": "111", "debit": "20"},
        {"account": "1111", "debit": "20"},
        {"account": "141", "debit": "30"},
        {"account": "141", "debit": "30"},
        {"account": "222", "debit": "5"},
        {"account": "222", "debit": "6"},
    ])
    marked = inspect_trial_balance(table)
    assert marked.loc[0, "parent_candidate"]
    reference = prepare_trial_balance_reference(table).set_index("account")
    assert reference.loc["141", "reference_status"] == "identical_repeat"
    assert reference.loc["141", "debit"] == Decimal("30")
    assert reference.loc["141", "source_rows"] == (6, 7)
    assert reference.loc["222", "reference_status"] == "conflict"
    assert reference.loc["222", "debit"] is None


def test_account_reconciliation_keeps_missing_vouchers_and_unallocated_rows(ledger_factory, trial_balance_factory):
    ledger = ledger_factory([
        {"account": "1111", "debit": "60", "credit": "10"},
        {"account": "1111", "voucher_no": None, "debit": "40"},
        {"account": "1112", "debit": "20", "credit": "5"},
        {"account": "9999", "debit": "7"},
        {"account": None, "debit": "3", "row_type": "unclassified"},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([
        {"account": "111", "debit": "120", "credit": "15"},
        {"account": "1111", "debit": "100", "credit": "10"},
        {"account": "1112", "debit": "20", "credit": "4"},
        {"account": "222"},
    ]))
    checks, unallocated = reconcile_account_movements(ledger, reference)
    by_account = checks.set_index("account")
    assert by_account.loc["111", "ledger_debit"] == Decimal("120")
    assert by_account.loc["1111", "ledger_debit"] == Decimal("100")
    assert by_account.loc["1112", "credit_difference"] == Decimal("1")
    assert by_account.loc["9999", "reconciliation_status"] == "missing_tb_reference"
    assert by_account.loc["222", "ledger_debit"] is None
    assert list(unallocated["source_row"]) == [6]


def test_net_balance_agreement_does_not_prove_both_balance_sides(ledger_factory, trial_balance_factory):
    ledger = ledger_factory([
        {"row_type": "opening_balance", "balance_debit": "100", "balance_credit": "40"},
        {"debit": "30"}, {"credit": "25"},
        {"row_type": "subtotal", "debit": "30", "credit": "25"},
        {"row_type": "closing_balance", "balance_debit": "90", "balance_credit": "25"},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([{
        "opening_debit": "100", "opening_credit": "40", "debit": "30", "credit": "25",
        "closing_debit": "80", "closing_credit": "15",
    }]))
    checks = check_source_controls(ledger, reference).set_index("check_name")
    assert checks.loc["ledger_net_rollforward", "check_status"] == "matched"
    assert checks.loc["closing_debit_vs_tb", "difference"] == Decimal("10")
    assert checks.loc["closing_credit_vs_tb", "difference"] == Decimal("10")
    duplicated = pd.concat([ledger, ledger.iloc[:1]], ignore_index=True)
    checks = check_source_controls(duplicated, reference).set_index("check_name")
    assert checks.loc["opening_debit_vs_tb", "check_status"] == "unavailable"


def test_declared_totals_use_non_overlapping_roots(ledger_factory, trial_balance_factory):
    ledger = ledger_factory([{"debit": "30", "credit": "25"}])
    values = {"opening_debit": "100", "opening_credit": "40", "debit": "30", "credit": "25",
              "closing_debit": "80", "closing_credit": "15"}
    client = trial_balance_factory([
        dict(values, account="111"), dict(values, account="1111"),
        dict(values, account="1111"), dict(values, account=None, account_name="Total"),
    ])
    reference = prepare_trial_balance_reference(client)
    checks, roots = reconcile_control_totals(ledger, reference, client, 7)
    assert roots == ("111",)
    assert checks["check_status"].eq("matched").all()
    assert len(checks) == 8
    with pytest.raises(SchemaError):
        reconcile_control_totals(ledger, reference, client, 4)
    client.loc[client["source_row"].eq(7), "debit_status"] = "missing"
    checks, _ = reconcile_control_totals(ledger, reference, client, 7)
    assert checks.set_index("check_name").loc["tb_roots_debit", "check_status"] == "unavailable"


def test_catalogue_membership_is_exact_and_counts_rows_once(ledger_factory):
    ledger = ledger_factory([
        {"account": "1111"},
        {"account": "1119", "counter_account": "1119"},
        {"account": None, "row_type": "unclassified"},
    ])
    checked, summary = check_account_catalogue(ledger, {"111", "1111", "1121"}, "toy_catalogue")
    assert list(checked["account_catalogue_status"]) == ["in_catalogue", "not_in_catalogue", "missing"]
    assert int(checked["has_unmatched_account"].sum()) == 1
    assert int(summary["not_in_catalogue"].sum()) == 2
    with pytest.raises(SchemaError):
        check_account_catalogue(ledger, set(), "empty_catalogue")


@pytest.mark.parametrize("actual,expected,tolerance,status,difference", [
    (None, "0", "0", "unavailable", None),
    ("0", None, "0", "unavailable", None),
    ("0", "0", "0", "matched", "0"),
    ("1", "1.01", "0.01", "matched", "-0.01"),
    ("1", "1.0100001", "0.01", "difference", "-0.0100001"),
    ("-3", "-3", "0", "matched", "0"),
])
def test_shared_comparison_preserves_missing_and_exact_tolerance(
    actual, expected, tolerance, status, difference,
):
    from auditpy.utils import compare_amounts
    actual = None if actual is None else Decimal(actual)
    expected = None if expected is None else Decimal(expected)
    result = compare_amounts("sample", actual, expected, Decimal(tolerance))
    assert result["actual"] == actual and result["expected"] == expected
    assert result["check_status"] == status
    assert result["difference"] == (None if difference is None else Decimal(difference))


def test_shared_comparison_rejects_negative_tolerance():
    from auditpy.utils import compare_amounts
    with pytest.raises(ValueError):
        compare_amounts("sample", Decimal("1"), Decimal("1"), Decimal("-1"))


@pytest.mark.parametrize("value,expected", [
    (Decimal("0"), True), (Decimal("-0.01"), True),
    (Decimal("NaN"), False), (Decimal("Infinity"), False),
    (None, False), (True, False), (1.0, False),
])
def test_shared_amount_guard_does_not_convert_values(value, expected):
    from auditpy.utils import is_finite_amount
    assert is_finite_amount(value) is expected
