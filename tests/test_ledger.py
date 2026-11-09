"""Verify immutable records, ledger ordering, scopes and source-backed balances."""

from dataclasses import FrozenInstanceError, replace
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from auditpy.ledger import GeneralLedger, TrialBalance
from auditpy.models import Account, JournalEntry
from auditpy.utils import SchemaError
from auditpy.validators import prepare_opening_balance_reference, prepare_trial_balance_reference


def test_frozen_lines_keep_codes_and_validate_entry_membership(ledger_factory):
    ledger = GeneralLedger.from_classified_table(ledger_factory([{"account": "00111"}]))
    assert ledger[0].account == "00111"
    with pytest.raises(FrozenInstanceError):
        ledger[0].debit = Decimal("1")
    with pytest.raises(ValueError):
        replace(ledger[0], debit=1.0)
    with pytest.raises(ValueError):
        JournalEntry("other.csv", None, date(2024, 1, 3), "0001", (ledger[0],))
    with pytest.raises(ValueError):
        Account(" 1111 ")


def test_running_balance_is_stable_signed_and_missing_aware(ledger_factory):
    table = ledger_factory([
        {"posting_date": date(2024, 1, 2), "debit": "3"},
        {"posting_date": date(2024, 1, 1), "debit": "10"},
        {"posting_date": date(2024, 1, 1), "credit": "-2"},
        {"posting_date": date(2024, 1, 3), "debit": None},
    ])
    ledger = GeneralLedger.from_classified_table(table)
    detail = ledger.account_ledger(Account("1111"), Decimal("100"), Decimal("0"))
    assert detail["source_row"].tolist() == [3, 4, 2, 5]
    assert detail["running_net_balance"].tolist() == [Decimal("110"), Decimal("112"), Decimal("115"), None]
    assert ledger.account_ledger(Account("1111"), None, Decimal("0"))["running_net_balance"].isna().all()
    assert ledger.account_ledger(Account("9999"), Decimal("0"), Decimal("0")).empty
    invalid = GeneralLedger([replace(ledger[0], posting_date=None)])
    with pytest.raises(SchemaError):
        invalid.account_ledger(Account("1111"), Decimal("0"), Decimal("0"))


def test_trial_balance_keeps_inactive_accounts_and_unallocated_records(
    ledger_factory, trial_balance_factory,
):
    table = ledger_factory([
        {"debit": "10"},
        {"account": "8211"},
        {"account": None, "row_type": "unclassified"},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([
        {"account": "111", "opening_debit": "5"},
        {"account": "1111", "opening_debit": "5"},
        {"account": "2111", "opening_debit": "100"},
    ]))
    ledger = GeneralLedger.from_classified_table(table)
    trial = TrialBalance.from_general_ledger(ledger, reference)
    result = trial.table.set_index("account")
    assert set(result.index) == {"1111", "2111", "8211"}
    assert result.loc["1111", "closing_net"] == Decimal("15")
    assert result.loc["2111", "closing_net"] == Decimal("100")
    assert result.loc["8211", "opening_debit"] is None
    assert result.loc["8211", "closing_net"] is None
    exported = ledger.to_table(trial, reference)
    assert len(exported) == 3
    unallocated = exported.loc[exported["account"].isna()]
    assert unallocated["source_row"].tolist() == [4]
    assert unallocated["balance_status"].tolist() == ["unallocated_account"]
    assert unallocated["running_net_balance"].isna().all()
    copy = trial.table
    copy.loc[:, "debit"] = None
    assert trial.total_debit == Decimal("10")


def test_entity_movements_keep_missing_keys_and_signed_activity(ledger_factory):
    ledger = GeneralLedger.from_classified_table(ledger_factory([
        {"account": "1311", "entity_code": "A", "debit": "100"},
        {"account": "1311", "entity_code": "A", "credit": "20"},
        {"account": "1311", "entity_code": None, "debit": "-5", "voucher_no": None},
        {"account": "1111", "entity_code": "A", "debit": "999"},
    ]))
    result = ledger.entity_movements(["1311"])
    known = result.loc[result["entity_code"].eq("A")].iloc[0]
    unknown = result.loc[result["entity_code"].isna()].iloc[0]
    assert known["net_movement"] == Decimal("80")
    assert known["source_rows"] == (2, 3)
    assert unknown["debit"] == Decimal("-5")
    assert unknown["missing_voucher_rows"] == 1
    assert sum(result["line_count"]) == 3
    assert ledger.entity_movements([]).empty


def test_opening_fallback_requires_explicit_unambiguous_evidence(
    ledger_factory, trial_balance_factory,
):
    controls = ledger_factory([
        {"account": "8211", "row_type": "opening_balance", "balance_debit": "0", "balance_credit": "0"},
    ])
    reference = prepare_trial_balance_reference(trial_balance_factory([{"account": "1111"}]))
    selected = prepare_opening_balance_reference(controls, reference).set_index("account")
    assert selected.loc["8211", "opening_debit"] == Decimal("0")
    assert selected.loc["8211", "opening_basis"] == "ledger_opening"
    duplicated = prepare_opening_balance_reference(pd.concat([controls, controls]), reference).set_index("account")
    assert duplicated.loc["8211", "opening_debit"] is None
    minimal = controls.drop(columns=["balance_debit", "balance_credit", "balance_debit_status", "balance_credit_status"])
    unavailable = prepare_opening_balance_reference(minimal, reference).set_index("account")
    assert unavailable.loc["8211", "opening_debit"] is None
    conflicting = prepare_trial_balance_reference(trial_balance_factory([{"account": "8211"}]))
    conflicting.loc[:, "reference_status"] = "conflict"
    blocked = prepare_opening_balance_reference(controls, conflicting).set_index("account")
    assert blocked.loc["8211", "opening_debit"] is None


@pytest.mark.parametrize("change", ["period", "unit", "basis", "overlap", "parent", "unknown_unit"])
def test_trial_addition_rejects_incompatible_scopes(change):
    scope = dict(period_start=date(2024, 1, 1), period_end=date(2024, 12, 31), unit="sample_units")
    table = pd.DataFrame([{"account": "1111", "debit": Decimal("10"), "credit": Decimal("0")}])
    other_table = table.assign(account="1121", debit=Decimal("0"), credit=Decimal("10"))
    left = TrialBalance(table, **scope)
    right_scope = scope.copy()
    if change == "period":
        right_scope.update(period_start=date(2025, 1, 1), period_end=date(2025, 12, 31))
    elif change == "unit":
        right_scope["unit"] = "other_units"
    elif change == "basis":
        right_scope["basis"] = "other_basis"
    elif change == "overlap":
        other_table = other_table.assign(account="1111")
    elif change == "parent":
        other_table = other_table.assign(account="111")
    else:
        right_scope["unit"] = None
        left = TrialBalance(table, **right_scope)
    with pytest.raises((ValueError, SchemaError)):
        left + TrialBalance(other_table, **right_scope)


def test_trial_addition_is_new_and_preserves_disjoint_amounts():
    scope = dict(period_start=date(2024, 1, 1), period_end=date(2024, 12, 31), unit="sample_units")
    left = TrialBalance(pd.DataFrame([{"account": "1111", "debit": Decimal("10"), "credit": Decimal("0")}]), **scope)
    right = TrialBalance(pd.DataFrame([{"account": "1121", "debit": Decimal("0"), "credit": Decimal("10")}]), **scope)
    combined = left + right
    assert combined.total_debit == combined.total_credit == Decimal("10")
    assert len(left.table) == len(right.table) == 1
    with pytest.raises(TypeError):
        left + 1
