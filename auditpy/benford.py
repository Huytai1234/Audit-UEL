"""Select traceable positive posting amounts and compare first-digit frequencies."""

from collections import Counter
from decimal import Decimal
from math import log10

import pandas as pd

from auditpy.utils import SchemaError, is_missing, is_finite_amount


BENFORD_POPULATION_SIDES = {
    "all_ledger_debit": "debit",
    "all_ledger_credit": "credit",
    "revenue": "credit",
    "expenses": "debit",
}


def first_significant_digit(amount):
    """Return the first significant digit of a positive finite Decimal."""
    if not is_finite_amount(amount) or amount <= 0:
        return None
    return amount.as_tuple().digits[0]


def select_benford_candidates(closing_lines, amount_column, account_prefixes=()):
    """Select one declared posting side; keep repeated amounts and source references."""
    required = {
        "account", "voucher_no", "source_file", "source_sheet", "source_row", "source_side",
        "exclude_from_income_statement", "closing_review_required", amount_column,
    }
    if amount_column not in ("debit", "credit") or not required.issubset(closing_lines.columns):
        raise SchemaError("Benford requires a debit/credit column and classified closing-line evidence.")
    source_key = ["source_file", "source_sheet", "source_row", "source_side"]
    if closing_lines.duplicated(source_key).any():
        raise SchemaError("Benford source keys must identify normalized records uniquely.")
    for column in ("exclude_from_income_statement", "closing_review_required"):
        if closing_lines[column].isna().any() or not closing_lines[column].isin([True, False]).all():
            raise SchemaError("Closing flags must be present and boolean.")
    prefixes = tuple(account_prefixes)
    if any(not isinstance(prefix, str) or not prefix.strip() for prefix in prefixes):
        raise ValueError("Account prefixes must be non-empty strings.")
    reasons = (
        "missing_account", "outside_account_scope", "identified_closing",
        "unavailable_amount", "zero_amount", "negative_amount", "selected",
    )
    counts = Counter({reason: 0 for reason in reasons})
    rows = []
    for record in closing_lines.to_dict("records"):
        account, amount = record["account"], record[amount_column]
        if is_missing(account):
            reason = "missing_account"
        elif prefixes and not account.startswith(prefixes):
            reason = "outside_account_scope"
        elif record["exclude_from_income_statement"]:
            reason = "identified_closing"
        elif not is_finite_amount(amount):
            reason = "unavailable_amount"
        elif amount == 0:
            reason = "zero_amount"
        elif amount < 0:
            reason = "negative_amount"
        else:
            reason = "selected"
        counts[reason] += 1
        if reason == "selected":
            record.update(
                benford_amount=amount,
                first_digit=first_significant_digit(amount),
                amount_column=amount_column,
            )
            rows.append(record)
    selected = pd.DataFrame(
        rows,
        columns=list(closing_lines.columns) + ["benford_amount", "first_digit", "amount_column"],
        dtype=object,
    )
    coverage = dict(
        review_records=len(closing_lines), amount_column=amount_column,
        account_prefixes=prefixes,
        selected_closing_review_records=sum(bool(row["closing_review_required"]) for row in rows),
        selected_without_voucher=sum(is_missing(row["voucher_no"]) for row in rows),
    )
    coverage.update(counts)
    return selected, coverage


def benford_frequency(amounts):
    """Return all nine digits, even when a digit has no observed occurrences."""
    digits = [first_significant_digit(amount) for amount in amounts]
    if any(digit is None for digit in digits):
        raise ValueError("Frequency inputs must be positive finite Decimal amounts.")
    counts, size = Counter(digits), len(digits)
    return pd.DataFrame([
        {
            "digit": digit,
            "observed_count": counts[digit],
            "observed_rate": counts[digit] / size if size else None,
            "expected_rate": log10(1 + 1 / digit),
            "expected_count": size * log10(1 + 1 / digit),
        }
        for digit in range(1, 10)
    ]).astype({"observed_rate": float})

def classify_mad(mad):
    """Apply the assignment's first-digit MAD bands without overlapping bounds."""
    if mad is None:
        return "unavailable"

    if isinstance(mad, bool) or not isinstance(mad, (int, float, Decimal)):
        raise ValueError("MAD must be a finite number between zero and one.")

    value = Decimal(str(mad))
    if not value.is_finite() or not 0 <= value <= 1:
        raise ValueError("MAD must be a finite number between zero and one.")

    if value < Decimal("0.006"):
        return "close_conformity"
    if value < Decimal("0.012"):
        return "acceptable_conformity"
    if value <= Decimal("0.015"):
        return "marginal_conformity"
    return "nonconformity"


def benford_statistics(amounts):
    """Calculate descriptive statistics and a conditional chi-square screen."""
    frequency = benford_frequency(amounts)
    size = int(frequency["observed_count"].sum())

    chi_square = mad = None
    screen = "empty_population"
    minimum_expected = float(frequency["expected_count"].min())
    critical_value = 15.507  # NIST: 8 degrees of freedom, upper-tail alpha 0.05.

    if size:
        observed = frequency["observed_count"]
        expected = frequency["expected_count"]

        chi_square = float((((observed - expected) ** 2) / expected).sum())
        mad = float(
            (frequency["observed_rate"] - frequency["expected_rate"]).abs().mean()
        )

        if minimum_expected < 5:
            screen = "small_expected_counts"
        elif chi_square > critical_value:
            screen = "above_critical"
        else:
            screen = "not_above_critical"

    return {
        "observation_count": size,
        "chi_square": chi_square,
        "degrees_of_freedom": 8,
        "alpha": 0.05,
        "critical_value": critical_value,
        "minimum_expected_count": minimum_expected,
        "chi_square_screen": screen,
        "mad": mad,
        "mad_band": classify_mad(mad),
        "interpretation_basis": (
            "Conditional screen; independence and Benford suitability are unverified."
        ),
    }
