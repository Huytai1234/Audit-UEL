"""Calculate financial ratios using explicit inputs and a simple registry."""

from decimal import Decimal
from .models import RatioResult
from .ledger import attach_source_classifications, review_cash_flows, summarize_cash_flow_estimate
from unicodedata import normalize
from dataclasses import asdict

import pandas as pd

from .utils import SchemaError, normalize_amount, normalize_code, is_finite_amount


_RATIO_REGISTRY = {}


def register_ratio(function):
    """Register a ratio function once and leave its calculation unchanged."""
    code = function.__name__
    if code in _RATIO_REGISTRY:
        raise ValueError("Duplicate ratio code: " + code)

    _RATIO_REGISTRY[code] = function
    return function


def get_ratio_registry():
    """Return a copy so callers cannot remove registered functions."""
    return _RATIO_REGISTRY.copy()


# Ratio formulas

def _input_issue(inputs, nonnegative=False):
    """Return the first missing, invalid or disallowed negative input name."""
    for name, value in inputs.items():
        if value is None:
            return "missing_" + name
        if not is_finite_amount(value):
            return "invalid_" + name
        if nonnegative and value < 0:
            return "negative_" + name
    return None


def _divide(code, numerator, denominator, unit):
    reason = _input_issue(dict(numerator=numerator, denominator=denominator))
    if reason is None and denominator <= 0:
        reason = "non_positive_denominator"
    if reason:
        return RatioResult(code, None, unit, "unavailable", reason)
    return RatioResult(code, numerator / denominator, unit, "calculated")


def _ratio_on_average(code, numerator, opening, closing, unit):
    reason = _input_issue(dict(opening_balance=opening, closing_balance=closing), True)
    if reason:
        return RatioResult(code, None, unit, "unavailable", reason)
    average = (opening + closing) / Decimal("2")
    return _divide(code, numerator, average, unit)


@register_ratio
def current_ratio(current_assets, current_liabilities):
    return _divide("current_ratio", current_assets, current_liabilities, "times")


@register_ratio
def gross_margin(gross_profit, net_revenue):
    return _divide("gross_margin", gross_profit, net_revenue, "fraction")


@register_ratio
def net_margin(profit_after_tax, net_revenue):
    return _divide("net_margin", profit_after_tax, net_revenue, "fraction")


@register_ratio
def quick_ratio(current_assets, inventories, current_liabilities):
    reason = _input_issue(dict(current_assets=current_assets, inventories=inventories))
    if reason:
        return RatioResult("quick_ratio", None, "times", "unavailable", reason)
    return _divide("quick_ratio", current_assets - inventories, current_liabilities, "times")


@register_ratio
def cash_ratio(cash_and_cash_equivalents, current_liabilities):
    return _divide("cash_ratio", cash_and_cash_equivalents, current_liabilities, "times")


@register_ratio
def debt_to_assets(total_liabilities, total_assets):
    return _divide("debt_to_assets", total_liabilities, total_assets, "fraction")


@register_ratio
def debt_to_equity(total_liabilities, equity):
    return _divide("debt_to_equity", total_liabilities, equity, "times")


@register_ratio
def interest_coverage(ebit, interest_expense):
    return _divide("interest_coverage", ebit, interest_expense, "times")


@register_ratio
def asset_turnover(net_revenue, opening_assets, closing_assets):
    return _ratio_on_average("asset_turnover", net_revenue, opening_assets, closing_assets, "times")


@register_ratio
def return_on_assets(profit_after_tax, opening_assets, closing_assets):
    return _ratio_on_average(
        "return_on_assets", profit_after_tax, opening_assets, closing_assets, "fraction",
    )


@register_ratio
def return_on_equity(profit_after_tax, opening_equity, closing_equity):
    return _ratio_on_average(
        "return_on_equity", profit_after_tax, opening_equity, closing_equity, "fraction",
    )


@register_ratio
def receivables_turnover(net_revenue, opening_receivables, closing_receivables):
    return _ratio_on_average(
        "receivables_turnover", net_revenue, opening_receivables, closing_receivables, "times",
    )


@register_ratio
def inventory_turnover(cost_of_sales, opening_inventory, closing_inventory):
    return _ratio_on_average(
        "inventory_turnover", cost_of_sales, opening_inventory, closing_inventory, "times",
    )


def _turnover_days(code, turnover, days):
    if type(days) is not int or days <= 0:
        raise ValueError("The day-count convention must be a positive integer.")
    return _divide(code, Decimal(days), turnover, "days")


@register_ratio
def days_sales_outstanding(turnover, days=365):
    return _turnover_days("days_sales_outstanding", turnover, days)


@register_ratio
def days_inventory_outstanding(turnover, days=365):
    return _turnover_days("days_inventory_outstanding", turnover, days)


@register_ratio
def days_payables_outstanding(cost_of_sales, opening_payables, closing_payables, days=365):
    code = "days_payables_outstanding"
    if type(days) is not int or days <= 0:
        raise ValueError("The day-count convention must be a positive integer.")
    reason = _input_issue(dict(opening_payables=opening_payables, closing_payables=closing_payables), True)
    if reason:
        return RatioResult(code, None, "days", "unavailable", reason)
    average = (opening_payables + closing_payables) / Decimal("2")
    return _divide(code, average * Decimal(days), cost_of_sales, "days")


@register_ratio
def cash_conversion_cycle(dso, dio, dpo):
    code = "cash_conversion_cycle"
    reason = _input_issue(dict(dso=dso, dio=dio, dpo=dpo), True)
    if reason:
        return RatioResult(code, None, "days", "unavailable", reason)
    return RatioResult(code, dso + dio - dpo, "days", "calculated")


@register_ratio
def altman_z_double_prime(
    working_capital, retained_earnings, ebit, book_equity, total_assets, total_liabilities,
):
    """Use the assignment's four-variable model, with no additive constant."""
    code = "altman_z_double_prime"
    inputs = dict(working_capital=working_capital, retained_earnings=retained_earnings,
                  ebit=ebit, book_equity=book_equity,
                  total_assets=total_assets, total_liabilities=total_liabilities)
    missing = [name for name, value in inputs.items() if value is None]
    invalid = [name for name, value in inputs.items()
               if not is_finite_amount(value)]
    if missing:
        return RatioResult(code, None, "score", "unavailable", "missing_inputs:" + ",".join(missing))
    if invalid:
        return RatioResult(code, None, "score", "unavailable", "invalid_inputs:" + ",".join(invalid))
    for name in ("total_assets", "total_liabilities"):
        if inputs[name] <= 0:
            return RatioResult(code, None, "score", "unavailable", "non_positive_" + name)
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities
    score = Decimal("6.56") * x1 + Decimal("3.26") * x2 + Decimal("6.72") * x3 + Decimal("1.05") * x4
    return RatioResult(code, score, "score", "calculated")


def classify_altman_zone(score):
    if not is_finite_amount(score):
        return None
    if score > Decimal("2.6"):
        return "safe"
    if score >= Decimal("1.1"):
        return "grey"
    return "distress"


@register_ratio
def accrual_ratio(profit_after_tax, operating_cash_flow, opening_assets, closing_assets):
    code = "accrual_ratio"
    reason = _input_issue(dict(profit_after_tax=profit_after_tax, operating_cash_flow=operating_cash_flow))
    if reason:
        return RatioResult(code, None, "fraction", "unavailable", reason)
    return _ratio_on_average(
        code, profit_after_tax - operating_cash_flow, opening_assets, closing_assets, "fraction",
    )


# Source inputs and supplementary estimates


def review_finance_costs(closing_lines, interest_terms, non_interest_terms=(), source_cost_categories=None):
    """Identify candidates without asserting complete interest expense."""
    required = {
        "account", "description", "debit", "credit",
        "exclude_from_income_statement", "closing_review_required",
        "source_file", "source_sheet", "source_row", "source_side",
    }
    if not required.issubset(closing_lines.columns):
        raise SchemaError(
            "Classified closing lines and source references are required."
        )

    for column in (
        "exclude_from_income_statement", "closing_review_required"
    ):
        if not pd.api.types.is_bool_dtype(closing_lines[column]):
            raise SchemaError("Closing flags must be boolean columns.")

    if not isinstance(interest_terms, tuple) or not interest_terms or any(
        not isinstance(term, str) or not term.strip()
        for term in interest_terms
    ):
        raise ValueError(
            "Provide a non-empty tuple of interest search terms."
        )

    terms = tuple(
        normalize("NFC", term).casefold().strip()
        for term in interest_terms
    )
    if not isinstance(non_interest_terms, tuple) or any(
        not isinstance(term, str) or not term.strip() for term in non_interest_terms
    ):
        raise ValueError("Non-interest search terms must be a tuple of non-empty strings.")
    other_terms = tuple(normalize("NFC", term).casefold().strip() for term in non_interest_terms)
    source_cost_categories = source_cost_categories or {}
    if any(not isinstance(code, str) or category not in ("interest", "non_interest")
           for code, category in source_cost_categories.items()):
        raise SchemaError("Source cost-item classifications must map codes to interest/non_interest.")
    result = closing_lines.loc[
        closing_lines["account"].fillna("").str.startswith("635")
    ].copy()

    def label(row):
        if row.exclude_from_income_statement:
            return "identified_closing", "closing_classifier"
        code = normalize_code(getattr(row, "cost_item_code", None))
        category = source_cost_categories.get(code)
        if category == "interest":
            return "interest_candidate", "source_cost_item"
        if category == "non_interest":
            return "non_interest_candidate", "source_cost_item"
        if row.closing_review_required:
            return "unresolved_classification", "closing_review_pending"

        text = (
            normalize("NFC", row.description).casefold()
            if isinstance(row.description, str) else ""
        )
        interest_match = any(term in text for term in terms)
        other_match = any(term in text for term in other_terms)
        if interest_match and not other_match:
            return "interest_candidate", "description_match"
        if other_match and not interest_match:
            return "non_interest_candidate", "description_match"
        return "other_finance_cost_review", "description_missing_or_ambiguous"

    labels = [label(row) for row in result.itertuples()]
    result["interest_review_status"] = [label for label, basis in labels]
    result["interest_review_basis"] = [basis for label, basis in labels]
    result["net_finance_cost"] = pd.Series([
        row.debit - row.credit
        if all(
            is_finite_amount(value)
            for value in (row.debit, row.credit)
        ) else None
        for row in result.itertuples()
    ], index=result.index, dtype=object)

    return result


def summarize_finance_review(details):
    """Keep signed net costs and distinguish missing amounts from zero."""
    rows = []
    for status, group in details.groupby(
        "interest_review_status", sort=True
    ):
        values = group["net_finance_cost"]
        rows.append({
            "interest_review_status": status,
            "line_count": len(group),
            "net_finance_cost": (
                None if values.isna().any()
                else sum(values, Decimal("0"))
            ),
        })

    return pd.DataFrame(rows, columns=[
        "interest_review_status", "line_count", "net_finance_cost",
    ], dtype=object)


def estimate_interest_expense(details):
    """Sum identified interest only; non-interest costs do not block the estimate."""
    active = details.loc[~details["interest_review_status"].eq("identified_closing")]
    if active.empty:
        return None, "no_finance_cost_evidence"
    if not all(is_finite_amount(x) for x in active["net_finance_cost"]):
        return None, "unavailable_finance_cost_amounts"
    unresolved = ~active["interest_review_status"].isin(("interest_candidate", "non_interest_candidate"))
    if {"debit", "credit"}.issubset(active.columns):
        exposure = active["debit"].ne(0) | active["credit"].ne(0)
    else:
        exposure = active["net_finance_cost"].ne(0)
    if (unresolved & exposure).any():
        return None, "unresolved_finance_cost_classification"
    interest = active.loc[active["interest_review_status"].eq("interest_candidate"), "net_finance_cost"]
    source_basis = "interest_review_basis" in active and active["interest_review_basis"].eq("source_cost_item").any()
    return sum(interest, Decimal("0")), "source_cost_item_estimate" if source_basis else "identified_description_estimate"

def account_side_total(table, prefix, amount_column):
    """Sum one balance side within a non-overlapping account scope."""
    if not {"account", amount_column}.issubset(table.columns):
        raise SchemaError("Account and balance columns are required.")

    if not isinstance(prefix, str) or not prefix.isdigit():
        raise ValueError("Provide a numeric account prefix as a string.")

    selected = table.loc[
        table["account"].fillna("").str.startswith(prefix)
    ]
    codes = selected["account"].tolist()

    if len(codes) != len(set(codes)) or any(
        a != b and b.startswith(a)
        for a in codes for b in codes
    ):
        raise SchemaError(
            "Account balances must be unique and non-overlapping."
        )

    values = selected[amount_column].tolist()
    if not values or any(
        not is_finite_amount(value)
        for value in values
    ):
        return None

    return sum(values, Decimal("0"))


def accrual_ratio_interval(profit_after_tax, cfo_lower, cfo_upper, opening_assets, closing_assets):
    """Propagate a conditional CFO envelope without inventing a point estimate."""
    if all(is_finite_amount(x) for x in (cfo_lower, cfo_upper)) and cfo_lower > cfo_upper:
        return {"lower": None, "upper": None, "calculation_status": "unavailable", "reason": "invalid_cfo_bounds"}
    lower = accrual_ratio(profit_after_tax, cfo_upper, opening_assets, closing_assets)
    upper = accrual_ratio(profit_after_tax, cfo_lower, opening_assets, closing_assets)
    if lower.value is None or upper.value is None:
        return {"lower": None, "upper": None, "calculation_status": "unavailable", "reason": lower.reason or upper.reason}
    status = "point_estimate_provisional" if cfo_lower == cfo_upper else "conditional_interval_estimate"
    return {"lower": lower.value, "upper": upper.value, "calculation_status": status, "reason": None}


def analyze_adjustment_effect(revenue, cost, adjustment_revenue, adjustment_cost):
    """Isolate an identified adjustment as a sensitivity, not a restatement."""
    result = {"revenue_before_identified_adjustment": None,
              "cost_before_identified_adjustment": None,
              "margin_before_identified_adjustment": None,
              "identified_adjustment_effect_pp": None}
    if not all(is_finite_amount(x) for x in (revenue, cost, adjustment_revenue, adjustment_cost)):
        return result
    before_revenue, before_cost = revenue - adjustment_revenue, cost - adjustment_cost
    observed = gross_margin(revenue - cost, revenue).value
    before = gross_margin(before_revenue - before_cost, before_revenue).value
    result.update(revenue_before_identified_adjustment=before_revenue,
                  cost_before_identified_adjustment=before_cost,
                  margin_before_identified_adjustment=before,
                  identified_adjustment_effect_pp=(observed - before) * 100 if observed is not None and before is not None else None)
    return result


# Reference policies and monthly analysis

def _validated_benchmark_cutoffs(policy):
    """Validate numeric rules and explicit gaps in reference context."""
    required = {
        "ratio_code", "unit", "direction",
        "good_cutoff", "risk_cutoff", "valid_minimum",
    }
    if not required.issubset(policy):
        missing = ", ".join(sorted(required - set(policy)))
        raise SchemaError("Missing benchmark columns: " + missing)

    if any(
        not isinstance(policy[field], str) or not policy[field].strip()
        for field in ("ratio_code", "unit")
    ) or policy["direction"] not in (
        "higher", "higher_exclusive", "lower", "lower_exclusive",
        "absolute_lower", "context",
    ):
        raise SchemaError("A ratio code, unit and valid direction are required.")

    good, good_status = normalize_amount(policy["good_cutoff"])
    risk, risk_status = normalize_amount(policy["risk_cutoff"])
    minimum, minimum_status = normalize_amount(policy["valid_minimum"])
    if "invalid" in (good_status, risk_status, minimum_status):
        raise SchemaError("Benchmark cutoffs must be finite numbers.")

    if policy["direction"] == "context":
        if good is not None or risk is not None:
            raise SchemaError("Context review requires blank numeric cutoffs.")
        return None, None, minimum, False

    higher = policy["direction"] in ("higher", "higher_exclusive")
    if risk is None or (good is None and not higher):
        raise SchemaError("A risk cutoff and a usable reference direction are required.")
    if good is not None and (
        (higher and good <= risk) or (not higher and good >= risk)
    ):
        raise SchemaError("Benchmark cutoffs have an invalid order.")
    if policy["direction"] == "absolute_lower" and good < 0:
        raise SchemaError("Absolute-deviation cutoffs cannot be negative.")
    cutoffs = [value for value in (good, risk) if value is not None]
    if minimum is not None and minimum > min(cutoffs):
        raise SchemaError("The valid minimum cannot exceed either cutoff.")
    return good, risk, minimum, higher


def prepare_benchmark_policies(table):
    """Prepare independently editable policies without requiring every ratio."""
    table = table.copy(deep=True)
    table.columns = [str(column).lstrip("\ufeff").strip() for column in table.columns]
    if table.columns.duplicated().any():
        raise SchemaError("Benchmark column names must be unique.")

    required = {
        "ratio_code", "unit", "direction",
        "good_cutoff", "risk_cutoff", "valid_minimum",
    }
    if not required.issubset(table.columns):
        missing = ", ".join(sorted(required - set(table.columns)))
        raise SchemaError("Missing benchmark columns: " + missing)

    policies = {}
    for policy in table.to_dict("records"):
        for field in ("ratio_code", "unit", "direction"):
            if isinstance(policy[field], str):
                policy[field] = policy[field].strip()
        _validated_benchmark_cutoffs(policy)
        code = policy["ratio_code"]
        if code == "altman_z_double_prime" and (
            policy["direction"] != "higher_exclusive"
            or normalize_amount(policy["good_cutoff"])[0] != Decimal("2.6")
            or normalize_amount(policy["risk_cutoff"])[0] != Decimal("1.1")
        ):
            raise SchemaError("Altman rules must preserve the assignment boundaries.")
        if code in policies:
            raise SchemaError("Duplicate benchmark ratio code: " + code)
        policies[code] = policy
    return policies


def assess_ratio(result, policy=None):
    """Apply a reference policy and retain the reason for every label."""
    output = dict(ratio_code=result.code, assessment=None, assessment_status="not_assessed",
                  assessment_reason=None, benchmark_basis=None, source_url=None, policy_note=None)
    if policy is not None:
        good, risk, minimum, higher = _validated_benchmark_cutoffs(policy)
        if policy["ratio_code"] != result.code or policy["unit"] != result.unit:
            raise SchemaError("The ratio code and reference unit must match.")
        for field in ("benchmark_basis", "source_url", "policy_note"):
            output[field] = policy.get(field) or None
        output["benchmark_basis"] = output["benchmark_basis"] or "configured_thresholds"
    if result.value is None:
        output.update(assessment_status="input_unavailable", assessment_reason=result.reason)
    elif policy is None:
        output.update(assessment_status="benchmark_unavailable", assessment_reason="no_reference_policy")
    elif minimum is not None and result.value < minimum:
        output.update(assessment="Needs attention", assessment_status="input_review_required",
                      assessment_reason="below_reference_valid_minimum")
    else:
        label, status, reason = "Needs attention", "assessed_policy", "configured_thresholds_only"
        direction = policy["direction"]
        if direction == "context":
            status, reason = "context_review_required", "reference_context_unverified"
        elif higher:
            if good is not None and (result.value > good or (direction == "higher" and result.value == good)):
                label = "Good"
            elif result.value < risk:
                label = "Risk"
            elif good is None:
                status, reason = "context_review_required", "good_reference_not_established"
        else:
            value = abs(result.value) if direction == "absolute_lower" else result.value
            if value <= good:
                label = "Good"
            elif value > risk or (direction in ("lower", "absolute_lower") and value == risk):
                label = "Risk"
        output.update(assessment=label, assessment_status=status, assessment_reason=reason)
    return output

def screen_monthly_series(table, value_column, expected_months, sd_multiplier=Decimal("2")):
    """Compare each observed month with the period mean +/- population SD."""
    if not {"month", value_column}.issubset(table.columns):
        raise SchemaError("Monthly period and value columns are required.")
    if (not isinstance(expected_months, tuple) or len(expected_months) < 3
            or not all(isinstance(month, str) and month for month in expected_months)
            or len(set(expected_months)) != len(expected_months)):
        raise ValueError("Provide at least three unique expected months.")
    if not is_finite_amount(sd_multiplier) or sd_multiplier <= 0:
        raise ValueError("The standard-deviation multiplier must be positive.")
    if table["month"].duplicated().any() or not table["month"].isin(expected_months).all():
        raise SchemaError("Monthly periods must be unique and within the declared scope.")

    details = table[["month", value_column]].copy().sort_values("month")
    details = details.rename(columns={value_column: "value"})
    details["metric"] = value_column
    values = details["value"].tolist()
    valid = [is_finite_amount(value) for value in values]
    missing = tuple(month for month in expected_months if month not in set(table["month"]))
    activity = "activity_status" not in table or bool(table["activity_status"].eq("observed_records").all())
    complete = not missing and all(valid) and activity
    mean = sd = lower = upper = None
    status = "incomplete_series"
    if complete:
        count = Decimal(len(values))
        mean = sum(values, Decimal("0")) / count
        variance = sum(((value - mean) ** 2 for value in values), Decimal("0")) / count
        sd = variance.sqrt()
        lower, upper = mean - sd_multiplier * sd, mean + sd_multiplier * sd
        status = "constant_series" if sd == 0 else "calculated"
    details["z_score"] = [(value - mean) / sd if complete and sd != 0 else None for value in values]
    details["outside_band"] = [value < lower or value > upper if complete else None for value in values]
    details["baseline_status"] = status
    summary = dict(
        metric=value_column, expected_month_count=len(expected_months), valid_months=sum(valid),
        missing_months=missing, activity_complete=activity, ddof=0, sd_multiplier=sd_multiplier,
        mean=mean, standard_deviation=sd, lower_bound=lower, upper_bound=upper,
        baseline_status=status, flagged_months=tuple(details.loc[details["outside_band"].eq(True), "month"]),
    )
    return details, summary

# Report tables

def build_ratio_report_tables(ratio_results, policies):
    """Summarize calculations separately from assessment coverage."""
    ratio_results = tuple(ratio_results)
    if not ratio_results or not all(
        isinstance(result, RatioResult) for result in ratio_results
    ):
        raise ValueError("Provide a non-empty collection of RatioResult objects.")

    codes = [result.code for result in ratio_results]
    if len(codes) != len(set(codes)):
        raise SchemaError("Each ratio must appear once in the report.")
    if set(policies) - set(codes):
        raise SchemaError("Configured ratios are missing from the report.")

    rows = []
    for result in ratio_results:
        assessment = assess_ratio(result, policies.get(result.code))
        assessment.pop("ratio_code")

        row = asdict(result)
        row.update({key: value for key, value in assessment.items()
                    if key not in ("source_url", "policy_note")})
        row["altman_zone"] = classify_altman_zone(result.value) if result.code == "altman_z_double_prime" else None
        rows.append(row)

    ratios = pd.DataFrame(rows, dtype=object)
    calculated = int(ratios["calculation_status"].eq("calculated").sum())
    assessed = int(ratios["assessment_status"].eq("assessed_policy").sum())
    without_policy = int(ratios["assessment_status"].eq("benchmark_unavailable").sum())

    summary = {
        "report_status": "provisional",
        "ratio_count": len(ratios),
        "calculated_ratios": calculated,
        "unavailable_ratios": len(ratios) - calculated,
        "configured_policies": len(policies),
        "assessed_ratios": assessed,
        "context_review_ratios": int(
            ratios["assessment_status"].eq("context_review_required").sum()
        ),
        "calculated_without_policy": without_policy,
        "assessment_coverage": (
            "partial" if assessed < len(ratios)
            else "assessed_under_configured_policies"
        ),
    }

    benchmark_columns = [
        "ratio_code", "unit", "direction",
        "good_cutoff", "risk_cutoff", "valid_minimum",
    ]
    optional_columns = {
        column for policy in policies.values() for column in policy
        if column not in benchmark_columns
    }

    return {
        "Summary": pd.DataFrame(list(summary.items()), columns=["metric", "value"], dtype=object),
        "Ratios": ratios,
        "Review_Items": ratios.loc[
            ~ratios["assessment_status"].eq("assessed_policy")
            | ratios["assessment"].isin(("Needs attention", "Risk"))
        ].copy(),
        "Benchmarks": pd.DataFrame(
            list(policies.values()),
            columns=benchmark_columns + sorted(optional_columns),
            dtype=object,
        ),
    }


def build_financial_analysis_tables(
    classified, statement_tables, settings, account_map, benchmark_table,
):
    """Build financial ratios, cash-flow estimates and monthly analysis."""
    from .ledger import GeneralLedger
    from .statements import build_opening_balance_sheet, classify_closing_lines

    required = {"Summary", "Trial_Balance", "Balance_Sheet", "Income_Statement", "Monthly"}
    if not required.issubset(statement_tables):
        raise SchemaError("Current ledger and statement tables are required.")
    source_scope = dict(statement_tables["Summary"].itertuples(index=False, name=None))
    if any(source_scope.get(field) != settings[field] for field in ("period_start", "period_end")):
        raise SchemaError("Financial-analysis and statement reporting periods must match.")

    policies = prepare_benchmark_policies(benchmark_table)
    registry = get_ratio_registry()
    if set(policies) - set(registry):
        raise SchemaError("Unknown benchmark ratio codes: " + ", ".join(sorted(set(policies) - set(registry))))
    policy = settings.get("financial_analysis", {})
    for code, benchmark in policies.items():
        for field in ("benchmark_basis", "source_url", "policy_note"):
            if not benchmark.get(field):
                benchmark[field] = policy.get("benchmark_references", {}).get(code, {}).get(field)
    days = policy.get("day_count", 365)
    multiplier = Decimal(str(policy.get("monthly_sd_multiplier", "2")))
    terms = tuple(policy.get("interest_search_terms", ["interest", "lãi vay", "lai vay"]))
    tolerance = Decimal(settings["balance_tolerance"])
    trial = statement_tables["Trial_Balance"].copy(deep=True)
    opening_statement, opening_details = build_opening_balance_sheet(trial, account_map, tolerance)
    opening = opening_statement.set_index("item")
    balance = statement_tables["Balance_Sheet"].set_index("item")
    income = statement_tables["Income_Statement"].set_index("item")

    def amount(table, item):
        if table.at[item, "calculation_status"] != "calculated":
            return None
        value = table.at[item, "amount"]
        if value is not None and (not is_finite_amount(value)):
            raise SchemaError("Use finite in-memory Decimal statement amounts.")
        return value

    columns = ["account", "reported_closing_debit", "reported_closing_credit"]
    if not set(columns).issubset(trial.columns):
        raise SchemaError("Reported closing balance sides and source evidence are required.")
    closing_balances = trial[columns].rename(columns={
        "reported_closing_debit": "closing_debit",
        "reported_closing_credit": "closing_credit",
    })
    ledger = GeneralLedger.from_classified_table(classified)
    entries, unassigned = ledger.group_entries(settings["voucher_key"])
    closing_lines = classify_closing_lines(entries, unassigned, account_map, tolerance)
    closing_lines = attach_source_classifications(closing_lines, classified)
    finance_details = closing_lines.pipe(review_finance_costs, terms, tuple(policy.get("non_interest_search_terms", ())), policy.get("source_cost_categories", {}))
    finance_summary = finance_details.pipe(summarize_finance_review)
    interest_expense, interest_reason = estimate_interest_expense(finance_details)
    unknown_accounts = closing_lines.loc[closing_lines["account"].isna()]
    cash_population_complete = all(r.debit == 0 and r.credit == 0 for r in unknown_accounts.itertuples())
    cash_flow_details = review_cash_flows(closing_lines, policy.get("cash_flow_rules", {}))
    cash_flow_summary = summarize_cash_flow_estimate(cash_flow_details, cash_population_complete)
    oa, ca = amount(opening, "total_assets"), amount(balance, "total_assets")
    oe, ce = amount(opening, "equity"), amount(balance, "equity")
    current_assets = amount(balance, "current_assets")
    current_liabilities = amount(balance, "current_liabilities")
    revenue, cost = amount(income, "net_revenue"), amount(income, "cost_of_sales")
    profit = amount(income, "profit_after_tax")
    pbt = amount(income, "profit_before_tax")
    ebit = pbt + interest_expense if pbt is not None and interest_expense is not None else None
    opening_ar = account_side_total(trial, "131", "opening_debit")
    closing_ar = account_side_total(closing_balances, "131", "closing_debit")
    opening_ap = account_side_total(trial, "331", "opening_credit")
    closing_ap = account_side_total(closing_balances, "331", "closing_credit")
    retained_credit = account_side_total(closing_balances, "421", "closing_credit")
    retained_debit = account_side_total(closing_balances, "421", "closing_debit")
    altman_inputs = {
        "working_capital": current_assets - current_liabilities
        if current_assets is not None and current_liabilities is not None else None,
        "retained_earnings": retained_credit - retained_debit
        if retained_credit is not None and retained_debit is not None else None,
        "ebit": ebit,
        "book_equity": ce, "total_assets": ca,
        "total_liabilities": amount(balance, "total_liabilities"),
    }
    estimated_cfo = cash_flow_summary["cfo_point_estimate"]
    cfo_reason = None if estimated_cfo is not None else cash_flow_summary["estimate_status"]
    ratio_results, input_rows = {}, []

    def calculate(code, **inputs):
        result = registry[code](**inputs)
        ratio_results[code] = result
        input_rows.extend({"ratio_code": code, "input_name": name, "value": value}
                          for name, value in inputs.items())
        return result

    calculate("current_ratio", current_assets=current_assets, current_liabilities=current_liabilities)
    calculate("gross_margin", gross_profit=amount(income, "gross_profit"), net_revenue=revenue)
    calculate("net_margin", profit_after_tax=profit, net_revenue=revenue)
    calculate("quick_ratio", current_assets=current_assets, inventories=amount(balance, "inventories"), current_liabilities=current_liabilities)
    calculate("cash_ratio", cash_and_cash_equivalents=amount(balance, "cash"), current_liabilities=current_liabilities)
    calculate("debt_to_assets", total_liabilities=amount(balance, "total_liabilities"), total_assets=ca)
    calculate("debt_to_equity", total_liabilities=amount(balance, "total_liabilities"), equity=ce)
    calculate("interest_coverage", ebit=ebit, interest_expense=interest_expense)
    calculate("asset_turnover", net_revenue=revenue, opening_assets=oa, closing_assets=ca)
    calculate("return_on_assets", profit_after_tax=profit, opening_assets=oa, closing_assets=ca)
    calculate("return_on_equity", profit_after_tax=profit, opening_equity=oe, closing_equity=ce)
    ar = calculate("receivables_turnover", net_revenue=revenue, opening_receivables=opening_ar, closing_receivables=closing_ar)
    inventory = calculate("inventory_turnover", cost_of_sales=cost, opening_inventory=amount(opening, "inventories"), closing_inventory=amount(balance, "inventories"))
    dso = calculate("days_sales_outstanding", turnover=ar.value, days=days)
    dio = calculate("days_inventory_outstanding", turnover=inventory.value, days=days)
    dpo = calculate("days_payables_outstanding", cost_of_sales=cost, opening_payables=opening_ap, closing_payables=closing_ap, days=days)
    calculate("cash_conversion_cycle", dso=dso.value, dio=dio.value, dpo=dpo.value)
    calculate("altman_z_double_prime", **altman_inputs)
    calculate("accrual_ratio", profit_after_tax=profit, operating_cash_flow=estimated_cfo, opening_assets=oa, closing_assets=ca)
    if set(ratio_results) != set(registry):
        raise SchemaError("Financial report coverage does not match the ratio registry.")
    tables = build_ratio_report_tables(tuple(ratio_results.values()), policies)
    maturity_assumed = "calculation_basis" in balance and any(
        balance.at[item, "calculation_basis"] == "includes_disclosed_maturity_assumptions"
        for item in ("current_assets", "current_liabilities")
    )
    maturity_ratios = {"current_ratio", "quick_ratio", "cash_ratio", "altman_z_double_prime"}
    ratios = tables["Ratios"]
    ratios["maturity_assumption_applied"] = [maturity_assumed and code in maturity_ratios for code in ratios["code"]]
    ratios["cash_flow_assumption_applied"] = ratios["code"].eq("accrual_ratio") & (estimated_cfo is not None)
    # A favorable screen does not resolve the assumptions used to calculate it.
    tables["Review_Items"] = ratios.loc[
        ~ratios["assessment_status"].eq("assessed_policy")
        | ratios["assessment"].isin(("Needs attention", "Risk"))
        | ratios["maturity_assumption_applied"]
        | ratios["cash_flow_assumption_applied"]
    ].copy()
    ratio_range = accrual_ratio_interval(profit, cash_flow_summary["cfo_lower"], cash_flow_summary["cfo_upper"], oa, ca)
    ratio_range.update(ratio_code="accrual_ratio", unit="fraction", basis="conditional_cash_flow_classification_envelope")
    for side in ("lower", "upper"):
        endpoint = RatioResult("accrual_ratio", ratio_range[side], "fraction", "calculated") if ratio_range[side] is not None else RatioResult("accrual_ratio", None, "fraction", "unavailable", ratio_range["reason"])
        ratio_range[side + "_endpoint_assessment"] = assess_ratio(endpoint, policies.get("accrual_ratio"))["assessment"]
    ratio_range["interpretation"] = "Endpoint labels do not classify the whole interval; absolute-deviation policies can change label inside an interval containing zero."

    monthly = statement_tables["Monthly"].copy(deep=True)
    monthly["gross_margin"] = [gross_margin(g, r).value for g, r in zip(monthly["gross_profit"], monthly["net_revenue"])]
    expected_months = tuple(str(period) for period in pd.period_range(settings["period_start"], settings["period_end"], freq="M"))
    trend_tables, baselines = [], []
    for metric in ("net_revenue", "gross_margin"):
        details, baseline = screen_monthly_series(monthly, metric, expected_months, multiplier)
        trend_tables.append(details)
        baselines.append(baseline)
    trend_details = pd.concat(trend_tables, ignore_index=True)
    trend_baselines = pd.DataFrame(baselines, dtype=object)
    flagged = trend_details.loc[trend_details["outside_band"].eq(True)]
    monthly_review = flagged.merge(
        monthly[["month", "net_revenue", "cost_of_sales", "gross_profit", "gross_margin"]],
        on="month", how="left", validate="many_to_one",
    )
    monthly_review["business_explanation_status"] = "unverified"
    monthly_review["analytical_observation"] = [
        "Observed gross profit / net revenue = " + format(row.gross_margin, ".2%")
        + "; this is outside the observed-period band, not a verified business cause."
        if row.metric == "gross_margin" else
        "Observed revenue is outside the observed-period band; a business explanation requires evidence."
        for row in monthly_review.itertuples()
    ]
    monthly_review["follow_up"] = "Inspect revenue/cost recognition, job/product mix, cut-off and supporting documents."
    previous_rows = monthly.set_index("month").to_dict("index")
    changes = []
    for r in monthly_review.itertuples():
        previous_month = str(pd.Period(r.month, freq="M") - 1)
        previous = previous_rows.get(previous_month, {})
        record = {"previous_month": previous_month}
        for name in ("net_revenue", "cost_of_sales", "gross_profit"):
            old, new = previous.get(name), getattr(r, name)
            record[name + "_change_pct"] = (new / old - 1) * 100 if isinstance(old, Decimal) and old > 0 and isinstance(new, Decimal) else None
        old_margin = previous.get("gross_margin")
        record["gross_margin_change_pp"] = (r.gross_margin - old_margin) * 100 if isinstance(old_margin, Decimal) and isinstance(r.gross_margin, Decimal) else None
        record["hypotheses_to_verify"] = "Contract/product mix, revenue and cost timing, inventory/job-cost allocation; none is established by the statistical flag."
        changes.append(record)
    for name in ("previous_month", "net_revenue_change_pct", "cost_of_sales_change_pct", "gross_profit_change_pct", "gross_margin_change_pp", "hypotheses_to_verify"):
        monthly_review[name] = [r[name] for r in changes]
    review_months = set(monthly_review["month"])
    evidence_months = {str(pd.Period(month, freq="M") + offset)
                       for month in review_months for offset in (-1, 0, 1)} & set(previous_rows)
    monthly_detail = closing_lines.loc[
        ~closing_lines["exclude_from_income_statement"]
        & closing_lines["account"].fillna("").str.startswith(("511", "521", "632"))
        & closing_lines["posting_date"].map(lambda d: str(d)[:7] in evidence_months)
    ].copy()
    monthly_detail["month"] = monthly_detail["posting_date"].map(lambda d: str(d)[:7])
    monthly_detail["review_role"] = monthly_detail["month"].map(lambda m: "flagged_month" if m in review_months else "adjacent_month")
    monthly_detail["metric_group"] = monthly_detail["account"].map(lambda a: "cost_of_sales" if a.startswith("632") else "net_revenue")
    monthly_detail["signed_statement_amount"] = [
        (r.debit - r.credit if r.metric_group == "cost_of_sales" else r.credit - r.debit)
        if isinstance(r.debit, Decimal) and isinstance(r.credit, Decimal) else None
        for r in monthly_detail.itertuples()
    ]
    evidence_rows = []
    for key, group in monthly_detail.groupby(["month", "metric_group", "account"], sort=True):
        values = group["signed_statement_amount"].tolist()
        evidence_rows.append(dict(zip(("month", "metric_group", "account"), key),
            line_count=len(group), amount=sum(values, Decimal(0)) if all(isinstance(x, Decimal) for x in values) else None,
            source_rows=tuple(group["source_row"])))
    adjustment_policy = policy.get("monthly_adjustment_review", {})
    adjustment_results = {}
    for month in sorted(evidence_months):
        rows = monthly_detail.loc[monthly_detail["month"].eq(month)]
        counter_prefix = adjustment_policy.get("counter_prefix", "")
        terms = tuple(term.casefold() for term in adjustment_policy.get("description_terms", ()))
        selected = rows.loc[
            rows["counter_account"].fillna("").str.startswith(counter_prefix)
            & rows["description"].fillna("").map(lambda s: any(t in str(s).casefold() for t in terms))
        ] if counter_prefix and terms else rows.iloc[:0]
        adjustments = {}
        for metric, field in (("net_revenue", "revenue_accounts"), ("cost_of_sales", "cost_accounts")):
            values = selected.loc[selected["metric_group"].eq(metric) & selected["account"].isin(adjustment_policy.get(field, ())), "signed_statement_amount"].tolist()
            adjustments[metric] = sum(values, Decimal(0)) if all(isinstance(x, Decimal) for x in values) else None
        values = previous_rows[month]
        adjustment_results[month] = dict(
            analyze_adjustment_effect(values["net_revenue"], values["cost_of_sales"], adjustments["net_revenue"], adjustments["cost_of_sales"]),
            identified_adjustment_revenue=adjustments["net_revenue"], identified_adjustment_cost=adjustments["cost_of_sales"],
            identified_adjustment_source_rows=tuple(selected.loc[selected["account"].isin(adjustment_policy.get("revenue_accounts", []) + adjustment_policy.get("cost_accounts", [])), "source_row"]),
        )
    for name in ("revenue_before_identified_adjustment", "cost_before_identified_adjustment", "margin_before_identified_adjustment",
                 "identified_adjustment_effect_pp", "identified_adjustment_revenue", "identified_adjustment_cost", "identified_adjustment_source_rows"):
        monthly_review[name] = [adjustment_results[r.month][name] for r in monthly_review.itertuples()]
    monthly_review["adjustment_interpretation"] = "Source-described tax entries identified by configured accounts/counterpart/description. Exclusion is an analytical sensitivity, not a correction or validated tax treatment."
    analysis_checks = []
    def check(name, actual, expected):
        status = "unavailable" if actual is None or expected is None else "matched" if actual == expected else "difference"
        analysis_checks.append({"check_name": name, "actual": actual, "expected": expected, "check_status": status})
    check("ratio_coverage", len(ratio_results), len(registry))
    for label, table in (("opening", opening), ("closing", balance)):
        for domain, parts, total_item in (
            ("assets", ("current_assets", "noncurrent_assets"), "total_assets"),
            ("liabilities", ("current_liabilities", "noncurrent_liabilities"), "total_liabilities"),
        ):
            values = [amount(table, item) for item in parts]
            check(label + "_" + domain + "_maturity_partition",
                  sum(values, Decimal("0")) if all(value is not None for value in values) else None,
                  amount(table, total_item))
    check("cash_record_coverage", sum(cash_flow_details["cash_flow_category"].value_counts()), len(cash_flow_details))
    check("cash_amount_conservation", sum((cash_flow_summary[name] for name in (
        "identified_operating_cash_flow", "investing_cash_flow_proxy", "financing_cash_flow_proxy", "internal_transfer_net", "noncash_adjustment_net", "unclassified_cash_net")), Decimal(0)) if cash_flow_summary["amounts_complete"] else None,
        cash_flow_summary["observed_cash_movement"])
    monthly_cash_change = monthly["cash_net_movement"].tolist()
    if monthly_cash_change:
        check("cash_movement_vs_monthly", cash_flow_summary["observed_cash_movement"], sum(monthly_cash_change, Decimal(0)) if all(isinstance(x, Decimal) for x in monthly_cash_change) else None)
        opening_cash, closing_cash = monthly.iloc[0]["opening_cash"], monthly.iloc[-1]["closing_cash"]
        movement = cash_flow_summary["observed_cash_movement"]
        check("cash_rollforward", opening_cash + movement if isinstance(opening_cash, Decimal) and isinstance(movement, Decimal) else None, closing_cash)
    if cash_flow_summary["cfo_lower"] is not None:
        check("cfo_interval_order", cash_flow_summary["cfo_lower"] <= cash_flow_summary["cfo_upper"], True)
    for month in sorted(evidence_months):
        for metric in ("net_revenue", "cost_of_sales"):
            values = monthly_detail.loc[monthly_detail["month"].eq(month) & monthly_detail["metric_group"].eq(metric), "signed_statement_amount"].tolist()
            actual = sum(values, Decimal(0)) if all(isinstance(x, Decimal) for x in values) else None
            check(month + "_" + metric + "_source_conservation", actual, previous_rows[month][metric])
    context = {
        "period_start": settings["period_start"], "period_end": settings["period_end"],
        "unit_status": source_scope["unit_status"], "day_count_convention": days,
        "monthly_screening": "Observed-period population SD; ddof=0; strict outside the configured band.",
        "monthly_sd_multiplier": multiplier,
        "scope_note": "Source and statement limitations remain unresolved; missing inputs are not zero.",
        "altman_model": "Assignment four-variable Z-double-prime; no additive constant.",
        "altman_applicability": "Unverified entity/industry suitability; no score inferred from partial inputs.",
        "retained_earnings_basis": "Reported closing net credit of account 421; provisional.",
        "interest_estimation_basis": interest_reason,
        "cfo_estimation_status": cash_flow_summary["estimate_status"],
        "estimated_operating_cash_flow": estimated_cfo,
        "cash_flow_assumptions": policy.get("cash_flow_rules", {}).get("assumption"),
        "cfo_input_limitation": cfo_reason,
        "cfo_lower": cash_flow_summary["cfo_lower"],
        "cfo_upper": cash_flow_summary["cfo_upper"],
        "reference_policy_scope": "Preliminary project screens with documented assumptions, not verified industry averages or probabilities of default.",
        "data_gap_note": "Unverified classifications are not a claim that the raw files contain no relevant data.",
        "maturity_assumption_applied": maturity_assumed,
        "maturity_interpretation": "Current/noncurrent totals follow account_map maturity assumptions where marked; zero noncurrent liabilities do not establish absence of long-term debt.",
    }
    tables["Summary"] = pd.concat([
        tables["Summary"], pd.DataFrame(list(context.items()), columns=["metric", "value"]),
    ], ignore_index=True)
    tables.update({
        "Ratio_Inputs": pd.DataFrame([
            dict(row, maturity_assumption_applied=maturity_assumed and row["ratio_code"] in maturity_ratios)
            for row in input_rows
        ], dtype=object),
        "Opening_Details": opening_details,
        "Finance_Summary": finance_summary, "Finance_Details": finance_details,
        "CFO_Details": cash_flow_details,
        "CFO_Summary": pd.DataFrame(list(cash_flow_summary.items()), columns=["metric", "value"], dtype=object),
        "Ratio_Ranges": pd.DataFrame([ratio_range], dtype=object),
        "Analysis_Checks": pd.DataFrame(analysis_checks, dtype=object),
        "Monthly": monthly, "Monthly_Trend": trend_details,
        "Monthly_Baselines": trend_baselines, "Monthly_Review": monthly_review,
        "Monthly_Review_Details": monthly_detail,
        "Monthly_Account_Summary": pd.DataFrame(evidence_rows, columns=["month", "metric_group", "account", "line_count", "amount", "source_rows"], dtype=object),
    })
    return tables
