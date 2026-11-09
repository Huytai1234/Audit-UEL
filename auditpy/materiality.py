"""Calculate preliminary materiality and assemble its evidence tables."""

from datetime import date
from decimal import Decimal, ROUND_FLOOR

import pandas as pd

from auditpy.utils import SchemaError, normalize_amount, is_finite_amount
from auditpy.validators import make_reconciliation_check


class Materiality:
    """Keep a benchmark and rate without automatically selecting an option."""

    def __init__(self, benchmark, benchmark_amount, rate):
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise ValueError("A benchmark name is required.")
        if benchmark_amount is not None and (
            not is_finite_amount(benchmark_amount)
        ):
            raise ValueError("Benchmark amounts must be finite Decimal values or None.")
        if not is_finite_amount(rate) or not 0 < rate < 1:
            raise ValueError("The rate must be a finite Decimal between 0 and 1.")
        self._benchmark = benchmark
        self._amount = benchmark_amount
        self._rate = rate

    @staticmethod
    def round_amount(amount, increment=Decimal("1")):
        """Round non-negative amounts down to a positive increment."""
        if not is_finite_amount(increment) or increment <= 0:
            raise ValueError("The rounding increment must be a positive finite Decimal.")
        if amount is None:
            return None
        if not is_finite_amount(amount) or amount < 0:
            raise ValueError("The amount must be a non-negative finite Decimal or None.")
        return (amount / increment).to_integral_value(rounding=ROUND_FLOOR) * increment

    def calculate(self, rounding_increment=Decimal("1")):
        if self._amount is None:
            status = "missing_benchmark"
        elif self._amount <= 0:
            status = "non_positive_benchmark"
        else:
            status = "calculated"
        raw = self._amount * self._rate if status == "calculated" else None
        rounded = self.round_amount(raw, rounding_increment)
        if rounded == 0:
            status = "rounded_to_zero"
        return {
            "benchmark": self._benchmark,
            "benchmark_amount": self._amount,
            "rate": self._rate,
            "unrounded_materiality": raw,
            "overall_materiality": rounded,
            "rounding_increment": rounding_increment,
            "calculation_status": status,
        }

    def calculate_levels(
        self, performance_rate, clearly_trivial_rate,
        rounding_increment=Decimal("1"),
    ):
        """Calculate both secondary levels from rounded overall materiality."""
        for rate in (performance_rate, clearly_trivial_rate):
            if not is_finite_amount(rate):
                raise ValueError("Secondary rates must be finite Decimal values.")
        if not 0 < clearly_trivial_rate < performance_rate < 1:
            raise ValueError("Require 0 < clearly trivial rate < performance rate < 1.")

        result = self.calculate(rounding_increment)
        overall = result["overall_materiality"] if result["calculation_status"] == "calculated" else None
        for name, rate in (
            ("performance_materiality", performance_rate),
            ("clearly_trivial_threshold", clearly_trivial_rate),
        ):
            raw = overall * rate if overall is not None else None
            result["unrounded_" + name] = raw
            result[name] = self.round_amount(raw, rounding_increment)
        result.update(
            performance_rate=performance_rate,
            clearly_trivial_rate=clearly_trivial_rate,
            secondary_percentage_basis="rounded_overall_materiality",
        )
        if result["calculation_status"] == "calculated" and not (
            0 < result["clearly_trivial_threshold"] < result["performance_materiality"] < overall
        ):
            result["calculation_status"] = "rounded_levels_invalid"
        return result


def build_materiality_tables(statement_tables, settings):
    """Calculate materiality from financial statement tables."""
    required_tables = {"Summary", "Income_Statement", "Balance_Sheet"}
    if not required_tables.issubset(statement_tables):
        raise SchemaError("Materiality requires statement and scope tables.")
    summary = statement_tables["Summary"]
    if not {"metric", "value"}.issubset(summary.columns) or summary["metric"].duplicated().any():
        raise SchemaError("Statement summary metrics must be present and unique.")
    source_summary = dict(zip(summary["metric"], summary["value"]))
    if not {"period_start", "period_end", "unit_status", "report_status"}.issubset(source_summary):
        raise SchemaError("Statement reporting scope is incomplete.")
    try:
        start = date.fromisoformat(settings["period_start"])
        end = date.fromisoformat(settings["period_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise SchemaError("A valid materiality reporting period is required.") from error
    if start > end or any(
        source_summary[field] != settings[field] for field in ("period_start", "period_end")
    ):
        raise SchemaError("Statement and configuration periods do not match.")

    policy = settings.get("materiality")
    if not isinstance(policy, dict):
        raise SchemaError("Materiality configuration is required.")
    sources = {
        "profit_before_tax": "Income_Statement",
        "net_revenue": "Income_Statement",
        "total_assets": "Balance_Sheet",
    }
    benchmark = policy.get("benchmark")
    if not isinstance(benchmark, str) or benchmark not in sources:
        raise SchemaError("Unsupported materiality benchmark.")
    rationale_fields = ("benchmark_rationale", "performance_rationale", "clearly_trivial_rationale")
    for field in rationale_fields:
        if not isinstance(policy.get(field), str) or not policy[field].strip():
            raise SchemaError("Document the materiality rationale: " + field)

    def configured_number(value, label):
        if not isinstance(value, str):
            raise SchemaError("Use a finite decimal string for " + label)
        amount, status = normalize_amount(value)
        if status != "ok":
            raise SchemaError("Use a finite decimal string for " + label)
        return amount

    numbers = {
        field: configured_number(policy.get(field), field)
        for field in ("overall_rate", "performance_rate", "clearly_trivial_rate", "rounding_increment")
    }
    amounts = {}
    source_rows = {}
    for item, sheet in sources.items():
        table = statement_tables[sheet]
        if not {"item", "amount"}.issubset(table.columns):
            raise SchemaError("Statement item and amount columns are required.")
        positions = [i for i, value in enumerate(table["item"]) if value == item]
        if len(positions) != 1:
            raise SchemaError("A unique statement item is required: " + item)
        amounts[item] = table.iloc[positions[0]]["amount"]
        source_rows[item] = positions[0] + 2

    selected = Materiality(benchmark, amounts[benchmark], numbers["overall_rate"]).calculate_levels(
        numbers["performance_rate"], numbers["clearly_trivial_rate"], numbers["rounding_increment"],
    )
    comparison_rates = policy.get("comparison_rates")
    if not isinstance(comparison_rates, dict) or set(comparison_rates) != set(sources):
        raise SchemaError("Comparison rates must cover profit, revenue and total assets.")
    plans = []
    for item, rates in comparison_rates.items():
        if not isinstance(rates, list) or not rates:
            raise SchemaError("Provide a non-empty comparison-rate list for " + item)
        plans.extend((item, configured_number(rate, item)) for rate in rates)
    if len(plans) != len(set(plans)):
        raise SchemaError("Materiality comparison options must be unique.")
    if (benchmark, numbers["overall_rate"]) not in plans:
        plans.append((benchmark, numbers["overall_rate"]))
    options = []
    for item, rate in plans:
        option = Materiality(item, amounts[item], rate).calculate(numbers["rounding_increment"])
        option.update(
            source_file="03_ledger_statements.xlsx", source_sheet=sources[item],
            source_item=item, source_row=source_rows[item], report_status="provisional",
            is_selected=item == benchmark and rate == numbers["overall_rate"],
        )
        options.append(option)

    selection = dict(
        selected, period_start=source_summary["period_start"], period_end=source_summary["period_end"],
        unit_status=source_summary["unit_status"], report_status="provisional",
        source_file="03_ledger_statements.xlsx", source_sheet=sources[benchmark],
        source_item=benchmark, source_row=source_rows[benchmark],
        benchmark_source_status=source_summary["report_status"], rounding_method="floor",
        limitation="Benchmarks remain provisional; reassess materiality when source evidence changes.",
        threshold_use="Performance materiality supports JE screening; transaction size alone does not establish a misstatement.",
    )
    selection.update({field: policy[field] for field in rationale_fields})
    definitions = (
        ("overall_materiality", "benchmark_amount", "rate", "unrounded_materiality"),
        ("performance_materiality", "overall_materiality", "performance_rate", "unrounded_performance_materiality"),
        ("clearly_trivial_threshold", "overall_materiality", "clearly_trivial_rate", "unrounded_clearly_trivial_threshold"),
    )
    levels = pd.DataFrame([
        {
            "level": name, "percentage_basis": basis, "basis_amount": selected[basis],
            "rate": selected[rate], "unrounded_amount": selected[raw], "amount": selected[name],
            "rounding_increment": numbers["rounding_increment"],
            "calculation_status": selected["calculation_status"], "report_status": "provisional",
        }
        for name, basis, rate, raw in definitions
    ], dtype=object)
    overall = selected["overall_materiality"]
    performance = selected["performance_materiality"]
    trivial = selected["clearly_trivial_threshold"]
    ordered = int(0 < trivial < performance < overall) if all(
        value is not None for value in (overall, performance, trivial)
    ) else None
    check_inputs = (
        ("source", "period_start", int(source_summary["period_start"] == settings["period_start"]), 1),
        ("source", "period_end", int(source_summary["period_end"] == settings["period_end"]), 1),
        ("source", "benchmark_amount", selected["benchmark_amount"], amounts[benchmark]),
        ("basis", "performance_percentage", selected["unrounded_performance_materiality"],
         overall * numbers["performance_rate"] if overall is not None else None),
        ("basis", "clearly_trivial_percentage", selected["unrounded_clearly_trivial_threshold"],
         overall * numbers["clearly_trivial_rate"] if overall is not None else None),
        ("levels", "positive_ordered_levels", ordered, 1),
    )
    return {
        "Options": pd.DataFrame(options, dtype=object),
        "Selected": pd.DataFrame(list(selection.items()), columns=["metric", "value"], dtype=object),
        "Levels": levels,
        "Checks": pd.DataFrame([
            make_reconciliation_check(name, actual, expected, group=group)
            for group, name, actual, expected in check_inputs
        ], dtype=object),
    }
