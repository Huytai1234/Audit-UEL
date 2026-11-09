"""Run the accounting audit pipeline, reports, charts and offline dashboard."""

import json
from datetime import date
from decimal import Decimal
from functools import partial
from pathlib import Path
from uuid import uuid4
from auditpy.charts import build_chart_outputs

from auditpy.loaders import create_loader, read_trial_balance, read_benchmark_table
from auditpy.materiality import build_materiality_tables
from auditpy.ratios import build_financial_analysis_tables
from auditpy.je_tests import (
    UnbalancedVoucherTest, DuplicateEntryTest, VoucherSequenceGapTest,
    WeekendHolidayTest, RoundAmountTest, NearApprovalLimitTest, LargeYearEndEntryTest,
    UnusualCounterAccountTest, MissingOrVagueDescriptionTest, BenfordTest,
    run_je_tests, build_je_report_tables,
)
from auditpy.statements import build_statement_tables
from auditpy.utils import (
    is_finite_amount,
    write_excel_report, run_audited, read_json_config, table_row_counts,
)
from auditpy.validators import (
    classify_ledger_rows, prepare_trial_balance_reference, build_integrity_tables,
)
from auditpy.acceptance import build_acceptance_tables
from auditpy.models import Client, AuditEngagement
from auditpy.ledger import GeneralLedger


def print_checks(label, checks):
    counts = checks["check_status"].value_counts()
    print(label + ":", ", ".join(f"{status}={count}" for status, count in counts.items()))


def build_je_tables(
    classified, settings, account_map, calendar, selection, log_path,
    *, run_id=None, engagement=None,
):
    """Run ten JE procedures with the selected performance materiality."""
    threshold = selection.get("performance_materiality")
    if (
        selection.get("calculation_status") != "calculated"
        or not is_finite_amount(threshold) or threshold <= 0
    ):
        raise ValueError("JE testing requires positive, finite performance materiality.")
    if any(selection.get(field) != settings[field] for field in ("period_start", "period_end")):
        raise ValueError("Materiality and JE reporting periods must match.")
    if any(
        date.fromisoformat(settings[field]).year != calendar["year"]
        for field in ("period_start", "period_end")
    ):
        raise ValueError("The reference calendar must cover the reporting year.")

    policy = settings["je_tests"]
    approval = policy["approval_screening"]
    tolerance = Decimal(settings["balance_tolerance"])
    benford = BenfordTest(policy["benford_populations"], settings["voucher_key"], account_map, tolerance)
    procedures = [
        UnbalancedVoucherTest(settings["voucher_key"], tolerance),
        DuplicateEntryTest(),
        VoucherSequenceGapTest(policy["voucher_sequence_patterns"], policy["voucher_sequence_assumption"]),
        WeekendHolidayTest(calendar),
        RoundAmountTest([Decimal(value) for value in policy["round_amount_bases"]], policy["round_amount_assumption"]),
        NearApprovalLimitTest(
            settings["voucher_key"], Decimal(approval["limit"]), Decimal(approval["lower_rate"]),
            approval["cash_account_prefixes"], approval["assumption"],
        ),
        LargeYearEndEntryTest(
            settings["voucher_key"], date.fromisoformat(settings["period_end"]),
            policy["year_end_window_days"], date.fromisoformat(settings["period_start"]),
            date.fromisoformat(settings["period_end"]),
        ),
        UnusualCounterAccountTest(policy["counter_account_rules"]),
        MissingOrVagueDescriptionTest(policy["vague_description_phrases"]),
        benford,
    ]
    if engagement is not None:
        engagement.tests = procedures
    results, run_id = run_je_tests(procedures, classified, threshold, log_path, run_id=run_id)
    with Path(log_path).open(encoding="utf-8") as file:
        events = [json.loads(line) for line in file]
    run_events = [event for event in events if event["run_id"] == run_id and event.get("event_type") == "je_procedure"]
    tables = build_je_report_tables(
        results, benford.results, run_events, settings, calendar,
    )
    if engagement is not None:
        engagement.results["journal_entries"] = tables
    return tables


def _load_journal(path, config):
    return create_loader(path, config).load()


def _source_row_counts(table):
    """Count source data rows independently of format-B debit/credit expansion."""
    return {"source_data_rows": len(table[["source_file", "source_sheet", "source_row"]].drop_duplicates())}


def _run_pipeline(project_dir, pipeline_id, audit_path):
    """Generate audit reports and visuals under one run ID."""
    def step(name, function, *args, **kwargs):
        return run_audited(function, *args, audit_name=name, log_path=audit_path, run_id=pipeline_id, **kwargs)

    def configuration(name, filename):
        path = project_dir / "config" / filename
        return step(name, read_json_config, path)

    def export(name, path, tables):
        counts = step(
            name, write_excel_report, path, tables, input_counts=table_row_counts(tables),
            output_counts=lambda written: written,
        )
        print("Report:", path, f"({len(counts)} sheets)")
        return counts

    config = configuration("LOAD_COLUMN_CONFIG", "column_map.json")
    settings = configuration("LOAD_AUDIT_CONFIG", "audit_config.json")

    raw_dir = project_dir / "data/raw"

    print("Loading source files...")
    journal_path = raw_dir / "SO CHI TIET 01.01-31.12.2024.xlsx"
    client_path = raw_dir / "BANG CAN DOI PS TK 01.01-31.12.2024.xlsx"
    journal = step(
        "LOAD_NORMALIZE_LEDGER", _load_journal, journal_path, config["general_ledger"],
        input_counts=_source_row_counts,
        output_counts=lambda table: {"normalized_records": len(table)},
    )
    client = step(
        "LOAD_NORMALIZE_TRIAL_BALANCE", read_trial_balance, client_path, config["trial_balance"],
        input_counts=_source_row_counts,
        output_counts=lambda table: {"client_trial_balance_rows": len(table)},
    )
    classified = step(
        "CLASSIFY_LEDGER", classify_ledger_rows, journal, config["general_ledger"]["row_labels"],
        input_counts={"normalized_records": len(journal)}, output_counts=lambda table: {"classified_records": len(table)},
    )
    reference = step(
        "PREPARE_TB_REFERENCE", prepare_trial_balance_reference, client,
        input_counts={"client_trial_balance_rows": len(client)}, output_counts=lambda table: {"account_references": len(table)},
    )

    print("Running data-integrity checks...")
    integrity_tables = build_integrity_tables(classified, client, reference, config, settings, step)
    summary = dict(integrity_tables["Summary"].itertuples(index=False, name=None))
    print("Records:", len(classified), "normalized;", summary["review_rows"], "reviewed;",
          len(integrity_tables["Record_Issues"]), "with integrity issues")
    print("Observed months:", len(summary["observed_months"]))
    print("Ledger debit-credit difference:", summary["difference"])

    output_path = project_dir / "outputs/01_data_integrity.xlsx"
    export("EXPORT_INTEGRITY", output_path, integrity_tables)

    print("Building ledger, financial statements and monthly observations...")
    account_map = configuration("LOAD_ACCOUNT_MAP", "account_map.json")
    statement_tables = step(
        "BUILD_LEDGER_STATEMENTS_MONTHLY", build_statement_tables,
        classified, client, reference, config, settings, account_map,
        input_counts={"classified_records": len(classified), "client_rows": len(client), "account_references": len(reference)},
        output_counts=table_row_counts,
    )
    statement_path = project_dir / "outputs/03_ledger_statements.xlsx"
    export("EXPORT_LEDGER_STATEMENTS", statement_path, statement_tables)
    print_checks("Statement checks", statement_tables["Checks"])

    print("Calculating preliminary materiality...")
    materiality_tables = step(
        "CALCULATE_MATERIALITY", build_materiality_tables, statement_tables, settings,
        input_counts=table_row_counts(statement_tables), output_counts=table_row_counts,
    )
    materiality_path = project_dir / "outputs/04_materiality.xlsx"
    export("EXPORT_MATERIALITY", materiality_path, materiality_tables)
    selection = dict(materiality_tables["Selected"].itertuples(index=False, name=None))
    for field in (
        "overall_materiality", "performance_materiality", "clearly_trivial_threshold",
        "calculation_status",
    ):
        print(field.replace("_", " ").capitalize() + ":", selection[field])
    print_checks("Materiality checks", materiality_tables["Checks"])

    print("Analyzing financial ratios, cash flows and monthly trends...")
    benchmark_path = project_dir / "config/benchmarks.csv"
    benchmark_table = step(
        "LOAD_BENCHMARKS", read_benchmark_table, benchmark_path,
        output_counts=lambda table: {"benchmark_rows": len(table)},
    )
    analysis_tables = step(
        "BUILD_FINANCIAL_ANALYSIS", build_financial_analysis_tables,
        classified, statement_tables, settings, account_map, benchmark_table,
        input_counts={"classified_records": len(classified), "benchmark_rows": len(benchmark_table),
                      **{"statement_" + name: count for name, count in table_row_counts(statement_tables).items()}},
        output_counts=table_row_counts,
    )
    analysis_path = project_dir / "outputs/05_financial_analysis.xlsx"
    export("EXPORT_FINANCIAL_ANALYSIS", analysis_path, analysis_tables)
    analysis_summary = dict(analysis_tables["Summary"].itertuples(index=False, name=None))
    for field, label in (
        ("calculated_ratios", "Calculated ratios"),
        ("unavailable_ratios", "Unavailable ratios"),
        ("assessment_coverage", "Ratio assessment coverage"),
        ("cfo_estimation_status", "CFO estimation status"),
        ("estimated_operating_cash_flow", "Estimated operating cash flow"),
    ):
        print(label + ":", analysis_summary[field])
    print_checks("Financial analysis checks", analysis_tables["Analysis_Checks"])

    threshold = selection.get("performance_materiality")
    if (
        selection.get("calculation_status") != "calculated"
        or not is_finite_amount(threshold) or threshold <= 0
    ):
        print("JE status: not_run_unusable_materiality")
        print("JE, acceptance and visuals were not generated: materiality is unavailable.")
        return

    client_info = step(
        "LOAD_CLIENT", Client.from_config, project_dir / "config/audit_config.json",
        output_counts=lambda value: {"clients": 1},
    )
    engagement_ledger = step(
        "BUILD_ENGAGEMENT_LEDGER", GeneralLedger.from_classified_table, classified,
        input_counts={"classified_records": len(classified)},
        output_counts=lambda value: {"ledger_records": len(value)},
    )
    engagement = AuditEngagement(client_info, engagement_ledger)
    engagement.results.update(
        statements=statement_tables,
        materiality=materiality_tables,
        financial_analysis=analysis_tables,
    )

    print("Running ten journal-entry procedures...")
    calendar = configuration("LOAD_REFERENCE_CALENDAR", "calendar_2024.json")
    je_tables = step(
        "RUN_BUILD_JE_REPORT", partial(
            build_je_tables, run_id=pipeline_id, engagement=engagement,
        ),
        classified, settings, account_map, calendar, selection, audit_path,
        input_counts={"classified_records": len(classified)},
        output_counts=table_row_counts,
    )
    je_path = project_dir / "outputs/02_je_findings.xlsx"
    export("EXPORT_JE_FINDINGS", je_path, je_tables)
    print(je_tables["Summary"][["test_code", "finding_count", "HIGH", "MEDIUM", "LOW"]].to_string(index=False))
    print("JE performance materiality:", threshold)
    print("Unique flagged records:", len(je_tables["Flagged_Records"]))

    print("Combining acceptance scores and sensitivity scenarios...")
    risk_config = configuration("LOAD_RISK_CONFIG", "risk_weights.json")
    acceptance_tables = build_acceptance_tables(
        engagement, integrity_tables, risk_config, selection, step,
    )
    export(
        "EXPORT_ACCEPTANCE_REPORT",
        project_dir / "outputs/06_acceptance_sensitivity.xlsx",
        acceptance_tables,
    )

    acceptance_summary = dict(
        acceptance_tables["Summary"].itertuples(index=False, name=None)
    )
    for field, label in (
        ("weighted_score", "Client risk score"),
        ("decision_after_knockouts", "Preliminary decision"),
        ("knockout_review_complete", "Knock-out review complete"),
        ("open_actions", "Open actions"),
    ):
        print(label + ":", acceptance_summary[field])
    print("Sensitivity scenarios changing the decision:",
          len(acceptance_summary["decision_changed_scenarios"]))

    print_checks("Acceptance checks", acceptance_tables["Checks"])
    print("Creating charts and the offline dashboard...")
    chart_result = step(
        "BUILD_CHARTS_DASHBOARD",
        build_chart_outputs,
        statement_tables,
        je_tables,
        analysis_tables,
        acceptance_tables,
        project_dir / "outputs",
        input_counts={
            "ledger_records": len(statement_tables["General_Ledger"]),
            "monthly_rows": len(analysis_tables["Monthly"]),
            "ratio_rows": len(analysis_tables["Ratios"]),
            "benford_frequency_rows": len(je_tables["Benford_Frequency"]),
            "je_procedure_rows": len(je_tables["Summary"]),
            "acceptance_group_rows": len(acceptance_tables["Group_Scores"]),
        },
        output_counts=lambda result: {
            "png_files": sum(p.suffix == ".png" for p in result["paths"]),
            "html_files": sum(p.suffix == ".html" for p in result["paths"]),
        },
    )

    print("Charts:", sum(path.suffix == ".png" for path in chart_result["paths"]),
          "PNG files in", project_dir / "outputs/charts")
    for path in chart_result["paths"]:
        if path.suffix == ".html":
            print("Dashboard:", path)
    coverage = chart_result["weekday_coverage"]
    print("Weekday chart coverage:", coverage["dated_records"], "dated records;",
          coverage["unavailable_date_records"], "missing or invalid dates")
    print("Skipped visuals:", ", ".join(chart_result["skipped"]) or "none")
    print("Results are preliminary. Review source differences, assumptions and open actions.")
    print("Screening flags do not confirm errors or fraud; the decision is not engagement approval.")


def main():
    project_dir = Path(__file__).resolve().parent
    run_id = uuid4().hex
    log_path = project_dir / "outputs/audit_log.jsonl"
    print("Audit run:", run_id)
    print("Audit log:", log_path)
    run_audited(
        _run_pipeline, project_dir, run_id, log_path,
        log_path=log_path, run_id=run_id, audit_name="RUN_PIPELINE", event_type="pipeline_run",
    )


if __name__ == "__main__":
    main()
