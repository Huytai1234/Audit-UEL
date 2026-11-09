"""Run the real entry point using small synthetic sources and another working directory."""

import importlib.util
from decimal import Decimal
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from openpyxl import load_workbook

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_importing_main_does_not_run_pipeline(capsys):
    spec = importlib.util.spec_from_file_location("entry_import_test", PROJECT_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("format_name", ["A", "B"])
@pytest.mark.parametrize("profitable", [False, True])
def test_main_runs_with_both_formats_from_another_directory(tmp_path, source_writer, format_name, profitable):
    project = tmp_path / "project"
    raw = project / "data/raw"
    raw.mkdir(parents=True)
    (project / "config").mkdir()
    shutil.copy2(PROJECT_DIR / "main.py", project / "main.py")
    shutil.copytree(PROJECT_DIR / "auditpy", project / "auditpy", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(PROJECT_DIR / "config/benchmarks.csv", project / "config/benchmarks.csv")
    settings = json.loads((PROJECT_DIR / "config/audit_config.json").read_text())
    settings["materiality"].update(benchmark="profit_before_tax", overall_rate="0.05")
    tb_config = json.loads((PROJECT_DIR / "config/column_map.json").read_text())["trial_balance"]
    if format_name == "A":
        columns = ["voucher_no", "posting_date", "document_date", "description", "account", "debit", "credit"]
        rows = [
            ["0001", "03/01/2024", "03/01/2024", "Deposit", "1121", "100.10", "0"],
            ["0001", "03/01/2024", "03/01/2024", "Deposit", "1111", "0", "100.10"],
        ]
        if profitable:
            columns.append("counter_account")
            rows[0].append("1111")
            rows[1].append("1121")
            rows.extend([
                ["PT24010001", "06/01/2024", "06/01/2024", "Revenue", "1121", "1000", "0", "5111"],
                ["PT24010001", "06/01/2024", "06/01/2024", "Revenue", "5111", "0", "1000", "1121"],
                ["GL24120001", "31/12/2024", "31/12/2024", "Close revenue", "5111", "1000", "0", "911"],
                ["GL24120001", "31/12/2024", "31/12/2024", "Close revenue", "911", "0", "1000", "5111"],
                ["GL24120002", "31/12/2024", "31/12/2024", "Transfer profit", "911", "1000", "0", "4212"],
                ["GL24120002", "31/12/2024", "31/12/2024", "Transfer profit", "4212", "0", "1000", "911"],
            ])
    else:
        columns = ["voucher_no", "posting_date", "document_date", "description", "debit_account", "credit_account", "amount"]
        rows = [["0001", "03/01/2024", "03/01/2024", "Deposit", "1121", "1111", "100.10"]]
        if profitable:
            rows.extend([
                ["PT24010001", "06/01/2024", "06/01/2024", "Revenue", "1121", "5111", "1000"],
                ["GL24120001", "31/12/2024", "31/12/2024", "Close revenue", "5111", "911", "1000"],
                ["GL24120002", "31/12/2024", "31/12/2024", "Transfer profit", "911", "4212", "1000"],
            ])
    source_writer(".xlsx", columns, rows).replace(raw / "SO CHI TIET 01.01-31.12.2024.xlsx")
    bank_total = "1100.10" if profitable else "100.10"
    tb_rows = [
        ["1111", "Cash", "100.10", "0", "0", "100.10", "0", "0"],
        ["1121", "Bank", "0", "0", bank_total, "0", bank_total, "0"],
        ["4111", "Capital", "0", "100.10", "0", "0", "0", "100.10"],
    ]
    if profitable:
        tb_rows.extend([
            ["5111", "Revenue", "0", "0", "1000", "1000", "0", "0"],
            ["911", "Income summary", "0", "0", "1000", "1000", "0", "0"],
            ["4212", "Retained profit", "0", "0", "0", "1000", "0", "1000"],
        ])
    movements_total = "3100.10" if profitable else "100.10"
    tb_rows.append([None, "Total", "100.10", "100.10", movements_total, movements_total, bank_total, bank_total])
    source_writer(".xlsx", ["Column"] * 8, tb_rows, leading_rows=[["Toy trial balance"], ["Account headings"]]).replace(
        raw / "BANG CAN DOI PS TK 01.01-31.12.2024.xlsx"
    )
    config = {
        "general_ledger": {
            "format": format_name, "sheet": "Sheet", "header_row": 1,
            "date_format": "%d/%m/%Y", "row_labels": {},
            "columns": {column: column for column in columns},
        },
        "trial_balance": dict(tb_config, total_row=10 if profitable else 7),
    }
    (project / "config/column_map.json").write_text(json.dumps(config), encoding="utf-8")
    (project / "config/audit_config.json").write_text(json.dumps(settings), encoding="utf-8")
    shutil.copy2(PROJECT_DIR / "config/account_map.json", project / "config/account_map.json")
    shutil.copy2(PROJECT_DIR / "config/calendar_2024.json", project / "config/calendar_2024.json")
    shutil.copy2(PROJECT_DIR / "config/risk_weights.json", project / "config/risk_weights.json")
    process = subprocess.run(
        [sys.executable, str(project / "main.py")], cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    events = [json.loads(line) for line in (project / "outputs/audit_log.jsonl").read_text().splitlines()]
    assert len({event["run_id"] for event in events}) == 1
    assert all(event["status"] == "completed" for event in events)
    by_name = {event["procedure"]: event for event in events}
    assert {
        "LOAD_NORMALIZE_LEDGER", "LOAD_NORMALIZE_TRIAL_BALANCE", "CLASSIFY_LEDGER",
        "CHECK_BASIC_INTEGRITY", "RECONCILE_ACCOUNT_MOVEMENTS", "CHECK_SOURCE_CONTROLS",
        "BUILD_LEDGER_STATEMENTS_MONTHLY", "CALCULATE_MATERIALITY",
        "EXPORT_INTEGRITY", "EXPORT_LEDGER_STATEMENTS", "EXPORT_MATERIALITY", "RUN_PIPELINE",
        "LOAD_BENCHMARKS", "BUILD_FINANCIAL_ANALYSIS", "EXPORT_FINANCIAL_ANALYSIS",
    }.issubset(by_name)
    loaded = by_name["LOAD_NORMALIZE_LEDGER"]
    source_rows = (8 if profitable else 2) if format_name == "A" else (4 if profitable else 1)
    assert loaded["input_rows"] == source_rows
    assert loaded["output_rows"] == (8 if profitable else 2)
    assert by_name["BUILD_LEDGER_STATEMENTS_MONTHLY"]["output_rows"] is None
    assert by_name["BUILD_LEDGER_STATEMENTS_MONTHLY"]["output_counts"]["Monthly"] == 12
    assert by_name["BUILD_FINANCIAL_ANALYSIS"]["output_counts"]["Ratios"] == 19
    assert by_name["BUILD_FINANCIAL_ANALYSIS"]["output_rows"] is None
    financial_path = project / "outputs/05_financial_analysis.xlsx"
    book = load_workbook(financial_path, read_only=True)
    try:
        rows = list(book["Ratios"].values)
        headers = rows[0]
        ratio_records = {row[headers.index("code")]: dict(zip(headers, row)) for row in rows[1:]}
        assert len(ratio_records) == 19
        assert ratio_records["net_margin"]["value"] == ("1" if profitable else None)
        assert ratio_records["net_margin"]["assessment_status"] == ("assessed_policy" if profitable else "input_unavailable")
        assert len(list(book["Monthly_Trend"].values)) == 25
        summary = dict(book["Summary"].iter_rows(min_row=2, max_col=2, values_only=True))
        assert summary["assessment_coverage"] == "partial"
        assert summary["cfo_estimation_status"] == (
            "point_estimate_provisional" if profitable or format_name == "B" else "interval_estimate"
        )
        cfo = dict(book["CFO_Summary"].iter_rows(min_row=2, max_col=2, values_only=True))
        assert Decimal(cfo["observed_cash_movement"]) == (Decimal("1000") if profitable else Decimal("0"))
        assert "CFO_Details" in book.sheetnames and "Ratio_Ranges" in book.sheetnames
    finally:
        book.close()
    book = load_workbook(project / "outputs/01_data_integrity.xlsx", read_only=True)
    try:
        assert len(book.sheetnames) == 9
        summary = dict(list(book["Summary"].values)[1:])
        assert summary["normalized_rows"] == (8 if profitable else 2)
        assert summary["difference"] == "0.00"
        assert summary["rows_with_record_or_catalogue_issues"] == 0
        assert summary["observed_month_count"] == (2 if profitable else 1)
        assert len(list(book["Voucher_Balances"].values)) == (5 if profitable else 2)
        assert len(list(book["Account_Catalogue"].values)) == (2 if format_name == "A" and not profitable else 3)
    finally:
        book.close()

    book = load_workbook(project / "outputs/04_materiality.xlsx", read_only=True)
    try:
        assert book.sheetnames == ["Options", "Selected", "Levels", "Checks"]
        selected = dict(book["Selected"].iter_rows(min_row=2, max_col=2, values_only=True))
        assert selected["benchmark"] == "profit_before_tax"
        if profitable:
            assert selected["calculation_status"] == "calculated"
            assert selected["overall_materiality"] == "50"
            assert selected["performance_materiality"] == "25"
        else:
            assert selected["calculation_status"] == "non_positive_benchmark"
            assert selected["overall_materiality"] is None
            assert selected["performance_materiality"] is None
            assert selected["clearly_trivial_threshold"] is None
    finally:
        book.close()
    book = load_workbook(project / "outputs/03_ledger_statements.xlsx", read_only=True)
    try:
        assert len(book.sheetnames) == 10
        summary = dict(list(book["Summary"].values)[1:])
        assert summary["general_ledger_records"] == (8 if profitable else 2)
        assert summary["trial_balance_accounts"] == (6 if profitable else 3)
        assert summary["monthly_periods"] == 12
        assert summary["months_with_records"] == (2 if profitable else 1)
        assert summary["closing_cash"] == bank_total
        assert summary["check_matched"] > 0
        assert "check_difference" not in summary
        assert len(list(book["General_Ledger"].values)) == (9 if profitable else 3)
        assert len(list(book["Monthly"].values)) == 13
        balances = dict((row[0], row[1]) for row in list(book["Balance_Sheet"].values)[1:])
        assert balances["total_assets"] == bank_total
        assert balances["total_funding"] == bank_total
    finally:
        book.close()

    if not profitable:
        assert "not_run_unusable_materiality" in process.stdout
        assert not (project / "outputs/02_je_findings.xlsx").exists()
        assert not (project / "outputs/06_acceptance_sensitivity.xlsx").exists()
        assert not any(event["event_type"] == "je_procedure" for event in events)
        assert "BUILD_CHARTS_DASHBOARD" not in by_name
        return
    assert len([event for event in events if event["event_type"] == "je_procedure"]) == 10
    assert "EXPORT_JE_FINDINGS" in by_name
    assert {
        "LOAD_RISK_CONFIG", "SCORE_DATA_QUALITY", "SCORE_JOURNAL_ENTRIES",
        "SCORE_FINANCIAL_HEALTH", "SCORE_REPORTING_RISK", "SCORE_QUALITATIVE",
        "COMBINE_RISK_GROUPS", "APPLY_KNOCKOUT_RULES",
        "ANALYZE_ACCEPTANCE_SENSITIVITY", "BUILD_ACCEPTANCE_ACTIONS",
        "BUILD_ACCEPTANCE_REPORT", "EXPORT_ACCEPTANCE_REPORT",
    }.issubset(by_name)
    assert by_name["ANALYZE_ACCEPTANCE_SENSITIVITY"]["output_counts"]["scenarios"] == 15

    visual_event = by_name["BUILD_CHARTS_DASHBOARD"]
    pngs = list((project / "outputs/charts").glob("*.png"))
    htmls = list((project / "outputs").glob("dashboard.html"))
    assert visual_event["output_counts"] == {"png_files": len(pngs), "html_files": len(htmls)}
    assert visual_event["input_counts"]["ledger_records"] == 8
    assert visual_event["input_counts"]["ratio_rows"] == 19
    assert visual_event["output_rows"] is None
    assert pngs and all(path.read_bytes().startswith(b"\x89PNG") for path in pngs)
    assert (project / "outputs/charts/09_posting_weekday_heatmap.png").is_file()
    assert "Weekday chart coverage:" in process.stdout
    assert "acceptance_risk" in process.stdout

    book = load_workbook(
        project / "outputs/06_acceptance_sensitivity.xlsx", read_only=True,
    )
    try:
        assert len(book.sheetnames) == 8
        acceptance = dict(
            book["Summary"].iter_rows(min_row=2, max_col=2, values_only=True)
        )
        assert acceptance["report_status"] == "provisional"
        assert acceptance["weighted_score"] is None
        assert acceptance["decision_after_knockouts"] is None
        assert json.loads(acceptance["missing_groups"])
        assert len(list(book["Sensitivity"].values)) == 16
        checks = list(book["Checks"].values)
        status_column = checks[0].index("check_status")
        assert all(row[status_column] != "difference" for row in checks[1:])
    finally:
        book.close()

    book = load_workbook(project / "outputs/02_je_findings.xlsx", read_only=True)
    try:
        assert len(book.sheetnames) == 20
        assert len(list(book["Summary"].values)) == 11
        assert len(list(book["Run_Log"].values)) == 11
        assert len(list(book["JE04"].values)) == 3
        assert len(list(book["Benford_Frequency"].values)) == 37
        assert len(list(book["Benford_Statistics"].values)) == 5
        assert "JE performance materiality: 25" in process.stdout
    finally:
        book.close()


def test_missing_source_leaves_failed_step_and_pipeline_trace(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(PROJECT_DIR / "main.py", project / "main.py")
    shutil.copytree(PROJECT_DIR / "auditpy", project / "auditpy", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT_DIR / "config", project / "config")
    process = subprocess.run([sys.executable, str(project / "main.py")], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode != 0
    events = [json.loads(line) for line in (project / "outputs/audit_log.jsonl").read_text().splitlines()]
    assert len({event["run_id"] for event in events}) == 1
    failed = [event for event in events if event["status"] == "error"]
    assert [event["procedure"] for event in failed] == ["LOAD_NORMALIZE_LEDGER", "RUN_PIPELINE"]
    assert all(event["error_type"] == "FileNotFoundError" and event["output_rows"] is None for event in failed)
    assert not (project / "outputs/01_data_integrity.xlsx").exists()
