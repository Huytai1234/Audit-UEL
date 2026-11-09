"""Check monetary meaning, overlap, full references and the Excel report contract."""

from copy import deepcopy
from decimal import Decimal as D

import pytest
from openpyxl import load_workbook

from auditpy.je_tests import build_je_report_tables
from auditpy.models import Finding, RiskLevel
from auditpy.utils import write_excel_report, SchemaError


def sample_report_inputs():
    """Build report-only examples, not additional journal-entry procedures."""
    results = {f"JE{n:02d}": {"findings": [], "coverage": {"finding_count": 0}} for n in range(1, 11)}
    def finding(code, refs, amount=D("0.10"), risk=RiskLevel.MEDIUM):
        return Finding(code, None, None, ("1111",), amount, "synthetic_measure",
                       "Synthetic report-only evidence", risk, "sample.csv", None, refs)
    results["JE04"]["findings"] = [finding("JE04", ((2, None),))]
    results["JE05"]["findings"] = [finding("JE05", ((2, None),))]
    results["JE03"]["findings"] = [finding("JE03", ((997, None), (998, None)), None)]
    results["JE10"]["findings"] = [finding("JE10", ((999, None),), None)]
    results["JE09"]["findings"] = [
        finding("JE09", ((4, "debit"),), None, RiskLevel.LOW),
        finding("JE09", ((4, "credit"),), D("0.20"), RiskLevel.HIGH),
    ]
    for result in results.values():
        result["coverage"]["finding_count"] = len(result["findings"])
    events = [{"test_code": code, "run_id": "sample", "status": "completed", "output_rows": len(result["findings"])}
              for code, result in results.items()]
    return results, events


def test_report_deduplicates_sources_but_keeps_both_pair_sides():
    results, events = sample_report_inputs()
    before = deepcopy(results)
    tables = build_je_report_tables(results, {}, events, {}, {})
    assert results == before
    assert len(tables["Source_References"]) == 7
    assert len(tables["Flagged_Records"]) == 3
    assert set(tables["Flagged_Records"]["source_row"]) == {2, 4}
    assert tables["Flagged_Records"].iloc[0]["test_codes"] == ("JE04", "JE05")
    assert tables["JE09"]["risk_level"].tolist() == ["HIGH", "LOW"]
    summary = tables["Summary"].set_index("test_code")
    assert summary.loc["JE03", "known_amount_sum"] is None
    assert summary.loc["JE10", "known_amount_sum"] is None
    assert summary.loc["JE09", "known_amount_sum"] == D("0.20")
    assert summary.loc["JE09", "unknown_amount_findings"] == 1
    assert summary.loc["JE08", "known_amount_sum"] == D("0")
    assert summary.loc["JE03", "flagged_source_records"] is None


@pytest.mark.parametrize("problem", ["missing_event", "different_runs", "failed_event", "wrong_count", "duplicate_code"])
def test_report_rejects_incomplete_or_inconsistent_runs(problem):
    results, events = sample_report_inputs()
    if problem == "missing_event":
        events.pop()
    elif problem == "different_runs":
        events[0]["run_id"] = "other"
    elif problem == "failed_event":
        events[0]["status"] = "error"
    elif problem == "wrong_count":
        events[0]["output_rows"] = 999
    else:
        events[0]["test_code"] = events[1]["test_code"]
    with pytest.raises(ValueError):
        build_je_report_tables(results, {}, events, {}, {})


def test_report_rejects_broken_procedure_coverage():
    results, events = sample_report_inputs()
    results["JE01"]["coverage"]["finding_count"] = 999
    with pytest.raises(SchemaError):
        build_je_report_tables(results, {}, events, {}, {})


def test_report_keeps_empty_sheets_and_exact_decimal_text(tmp_path):
    results, events = sample_report_inputs()
    tables = build_je_report_tables(results, {}, events, {}, {})
    path = tmp_path / "report.xlsx"
    row_counts = write_excel_report(path, tables)
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        assert len(book.sheetnames) == 20
        rows = list(book["Summary"].values)
        amount_index = list(rows[0]).index("known_amount_sum")
        assert rows[3][amount_index] is None
        assert rows[4][amount_index] == "0.10"
        assert row_counts["JE08"] == 0
        assert next(book["JE08"].values)[0] == "finding_id"
        assert list(book["JE08"].values)[1:] == []
    finally:
        book.close()
