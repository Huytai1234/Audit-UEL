"""Verify audit trails, fail-fast execution and same-run materiality integration."""

import json
from decimal import Decimal as D

import pandas as pd
import pytest

from auditpy.je_tests import JournalEntryTest, run_je_procedure, run_je_tests
from auditpy.utils import SchemaError
from main import build_je_tables


class SyntheticProcedure(JournalEntryTest):
    """Exercise the runner contract without introducing another production procedure."""
    def __init__(self, code, behavior="ok"):
        self.test_code, self.behavior = code, behavior
        self._coverage = {"review_records": 2, "finding_count": 0}

    def run(self, classified, performance_materiality):
        if self.behavior == "wrong_type":
            return ["not a Finding"]
        if self.behavior == "wrong_count":
            self._coverage["finding_count"] = 1
        if self.behavior == "fail":
            raise ValueError("Synthetic failure")
        return []


def test_audit_trail_keeps_zero_outputs_and_separate_runs(tmp_path):
    path = tmp_path / "nested/events.jsonl"
    frame = pd.DataFrame({"account": ["1111", "1121"]})
    procedures = [SyntheticProcedure(f"JE{n:02d}") for n in range(1, 11)]
    _, first_id = run_je_tests(procedures, frame, D("500"), path)
    _, second_id = run_je_tests(procedures, frame, D("500"), path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert first_id != second_id
    assert len(events) == 20
    assert all(event["output_rows"] == 0 and event["status"] == "completed" for event in events)
    assert all(event["elapsed_seconds"] >= 0 and event["input_rows"] == 2 for event in events)
    assert all(event["started_at_utc"].endswith("+00:00") for event in events)
    assert run_je_procedure.__name__ == "run_je_procedure"


@pytest.mark.parametrize("behavior,error", [("wrong_type", TypeError), ("wrong_count", SchemaError), ("fail", ValueError)])
def test_audit_trail_records_and_reraises_failures(tmp_path, behavior, error):
    path = tmp_path / "events.jsonl"
    with pytest.raises(error):
        run_je_procedure(SyntheticProcedure("JE01", behavior), pd.DataFrame({"x": [1, 2]}), D("1"), log_path=path, run_id="failure")
    event = json.loads(path.read_text())
    assert event["status"] == "error"
    assert event["output_rows"] is None
    assert event["error_type"] == error.__name__


def test_runner_rejects_missing_or_duplicate_procedures_before_execution(tmp_path):
    procedures = [SyntheticProcedure(f"JE{n:02d}") for n in range(1, 11)]
    path = tmp_path / "events.jsonl"
    for invalid in (procedures[:9], procedures[:9] + [procedures[0]]):
        with pytest.raises(ValueError):
            run_je_tests(invalid, pd.DataFrame(), D("1"), path)
    assert not path.exists()


def test_failure_stops_batch_before_later_procedures(tmp_path):
    procedures = [SyntheticProcedure(f"JE{n:02d}") for n in range(1, 11)]
    procedures[1].behavior = "fail"
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        run_je_tests(procedures, pd.DataFrame({"x": [1, 2]}), D("1"), path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["test_code"] for event in events] == ["JE01", "JE02"]
    assert [event["status"] for event in events] == ["completed", "error"]


def test_je_report_uses_in_memory_threshold_without_materiality_workbook(tmp_path, ledger_factory, audit_settings, calendar_reference, account_map):
    frame = ledger_factory([{"posting_date": "06/01/2024", "debit": "100"}])
    before = frame.copy(deep=True)
    selection = {"calculation_status": "calculated", "performance_materiality": D("50"),
                 "period_start": audit_settings["period_start"], "period_end": audit_settings["period_end"]}
    tables = build_je_tables(frame, audit_settings, account_map, calendar_reference, selection, tmp_path / "events.jsonl")
    assert tables["JE04"].iloc[0]["risk_level"] == "HIGH"
    assert len(tables["Run_Log"]) == 10
    assert set(tables["Run_Log"]["performance_materiality"]) == {"50"}
    assert not list(tmp_path.glob("*.xlsx"))
    pd.testing.assert_frame_equal(frame, before)
    selection["performance_materiality"] = D("500")
    second = build_je_tables(frame, audit_settings, account_map, calendar_reference, selection, tmp_path / "events.jsonl")
    assert second["JE04"].iloc[0]["risk_level"] == "MEDIUM"


@pytest.mark.parametrize("problem", ["missing_threshold", "zero_threshold", "wrong_period", "unavailable", "wrong_calendar"])
def test_je_report_rejects_unusable_or_incompatible_context(tmp_path, ledger_factory, audit_settings, calendar_reference, account_map, problem):
    selection = {"calculation_status": "calculated", "performance_materiality": D("50"),
                 "period_start": audit_settings["period_start"], "period_end": audit_settings["period_end"]}
    if problem == "missing_threshold":
        selection["performance_materiality"] = None
    elif problem == "zero_threshold":
        selection["performance_materiality"] = D("0")
    elif problem == "wrong_period":
        selection["period_end"] = "2023-12-31"
    elif problem == "unavailable":
        selection["calculation_status"] = "non_positive_benchmark"
    else:
        calendar_reference["year"] = 2025
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        build_je_tables(ledger_factory([{}]), audit_settings, account_map, calendar_reference, selection, path)
    assert not path.exists()
