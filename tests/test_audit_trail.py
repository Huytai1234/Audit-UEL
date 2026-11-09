"""Verify application-wide, explicitly counted audit events and failure reporting."""

import json

import pandas as pd
import pytest

from auditpy.utils import audit_log, run_audited, table_row_counts


def test_decorator_logs_a_regular_function_and_preserves_its_identity(tmp_path):
    @audit_log
    def transform(table):
        return table.assign(amount=table["amount"] * 2)

    path = tmp_path / "events.jsonl"
    result = transform(
        pd.DataFrame({"amount": [10, 20]}), log_path=path, run_id="one",
        input_counts={"source_rows": 2}, output_counts=lambda table: {"result_rows": len(table)},
    )
    event = json.loads(path.read_text())
    assert transform.__name__ == "transform"
    assert result["amount"].tolist() == [20, 40]
    assert event["input_rows"] == event["output_rows"] == 2
    assert event["procedure"] == "transform"
    assert event["event_type"] == "pipeline_step"
    assert event["status"] == "completed"


def test_report_dictionary_uses_per_table_counts_without_summing_units(tmp_path):
    path = tmp_path / "events.jsonl"
    tables = {"records": pd.DataFrame({"x": [1, 2, 3]}), "summary": pd.DataFrame({"x": [6]})}
    run_audited(lambda: tables, log_path=path, run_id="one", input_counts={"records": 3}, output_counts=table_row_counts)
    event = json.loads(path.read_text())
    assert event["output_counts"] == {"records": 3, "summary": 1}
    assert event["output_rows"] is None
    with pytest.raises(TypeError):
        table_row_counts({"worksheet_count": 2})


def test_non_tabular_configuration_has_no_invented_row_counts(tmp_path):
    path = tmp_path / "events.jsonl"
    run_audited(lambda: {"rate": 0.05, "year": 2024}, log_path=path, run_id="one")
    event = json.loads(path.read_text())
    assert event["input_rows"] is None and event["output_rows"] is None
    assert event["output_counts"] == {}


def test_loader_counts_source_rows_separately_from_normalized_rows(tmp_path):
    table = pd.DataFrame({"source_row": [2, 2, 3, 3]})
    path = tmp_path / "events.jsonl"
    run_audited(
        lambda: table, log_path=path, run_id="one",
        input_counts=lambda result: {"source_rows": int(result["source_row"].nunique())},
        output_counts=lambda result: {"normalized_rows": len(result)},
    )
    event = json.loads(path.read_text())
    assert event["input_rows"] == 2
    assert event["output_rows"] == 4


def test_missing_file_is_logged_before_processing_and_error_is_preserved(tmp_path):
    path = tmp_path / "events.jsonl"
    with pytest.raises(FileNotFoundError):
        run_audited(lambda: (tmp_path / "missing.xlsx").read_text(), log_path=path, run_id="one")
    event = json.loads(path.read_text())
    assert event["status"] == "error"
    assert event["error_type"] == "FileNotFoundError"
    assert "missing.xlsx" in event["error_message"]
    assert event["output_rows"] is None


@pytest.mark.parametrize("counts", [{"rows": True}, {"rows": -1}, {"rows": 1.5}])
def test_invalid_declared_counts_fail_instead_of_creating_false_evidence(tmp_path, counts):
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        run_audited(lambda: None, log_path=path, run_id="one", input_counts=counts)
    assert json.loads(path.read_text())["status"] == "error"


def test_metadata_cannot_claim_a_different_run_or_success_status(tmp_path):
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        run_audited(lambda: None, log_path=path, run_id="one", metadata={"run_id": "other", "status": "completed"})
    event = json.loads(path.read_text())
    assert event["run_id"] == "one" and event["status"] == "error"
