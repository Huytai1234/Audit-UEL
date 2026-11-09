"""Build small, explicitly synthetic accounting inputs for regression tests."""

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from auditpy.loaders import normalize_codes_and_amounts, normalize_date_columns
from auditpy.utils import normalize_amount


@pytest.fixture
def audit_settings():
    path = Path(__file__).resolve().parents[1] / "config/audit_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def calendar_reference():
    path = Path(__file__).resolve().parents[1] / "config/calendar_2024.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def account_map():
    path = Path(__file__).resolve().parents[1] / "config/account_map.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def ledger_factory():
    """Create normalized journal/control records with source references."""
    def build(rows):
        records = []
        for source_row, changes in enumerate(rows, start=2):
            record = {
                "voucher_no": "0001", "account": "1111",
                "counter_account": "1121", "description": "Sample entry",
                "posting_date": date(2024, 1, 3),
                "document_date": date(2024, 1, 3),
                "debit": "0", "credit": "0",
                "balance_debit": None, "balance_credit": None,
                "row_type": "journal_candidate", "source_file": "sample.csv",
                "source_sheet": None, "source_row": source_row,
            }
            record.update(changes)
            records.append(record)
        table = pd.DataFrame(records, dtype=object)
        table["debit_raw"] = table["debit"]
        table["credit_raw"] = table["credit"]
        return normalize_date_columns(normalize_codes_and_amounts(table), "%d/%m/%Y")

    return build


@pytest.fixture
def trial_balance_factory():
    """Create normalized client reference records, including a declared total if needed."""
    def build(rows):
        records = []
        for source_row, changes in enumerate(rows, start=4):
            record = {
                "account": "1111", "account_name": "Sample account",
                "opening_debit": "0", "opening_credit": "0",
                "debit": "0", "credit": "0",
                "closing_debit": "0", "closing_credit": "0",
                "source_file": "sample_tb.xlsx", "source_sheet": "Sheet",
                "source_row": source_row,
            }
            record.update(changes)
            for field in (
                "opening_debit", "opening_credit", "debit", "credit",
                "closing_debit", "closing_credit",
            ):
                record[field], record[field + "_status"] = normalize_amount(record[field])
            records.append(record)
        return pd.DataFrame(records, dtype=object)

    return build


@pytest.fixture
def source_writer(tmp_path):
    """Write a temporary CSV/Excel source without touching the supplied raw files."""
    def write(suffix, headers, rows, leading_rows=()):
        path = tmp_path / ("sample" + suffix)
        if suffix == ".csv":
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(leading_rows)
                writer.writerow(headers)
                writer.writerows(rows)
        else:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sheet"
            for row in leading_rows:
                sheet.append(row)
            sheet.append(headers)
            for row in rows:
                sheet.append(row)
            workbook.save(path)
            workbook.close()
        return path

    return write
