"""Normalize data, export reports and record audit events."""
import json
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import chain
from pathlib import Path
from functools import wraps
from time import perf_counter
from tempfile import TemporaryDirectory

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def is_finite_amount(value):
    """Recognize a finite Decimal; do not convert or fill the input."""
    return isinstance(value, Decimal) and value.is_finite()


def is_missing(value):
    """Identify null values and blank strings without treating zero as missing."""
    if pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def normalize_code(value):
    """Normalize an identifier while preserving existing leading zeros."""
    if is_missing(value):
        return None

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return str(value).strip()


def normalize_amount(value):
    """Return a Decimal amount and its conversion status."""
    if is_missing(value):
        return None, "missing"

    if isinstance(value, bool):
        return None, "invalid"

    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        return None, "invalid"

    if not amount.is_finite():
        return None, "invalid"

    return amount, "ok"


def normalize_date(value, date_format="%d/%m/%Y"):
    """Return a date and its conversion status using the specified text format."""
    if is_missing(value):
        return None, "missing"

    if isinstance(value, datetime):
        return value.date(), "ok"

    if isinstance(value, date):
        return value, "ok"

    if isinstance(value, str):
        try:
            parsed_date = datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            return None, "invalid"

        return parsed_date, "ok"

    return None, "invalid"


class SchemaError(Exception):
    """The source schema does not meet the loader requirements."""


def compare_amounts(name, actual, expected, tolerance=Decimal("0")):
    """Compare numeric evidence while preserving missing values and tolerance."""
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative.")
    difference = actual - expected if actual is not None and expected is not None else None
    status = "unavailable" if difference is None else (
        "matched" if abs(difference) <= tolerance else "difference"
    )
    return dict(check_name=name, actual=actual, expected=expected,
                difference=difference, check_status=status)


def write_excel_report(path, tables):
    """Write DataFrames while preserving exact amounts and source formulas as text."""
    path = Path(path)
    if not tables or path.suffix.lower() != ".xlsx":
        raise ValueError("Provide at least one table and an .xlsx output path.")

    names = list(tables)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Worksheet names must be unique ignoring case.")

    for name, table in tables.items():
        if not name or len(name) > 31 or any(char in name for char in "\\/*?:[]"):
            raise ValueError("Invalid worksheet name: " + name)
        if len(table) >= 1048576 or not 0 < len(table.columns) <= 16384:
            raise ValueError("Table exceeds Excel worksheet limits: " + name)

    def make_cell(sheet, value):
        if isinstance(value, (tuple, list, dict)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        elif not isinstance(value, str) and pd.isna(value):
            value = None
        elif isinstance(value, Decimal):
            value = format(value, "f")
        elif isinstance(value, float):
            value = str(value)
        elif isinstance(value, (date, datetime)):
            value = value.isoformat()

        if isinstance(value, str) and len(value) > 32767:
            raise ValueError("Cell text exceeds Excel limits in " + sheet.title)

        cell = WriteOnlyCell(sheet, value=value)
        if isinstance(value, str):
            cell.data_type = "s"
        return cell

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)
    row_counts = {}

    try:
        for name, table in tables.items():
            sheet = workbook.create_sheet(name)
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = (
                "A1:" + get_column_letter(len(table.columns))
                + str(len(table) + 1)
            )
            for position, column in enumerate(table.columns, start=1):
                width = 65 if column in ("value", "description") else 24
                sheet.column_dimensions[get_column_letter(position)].width = width

            records = chain(
                [table.columns], table.itertuples(index=False, name=None),
            )
            for row_number, values in enumerate(records, start=1):
                cells = [make_cell(sheet, value) for value in values]
                if row_number == 1:
                    for cell in cells:
                        cell.font = Font(bold=True)
                sheet.append(cells)

            row_counts[name] = len(table)

        with TemporaryDirectory(dir=path.parent) as temporary_dir:
            temporary_path = Path(temporary_dir) / "report.xlsx"
            workbook.save(temporary_path)
            temporary_path.replace(path)
    finally:
        for sheet in workbook.worksheets:
            if not sheet.closed:
                sheet.close()
        workbook.close()

    return row_counts

def table_row_counts(tables):
    """Count each DataFrame separately; never count dictionary keys as rows."""
    if not isinstance(tables, dict) or any(not isinstance(table, pd.DataFrame) for table in tables.values()):
        raise TypeError("Provide a mapping of table names to DataFrames.")
    return {name: len(table) for name, table in tables.items()}


def read_json_config(path):
    """Read a configuration inside an audited operation."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _checked_row_counts(counts):
    if counts is None:
        return {}
    if not isinstance(counts, dict) or any(
        not isinstance(name, str) or not name.strip() or type(count) is not int or count < 0
        for name, count in counts.items()
    ):
        raise ValueError("Row counts require named non-negative integers.")
    return counts.copy()


def audit_log(function):
    """Record the procedure, timing, row counts and errors in one JSONL log."""
    @wraps(function)
    def wrapper(
        *args, log_path, run_id, audit_name=None, input_counts=None, output_counts=None,
        metadata=None, event_type="pipeline_step", **kwargs,
    ):
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        started_clock = perf_counter()
        record = {
            "run_id": run_id, "event_type": event_type,
            "procedure": audit_name or function.__name__,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_counts": {}, "output_counts": None,
            "input_rows": None, "output_rows": None,
            "status": "error", "error_type": None, "error_message": None,
        }
        reserved = set(record) | {"elapsed_seconds"}
        try:
            if metadata and reserved.intersection(metadata):
                raise ValueError("Audit metadata cannot overwrite lifecycle fields.")
            record.update(metadata or {})
            if not callable(input_counts):
                record["input_counts"] = _checked_row_counts(input_counts)
            result = function(*args, **kwargs)
            if callable(input_counts):
                record["input_counts"] = _checked_row_counts(input_counts(result))
            record["output_counts"] = _checked_row_counts(output_counts(result) if output_counts else None)
            record["status"] = "completed"
            return result
        except Exception as error:
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
            record["output_counts"] = None
            raise
        finally:
            record["elapsed_seconds"] = round(perf_counter() - started_clock, 6)
            for direction in ("input", "output"):
                counts = record[direction + "_counts"]
                record[direction + "_rows"] = (
                    next(iter(counts.values()))
                    if isinstance(counts, dict) and len(counts) == 1
                    and all(type(value) is int and value >= 0 for value in counts.values())
                    else None
                )
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return wrapper


@audit_log
def run_audited(function, *args, **kwargs):
    """Execute a callable through the shared audit decorator."""
    return function(*args, **kwargs)
