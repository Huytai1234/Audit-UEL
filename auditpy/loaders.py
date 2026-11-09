"""Read CSV/Excel journals in formats A/B and client trial balance records."""

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from auditpy.utils import (
    SchemaError,
    is_missing,
    normalize_amount,
    normalize_code,
    normalize_date,
)


def read_benchmark_table(path):
    """Read editable numeric policies without requiring descriptive metadata."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def read_excel_table(
    path, sheet_name, header_row=1, max_rows=None, use_column_letters=False,
):
    """Read Excel records with source names or Excel column letters."""
    path = Path(path)
    workbook = load_workbook(path, read_only=True, data_only=False)

    try:
        sheet = workbook[sheet_name]
        if use_column_letters:
            column_names = [
                get_column_letter(index)
                for index in range(1, sheet.max_column + 1)
            ]
        else:
            column_names = [cell.value for cell in sheet[header_row]]
        validate_column_names(column_names)

        last_row = sheet.max_row
        if max_rows is not None:
            last_row = min(last_row, header_row + max_rows)

        rows = sheet.iter_rows(
            min_row=header_row + 1,
            max_row=last_row,
            max_col=sheet.max_column,
            values_only=True,
        )
        table = pd.DataFrame(rows, columns=column_names, dtype=object)
        table["source_file"] = path.name
        table["source_sheet"] = sheet.title
        table["source_row"] = range(
            header_row + 1, header_row + 1 + len(table),
        )
        return table
    finally:
        workbook.close()

def apply_column_mapping(table, column_map):
    """Rename mapped columns while retaining the other source columns."""
    return table.rename(columns=column_map, errors="raise")


def normalize_codes_and_amounts(table):
    """Normalize available identifier and amount fields on a copy of the table."""
    result = table.copy()

    for column in (
        "voucher_no", "account", "counter_account",
        "entity_code", "currency_code",
    ):
        if column not in result.columns:
            continue

        result[column] = result[column].map(normalize_code)

    for column in (
        "debit", "credit", "foreign_debit",
        "foreign_credit", "balance_debit", "balance_credit",
    ):
        if column not in result.columns:
            continue

        parsed_values = result[column].map(normalize_amount)

        result[column] = pd.Series(
            [item[0] for item in parsed_values],
            index=result.index,
            dtype=object,
        )
        result[column + "_status"] = [item[1] for item in parsed_values]

    return result


def normalize_date_columns(table, date_format):
    """Normalize posting and document dates while retaining conversion status."""
    result = table.copy()

    for column in ("posting_date", "document_date"):
        parsed_values = [
            normalize_date(value, date_format)
            for value in result[column]
        ]

        result[column] = pd.Series(
            [item[0] for item in parsed_values],
            index=result.index,
            dtype=object,
        )
        result[column + "_status"] = [item[1] for item in parsed_values]

    return result


def validate_column_names(column_names):
    """Reject missing, duplicate, or reserved source column names."""
    if not column_names or any(is_missing(column) for column in column_names):
        raise SchemaError("Missing column names.")
    if len(set(column_names)) != len(column_names):
        raise SchemaError("Duplicate column names.")
    if set(column_names) & {"source_file", "source_sheet", "source_row"}:
        raise SchemaError("Source columns conflict with reserved source-reference fields.")


def read_table(path, config, max_rows=None):
    """Read Excel or CSV records and attach source references."""
    path = Path(path)

    if path.suffix.lower() == ".xlsx":
        return read_excel_table(
            path,
            config["sheet"],
            config.get("header_row", 1),
            max_rows,
        )

    if path.suffix.lower() != ".csv":
        raise SchemaError("Supported file types are .xlsx and .csv.")

    rows, source_rows = [], []
    skip_rows = config.get("header_row", 1) - 1

    with path.open(
        encoding=config.get("encoding", "utf-8-sig"),
        newline="",
    ) as file:
        for _ in range(skip_rows):
            next(file, "")

        reader = csv.DictReader(
            file,
            delimiter=config.get("delimiter", ","),
            strict=True,
        )
        column_names = reader.fieldnames
        validate_column_names(column_names)

        for row in reader:
            if None in row:
                raise SchemaError("A CSV record contains more fields than the header.")

            rows.append(row)
            source_rows.append(reader.line_num + skip_rows)

            if max_rows is not None and len(rows) >= max_rows:
                break

    table = pd.DataFrame(rows, columns=column_names, dtype=object)
    table["source_file"] = path.name
    table["source_sheet"] = None
    table["source_row"] = source_rows
    return table


class BaseLoader(ABC):
    """Define the common interface and column mapping for journal loaders."""

    def __init__(self, path, config):
        self._path = Path(path)
        self._config = config

    def _read_and_map_columns(self, max_rows):
        table = read_table(self._path, self._config, max_rows)

        try:
            table = apply_column_mapping(table, self._config["columns"])
        except KeyError as error:
            raise SchemaError("Column mapping does not match the source: " + str(error))

        if not table.columns.is_unique:
            raise SchemaError("Column mapping produces duplicate column names.")

        return table

    @abstractmethod
    def load(self, max_rows=None):
        """Return normalized records without discarding source rows."""
        raise NotImplementedError


class LineFormatLoader(BaseLoader):
    """Load journal records with one account per row (format A)."""

    def load(self, max_rows=None):
        table = self._read_and_map_columns(max_rows)

        required_columns = {
            "voucher_no", "posting_date", "document_date",
            "description", "account", "debit", "credit",
        }
        missing_columns = required_columns - set(table.columns)

        if missing_columns:
            raise SchemaError("Missing required columns: " + ", ".join(sorted(missing_columns)))

        table["debit_raw"] = table["debit"]
        table["credit_raw"] = table["credit"]
        table = normalize_codes_and_amounts(table)
        return normalize_date_columns(
            table,
            self._config["date_format"],
        )

class PairFormatLoader(BaseLoader):
    """Expand paired journal records into format A without inventing missing data."""

    def load(self, max_rows=None):
        table = self._read_and_map_columns(max_rows)
        required_columns = {
            "voucher_no", "posting_date", "document_date", "description",
            "debit_account", "credit_account", "amount",
        }
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise SchemaError(
                "Missing required columns: " + ", ".join(sorted(missing_columns))
            )

        output_columns = [
            "account", "counter_account", "debit", "credit",
            "source_side", "pair_status", "amount_status",
        ]
        if set(output_columns) & set(table.columns):
            raise SchemaError("Format B source contains reserved output columns.")

        rows = []
        for row in table.to_dict("records"):
            debit_account = normalize_code(row["debit_account"])
            credit_account = normalize_code(row["credit_account"])
            amount, amount_status = normalize_amount(row["amount"])
            row["amount_status"] = amount_status
            row["pair_status"] = "expanded"

            if debit_account is None or credit_account is None:
                row["pair_status"] = "missing_account"
            elif amount_status != "ok":
                row["pair_status"] = amount_status + "_amount"

            if row["pair_status"] != "expanded":
                rows.append(dict(
                    row, account=None, counter_account=None,
                    debit=None, credit=None, source_side="unresolved",
                ))
                continue

            for side, account, counter_account in (
                ("debit", debit_account, credit_account),
                ("credit", credit_account, debit_account),
            ):
                line = dict(
                    row, account=account, counter_account=counter_account,
                    debit=0, credit=0, source_side=side,
                )
                line[side] = amount
                rows.append(line)

        result = pd.DataFrame(
            rows, columns=list(table.columns) + output_columns, dtype=object,
        )
        result = normalize_codes_and_amounts(result)
        return normalize_date_columns(result, self._config["date_format"])


def create_loader(path, config):
    """Select the accounting representation; file type is handled by read_table."""
    if config.get("format") == "A":
        return LineFormatLoader(path, config)
    if config.get("format") == "B":
        return PairFormatLoader(path, config)
    raise SchemaError("Accounting data format must be 'A' or 'B'.")

def read_trial_balance(path, config):
    """Read the client-supplied trial balance without removing control rows."""
    table = read_excel_table(
        path, config["sheet"],
        header_row=config["data_start_row"] - 1,
        use_column_letters=True,
    )
    table = apply_column_mapping(table, config["columns"])
    if not table.columns.is_unique:
        raise SchemaError("Trial balance mapping produces duplicate columns.")

    amount_columns = (
        "opening_debit", "opening_credit", "debit", "credit",
        "closing_debit", "closing_credit",
    )
    required_columns = {"account", "account_name"} | set(amount_columns)
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise SchemaError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    table["account"] = table["account"].map(normalize_code)
    for column in amount_columns:
        parsed_values = table[column].map(normalize_amount)
        table[column] = pd.Series(
            [item[0] for item in parsed_values],
            index=table.index, dtype=object,
        )
        table[column + "_status"] = [item[1] for item in parsed_values]

    return table
