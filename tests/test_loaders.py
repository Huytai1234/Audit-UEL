"""Check format selection, source traceability, normalization, and schema failures."""

from datetime import date
from decimal import Decimal

import pytest

from auditpy.loaders import BaseLoader, LineFormatLoader, PairFormatLoader, create_loader
from auditpy.utils import SchemaError, normalize_amount, normalize_code, normalize_date


def loader_config(format_name, headers):
    return {
        "format": format_name, "sheet": "Sheet", "header_row": 1,
        "date_format": "%d/%m/%Y", "columns": {name: name for name in headers},
    }


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_line_format_preserves_identifiers_amounts_and_source_rows(source_writer, suffix):
    headers = [
        "voucher_no", "posting_date", "document_date", "description",
        "account", "debit", "credit", "extra_note",
    ]
    path = source_writer(suffix, headers, [
        ["0001", "03/04/2024", "03/04/2024", "Deposit", "00111", "100.10", "0", "keep"],
        ["0001", "03/04/2024", "03/04/2024", "Deposit", "1121", "0", "100.10", "keep"],
    ])
    loader = create_loader(path, loader_config("A", headers))
    table = loader.load()
    assert isinstance(loader, LineFormatLoader)
    assert list(table["voucher_no"]) == ["0001", "0001"]
    assert table.loc[0, "account"] == "00111"
    assert table.loc[0, "debit"] == Decimal("100.10")
    assert table.loc[0, "posting_date"] == date(2024, 4, 3)
    assert list(table["source_row"]) == [2, 3]
    assert table.loc[0, "source_sheet"] == ("Sheet" if suffix == ".xlsx" else None)
    assert table["extra_note"].eq("keep").all()


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_pair_format_expands_valid_pairs_and_retains_unresolved_inputs(source_writer, suffix):
    headers = [
        "voucher_no", "posting_date", "document_date", "description",
        "debit_account", "credit_account", "amount",
    ]
    path = source_writer(suffix, headers, [
        ["0001", "03/01/2024", "03/01/2024", "Deposit", "1121", "1111", "100.10"],
        ["0002", "03/01/2024", "03/01/2024", "Missing account", "1121", "", "20"],
        ["0003", "03/01/2024", "03/01/2024", "Missing amount", "1121", "1111", ""],
    ])
    loader = create_loader(path, loader_config("B", headers))
    table = loader.load()
    assert isinstance(loader, PairFormatLoader)
    assert list(table["source_row"]) == [2, 2, 3, 4]
    assert list(table["source_side"]) == ["debit", "credit", "unresolved", "unresolved"]
    assert list(table["pair_status"]) == [
        "expanded", "expanded", "missing_account", "missing_amount",
    ]
    assert list(table["debit"]) == [Decimal("100.10"), Decimal("0"), None, None]
    assert list(table["credit"]) == [Decimal("0"), Decimal("100.10"), None, None]


def test_missing_required_column_and_unknown_format_are_rejected(source_writer):
    headers = ["voucher_no", "posting_date", "document_date", "description", "account", "debit"]
    path = source_writer(".csv", headers, [["1", "03/01/2024", "03/01/2024", "Entry", "1111", "0"]])
    with pytest.raises(SchemaError, match="credit"):
        create_loader(path, loader_config("A", headers)).load()
    with pytest.raises(SchemaError):
        create_loader(path, loader_config("C", headers))
    with pytest.raises(TypeError):
        BaseLoader(path, loader_config("A", headers))


def test_ambiguous_amounts_missing_values_and_dates_remain_distinct():
    assert normalize_code(" 00111 ") == "00111"
    assert normalize_code(1111.0) == "1111"
    assert normalize_amount(None) == (None, "missing")
    assert normalize_amount("0") == (Decimal("0"), "ok")
    assert normalize_amount("-5.25") == (Decimal("-5.25"), "ok")
    for value in (True, "1,234.56", "NaN", "Infinity"):
        assert normalize_amount(value) == (None, "invalid")
    assert normalize_date("03/04/2024") == (date(2024, 4, 3), "ok")
    assert normalize_date("31/02/2024") == (None, "invalid")
    assert normalize_date("12/31/2024") == (None, "invalid")


def test_csv_multiline_description_retains_physical_source_reference(source_writer):
    headers = ["voucher_no", "posting_date", "document_date", "description", "account", "debit", "credit"]
    path = source_writer(".csv", headers, [[
        "0001", "03/01/2024", "03/01/2024", "First line\nSecond line", "1111", "0", "0",
    ]])
    table = create_loader(path, loader_config("A", headers)).load()
    assert table.loc[0, "source_row"] == 3
    assert table.loc[0, "description"] == "First line\nSecond line"
