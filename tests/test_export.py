"""Check exact-value export, literal formulas, and output replacement behavior."""

from decimal import Decimal

import pandas as pd
import pytest
from openpyxl import load_workbook

from auditpy.utils import write_excel_report


def test_excel_round_trip_keeps_exact_money_identifiers_and_literal_formulas(tmp_path):
    table = pd.DataFrame({
        "account": ["00111"], "amount": [Decimal("4549575320108.53234660")],
        "source_formula": ["=SUM(A1:A2)"], "source_rows": [(2, 3)],
        "missing": [None], "zero": [Decimal("0")], "negative": [Decimal("-0.03507850")],
    }, dtype=object)
    before = table.copy(deep=True)
    path = tmp_path / "report.xlsx"
    counts = write_excel_report(path, {"Sample": table, "Empty": table.iloc[:0]})
    book = load_workbook(path, data_only=False)
    try:
        sheet = book["Sample"]
        assert sheet["A2"].value == "00111"
        assert sheet["B2"].value == "4549575320108.53234660"
        assert sheet["C2"].value == "=SUM(A1:A2)" and sheet["C2"].data_type == "s"
        assert sheet["D2"].value == "[2, 3]"
        assert sheet["E2"].value is None
        assert sheet["F2"].value == "0" and sheet["G2"].value == "-0.03507850"
        assert book["Empty"].max_row == 1
        assert counts == {"Sample": 1, "Empty": 0}
    finally:
        book.close()
    pd.testing.assert_frame_equal(table, before)


def test_failed_export_does_not_replace_report_or_silently_truncate_text(tmp_path):
    path = tmp_path / "report.xlsx"
    write_excel_report(path, {"Original": pd.DataFrame({"value": ["keep"]})})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Cell text exceeds"):
        write_excel_report(path, {"TooLong": pd.DataFrame({"value": ["x" * 32768]})})
    assert path.read_bytes() == before
