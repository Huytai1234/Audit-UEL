"""Check journal-entry evidence across the supported source representations."""

from decimal import Decimal

import pytest

from auditpy.je_tests import VoucherSequenceGapTest
from auditpy.loaders import create_loader
from auditpy.validators import classify_ledger_rows


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_sequence_gap_retains_unresolved_pair_boundary(source_writer, suffix):
    """A valid voucher number remains sequence evidence when its pair is incomplete."""
    headers = [
        "voucher_no", "posting_date", "document_date", "description",
        "debit_account", "credit_account", "amount",
    ]
    path = source_writer(suffix, headers, [
        ["PC24010001", "03/01/2024", "03/01/2024", "Deposit", "1121", "1111", "100"],
        ["PC24010004", "04/01/2024", "04/01/2024", "Incomplete pair", "1121", "", "20"],
    ])
    config = {
        "format": "B", "sheet": "Sheet", "header_row": 1,
        "date_format": "%d/%m/%Y", "columns": {name: name for name in headers},
    }
    classified = classify_ledger_rows(create_loader(path, config).load(), {})
    procedure = VoucherSequenceGapTest(
        [r"(?P<series>PC[0-9]{4})(?P<sequence>[0-9]{4})"],
        "Synthetic sample: the captured series is a sequential register.",
    )

    findings = procedure.run(classified, Decimal("500"))

    assert len(findings) == 1
    assert findings[0].amount is None
    assert findings[0].risk_amount is None
    assert findings[0].source_file == path.name
    assert findings[0].source_sheet == ("Sheet" if suffix == ".xlsx" else None)
    assert findings[0].source_refs == ((2, "debit"), (3, "unresolved"))
    assert procedure.coverage["parsed_identifiers"] == 2
    assert procedure.coverage["missing_sequence_numbers"] == 2
