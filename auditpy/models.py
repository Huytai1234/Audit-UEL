"""Represent normalized accounting records without changing source evidence."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional
from enum import Enum
from auditpy.utils import read_json_config, is_finite_amount


@dataclass(frozen=True)
class JournalLine:
    """Store one normalized journal record, including unresolved information."""

    source_file: str
    source_sheet: Optional[str]
    source_row: int
    row_type: str
    voucher_no: Optional[str]
    posting_date: Optional[date]
    document_date: Optional[date]
    description: Optional[str]
    account: Optional[str]
    debit: Optional[Decimal]
    credit: Optional[Decimal]
    counter_account: Optional[str] = None
    entity_code: Optional[str] = None
    source_side: Optional[str] = None

    def __post_init__(self):
        for amount in (self.debit, self.credit):
            if amount is not None and (
                not is_finite_amount(amount)
            ):
                raise ValueError(
                    "Journal amounts must be finite Decimal values or None."
                )

@dataclass(frozen=True)
class Account:
    """Identify an account without asserting catalogue membership."""

    code: str
    name: Optional[str] = None

    def __post_init__(self):
        if (
            not isinstance(self.code, str)
            or not self.code
            or self.code != self.code.strip()
        ):
            raise ValueError("Account codes must be non-empty normalized strings.")

@dataclass(frozen=True)
class JournalEntry:
    """Represent a group under the declared voucher key."""

    source_file: str
    source_sheet: Optional[str]
    posting_date: date
    voucher_no: str
    lines: tuple

    def __post_init__(self):
        if not isinstance(self.lines, tuple) or not self.lines:
            raise ValueError("JournalEntry requires a non-empty tuple of journal lines.")

        if not all(isinstance(line, JournalLine) for line in self.lines):
            raise TypeError("JournalEntry requires JournalLine objects.")

        if (
            not isinstance(self.source_file, str) or not self.source_file.strip()
            or not isinstance(self.voucher_no, str) or not self.voucher_no.strip()
            or type(self.posting_date) is not date
        ):
            raise ValueError("JournalEntry requires a file, voucher number and posting date.")

        fields = ("source_file", "source_sheet", "posting_date", "voucher_no")
        expected = tuple(getattr(self, field) for field in fields)

        if any(
            tuple(getattr(line, field) for field in fields) != expected
            for line in self.lines
        ):
            raise ValueError("Journal lines do not match the entry key.")

    @property
    def difference(self):
        if any(line.debit is None or line.credit is None for line in self.lines):
            return None

        return sum(
            (line.debit - line.credit for line in self.lines), Decimal("0"),
        )

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Decision(Enum):
    ACCEPT = 0
    CONDITIONAL = 1
    REJECT = 2


@dataclass(frozen=True)
class Finding:
    """Store immutable findings, retaining unresolved format-B source references."""

    test_code: str
    voucher_no: Optional[str]
    posting_date: Optional[date]
    accounts: tuple
    amount: Optional[Decimal]
    amount_basis: str
    description: str
    risk_level: RiskLevel
    source_file: str
    source_sheet: Optional[str]
    source_refs: tuple = ()
    risk_amount: Optional[Decimal] = None

    def __post_init__(self):
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError("A RiskLevel member is required.")
        for value in (self.amount, self.risk_amount):
            if value is not None and (
                not is_finite_amount(value) or value < 0
            ):
                raise ValueError("Finding amounts must be non-negative finite Decimal values or None.")
        if not isinstance(self.accounts, tuple) or any(
            not isinstance(code, str) or not code.strip() for code in self.accounts
        ):
            raise ValueError("Accounts must be a tuple of non-empty strings.")
        if not isinstance(self.source_refs, tuple) or any(
            not isinstance(ref, tuple) or len(ref) != 2
            or type(ref[0]) is not int or ref[0] < 1
            or ref[1] not in (None, "debit", "credit", "unresolved")
            for ref in self.source_refs
        ):
            raise ValueError("Source references must be tuples of positive row numbers and sides.")
        for value in (self.test_code, self.amount_basis, self.description, self.source_file):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Finding identifiers, description and amount basis are required.")

    def __lt__(self, other):
        if not isinstance(other, Finding):
            return NotImplemented
        return self.risk_level.value < other.risk_level.value

@dataclass(frozen=True)
class RatioResult:
    """Store a calculation result; availability is not a financial risk rating."""

    code: str
    value: Optional[Decimal]
    unit: str
    calculation_status: str
    reason: Optional[str] = None

    def __post_init__(self):
        if self.calculation_status == "calculated":
            if (
                not is_finite_amount(self.value)
                or self.reason is not None
            ):
                raise ValueError(
                    "Calculated ratios require a finite Decimal and no reason."
                )
        elif self.calculation_status == "unavailable":
            if (
                self.value is not None
                or not isinstance(self.reason, str)
                or not self.reason.strip()
            ):
                raise ValueError("Unavailable ratios require None and a reason.")
        else:
            raise ValueError("Unknown ratio calculation status.")

@dataclass(frozen=True)
class Client:
    """Identify the case while retaining unverified client information."""

    reference: str
    name: Optional[str] = None
    industry: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("A nonempty client reference is required.")

    @classmethod
    def from_config(cls, path):
        info = read_json_config(path)["client"]
        return cls(
            reference=info["reference"],
            name=info.get("name"),
            industry=info.get("industry"),
        )


class AuditEngagement:
    """Keep the client, ledger, JE procedures and results together."""

    def __init__(self, client, ledger):
        self.client = client
        self.ledger = ledger
        self.tests = []
        self.results = {}