"""Reconstruct provisional financial statements and monthly observations."""

from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from datetime import date
from decimal import Decimal

import pandas as pd

from auditpy.ledger import GeneralLedger, TrialBalance
from auditpy.validators import prepare_opening_balance_reference, make_reconciliation_check
from auditpy.models import JournalLine
from auditpy.utils import SchemaError, is_finite_amount, compare_amounts

def classify_closing_lines(
    entries, unassigned, account_map, balance_tolerance=Decimal("0"),
):
    """Annotate every line; preserve amounts and flag uncertain closing evidence."""
    pl_prefixes = tuple(account_map["income_statement_accounts"])
    summary = account_map["closing_accounts"]["income_summary"]
    equity = account_map["closing_accounts"]["retained_earnings"]

    prefixes = pl_prefixes + (summary, equity)
    if not pl_prefixes or any(
        not isinstance(prefix, str) or len(prefix) != 3 or not prefix.isdigit()
        for prefix in prefixes
    ):
        raise SchemaError("Three-digit account prefixes are required.")

    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    rows = []

    def add_line(line, entry_difference, grouping_status, peers=()):
        account = line.account or ""
        counter = line.counter_account or ""
        own_pl = account.startswith(pl_prefixes)
        counter_pl = counter.startswith(pl_prefixes)

        if own_pl and counter.startswith(summary):
            kind = "pl_closing"
        elif account.startswith(summary) and counter_pl:
            kind = "summary_counterpart"
        elif (
            account.startswith(summary) and counter.startswith(equity)
            or account.startswith(equity) and counter.startswith(summary)
        ):
            kind = "profit_transfer"
        elif (
            own_pl and counter.startswith(equity)
            or account.startswith(equity) and counter_pl
        ):
            kind = "direct_equity_review"
        elif account.startswith(summary) or counter.startswith(summary):
            kind = "unmapped_summary_review"
        elif own_pl and not counter:
            kind = "missing_counter_review"
        else:
            kind = "not_identified"

        excluded = kind in (
            "pl_closing", "summary_counterpart", "profit_transfer",
        )

        reciprocal = None
        if excluded and grouping_status == "assigned":
            reciprocal = line.debit is not None and line.credit is not None and any(
                peer.account == line.counter_account and peer.counter_account == line.account
                and peer.debit == line.credit and peer.credit == line.debit
                for peer in peers
            )

        review = kind.endswith("_review") or excluded and (
            grouping_status == "unassigned"
            or entry_difference is None
            or abs(entry_difference) > balance_tolerance
            or reciprocal is False
        )

        row = asdict(line)
        row.update(
            closing_kind=kind,
            exclude_from_income_statement=excluded,
            closing_review_required=bool(review),
            reciprocal_line_found=reciprocal,
            grouping_status=grouping_status,
            entry_difference=entry_difference,
        )
        rows.append(row)

    for entry in entries:
        difference = entry.difference
        for line in entry.lines:
            add_line(line, difference, "assigned", entry.lines)

    for line in unassigned:
        add_line(line, None, "unassigned")

    columns = [field.name for field in fields(JournalLine)] + [
        "closing_kind", "exclude_from_income_statement", "closing_review_required",
        "grouping_status", "entry_difference", "reciprocal_line_found",
    ]
    result = pd.DataFrame(rows, columns=columns, dtype=object)

    for column in ("exclude_from_income_statement", "closing_review_required"):
        result[column] = result[column].astype(bool)

    return result

class FinancialStatement(ABC):
    """Define a common interface for reconstructed financial statements."""

    @abstractmethod
    def build(self):
        """Return the reconstructed statement table."""
        raise NotImplementedError


class IncomeStatement(FinancialStatement):
    """Calculate provisional results from mapped, non-closing movements."""

    ITEMS = (
        "revenue", "revenue_deductions", "financial_income", "cost_of_sales",
        "finance_costs", "selling_expenses", "administrative_expenses",
        "other_income", "other_expenses", "income_tax_expense",
    )
    CREDIT_ITEMS = {"revenue", "financial_income", "other_income"}

    def __init__(self, closing_lines, account_map):
        required = {
            "account", "debit", "credit",
            "exclude_from_income_statement", "closing_review_required",
            "grouping_status",
        }
        if not required.issubset(closing_lines.columns) or closing_lines.empty:
            raise SchemaError("A non-empty classified ledger is required.")

        if not all(
            pd.api.types.is_bool_dtype(closing_lines[column])
            for column in ("exclude_from_income_statement", "closing_review_required")
        ):
            raise SchemaError("Closing flags must be boolean columns.")

        mapping = dict(account_map["income_statement_accounts"])
        if set(mapping.values()) != set(self.ITEMS) or any(
            not isinstance(code, str) or len(code) != 3 or not code.isdigit()
            for code in mapping
        ):
            raise SchemaError("The income-statement mapping is incomplete or invalid.")

        active = closing_lines.loc[
            ~closing_lines["exclude_from_income_statement"]
        ].copy()
        roots = active["account"].str[:3]
        active["statement_item"] = roots.map(mapping)

        production = account_map["production_cost_accounts"]
        unexpected = (
            active["account"].fillna("").str.startswith(("5", "6", "7", "8"))
            & active["statement_item"].isna()
            & ~roots.isin(production)
        )
        if unexpected.any():
            raise SchemaError("Unmapped income-statement account prefixes.")

        unknown = active.loc[active["account"].isna(), ["debit", "credit"]]
        if any(
            value is None or value != 0
            for row in unknown.itertuples(index=False)
            for value in row
        ):
            raise SchemaError("Unallocated nonzero or missing amounts require review.")

        self._data = active.loc[active["statement_item"].notna()].copy()
        self._review = closing_lines.loc[
            closing_lines["closing_review_required"]
            | closing_lines["grouping_status"].eq("unassigned")
            | closing_lines["account"].isna()
        ].copy()

    @property
    def review_lines(self):
        return self._review.copy(deep=True)

    @property
    def components(self):
        rows = []
        for item in self.ITEMS:
            group = self._data.loc[self._data["statement_item"].eq(item)]
            complete = group[["debit", "credit"]].notna().all().all()

            debit = sum(group["debit"], Decimal("0")) if complete else None
            credit = sum(group["credit"], Decimal("0")) if complete else None
            amount = (
                credit - debit if item in self.CREDIT_ITEMS else debit - credit
            ) if complete else None

            rows.append({
                "item": item,
                "amount": amount,
                "line_count": len(group),
                "calculation_status": (
                    "unavailable" if not complete else
                    "calculated" if len(group) else "no_observed_records"
                ),
            })

        return pd.DataFrame(rows, dtype=object)

    def build(self):
        values = self.components.set_index("item")["amount"].to_dict()

        def combine(positive, negative=()):
            if any(values[item] is None for item in positive + negative):
                return None
            return (
                sum((values[item] for item in positive), Decimal("0"))
                - sum((values[item] for item in negative), Decimal("0"))
            )

        values["net_revenue"] = combine(("revenue",), ("revenue_deductions",))
        values["gross_profit"] = combine(("net_revenue",), ("cost_of_sales",))
        values["operating_profit"] = combine(
            ("gross_profit", "financial_income"),
            ("finance_costs", "selling_expenses", "administrative_expenses"),
        )
        values["other_profit"] = combine(("other_income",), ("other_expenses",))
        values["profit_before_tax"] = combine(("operating_profit", "other_profit"))
        values["profit_after_tax"] = combine(
            ("profit_before_tax",), ("income_tax_expense",),
        )

        order = (
            "net_revenue", "cost_of_sales", "gross_profit", "selling_expenses",
            "administrative_expenses", "financial_income", "finance_costs",
            "operating_profit", "other_income", "other_expenses", "other_profit",
            "profit_before_tax", "income_tax_expense", "profit_after_tax",
        )

        return pd.DataFrame([
            {
                "item": item,
                "amount": values[item],
                "calculation_status": (
                    "calculated" if values[item] is not None else "unavailable"
                ),
                "report_status": "provisional",
            }
            for item in order
        ], dtype=object)

def prepare_balance_sheet_balances(
    classified, reference, trial_balance, balance_tolerance=Decimal("0"),
):
    """Preserve reported debit/credit balances and compare reconstructed net amounts."""
    required = {
        "account", "row_type", "source_file", "source_sheet", "source_row",
    }
    control_fields = {
        "balance_debit", "balance_credit", "balance_debit_status", "balance_credit_status",
    }
    reference_fields = {
        "account", "closing_debit", "closing_credit", "reference_status",
        "source_file", "source_sheet", "source_rows",
    }
    if not required.issubset(classified.columns) or not reference_fields.issubset(reference.columns):
        raise SchemaError("Balance-sheet source columns are missing.")

    if reference["account"].isna().any() or reference["account"].duplicated().any():
        raise SchemaError("Client references require unique account codes.")

    for source in (classified, reference):
        if len(source[["source_file", "source_sheet"]].drop_duplicates()) > 1:
            raise SchemaError("Prepare one ledger and one client source at a time.")

    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    clients = reference.set_index("account").to_dict("index")
    groups = {
        account: group for account, group in classified.loc[
            classified["account"].notna()
        ].groupby("account", sort=False)
    }
    empty = classified.iloc[:0]
    rows = []

    for calculated in trial_balance.table.to_dict("records"):
        account = calculated["account"]
        group = groups.get(account, empty)
        client = clients.get(account)
        closing = group.loc[group["row_type"].eq("closing_balance")]

        debit = credit = None
        source_file = source_sheet = None
        source_rows = ()
        basis = "unavailable"

        if client is not None and client["reference_status"] in ("unique", "identical_repeat"):
            debit, credit = client["closing_debit"], client["closing_credit"]
            source_file, source_sheet = client["source_file"], client["source_sheet"]
            source_rows = client["source_rows"]
            basis = "client_trial_balance"

        elif client is None and len(closing) == 1 and control_fields.issubset(classified.columns):
            control = closing.iloc[0]
            if all(
                control[field + "_status"] == "ok" and control[field] is not None
                for field in ("balance_debit", "balance_credit")
            ):
                debit, credit = control["balance_debit"], control["balance_credit"]
                source_file, source_sheet = control["source_file"], control["source_sheet"]
                source_rows = (control["source_row"],)
                basis = "ledger_closing"

        reported_net = debit - credit if debit is not None and credit is not None else None
        reconstructed_net = calculated["closing_net"]
        difference = (
            reconstructed_net - reported_net
            if reconstructed_net is not None and reported_net is not None else None
        )
        status = "unavailable" if difference is None else (
            "matched_net" if abs(difference) <= balance_tolerance else "difference"
        )

        postings = group.loc[group["row_type"].isin([
            "journal_candidate", "unclassified", "unresolved_pair",
        ])]
        opening = group.loc[group["row_type"].eq("opening_balance")]

        rows.append({
            "account": account,
            "closing_debit": debit,
            "closing_credit": credit,
            "reported_closing_net": reported_net,
            "reconstructed_closing_net": reconstructed_net,
            "net_difference": difference,
            "net_check_status": status,
            "balance_basis": basis,
            "balance_source_file": source_file,
            "balance_source_sheet": source_sheet,
            "balance_source_rows": source_rows,
            "ledger_line_count": len(postings),
            "entity_count": postings["entity_code"].nunique() if "entity_code" in postings else None,
            "missing_entity_rows": (
                int(postings["entity_code"].isna().sum()) if "entity_code" in postings else None
            ),
            "entity_opening_rows": (
                int(opening["entity_code"].notna().sum()) if "entity_code" in opening else None
            ),
        })

    columns = [
        "account", "closing_debit", "closing_credit", "reported_closing_net",
        "reconstructed_closing_net", "net_difference", "net_check_status",
        "balance_basis", "balance_source_file", "balance_source_sheet",
        "balance_source_rows", "ledger_line_count", "entity_count",
        "missing_entity_rows", "entity_opening_rows",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object)

BALANCE_SHEET_ASSET_ITEMS = (
    "cash", "receivables", "inventories", "other_current_assets",
    "net_fixed_assets", "other_noncurrent_assets", "unclassified_assets",
)
BALANCE_SHEET_FUNDING_ITEMS = (
    "current_liabilities", "noncurrent_liabilities",
    "equity", "unclassified_liabilities",
)


def default_balance_maturity(item):
    """Use an item's explicit maturity; leave dual-purpose balances unresolved."""
    if item in ("cash", "inventories", "other_current_assets", "current_liabilities"):
        return "current"
    if item in ("net_fixed_assets", "other_noncurrent_assets", "noncurrent_liabilities"):
        return "noncurrent"
    return "not_applicable" if item == "equity" else "unresolved"


def map_balance_sheet_items(balances, account_map):
    """Map each reported balance side and retain its source evidence."""
    required = {"account", "closing_debit", "closing_credit"}
    if not required.issubset(balances.columns) or balances.empty:
        raise SchemaError("Non-empty presentation balances are required.")

    codes = balances["account"].tolist()
    if (
        not all(isinstance(code, str) and code.strip() for code in codes)
        or len(codes) != len(set(codes))
        or any(a != b and b.startswith(a) for a in codes for b in codes)
    ):
        raise SchemaError("Account scopes must be valid and non-overlapping.")

    rules = account_map["balance_sheet_accounts"]
    items = BALANCE_SHEET_ASSET_ITEMS + BALANCE_SHEET_FUNDING_ITEMS
    if any(
        not isinstance(prefix, str) or not prefix.isdigit()
        or not isinstance(targets, list) or len(targets) != 2
        or any(target not in items for target in targets)
        for prefix, targets in rules.items()
    ):
        raise SchemaError("Invalid balance-sheet mapping rules.")

    maturity_rules = account_map.get("balance_sheet_maturity", {})
    if not isinstance(maturity_rules, dict) or any(
        not isinstance(prefix, str) or not prefix.isdigit()
        or maturity not in ("current", "noncurrent", "unresolved")
        for prefix, maturity in maturity_rules.items()
    ):
        raise SchemaError("Invalid balance-sheet maturity assumptions.")
    prefixes = sorted(rules, key=len, reverse=True)
    maturity_prefixes = sorted(maturity_rules, key=len, reverse=True)
    rows = []

    for balance in balances.to_dict("records"):
        account = balance["account"]
        prefix = next((p for p in prefixes if account.startswith(p)), None)

        if prefix is None:
            if balance["closing_debit"] == 0 and balance["closing_credit"] == 0:
                continue
            raise SchemaError(f"Map the nonzero or unavailable account balance: {account}")

        for index, side in enumerate(("closing_debit", "closing_credit")):
            target = rules[prefix][index]
            original_item = target
            maturity = default_balance_maturity(target)
            maturity_prefix = next((p for p in maturity_prefixes if account.startswith(p)), None)
            maturity_basis = "unresolved" if maturity == "unresolved" else "account_map"
            if maturity_prefix is not None:
                if target not in ("receivables", "unclassified_assets", "unclassified_liabilities", "current_liabilities", "noncurrent_liabilities"):
                    raise SchemaError("Maturity overrides apply only to receivables and unresolved/dated liabilities.")
                maturity = maturity_rules[maturity_prefix]
                maturity_basis = "unresolved" if maturity == "unresolved" else "configured_assumption"
                if maturity == "unresolved" and target in ("current_liabilities", "noncurrent_liabilities"):
                    target = "unclassified_liabilities"
                if maturity != "unresolved":
                    if target == "unclassified_assets":
                        target = "other_current_assets" if maturity == "current" else "other_noncurrent_assets"
                    elif target in ("unclassified_liabilities", "current_liabilities", "noncurrent_liabilities"):
                        target = "current_liabilities" if maturity == "current" else "noncurrent_liabilities"
            amount = balance[side]

            if amount is not None and (
                not is_finite_amount(amount)
            ):
                raise SchemaError("Balances must be finite Decimal values or None.")

            positive_side = (
                "closing_debit" if target in BALANCE_SHEET_ASSET_ITEMS else "closing_credit"
            )
            if amount is not None and side != positive_side:
                amount = -amount

            rows.append(dict(
                balance,
                mapping_prefix=prefix,
                balance_side=side,
                item=target,
                amount=amount,
                item_before_maturity=original_item,
                maturity=maturity,
                maturity_prefix=maturity_prefix,
                maturity_basis=maturity_basis,
            ))

    columns = list(balances.columns) + [
        "mapping_prefix", "balance_side", "item", "amount",
        "item_before_maturity", "maturity", "maturity_prefix", "maturity_basis",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object)

class BalanceSheet(FinancialStatement):
    """Aggregate mapped balances while keeping classification gaps explicit."""

    def __init__(
        self, mapped_items, *, balance_tolerance=Decimal("0"), notes=(),
    ):
        required = {"item", "amount"}
        items = BALANCE_SHEET_ASSET_ITEMS + BALANCE_SHEET_FUNDING_ITEMS

        if not required.issubset(mapped_items.columns) or mapped_items.empty:
            raise SchemaError("Non-empty mapped balance-sheet items are required.")

        if not mapped_items["item"].isin(items).all():
            raise SchemaError("Unknown balance-sheet item names.")

        if any(
            value is not None and (
                not is_finite_amount(value)
            )
            for value in mapped_items["amount"]
        ):
            raise SchemaError("Mapped amounts must be finite Decimal values or None.")

        keys = ["account", "balance_side"]
        if set(keys).issubset(mapped_items.columns) and mapped_items.duplicated(keys).any():
            raise SchemaError("Duplicate balance-side contributions are not allowed.")

        if not balance_tolerance.is_finite() or balance_tolerance < 0:
            raise ValueError("Balance tolerance must be finite and non-negative.")

        self._details = mapped_items.copy(deep=True)
        if "maturity" not in self._details:
            self._details["maturity"] = self._details["item"].map(default_balance_maturity)
        if "maturity_basis" not in self._details:
            self._details["maturity_basis"] = "account_map"
        if not self._details["maturity"].isin(("current", "noncurrent", "unresolved", "not_applicable")).all():
            raise SchemaError("Unknown balance maturity classifications.")
        for row in self._details.itertuples():
            expected = default_balance_maturity(row.item)
            if expected != "unresolved" and row.maturity != expected:
                raise SchemaError("Balance-sheet item and maturity must agree.")
        self._tolerance = balance_tolerance
        self._notes = tuple(notes)

    @property
    def details(self):
        return self._details.copy(deep=True)

    @property
    def notes(self):
        return self._notes

    @staticmethod
    def _sum_amounts(amounts):
        amounts = tuple(amounts)
        return sum(amounts, Decimal("0")) if all(v is not None for v in amounts) else None

    def build(self):
        items = BALANCE_SHEET_ASSET_ITEMS + BALANCE_SHEET_FUNDING_ITEMS
        values = {
            item: self._sum_amounts(
                self._details.loc[self._details["item"].eq(item), "amount"]
            )
            for item in items
        }

        def total(names):
            return self._sum_amounts(values[name] for name in names)

        def maturity_total(names, maturity):
            selected = self._details["item"].isin(names) & self._details["maturity"].eq(maturity)
            return self._sum_amounts(self._details.loc[selected, "amount"])

        values["current_assets"] = maturity_total(BALANCE_SHEET_ASSET_ITEMS, "current")
        values["noncurrent_assets"] = maturity_total(BALANCE_SHEET_ASSET_ITEMS, "noncurrent")
        values["total_assets"] = total(BALANCE_SHEET_ASSET_ITEMS)
        values["total_liabilities"] = total((
            "current_liabilities", "noncurrent_liabilities", "unclassified_liabilities",
        ))
        values["total_funding"] = total(("total_liabilities", "equity"))
        values["balance_difference"] = (
            values["total_assets"] - values["total_funding"]
            if values["total_assets"] is not None
            and values["total_funding"] is not None else None
        )

        incomplete = set()
        assumed_totals = set(self._details.loc[
            self._details["maturity_basis"].eq("configured_assumption")
            & self._details["amount"].map(lambda value: value is None or value != 0), "item"
        ])
        for names, totals in (
            (BALANCE_SHEET_ASSET_ITEMS, ("current_assets", "noncurrent_assets")),
            (("current_liabilities", "noncurrent_liabilities", "unclassified_liabilities"), ("current_liabilities", "noncurrent_liabilities")),
        ):
            relevant = self._details.loc[self._details["item"].isin(names)]
            active = relevant.loc[relevant["amount"].map(lambda value: value is None or value != 0)]
            if active["maturity"].eq("unresolved").any():
                incomplete.update(totals)
            if active["maturity_basis"].eq("configured_assumption").any():
                # Both sides of a partition depend on the assumed allocation, even a zero side.
                assumed_totals.update(totals)

        order = (
            "cash", "receivables", "inventories", "other_current_assets",
            "current_assets", "net_fixed_assets", "other_noncurrent_assets",
            "noncurrent_assets", "unclassified_assets", "total_assets",
            "current_liabilities", "noncurrent_liabilities", "unclassified_liabilities",
            "total_liabilities", "equity", "total_funding", "balance_difference",
        )

        return pd.DataFrame([
            {
                "item": item,
                "amount": None if item in incomplete else values[item],
                "mapped_amount": values[item],
                "calculation_status": (
                    "unavailable" if values[item] is None else
                    "classification_incomplete" if item in incomplete else "calculated"
                ),
                "report_status": "provisional",
                "calculation_basis": (
                    "maturity_unresolved" if item in incomplete else
                    "includes_disclosed_maturity_assumptions" if item in assumed_totals else
                    "mapped_source_balances"
                ),
            }
            for item in order
        ], dtype=object)

    @property
    def balance_check(self):
        result = self.build().set_index("item")
        difference = result.loc["balance_difference", "amount"]
        return {
            "difference": difference,
            "tolerance": self._tolerance,
            "within_tolerance": (
                abs(difference) <= self._tolerance if difference is not None else None
            ),
        }

def build_monthly_summary(
    closing_lines, trial_balance, account_map, *,
    cash_closing_reference=None, balance_tolerance=Decimal("0"),
):
    """Summarize monthly results and roll forward observed cash balances."""
    start = trial_balance.scope["period_start"]
    end = trial_balance.scope["period_end"]

    if type(start) is not date or type(end) is not date or start > end:
        raise ValueError("A valid reporting period is required.")

    if "posting_date" not in closing_lines or any(
        type(value) is not date or not start <= value <= end
        for value in closing_lines["posting_date"]
    ):
        raise SchemaError("Posting dates do not fit the monthly reporting period.")

    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    if cash_closing_reference is not None and (
        not is_finite_amount(cash_closing_reference)
    ):
        raise ValueError("The cash closing reference must be a finite Decimal or None.")

    sum_amounts = BalanceSheet._sum_amounts
    annual = IncomeStatement(closing_lines, account_map).build()
    annual_values = annual.set_index("item")["amount"].to_dict()

    rules = account_map["balance_sheet_accounts"]
    prefixes = sorted(rules, key=len, reverse=True)
    cash_codes = []
    for account in trial_balance.table["account"]:
        prefix = next((p for p in prefixes if account.startswith(p)), None)
        if prefix is not None and "cash" in rules[prefix]:
            cash_codes.append(account)
    opening_cash = None
    if cash_codes:
        opening = trial_balance.table.loc[
            trial_balance.table["account"].isin(cash_codes),
            ["account", "opening_debit", "opening_credit"],
        ].rename(columns={
            "opening_debit": "closing_debit", "opening_credit": "closing_credit",
        })
        opening_items = map_balance_sheet_items(opening, account_map)
        opening_cash = sum_amounts(opening_items.loc[opening_items["item"].eq("cash"), "amount"])

    rows = []

    for period in pd.period_range(start, end, freq="M"):
        first = max(start, period.start_time.date())
        last = min(end, period.end_time.date())
        month_rows = closing_lines.loc[
            closing_lines["posting_date"].between(first, last)
        ]

        if month_rows.empty:
            income = {item: Decimal("0") for item in annual_values}
        else:
            income = IncomeStatement(
                month_rows, account_map,
            ).build().set_index("item")["amount"].to_dict()

        cash_rows = month_rows.loc[month_rows["account"].isin(cash_codes)]
        debit = sum_amounts(cash_rows["debit"]) if cash_codes else None
        credit = sum_amounts(cash_rows["credit"]) if cash_codes else None
        movement = debit - credit if debit is not None and credit is not None else None
        closing_cash = (
            opening_cash + movement
            if opening_cash is not None and movement is not None else None
        )

        rows.append({
            "month": str(period),
            "period_start": first,
            "period_end": last,
            "record_count": len(month_rows),
            "activity_status": (
                "observed_records" if len(month_rows) else "no_observed_records"
            ),
            **income,
            "operating_expenses": sum_amounts(
                income[item] for item in ("selling_expenses", "administrative_expenses")
            ),
            "opening_cash": opening_cash,
            "cash_debit": debit,
            "cash_credit": credit,
            "cash_net_movement": movement,
            "closing_cash": closing_cash,
            "cash_record_count": len(cash_rows),
            "cash_status": "calculated" if closing_cash is not None else "unavailable",
            "report_status": "provisional",
        })

        opening_cash = closing_cash

    monthly = pd.DataFrame(rows, dtype=object)
    if sum(monthly["record_count"]) != len(closing_lines):
        raise SchemaError("Monthly record coverage failed.")

    checks = []

    def add_check(name, actual, expected):
        checks.append(compare_amounts(name, actual, expected, balance_tolerance))

    for item, expected in annual_values.items():
        add_check(item, sum_amounts(monthly[item]), expected)

    all_cash = closing_lines.loc[closing_lines["account"].isin(cash_codes)]
    for side in ("debit", "credit"):
        add_check(
            "cash_" + side,
            sum_amounts(monthly["cash_" + side]),
            sum_amounts(all_cash[side]) if cash_codes else None,
        )

    add_check("period_end_cash", monthly.iloc[-1]["closing_cash"], cash_closing_reference)
    return monthly, pd.DataFrame(checks, dtype=object)

def build_opening_balance_sheet(
    trial_table, account_map, balance_tolerance=Decimal("0")
):
    """Reuse statement mapping for opening balances and retain evidence."""
    columns = [
        "account", "opening_debit", "opening_credit", "opening_basis",
        "opening_source_file", "opening_source_sheet",
        "opening_source_rows",
    ]
    if not set(columns).issubset(trial_table.columns):
        raise SchemaError(
            "Selected opening balances and source evidence are required."
        )

    aliases = {
        "opening_debit": "closing_debit",
        "opening_credit": "closing_credit",
    }
    presentation = trial_table[columns].rename(
        columns=aliases
    ).copy()

    details = map_balance_sheet_items(presentation, account_map)
    details = details.rename(
        columns={value: key for key, value in aliases.items()}
    )
    details["balance_side"] = details["balance_side"].replace({
        "closing_debit": "opening_debit",
        "closing_credit": "opening_credit",
    })

    statement = BalanceSheet(
        details, balance_tolerance=balance_tolerance
    ).build()
    statement["balance_basis"] = "selected_opening_balances"

    return statement, details


def build_statement_tables(classified, client, reference, config, settings, account_map):
    """Build the ledger, financial statements, monthly totals and reconciliations."""
    tolerance = Decimal(settings["balance_tolerance"])
    strict_sum = BalanceSheet._sum_amounts
    ledger = GeneralLedger.from_classified_table(classified)
    opening_reference = prepare_opening_balance_reference(classified, reference)
    trial = TrialBalance.from_general_ledger(
        ledger, opening_reference,
        period_start=date.fromisoformat(settings["period_start"]),
        period_end=date.fromisoformat(settings["period_end"]), unit=None,
    )
    ledger_table = ledger.to_table(trial, reference)
    entries, unassigned = ledger.group_entries(settings["voucher_key"])
    closing_lines = classify_closing_lines(entries, unassigned, account_map, tolerance)
    balances = prepare_balance_sheet_balances(classified, reference, trial, tolerance)
    bs_items = map_balance_sheet_items(balances, account_map)
    balance_sheet = BalanceSheet(
        bs_items, balance_tolerance=tolerance, notes=account_map["balance_sheet_notes"],
    )
    income_statement = IncomeStatement(closing_lines, account_map)
    statement_objects = {
        "Balance_Sheet": balance_sheet, "Income_Statement": income_statement,
    }
    statement_tables = {name: statement.build() for name, statement in statement_objects.items()}
    bs_values = statement_tables["Balance_Sheet"].set_index("item")["amount"].to_dict()
    monthly, monthly_checks = build_monthly_summary(
        closing_lines, trial, account_map,
        cash_closing_reference=bs_values["cash"], balance_tolerance=tolerance,
    )

    entity_accounts = tuple(
        code for code in trial.table["account"]
        if code.startswith(tuple(account_map["entity_review_prefixes"]))
    )
    entities = ledger.entity_movements(entity_accounts)
    entity_scope = GeneralLedger(line for line in ledger if line.account in entity_accounts)
    checks = []

    def check(group, name, actual, expected, account=None, allowed=Decimal("0")):
        checks.append(make_reconciliation_check(
            name, actual, expected, group=group, account=account, tolerance=allowed,
        ))

    original = ledger.movements()
    regrouped = GeneralLedger(
        tuple(line for entry in entries for line in entry.lines) + unassigned
    ).movements()
    check("coverage", "ledger_records", len(ledger_table), len(ledger))
    check("coverage", "voucher_records", regrouped["line_count"], len(ledger))
    check("coverage", "closing_records", len(closing_lines), len(ledger))
    check("coverage", "monthly_records", sum(monthly["record_count"]), len(ledger))
    check("coverage", "entity_records", sum(entities["line_count"]), len(entity_scope))
    for side in ("debit", "credit"):
        for name, actual in (
            ("ledger_" + side, strict_sum(ledger_table[side])),
            ("voucher_" + side, regrouped[side]),
            ("closing_" + side, strict_sum(closing_lines[side])),
        ):
            check("amount_conservation", name, actual, original[side], allowed=tolerance)
        check(
            "entity_movements", side, strict_sum(entities[side]),
            entity_scope.movements()[side] if len(entity_scope) else Decimal("0"),
            allowed=tolerance,
        )

    endings = ledger_table.loc[ledger_table["account"].notna()].groupby(
        "account", sort=False,
    ).tail(1).set_index("account")["running_net_balance"].to_dict()
    for row in trial.table.to_dict("records"):
        actual = endings.get(row["account"])
        if row["ledger_line_count"] == 0:
            actual = (
                row["opening_debit"] - row["opening_credit"]
                if row["opening_debit"] is not None and row["opening_credit"] is not None else None
            )
        check("running_balances", "closing_net", actual, row["closing_net"], row["account"], tolerance)
    for row in balances.to_dict("records"):
        check(
            "reported_balances", "closing_net", row["reconstructed_closing_net"],
            row["reported_closing_net"], row["account"], tolerance,
        )

    total_rows = client.loc[client["source_row"].eq(config["trial_balance"]["total_row"])]
    if len(total_rows) != 1:
        raise SchemaError("A unique client trial-balance control total is required.")
    footer = total_rows.iloc[0]
    for side in ("debit", "credit"):
        check(
            "source_totals", "opening_" + side, strict_sum(trial.table["opening_" + side]),
            footer["opening_" + side], allowed=tolerance,
        )
        check(
            "source_totals", "closing_" + side, strict_sum(balances["closing_" + side]),
            footer["closing_" + side], allowed=tolerance,
        )
    source_net = (
        strict_sum(balances["closing_debit"]) - strict_sum(balances["closing_credit"])
        if balances[["closing_debit", "closing_credit"]].notna().all().all() else None
    )
    check("mapping", "signed_balance_conservation", bs_values["balance_difference"], source_net, allowed=tolerance)
    check("balance_sheet", "assets_vs_funding", bs_values["total_assets"], bs_values["total_funding"], allowed=tolerance)
    for group, parts, target in (
        ("asset_maturity_partition", ("current_assets", "noncurrent_assets"), "total_assets"),
        ("liability_maturity_partition", ("current_liabilities", "noncurrent_liabilities"), "total_liabilities"),
    ):
        values = [bs_values[name] for name in parts]
        actual = sum(values, Decimal("0")) if all(value is not None for value in values) else None
        check("mapping", group, actual, bs_values[target], allowed=tolerance)
    check_table = pd.DataFrame(checks, dtype=object)
    monthly_checks = monthly_checks.assign(check_group="monthly", account=None, tolerance=tolerance)
    check_table = pd.concat([check_table, monthly_checks[check_table.columns]], ignore_index=True)

    reported = balances.rename(columns={
        "closing_debit": "reported_closing_debit",
        "closing_credit": "reported_closing_credit",
    }).drop(columns=["ledger_line_count"])
    trial_table = trial.table.merge(reported, on="account", validate="one_to_one")
    follow_up = (
        closing_lines["closing_review_required"]
        | closing_lines["grouping_status"].eq("unassigned")
        | closing_lines["account"].isna()
    )
    statement_details = closing_lines.loc[
        closing_lines["exclude_from_income_statement"] | follow_up
    ].copy()
    statement_details["follow_up_required"] = follow_up.loc[statement_details.index]
    income_values = statement_tables["Income_Statement"].set_index("item")["amount"].to_dict()
    summary = {
        "period_start": settings["period_start"], "period_end": settings["period_end"],
        "unit_status": "Not confirmed; amounts shown in source units.",
        "report_status": "provisional",
        "general_ledger_records": len(ledger_table), "trial_balance_accounts": len(trial_table),
        "unallocated_account_records": len(trial.unallocated_records),
        "voucher_groups": len(entries), "unassigned_voucher_records": len(unassigned),
        "identified_closing_records": int(closing_lines["exclude_from_income_statement"].sum()),
        "statement_review_records": int(follow_up.sum()),
        "entity_movement_groups": len(entities), "entity_movement_records": len(entity_scope),
        "entity_balance_basis": "Observed movements only; entity opening/closing balances are not reconstructed.",
        "cash_accounts": tuple(sorted(bs_items.loc[bs_items["item"].eq("cash"), "account"].unique())),
        "opening_cash": monthly.iloc[0]["opening_cash"],
        "closing_cash": monthly.iloc[-1]["closing_cash"],
        "observed_cash_net_movement": strict_sum(monthly["cash_net_movement"]),
        "monthly_periods": len(monthly),
        "months_with_records": int(monthly["record_count"].gt(0).sum()),
        "monthly_check_count": len(monthly_checks),
        "operating_expenses_definition": "Selling plus administrative expenses; other expense categories remain separate.",
        "cash_movement_definition": "Account debit/credit movements include internal transfers; they are not external receipts/payments or CFO.",
        "empty_month_rule": "Zero observed movements do not confirm inactivity; missing source amounts remain unavailable.",
        "amount_storage": "Decimal values are stored as text; calculations are performed in Python.",
        "balance_tolerance": tolerance, "balance_within_tolerance": balance_sheet.balance_check["within_tolerance"],
        "review_scope": "Selected statement follow-up only; consult 01_data_integrity.xlsx for all integrity flags.",
        "deductions_interpretation": "A zero net 521 component does not establish absence of deductions; internal 521/511 transfers can offset.",
        "tax_interpretation": "Observed tax expense does not confirm the tax liability.",
        "maturity_basis": "See Balance_Sheet.calculation_basis and BS_Details.maturity_basis; configured maturity is an assumption, not verified due-date evidence.",
    }
    summary.update({"income_" + name: amount for name, amount in income_values.items()})
    summary.update({"balance_" + name: amount for name, amount in bs_values.items()})
    for row in income_statement.components.to_dict("records"):
        summary["component_" + row["item"] + "_amount"] = row["amount"]
        summary["component_" + row["item"] + "_records"] = row["line_count"]
    for prefix, counts in (
        ("check_", check_table["check_status"].value_counts()),
        ("net_balance_", balances["net_check_status"].value_counts()),
        ("balance_source_", balances["balance_basis"].value_counts()),
    ):
        summary.update({prefix + name: int(count) for name, count in counts.items()})
    summary.update({"presentation_note_" + str(i): note for i, note in enumerate(balance_sheet.notes, 1)})

    return {
        "Summary": pd.DataFrame(list(summary.items()), columns=["metric", "value"]),
        "General_Ledger": ledger_table, "Trial_Balance": trial_table,
        **statement_tables, "Monthly": monthly, "Checks": check_table,
        "BS_Details": bs_items, "Statement_Details": statement_details,
        "Entity_Movements": entities,
    }
