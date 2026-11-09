"""Store journal evidence and reconstruct account balances with reporting scope."""

from dataclasses import asdict
from datetime import date
from decimal import Decimal
from unicodedata import normalize
from functools import reduce
from itertools import accumulate, chain

import pandas as pd

from auditpy.models import Account, JournalEntry, JournalLine
from auditpy.utils import SchemaError, is_missing, normalize_code, is_finite_amount


class GeneralLedger:
    """Keep an immutable sequence of journal records, including incomplete records."""

    TABLE_COLUMNS = (
        "sequence", "account", "account_name", "posting_date", "document_date",
        "voucher_no", "counter_account", "entity_code", "description", "debit", "credit",
        "running_net_balance", "balance_status", "source_file", "source_sheet",
        "source_row", "source_side",
    )

    def __init__(self, lines):
        self.__lines = tuple(lines)
        if not all(isinstance(line, JournalLine) for line in self.__lines):
            raise TypeError("GeneralLedger requires JournalLine objects.")

    def __len__(self):
        return len(self.__lines)

    def __iter__(self):
        yield from self.__lines

    def __getitem__(self, index):
        return self.__lines[index]

    def __repr__(self):
        return f"GeneralLedger(line_count={len(self)})"

    @classmethod
    def from_classified_table(cls, table):
        required = [
            "source_file", "source_sheet", "source_row", "row_type",
            "voucher_no", "posting_date", "document_date", "description",
            "account", "debit", "credit",
        ]
        if not set(required).issubset(table.columns):
            raise SchemaError("Required ledger columns are missing.")

        columns = required + [
            column for column in ("counter_account", "entity_code", "source_side")
            if column in table.columns
        ]
        scope = table.loc[table["row_type"].isin([
            "journal_candidate", "unclassified", "unresolved_pair",
        ]), columns]

        return cls(
            JournalLine(**row) for row in scope.to_dict("records")
        )

    def movements(self, account=None):
        selected = (
            self.__lines if account is None
            else tuple(line for line in self if line.account == account)
        )
        complete = bool(selected) and all(
            line.debit is not None and line.credit is not None
            for line in selected
        )
        debit = (
            sum((line.debit for line in selected), Decimal("0"))
            if complete else None
        )
        credit = (
            sum((line.credit for line in selected), Decimal("0"))
            if complete else None
        )
        return {
            "line_count": len(selected),
            "amounts_complete": complete,
            "debit": debit,
            "credit": credit,
            "net_movement": debit - credit if complete else None,
        }

    def account_ledger(self, account, opening_debit, opening_credit):
        """Calculate running net balances by posting date, preserving input ties."""
        if not isinstance(account, Account):
            raise TypeError("An Account object is required.")

        for amount in (opening_debit, opening_credit):
            if amount is not None and (
                not is_finite_amount(amount)
            ):
                raise ValueError("Opening balances must be finite Decimal values or None.")

        lines = [line for line in self if line.account == account.code]

        if any(type(line.posting_date) is not date for line in lines):
            raise SchemaError("Valid posting dates are required for a chronological ledger.")

        lines = sorted(lines, key=lambda line: line.posting_date)

        opening_net = (
            opening_debit - opening_credit
            if opening_debit is not None and opening_credit is not None else None
        )
        changes = (
            line.debit - line.credit
            if line.debit is not None and line.credit is not None else None
            for line in lines
        )

        def add_balance(balance, change):
            return (
                balance + change
                if balance is not None and change is not None else None
            )

        balances = accumulate(chain((opening_net,), changes), add_balance)
        next(balances)

        rows = [
            {
                "sequence": sequence,
                "account": account.code,
                "account_name": account.name,
                "posting_date": line.posting_date,
                "document_date": line.document_date,
                "voucher_no": line.voucher_no,
                "counter_account": line.counter_account,
                "entity_code": line.entity_code,
                "description": line.description,
                "debit": line.debit,
                "credit": line.credit,
                "running_net_balance": balance,
                "balance_status": "calculated" if balance is not None else "unavailable",
                "source_file": line.source_file,
                "source_sheet": line.source_sheet,
                "source_row": line.source_row,
                "source_side": line.source_side,
            }
            for sequence, (line, balance) in enumerate(zip(lines, balances), start=1)
        ]

        return pd.DataFrame(rows, columns=self.TABLE_COLUMNS, dtype=object)

    def group_entries(self, key_fields):
        """Group declared voucher keys and retain records with incomplete keys."""
        key_fields = tuple(key_fields)
        supported = {"source_file", "source_sheet", "posting_date", "voucher_no"}

        if len(key_fields) != len(supported) or set(key_fields) != supported:
            raise SchemaError(
                "JournalEntry supports file, sheet, posting date and voucher number."
            )

        groups = {}
        unassigned = []

        for line in self:
            key = tuple(getattr(line, field) for field in key_fields)
            missing_key = any(
                is_missing(value)
                for field, value in zip(key_fields, key)
                if field != "source_sheet"
            )

            if missing_key:
                unassigned.append(line)
            else:
                groups.setdefault(key, []).append(line)

        entries = tuple(
            JournalEntry(
                **dict(zip(key_fields, key)),
                lines=tuple(lines),
            )
            for key, lines in groups.items()
        )

        return entries, tuple(unassigned)

    def to_table(self, trial_balance, reference):
        """Assemble account ledgers and retain unallocated records in the same view."""
        names = reference.set_index("account")["account_name"].to_dict()
        frames = []
        for row in trial_balance.table.to_dict("records"):
            detail = self.account_ledger(
                Account(row["account"], names.get(row["account"])),
                row["opening_debit"], row["opening_credit"],
            )
            if not detail.empty:
                frames.append(detail)

        columns = self.TABLE_COLUMNS
        unallocated = [asdict(line) for line in self if line.account is None]
        if unallocated:
            detail = pd.DataFrame(unallocated, dtype=object).reindex(columns=columns)
            detail["balance_status"] = "unallocated_account"
            frames.append(detail)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        if len(result) != len(self):
            raise SchemaError("Account ledgers do not preserve all journal records.")
        return result

    def entity_movements(self, accounts):
        """Aggregate observed entity movements without inventing opening balances."""
        accounts = set(accounts)
        groups = {}
        for line in filter(lambda line: line.account in accounts, self):
            key = (line.source_file, line.source_sheet, line.account, line.entity_code)
            groups.setdefault(key, []).append(line)
        rows = []
        for key, lines in groups.items():
            movement = GeneralLedger(lines).movements()
            rows.append({
                "source_file": key[0], "source_sheet": key[1],
                "account": key[2], "entity_code": key[3],
                **movement,
                "missing_voucher_rows": sum(line.voucher_no is None for line in lines),
                "source_rows": tuple(line.source_row for line in lines),
            })
        columns = [
            "source_file", "source_sheet", "account", "entity_code",
            "line_count", "amounts_complete", "debit", "credit", "net_movement",
            "missing_voucher_rows", "source_rows",
        ]
        return pd.DataFrame(rows, columns=columns, dtype=object)


class TrialBalance:
    """Reconstruct net account balances from observed ledger movements."""

    def __init__(
        self, table, unallocated_records=(), *,
        period_start=None, period_end=None, unit=None,
        basis="selected_openings_and_observed_movements",
    ):
        if (period_start is None) != (period_end is None):
            raise ValueError("Both period dates must be provided together.")

        if period_start is not None and (
            type(period_start) is not date or type(period_end) is not date
            or period_start > period_end
        ):
            raise ValueError("The reporting period is invalid.")

        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise ValueError("The reporting unit must be a non-empty string or None.")

        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("The accounting basis must be a non-empty string.")

        self.__table = table.copy(deep=True)
        self.__unallocated_records = tuple(unallocated_records)
        self.__scope = {
            "period_start": period_start,
            "period_end": period_end,
            "unit": unit,
            "basis": basis,
        }

    @property
    def scope(self):
        """Return a copy of the declared reporting scope."""
        return self.__scope.copy()

    @property
    def table(self):
        return self.__table.copy(deep=True)

    @property
    def unallocated_records(self):
        return self.__unallocated_records

    @property
    def total_debit(self):
        amounts = self.__table["debit"].tolist()
        return (
            sum(amounts, Decimal("0"))
            if amounts and all(value is not None for value in amounts)
            else None
        )

    @property
    def total_credit(self):
        amounts = self.__table["credit"].tolist()
        return (
            sum(amounts, Decimal("0"))
            if amounts and all(value is not None for value in amounts)
            else None
        )

    @classmethod
    def from_general_ledger(
        cls, ledger, reference, *,
        period_start=None, period_end=None, unit=None,
        basis="selected_openings_and_observed_movements",
    ):
        """Reconstruct leaf-account balances and retain unallocated journal lines."""
        required = {
            "account", "opening_debit", "opening_credit",
            "source_file", "source_sheet", "source_rows",
        }
        if not required.issubset(reference.columns):
            raise SchemaError("Opening reference columns are missing.")

        if reference["account"].isna().any() or reference["account"].duplicated().any():
            raise SchemaError("Opening references require unique account codes.")

        groups = {}
        unallocated = []

        for line in ledger:
            if line.account is None:
                unallocated.append(line)
            else:
                groups.setdefault(line.account, []).append(line)

        references = {
            row["account"]: row for row in reference.to_dict("records")
        }
        codes = set(references) | set(groups)

        parents = {
            code for code in codes
            if any(other != code and other.startswith(code) for other in codes)
        }

        if parents & set(groups):
            raise SchemaError(
                "Direct postings to parent accounts require a separate allocation review."
            )

        def add_movement(totals, line):
            if None in (totals[0], totals[1], line.debit, line.credit):
                return None, None
            return totals[0] + line.debit, totals[1] + line.credit

        rows = []

        for account in sorted(codes - parents):
            lines = groups.get(account, [])
            debit, credit = reduce(
                add_movement, lines, (Decimal("0"), Decimal("0")),
            )

            opening = references.get(account, {})
            opening_debit = opening.get("opening_debit")
            opening_credit = opening.get("opening_credit")
            complete = None not in (
                opening_debit, opening_credit, debit, credit,
            )

            closing_net = (
                opening_debit - opening_credit + debit - credit
                if complete else None
            )

            rows.append({
                "account": account,
                "opening_debit": opening_debit,
                "opening_credit": opening_credit,
                "debit": debit,
                "credit": credit,
                "closing_net": closing_net,
                "ledger_line_count": len(lines),
                "calculation_status": "calculated" if complete else "unavailable",
                "opening_source_file": opening.get("source_file"),
                "opening_source_sheet": opening.get("source_sheet"),
                "opening_source_rows": opening.get("source_rows", ()),
                "opening_basis": opening.get(
                    "opening_basis",
                    "provided_reference" if opening else "unavailable",
                ),
            })

        columns = [
            "account", "opening_debit", "opening_credit", "debit", "credit",
            "closing_net", "ledger_line_count", "calculation_status",
            "opening_source_file", "opening_source_sheet", "opening_source_rows",
            "opening_basis",
        ]

        result = cls(
            pd.DataFrame(rows, columns=columns, dtype=object),
            unallocated,
            period_start=period_start,
            period_end=period_end,
            unit=unit,
            basis=basis,
        )

        if period_start is not None and any(
            type(line.posting_date) is not date
            or not period_start <= line.posting_date <= period_end
            for line in ledger
        ):
            raise ValueError("Ledger posting dates do not fit the declared period.")

        return result

    def __add__(self, other):
        """Combine disjoint account scopes with matching reporting metadata."""
        if not isinstance(other, TrialBalance):
            return NotImplemented

        if self.scope != other.scope:
            raise ValueError("Reporting periods, units and accounting bases must match.")

        if any(
            self.__scope[field] is None
            for field in ("period_start", "period_end", "unit")
        ):
            raise ValueError("A confirmed reporting period and unit are required.")

        if self.unallocated_records or other.unallocated_records:
            raise SchemaError("Resolve unallocated records before combining account scopes.")

        combined = pd.concat([self.table, other.table], ignore_index=True)
        codes = combined["account"].tolist()

        if (
            not all(isinstance(code, str) and code.strip() for code in codes)
            or len(codes) != len(set(codes))
        ):
            raise SchemaError("Combined account codes must be valid and unique.")

        if any(
            left != right and right.startswith(left)
            for left in codes for right in codes
        ):
            raise SchemaError("Parent and descendant account scopes overlap.")

        combined = combined.sort_values("account").reset_index(drop=True)
        return TrialBalance(combined, **self.scope)


# Cash movement observations used by financial analysis

def attach_source_classifications(ledger_table, source_table):
    """Recover existing source classifications by provenance, not row order."""
    keys = ["source_file", "source_sheet", "source_row", "source_side"]
    fields = [name for name in ("cost_item_code", "cost_item_name", "receipt_payment_code", "receipt_payment_name")
              if name in source_table and name not in ledger_table]
    if not fields:
        return ledger_table.copy(deep=True)
    source = source_table.reindex(columns=keys + fields).copy()
    if "source_side" not in source_table:
        source["source_side"] = pd.Series(None, index=source.index, dtype=object)
    if source.duplicated(keys).any():
        raise SchemaError("Source classification references must be unique.")
    left = ledger_table.copy(deep=True)
    for table in (left, source):
        for column in ("source_sheet", "source_side"):
            table[column] = table[column].astype(object).where(table[column].notna(), None)
    result = left.merge(
        source, on=keys, how="left", validate="one_to_one", sort=False,
        indicator="_source_match",
    )
    if len(result) != len(ledger_table):
        raise SchemaError("Source classifications changed ledger coverage.")
    if not result["_source_match"].eq("both").all():
        raise SchemaError("Some ledger references have no matching source row.")
    return result.drop(columns="_source_match")



def review_cash_flows(ledger_table, rules):
    """Classify observed cash legs using declared, provisional counterpart rules."""
    required = {
        "account", "counter_account", "entity_code", "description", "debit", "credit",
        "source_file", "source_sheet", "source_row", "source_side",
    }
    if not required.issubset(ledger_table):
        raise SchemaError("Cash-flow review requires ledger amounts and source references.")
    cash_prefixes = tuple(rules.get("cash_prefixes", ("111", "112", "113")))
    categories = {
        category: tuple(rules.get(category + "_prefixes", ()))
        for category in ("operating", "investing", "financing")
    }
    groups = {"internal_transfer": cash_prefixes, **categories}
    source_codes = rules.get("receipt_payment_codes", {})
    description_rules = rules.get("description_rules", ())
    if any(category not in categories for category in source_codes.values()):
        raise SchemaError("Receipt/payment codes must map to operating/investing/financing.")
    for rule in description_rules:
        if (rule["category"] not in (*categories, "noncash_adjustment")
                or not rule["counter_prefixes"] or not rule["terms"]):
            raise SchemaError("Cash description rules require a category, accounts and terms.")
    for name, prefixes in groups.items():
        if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
            raise SchemaError("Cash-flow account prefixes must be non-empty strings.")
        for other, other_prefixes in groups.items():
            if name < other and any(a.startswith(b) or b.startswith(a) for a in prefixes for b in other_prefixes):
                raise SchemaError("Cash-flow classification prefixes must not overlap.")
    if not cash_prefixes:
        raise SchemaError("Cash account prefixes are required.")
    suppliers = tuple(rules.get("supplier_prefixes", ("331",)))
    capital_assets = tuple(rules.get("capital_asset_prefixes", ("211", "213")))
    capital_suppliers = {
        (r.source_file, r.source_sheet, r.entity_code)
        for r in ledger_table.itertuples()
        if isinstance(r.account, str) and r.account.startswith(suppliers)
        and isinstance(r.counter_account, str) and r.counter_account.startswith(capital_assets)
        and isinstance(r.entity_code, str) and r.entity_code.strip()
    }
    cash = ledger_table.loc[ledger_table["account"].fillna("").str.startswith(cash_prefixes)].copy()
    statuses, reasons, net_amounts, receipts, payments = [], [], [], [], []
    valuation_terms = tuple(normalize("NFC", term).casefold() for term in rules.get("noncash_review_terms", ()))
    for r in cash.itertuples():
        amounts_ok = all(is_finite_amount(x) for x in (r.debit, r.credit))
        counter = r.counter_account if isinstance(r.counter_account, str) else ""
        matches = [name for name, prefixes in groups.items() if counter and counter.startswith(prefixes)]
        category = matches[0] if len(matches) == 1 else "unclassified"
        reason = "declared_counterpart_proxy" if matches else "counterpart_nature_unresolved"
        source_category = source_codes.get(normalize_code(getattr(r, "receipt_payment_code", None)))
        if source_category and (category == "unclassified" or counter.startswith(("131", "138", "141", "331", "338"))):
            category, reason = source_category, "source_receipt_payment_classification"
        if counter.startswith(suppliers) and (
            not isinstance(r.entity_code, str) or not r.entity_code.strip()
            or (r.source_file, r.source_sheet, r.entity_code) in capital_suppliers
        ):
            category, reason = "unclassified", "supplier_capital_or_missing_entity_review"
        description = normalize("NFC", str(r.description or "")).casefold()
        if counter.startswith(("515", "635")) and any(term in description for term in valuation_terms):
            category, reason = "unclassified", "possible_noncash_valuation"
        matched_rules = [rule for rule in description_rules
                         if counter.startswith(tuple(rule["counter_prefixes"]))
                         and any(normalize("NFC", term).casefold() in description
                                 for term in rule["terms"])]
        if matched_rules:
            outcomes = {rule["category"] for rule in matched_rules}
            category = next(iter(outcomes)) if len(outcomes) == 1 else "unclassified"
            reason = "description_rule:" + ",".join(rule["rule_id"] for rule in matched_rules)
            if len(outcomes) != 1:
                reason = "conflicting_description_rules"
        opening_date = rules.get("opening_adjustment_date")
        if (not counter and opening_date and r.posting_date == date.fromisoformat(opening_date)
                and str(r.voucher_no or "").startswith(tuple(rules.get("opening_voucher_prefixes", ())))
                and any(normalize("NFC", term).casefold() in description
                        for term in rules.get("opening_description_terms", ()))):
            category, reason = "noncash_adjustment", "disclosed_opening_adjustment_assumption"
        if not amounts_ok:
            category, reason = "unclassified", "cash_amount_unavailable"
        statuses.append(category)
        reasons.append(reason)
        net_amounts.append(r.debit - r.credit if amounts_ok else None)
        receipts.append(max(r.debit, Decimal(0)) + max(-r.credit, Decimal(0)) if amounts_ok else None)
        payments.append(max(r.credit, Decimal(0)) + max(-r.debit, Decimal(0)) if amounts_ok else None)
    cash["cash_flow_category"] = statuses
    cash["classification_reason"] = reasons
    cash["cash_net"] = net_amounts
    cash["receipt_exposure"] = receipts
    cash["payment_exposure"] = payments
    internal = cash["cash_flow_category"].eq("internal_transfer")
    if internal.any() and sum(cash.loc[internal, "cash_net"], Decimal(0)) != 0:
        cash.loc[internal, "cash_flow_category"] = "unclassified"
        cash.loc[internal, "classification_reason"] = "internal_transfer_does_not_net_to_zero"
    return cash



def summarize_cash_flow_estimate(cash, population_complete=True):
    """Return a point only when classified; otherwise bound observed uncertainty."""
    zero = Decimal(0)
    amounts_complete = all(is_finite_amount(x) for x in cash["cash_net"])
    available = bool(len(cash)) and amounts_complete and population_complete
    totals = {
        category: sum(cash.loc[cash["cash_flow_category"].eq(category), "cash_net"], zero)
        if amounts_complete else None
        for category in ("operating", "investing", "financing", "internal_transfer", "noncash_adjustment", "unclassified")
    }
    unresolved = cash.loc[cash["cash_flow_category"].eq("unclassified")]
    receipts = sum(unresolved["receipt_exposure"], zero) if amounts_complete else None
    payments = sum(unresolved["payment_exposure"], zero) if amounts_complete else None
    lower = totals["operating"] - payments if available else None
    upper = totals["operating"] + receipts if available else None
    point = lower if available and receipts == 0 and payments == 0 else None
    summary = {
        "method": "direct_source_and_description_estimate",
        "estimate_status": "point_estimate_provisional" if point is not None else "interval_estimate" if available else "unavailable",
        "cash_records": len(cash), "population_complete": population_complete,
        "amounts_complete": amounts_complete, "unclassified_records": len(unresolved),
        "identified_operating_cash_flow": totals["operating"],
        "investing_cash_flow_proxy": totals["investing"], "financing_cash_flow_proxy": totals["financing"],
        "internal_transfer_net": totals["internal_transfer"], "unclassified_cash_net": totals["unclassified"],
        "noncash_adjustment_net": totals["noncash_adjustment"],
        "noncash_adjustment_records": int(cash["cash_flow_category"].eq("noncash_adjustment").sum()),
        "unclassified_receipt_exposure": receipts, "unclassified_payment_exposure": payments,
        "cfo_lower": lower, "cfo_upper": upper, "cfo_point_estimate": point,
        "observed_cash_movement": sum(cash["cash_net"], zero) if amounts_complete and len(cash) else None,
        "scope_note": "Estimate from observed cash-account records, source codes and disclosed description/opening assumptions. Noncash and prior-opening adjustments reconcile ledger movements but are outside operating/investing/financing flows. Not an audited cash-flow statement.",
        "limitation": "Unclassified gross receipts/payments may be operating, non-operating or noncash. Missing amounts or unidentified nonzero ledger accounts prevent a finite envelope.",
    }
    return summary

