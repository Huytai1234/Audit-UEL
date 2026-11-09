"""Define journal-entry procedures and retain their review coverage."""

import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import date, timedelta
from decimal import Decimal
from functools import partial
from uuid import uuid4

import pandas as pd

from auditpy.benford import (
    BENFORD_POPULATION_SIDES, benford_frequency, benford_statistics,
    select_benford_candidates,
)
from auditpy.ledger import GeneralLedger
from auditpy.models import Finding, RiskLevel
from auditpy.statements import classify_closing_lines
from auditpy.utils import SchemaError, is_missing, run_audited, is_finite_amount
from auditpy.validators import check_voucher_balances

class JournalEntryTest(ABC):
    """Share review setup; each subclass supplies its own detection rule."""

    @property
    def coverage(self):
        return getattr(self, "_coverage", {}).copy()

    def _begin_review(self, materiality):
        self._coverage = {}
        if not is_finite_amount(materiality) or materiality <= 0:
            raise ValueError("Performance materiality must be a positive finite Decimal.")

    LINE_COLUMNS = ("row_type", "posting_date", "voucher_no", "account", "debit", "credit",
                    "source_file", "source_sheet", "source_row")

    def _scope(self, table, required=LINE_COLUMNS):
        if not set(required).issubset(table.columns):
            raise SchemaError(f"Required {self.test_code} columns are missing.")
        return table.loc[table["row_type"].isin([
            "journal_candidate", "unclassified", "unresolved_pair",
        ])]

    def _finish(self, findings):
        self._coverage["finding_count"] = len(findings)
        return sorted(findings, reverse=True)

    def _finding(self, source, *, amount, amount_basis, description, risk_level,
                 accounts=None, source_refs=None, risk_amount=None):
        """Add common source fields without changing a test's business decision."""
        if source_refs is None:
            source_refs = source_references([source])
        if accounts is None:
            accounts = () if is_missing(source.get("account")) else (source["account"],)
        return Finding(
            test_code=self.test_code, accounts=accounts, amount=amount,
            amount_basis=amount_basis, description=description, risk_level=risk_level,
            voucher_no=None if is_missing(source.get("voucher_no")) else source["voucher_no"],
            posting_date=None if is_missing(source.get("posting_date")) else source["posting_date"],
            source_file=source["source_file"],
            source_sheet=None if is_missing(source.get("source_sheet")) else source["source_sheet"],
            source_refs=source_refs, risk_amount=amount if risk_amount is None else risk_amount,
        )

    @abstractmethod
    def run(self, classified, performance_materiality):
        raise NotImplementedError


def amount_risk(amount, materiality, zero_low=False):
    if zero_low and amount == 0:
        return RiskLevel.LOW
    return RiskLevel.HIGH if amount is not None and amount > materiality else RiskLevel.MEDIUM


def source_references(records):
    """Keep physical rows and format-B sides in their original order."""
    return tuple((int(row["source_row"]), None if is_missing(row.get("source_side"))
                  else row["source_side"]) for row in records)


class UnbalancedVoucherTest(JournalEntryTest):
    test_code = "JE01"

    def __init__(self, key_columns, balance_tolerance=Decimal("0")):
        required = {"source_file", "source_sheet", "posting_date", "voucher_no"}
        if len(key_columns) != 4 or set(key_columns) != required:
            raise SchemaError("JE01 requires file, sheet, posting date and voucher number.")
        self._key_columns = list(key_columns)
        self._tolerance = balance_tolerance

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        checks, unassigned = check_voucher_balances(classified, self._key_columns, self._tolerance)
        scope = self._scope(classified, ("row_type",))
        findings = []
        for row in checks.loc[checks["balance_status"].eq("unbalanced")].to_dict("records"):
            group = scope
            for column in self._key_columns:
                value = row[column]
                group = group.loc[group[column].isna() if is_missing(value) else group[column].eq(value)]
            if len(group) != row["line_count"]:
                raise SchemaError("JE01 source-group reconstruction failed.")
            difference = row["difference"]
            activity = max(
                sum(map(abs, group["debit"]), Decimal("0")),
                sum(map(abs, group["credit"]), Decimal("0")),
            )
            risk_amount = max(abs(difference), activity)
            refs = source_references(group.to_dict("records"))
            findings.append(self._finding(
                row, accounts=tuple(sorted(group['account'].dropna().unique())), amount=abs(difference),
                amount_basis='absolute_voucher_imbalance',
                description=(f"Debit={row['total_debit']}; credit={row['total_credit']}; "
                    f"difference={difference}; tolerance={self._tolerance}. Verify voucher completeness."),
                risk_level=amount_risk(risk_amount, performance_materiality), source_refs=refs,
                risk_amount=risk_amount,
            ))
        counts = checks["balance_status"].value_counts()
        self._coverage = {
            "review_records": len(scope), "assigned_records": int(checks["line_count"].sum()),
            "unassigned_records": len(unassigned), "voucher_groups": len(checks),
            "balanced_groups": int(counts.get("balanced", 0)),
            "unbalanced_groups": int(counts.get("unbalanced", 0)),
            "unavailable_amount_groups": int(counts.get("unavailable_amounts", 0)),
            "balanced_requires_review_groups": int(counts.get("balanced_requires_review", 0)),
        }
        return self._finish(findings)

class DuplicateEntryTest(JournalEntryTest):
    test_code = "JE02"

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        signature = ["source_file", "source_sheet", "voucher_no", "posting_date", "account", "debit", "credit"]
        required = set(signature) | {"row_type", "debit_status", "credit_status", "source_row"}
        scope = self._scope(classified, required)
        key_ready = ~scope["source_file"].map(is_missing)
        for column in ("voucher_no", "posting_date", "account"):
            key_ready &= ~scope[column].map(is_missing)
        amounts_ready = scope["debit_status"].eq("ok") & scope["credit_status"].eq("ok")
        eligible = scope.loc[key_ready & amounts_ready]
        if "source_side" in scope.columns:
            signature.append("source_side")
        matching = eligible.loc[eligible.duplicated(subset=signature, keep=False)]
        detail_columns = [column for column in (
            "counter_account", "entity_code", "document_date", "description", "invoice_no",
            "cost_item_code", "organizational_unit_code",
        ) if column in scope.columns]
        findings = []
        varied_groups = 0
        for _, group in matching.groupby(signature, sort=False, dropna=False):
            first = group.iloc[0]
            count = len(group)
            unit_amount = max(abs(first["debit"]), abs(first["credit"]))
            possible_excess = unit_amount * (count - 1)
            risk_amount = unit_amount * count
            different_details = [column for column in detail_columns if group[column].nunique(dropna=False) > 1]
            varied_groups += int(bool(different_details))
            risk = amount_risk(risk_amount, performance_materiality, zero_low=True)
            refs = source_references(group.to_dict("records"))
            findings.append(self._finding(
                first, amount=possible_excess, amount_basis='possible_excess_line_amount',
                description=(f"Occurrences={count}; possible excess rows={count - 1}; "
                    f"differing details={different_details}. Verify invoices and allocations."),
                risk_level=risk, source_refs=refs, risk_amount=risk_amount,
            ))
        self._coverage = {
            "review_records": len(scope), "eligible_records": len(eligible),
            "excluded_records": len(scope) - len(eligible),
            "missing_key_records": int((~key_ready).sum()),
            "unavailable_amount_records": int((~amounts_ready).sum()),
            "duplicate_candidate_records": len(matching),
            "possible_excess_records": len(matching) - len(findings),
            "affected_vouchers": len(matching[["source_file", "source_sheet", "posting_date", "voucher_no"]].drop_duplicates()),
            "groups_with_detail_variations": int(varied_groups),
        }
        return self._finish(findings)

class VoucherSequenceGapTest(JournalEntryTest):
    test_code = "JE03"

    def __init__(self, patterns, assumption):
        if not patterns or not isinstance(assumption, str) or not assumption.strip():
            raise ValueError("JE03 requires patterns and a documented sequence assumption.")
        self._patterns = [re.compile(pattern) for pattern in patterns]
        if any(not {"series", "sequence"}.issubset(pattern.groupindex) for pattern in self._patterns):
            raise ValueError("Each pattern must define series and sequence groups.")
        self._assumption = assumption
        self._unparsed = []

    @property
    def unparsed_identifiers(self):
        return tuple(self._unparsed)

    def run(self, classified, performance_materiality):
        """Identify internal sequence gaps; no monetary amount is inferred."""
        self._begin_review(performance_materiality)
        self._unparsed = []
        required = {"row_type", "source_file", "source_sheet", "source_row", "voucher_no"}
        scope = self._scope(classified, required)
        ready = ~scope["source_file"].map(is_missing) & ~scope["voucher_no"].map(is_missing)
        identifiers = scope.loc[ready].drop_duplicates([
            "source_file", "source_sheet", "voucher_no",
        ])

        series = {}
        parsed = unmatched = ambiguous = 0
        for record in identifiers.to_dict("records"):
            matches = [pattern.fullmatch(record["voucher_no"]) for pattern in self._patterns]
            matches = [match for match in matches if match is not None]
            if len(matches) != 1:
                unmatched += int(len(matches) == 0)
                ambiguous += int(len(matches) > 1)
                self._unparsed.append((
                    record["source_file"], record["source_sheet"], int(record["source_row"]),
                    record["voucher_no"], "unmatched" if not matches else "ambiguous",
                ))
                continue

            match = matches[0]
            sequence = match.group("sequence")
            if not sequence or not sequence.isascii() or not sequence.isdigit() or not match.group("series"):
                raise ValueError("Sequence patterns must capture a series and ASCII digits.")
            sheet = None if is_missing(record["source_sheet"]) else record["source_sheet"]
            key = (record["source_file"], sheet, match.group("series"), len(sequence))
            series.setdefault(key, {})[int(sequence)] = record
            parsed += 1

        findings = []
        missing_count = 0
        for (file_name, sheet, series_name, width), records in series.items():
            numbers = sorted(records)
            for left, right in zip(numbers, numbers[1:]):
                if right - left <= 1:
                    continue
                count = right - left - 1
                missing_count += count
                refs = source_references((records[left], records[right]))
                findings.append(self._finding(
                    {'source_file': file_name, 'source_sheet': sheet}, accounts=(), amount=None,
                    amount_basis='unobserved_sequence_interval',
                    description=(f"Series={series_name}; sequence_width={width}; "
                        f"observed={records[left]['voucher_no']} -> {records[right]['voucher_no']}; "
                        f"missing range={str(left + 1).zfill(width)}..{str(right - 1).zfill(width)}; "
                        f"missing_count={count}. Confirm the register and cancellations; see sequence assumption in Parameters."),
                    risk_level=RiskLevel.MEDIUM, source_refs=refs,
                ))

        self._coverage = {
            "review_records": len(scope), "records_with_keys": int(ready.sum()),
            "missing_key_records": int((~ready).sum()), "distinct_identifiers": len(identifiers),
            "parsed_identifiers": parsed, "unmatched_identifiers": unmatched,
            "ambiguous_identifiers": ambiguous, "observed_series": len(series),
            "missing_sequence_numbers": missing_count,
            "unparsed_identifiers": tuple(self._unparsed),
        }
        return self._finish(findings)

class WeekendHolidayTest(JournalEntryTest):
    test_code = "JE04"

    def __init__(self, calendar):
        self._year = calendar["year"]
        self._assumption = calendar["assumption"]
        self._dates = {
            date.fromisoformat(day): label
            for day, label in calendar["reference_dates"].items()
        }
        self._working_notes = {
            date.fromisoformat(day): label
            for day, label in calendar["working_date_notes"].items()
        }
        if type(self._year) is not int or not self._dates or not self._assumption.strip():
            raise ValueError("JE04 requires a calendar year, dates and an assumption.")
        for mapping in (self._dates, self._working_notes):
            if any(day.year != self._year or not label.strip() for day, label in mapping.items()):
                raise ValueError("Calendar dates must match the year and have non-empty labels.")

    def run(self, classified, performance_materiality):
        """Screen ledger records by posting date, including missing voucher numbers."""
        self._begin_review(performance_materiality)
        scope = self._scope(classified)
        valid_date = scope["posting_date"].map(lambda day: type(day) is date).astype(bool)
        in_year = scope["posting_date"].map(
            lambda day: type(day) is date and day.year == self._year
        ).astype(bool)
        eligible = scope.loc[in_year]
        weekend = eligible["posting_date"].map(lambda day: day.weekday() >= 5).astype(bool)
        reference_day = eligible["posting_date"].isin(self._dates)
        flagged = eligible.loc[weekend | reference_day]

        findings = []
        for record in flagged.to_dict("records"):
            day = record["posting_date"]
            complete = all(
                isinstance(record[column], Decimal) and record[column].is_finite()
                for column in ("debit", "credit")
            )
            amount = max(abs(record["debit"]), abs(record["credit"])) if complete else None
            risk = amount_risk(amount, performance_materiality, zero_low=True)
            findings.append(self._finding(
                record, amount=amount, amount_basis='maximum_absolute_line_side',
                description=(f"Weekend={day.weekday() >= 5}; calendar reference={self._dates.get(day, 'none')}; "
                    f"working-day note={self._working_notes.get(day, 'none')}. Verify the client calendar and approval."),
                risk_level=risk,
            ))
        self._coverage = {
            "review_records": len(scope), "eligible_records": len(eligible),
            "unavailable_date_records": int((~valid_date).sum()),
            "outside_calendar_year_records": int((valid_date & ~in_year).sum()),
            "weekend_records": int(weekend.sum()),
            "calendar_reference_records": int(reference_day.sum()),
            "overlap_records": int((weekend & reference_day).sum()),
            "flagged_records": len(flagged),
        }
        return self._finish(findings)

def is_round_amount(amount, base):
    """Check exact nonzero multiples without rounding the source amount."""
    if not is_finite_amount(base) or base <= 0:
        raise ValueError("The round-amount base must be a positive finite Decimal.")
    return (
        is_finite_amount(amount)
        and amount != 0 and abs(amount) % base == 0
    )


class RoundAmountTest(JournalEntryTest):
    test_code = "JE05"

    def __init__(self, bases, assumption):
        bases = tuple(bases)
        if not bases or any(
            not is_finite_amount(base) or base <= 0
            for base in bases
        ):
            raise ValueError("Round-amount bases must be positive finite Decimal values.")
        if len(set(bases)) != len(bases) or not isinstance(assumption, str) or not assumption.strip():
            raise ValueError("Use unique bases and document the screening assumption.")
        self._checks = {
            base: partial(is_round_amount, base=base) for base in bases
        }
        self._assumption = assumption

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        scope = self._scope(classified)
        ready = (
            scope["debit"].map(lambda value: is_finite_amount(value))
            & scope["credit"].map(lambda value: is_finite_amount(value))
        )
        eligible = scope.loc[ready.astype(bool)]
        base_counts = {format(base, "f"): 0 for base in self._checks}
        findings = []
        multiple_base_records = negative_round_records = 0
        for record in eligible.to_dict("records"):
            matches = {
                side: tuple(
                    format(base, "f") for base, check in self._checks.items()
                    if check(record[side])
                )
                for side in ("debit", "credit")
            }
            matches = {side: bases for side, bases in matches.items() if bases}
            if not matches:
                continue
            matched_bases = {base for bases in matches.values() for base in bases}
            for base in matched_bases:
                base_counts[base] += 1
            multiple_base_records += int(len(matched_bases) > 1)
            negative_round_records += int(any(record[side] < 0 for side in matches))
            amount = max(abs(record[side]) for side in matches)
            risk_amount = max(abs(record["debit"]), abs(record["credit"]))
            findings.append(self._finding(
                record, amount=amount, risk_amount=risk_amount, amount_basis='maximum_absolute_round_side',
                description=(f"Debit={record['debit']}; credit={record['credit']}; matched bases={matches}. "
                    "Inspect contracts, estimates and approvals."),
                risk_level=amount_risk(risk_amount, performance_materiality),
            ))
        self._coverage = {
            "review_records": len(scope), "eligible_records": len(eligible),
            "unavailable_amount_records": len(scope) - len(eligible),
            "multiple_base_records": multiple_base_records,
            "negative_round_records": negative_round_records,
        }
        self._coverage.update({
            "base_" + base + "_records": count for base, count in base_counts.items()
        })
        return self._finish(findings)

def make_threshold_test(limit, lower_rate):
    """Create a predicate that retains its own lower and upper boundaries."""
    if not is_finite_amount(limit) or limit <= 0:
        raise ValueError("The approval limit must be a positive finite Decimal.")
    if not is_finite_amount(lower_rate) or not 0 < lower_rate < 1:
        raise ValueError("The lower-bound rate must be a finite Decimal between zero and one.")
    lower_bound = limit * lower_rate

    def is_near_limit(amount):
        return (
            is_finite_amount(amount)
            and lower_bound <= amount < limit
        )

    return is_near_limit


class NearApprovalLimitTest(JournalEntryTest):
    test_code = "JE06"

    def __init__(self, key_columns, limit, lower_rate, cash_prefixes, assumption):
        self._check = make_threshold_test(limit, lower_rate)
        self._limit, self._lower_rate = limit, lower_rate
        self._key_columns = tuple(key_columns)
        self._cash_prefixes = tuple(cash_prefixes)
        if not self._cash_prefixes or any(
            not isinstance(code, str) or not code.strip()
            for code in self._cash_prefixes
        ):
            raise ValueError("Provide non-empty cash-account prefixes.")
        if not isinstance(assumption, str) or not assumption.strip():
            raise ValueError("Document the approval-threshold and payment-scope assumption.")
        self._assumption = assumption

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        ledger = GeneralLedger.from_classified_table(classified)
        entries, unassigned = ledger.group_entries(self._key_columns)
        cash_groups = unavailable_groups = positive_groups = 0
        findings = []
        for entry in entries:
            cash_lines = [
                line for line in entry.lines
                if line.account is not None and line.account.startswith(self._cash_prefixes)
            ]
            if not cash_lines:
                continue
            cash_groups += 1
            if any(line.credit is None for line in cash_lines):
                unavailable_groups += 1
                continue
            positive_lines = [line for line in cash_lines if line.credit > 0]
            if not positive_lines:
                continue
            positive_groups += 1
            amount = sum((line.credit for line in positive_lines), Decimal("0"))
            if not self._check(amount):
                continue
            findings.append(self._finding(
                asdict(positive_lines[0]), accounts=tuple(sorted({line.account for line in positive_lines})),
                amount=amount, amount_basis='positive_cash_credit_by_voucher',
                description=(f"Positive cash credits={amount}; approval limit={self._limit}; "
                    f"lower rate={self._lower_rate}; positive records={len(positive_lines)}. Verify payment approval."),
                risk_level=amount_risk(amount, performance_materiality),
                source_refs=tuple((line.source_row, line.source_side) for line in positive_lines),
            ))
        self._coverage = {
            "review_records": len(ledger),
            "assigned_records": sum(len(entry.lines) for entry in entries),
            "unassigned_records": len(unassigned), "voucher_groups": len(entries),
            "cash_voucher_groups": cash_groups,
            "unavailable_cash_groups": unavailable_groups,
            "positive_outflow_groups": positive_groups,
        }
        return self._finish(findings)

class LargeYearEndEntryTest(JournalEntryTest):
    test_code = "JE07"

    def __init__(self, key_columns, closing_date, window_days, source_start, source_end):
        if any(type(day) is not date for day in (closing_date, source_start, source_end)):
            raise ValueError("Closing date and declared source dates must be date objects.")
        if source_start > source_end or type(window_days) is not int or window_days < 0:
            raise ValueError("Provide an ordered source period and non-negative integer window.")
        self._key_columns = tuple(key_columns)
        self._closing = closing_date
        self._start = closing_date - timedelta(days=window_days)
        self._end = closing_date + timedelta(days=window_days)
        self._source_start, self._source_end = source_start, source_end
        self._requested = tuple(
            self._start + timedelta(days=i) for i in range(2 * window_days + 1)
        )

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        ledger = GeneralLedger.from_classified_table(classified)
        entries, unassigned = ledger.group_entries(self._key_columns)
        units = [(entry.lines, "voucher") for entry in entries]
        units += [((line,), "unassigned_record") for line in unassigned]
        declared_dates = tuple(
            day for day in self._requested
            if self._source_start <= day <= self._source_end
        )
        uncovered_dates = tuple(day for day in self._requested if day not in declared_dates)
        status = "full" if not uncovered_dates else "partial" if declared_dates else "none"
        counts = dict(
            unavailable_date_records=0, outside_source_records=0, window_records=0,
            window_vouchers=0, window_unassigned_records=0, unavailable_amount_units=0,
            large_vouchers=0, large_unassigned_records=0,
        )
        findings, observed_dates = [], set()
        for lines, kind in units:
            first = lines[0]
            day = first.posting_date
            if type(day) is not date:
                counts["unavailable_date_records"] += len(lines)
                continue
            if not self._source_start <= day <= self._source_end:
                counts["outside_source_records"] += len(lines)
                continue
            if not self._start <= day <= self._end:
                continue
            observed_dates.add(day)
            counts["window_records"] += len(lines)
            counts["window_" + kind + "s"] += 1
            if any(line.debit is None or line.credit is None for line in lines):
                counts["unavailable_amount_units"] += 1
                continue
            debit_activity = sum((abs(line.debit) for line in lines), Decimal("0"))
            credit_activity = sum((abs(line.credit) for line in lines), Decimal("0"))
            amount = max(debit_activity, credit_activity)
            if amount <= performance_materiality:
                continue
            counts["large_" + kind + "s"] += 1
            findings.append(self._finding(
                asdict(first), accounts=tuple(sorted({line.account for line in lines if line.account is not None})),
                amount=amount, amount_basis='maximum_absolute_side_activity_' + kind,
                description=(f"Unit={kind}; days from closing={(day - self._closing).days}; "
                    f"absolute debit={debit_activity}; absolute credit={credit_activity}; window coverage={status}. "
                    "Inspect delivery, acceptance and service dates."),
                risk_level=RiskLevel.HIGH,
                source_refs=tuple((line.source_row, line.source_side) for line in lines),
            ))
        self._coverage = {
            "review_records": len(ledger),
            "assigned_records": sum(len(entry.lines) for entry in entries),
            "unassigned_records": len(unassigned), "voucher_groups": len(entries),
            "requested_window_start": self._start, "requested_window_end": self._end,
            "declared_source_start": self._source_start,
            "declared_source_end": self._source_end,
            "declared_window_coverage": status,
            "declared_window_days": len(declared_dates),
            "uncovered_window_dates": uncovered_dates,
            "observed_window_day_count": len(observed_dates),
        }
        self._coverage.update(counts)
        return self._finish(findings)

def is_single_account_code(value):
    """Recognize one numeric account code without interpreting compound strings."""
    return isinstance(value, str) and value.isascii() and value.isdigit()


class UnusualCounterAccountTest(JournalEntryTest):
    test_code = "JE08"

    def __init__(self, rules):
        required = {"rule_id", "debit_prefix", "credit_prefix", "material_only", "reason"}
        if not rules or any(
            not isinstance(rule, dict) or not required.issubset(rule) for rule in rules
        ):
            raise SchemaError("JE08 requires complete counter-account rules.")
        for rule in rules:
            if not all(
                isinstance(rule[field], str) and rule[field].strip()
                for field in ("rule_id", "reason")
            ):
                raise SchemaError("Rule identifiers and reasons must be non-empty strings.")
            if not all(
                is_single_account_code(rule[field])
                for field in ("debit_prefix", "credit_prefix")
            ) or type(rule["material_only"]) is not bool:
                raise SchemaError("Use numeric prefixes and a boolean material-only setting.")
        if len({rule["rule_id"] for rule in rules}) != len(rules):
            raise SchemaError("JE08 rule identifiers must be unique.")
        self._rules = tuple(rule.copy() for rule in rules)

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        ledger = GeneralLedger.from_classified_table(classified)
        counts = dict(
            missing_account_records=0, missing_counter_account_records=0,
            non_single_code_records=0, unavailable_amount_records=0, eligible_records=0,
        )
        findings = []
        for line in ledger:
            missing_code = line.account is None or line.counter_account is None
            single_codes = all(
                is_single_account_code(code) for code in (line.account, line.counter_account)
            )
            money_ready = line.debit is not None and line.credit is not None
            counts["missing_account_records"] += int(line.account is None)
            counts["missing_counter_account_records"] += int(line.counter_account is None)
            counts["non_single_code_records"] += int(not missing_code and not single_codes)
            counts["unavailable_amount_records"] += int(not money_ready)
            if not single_codes or not money_ready:
                continue
            counts["eligible_records"] += 1
            matched, amounts, details = {}, [], []
            for column, signed in (("debit", line.debit), ("credit", line.credit)):
                if signed == 0:
                    continue
                account_is_debit = (
                    (column == "debit" and signed > 0)
                    or (column == "credit" and signed < 0)
                )
                debit_account, credit_account = (
                    (line.account, line.counter_account) if account_is_debit
                    else (line.counter_account, line.account)
                )
                for rule in self._rules:
                    if not debit_account.startswith(rule["debit_prefix"]) or not credit_account.startswith(rule["credit_prefix"]):
                        continue
                    rule_id = rule["rule_id"]
                    if rule["material_only"] and abs(signed) <= performance_materiality:
                        continue
                    matched[rule_id] = rule["reason"]
                    amounts.append(abs(signed))
                    details.append(
                        f"{rule_id}: effective debit={debit_account}, "
                        f"credit={credit_account}, source {column}={signed}"
                    )
            if not matched:
                continue
            risk_amount = max(abs(line.debit), abs(line.credit))
            findings.append(self._finding(
                asdict(line), accounts=tuple(sorted({line.account, line.counter_account})), amount=max(amounts),
                risk_amount=risk_amount, amount_basis='maximum_absolute_matched_source_side',
                description=(f"Debit={line.debit}; credit={line.credit}; matches={details}; "
                    f"reasons={list(matched.values())}. Verify effective direction and supporting documents."),
                risk_level=amount_risk(risk_amount, performance_materiality),
            ))
        self._coverage = {
            "review_records": len(ledger),
            "counter_account_column_present": "counter_account" in classified.columns,
            "excluded_records": len(ledger) - counts["eligible_records"],
        }
        self._coverage.update(counts)
        return self._finish(findings)

def normalize_description_text(text):
    """Build a comparison key without replacing the source description."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = " ".join(text.split())
    return re.sub(r"^[\W_]+|[\W_]+$", "", text).strip()


class MissingOrVagueDescriptionTest(JournalEntryTest):
    test_code = "JE09"

    def __init__(self, generic_phrases):
        if not isinstance(generic_phrases, (list, tuple)) or not generic_phrases or any(
            not isinstance(value, str) for value in generic_phrases
        ):
            raise ValueError("Provide a non-empty list of generic description phrases.")
        phrases = [normalize_description_text(value) for value in generic_phrases]
        if any(not phrase for phrase in phrases):
            raise ValueError("Generic phrases must contain informative characters.")
        self._phrases = frozenset(phrases)

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        ledger = GeneralLedger.from_classified_table(classified)
        counts = dict(
            missing_description_records=0, invalid_description_type_records=0,
            punctuation_only_records=0, generic_phrase_records=0, not_flagged_records=0,
            unavailable_amount_findings=0, zero_amount_findings=0, negative_amount_findings=0,
        )
        findings = []
        flagged_keys, other_description_keys = set(), set()
        for line in ledger:
            key = (line.source_file, line.source_sheet, line.posting_date, line.voucher_no)
            key_ready = line.voucher_no is not None and type(line.posting_date) is date
            if is_missing(line.description):
                issue = "missing_description"
            elif not isinstance(line.description, str):
                issue = "invalid_description_type"
            else:
                text = normalize_description_text(line.description)
                if not text:
                    issue = "punctuation_only"
                elif text in self._phrases:
                    issue = "generic_phrase"
                else:
                    counts["not_flagged_records"] += 1
                    if key_ready:
                        other_description_keys.add(key)
                    continue
            counts[issue + "_records"] += 1
            if key_ready:
                flagged_keys.add(key)
            complete = line.debit is not None and line.credit is not None
            amount = max(abs(line.debit), abs(line.credit)) if complete else None
            counts["unavailable_amount_findings"] += int(not complete)
            counts["zero_amount_findings"] += int(amount == Decimal("0"))
            counts["negative_amount_findings"] += int(
                complete and (line.debit < 0 or line.credit < 0)
            )
            risk = amount_risk(amount, performance_materiality, zero_low=True)
            findings.append(self._finding(
                asdict(line), amount=amount, amount_basis='maximum_absolute_line_side',
                description=f"Description issue={issue}. Inspect source text, transaction purpose and supporting documents.",
                risk_level=risk,
            ))
        self._coverage = {
            "review_records": len(ledger), "configured_phrase_count": len(self._phrases),
            "affected_voucher_groups": len(flagged_keys),
            "affected_groups_with_other_descriptions": len(flagged_keys & other_description_keys),
        }
        self._coverage.update(counts)
        return self._finish(findings)

class BenfordTest(JournalEntryTest):
    """Screen first-digit distributions by source file, sheet and population."""
    test_code = "JE10"

    def __init__(self, populations, key_columns, account_map, balance_tolerance=Decimal("0")):
        if set(populations) != set(BENFORD_POPULATION_SIDES):
            raise ValueError("Benford requires ledger debit/credit, revenue and expense policies.")
        if any(populations[name].get("amount_column") != side
               for name, side in BENFORD_POPULATION_SIDES.items()):
            raise ValueError("Benford amount columns must match the declared population sides.")
        if any(populations[name].get("account_prefixes")
               for name in ("all_ledger_debit", "all_ledger_credit")):
            raise ValueError("Ledger-wide Benford populations must not restrict account prefixes.")

        self._populations = deepcopy(populations)
        self._key_columns = tuple(key_columns)
        self._account_map = deepcopy(account_map)
        self._tolerance = balance_tolerance
        self._results = {}

    @property
    def results(self):
        return deepcopy(self._results)

    @staticmethod
    def _risk_level(statistics):
        """Apply a declared statistical priority rule, not a monetary loss estimate."""
        if statistics["chi_square_screen"] in ("empty_population", "small_expected_counts"):
            return None
        if statistics["mad_band"] == "nonconformity":
            return RiskLevel.HIGH
        if (
            statistics["mad_band"] == "marginal_conformity"
            or statistics["chi_square_screen"] == "above_critical"
        ):
            return RiskLevel.MEDIUM
        return None

    def run(self, classified, performance_materiality):
        self._begin_review(performance_materiality)
        self._results = {}

        ledger = GeneralLedger.from_classified_table(classified)
        entries, unassigned = ledger.group_entries(self._key_columns)
        closing_lines = classify_closing_lines(
            entries, unassigned, self._account_map, self._tolerance,
        )
        evidence_columns = [
            "source_file", "source_sheet", "source_row", "source_side", "posting_date",
            "voucher_no", "account", "amount_column", "benford_amount", "first_digit",
            "closing_review_required",
        ]
        findings = []
        groups = closing_lines.groupby(
            ["source_file", "source_sheet"], dropna=False, sort=False,
        )

        for (source_file, source_sheet), source in groups:
            source_sheet = None if is_missing(source_sheet) else source_sheet

            for name, policy in self._populations.items():
                selected, coverage = select_benford_candidates(
                    source, policy["amount_column"], policy["account_prefixes"],
                )
                statistics = benford_statistics(selected["benford_amount"])
                observations = selected[evidence_columns].copy()

                self._results[(source_file, source_sheet, name)] = {
                    "coverage": coverage,
                    "statistics": statistics,
                    "frequency": benford_frequency(selected["benford_amount"]),
                    "observations": observations,
                }

                risk = self._risk_level(statistics)
                if risk is None:
                    continue

                refs = source_references(observations.head(3).to_dict("records"))
                findings.append(self._finding(
                    {'source_file': source_file, 'source_sheet': source_sheet},
                    accounts=tuple(sorted(observations['account'].unique())), amount=None, risk_amount=None,
                    amount_basis='population_distribution',
                    description=(f"Population={name}; N={statistics['observation_count']}; MAD={statistics['mad']}; "
                        f"band={statistics['mad_band']}; chi-square={statistics['chi_square']}; "
                        f"screen={statistics['chi_square_screen']}. References are population examples."),
                    risk_level=risk, source_refs=refs,
                ))

        self._coverage = {
            "review_records": len(closing_lines),
            "population_runs": len(self._results),
        }
        return self._finish(findings)

def _execute_je_procedure(procedure, classified, performance_materiality):
    """Run one strategy and validate its finding contract before logging success."""
    findings = procedure.run(classified, performance_materiality)

    if not isinstance(findings, list) or any(
        not isinstance(item, Finding) for item in findings
    ):
        raise TypeError("Each JE procedure must return a list of Finding objects.")
    if any(item.test_code != procedure.test_code for item in findings):
        raise SchemaError("Finding codes must match the executed procedure.")
    if len(findings) != procedure.coverage.get("finding_count"):
        raise SchemaError("Finding count does not match procedure coverage.")

    return findings


def run_je_procedure(procedure, classified, performance_materiality, *, log_path, run_id):
    """Retain the JE interface while using the shared application audit trail."""
    return run_audited(
        _execute_je_procedure, procedure, classified, performance_materiality,
        log_path=log_path, run_id=run_id, audit_name=type(procedure).__name__,
        event_type="je_procedure", input_counts={"normalized_records": len(classified)},
        output_counts=lambda findings: {"findings": len(findings)},
        metadata={"test_code": procedure.test_code,
                  "performance_materiality": str(performance_materiality)},
    )


def run_je_tests(procedures, classified, performance_materiality, log_path, *, run_id=None):
    """Run exactly the ten declared procedures and retain their separate results."""
    procedures = tuple(procedures)
    if any(not isinstance(procedure, JournalEntryTest) for procedure in procedures):
        raise TypeError("The runner requires JournalEntryTest instances.")

    codes = [procedure.test_code for procedure in procedures]
    if len(codes) != 10 or set(codes) != {
        f"JE{number:02d}" for number in range(1, 11)
    }:
        raise ValueError("Run JE01 through JE10 exactly once each.")

    run_id = run_id or uuid4().hex
    results = {}
    for procedure in procedures:
        findings = run_je_procedure(
            procedure, classified, performance_materiality,
            log_path=log_path, run_id=run_id,
        )
        results[procedure.test_code] = {
            "findings": findings,
            "coverage": procedure.coverage,
        }
    return results, run_id

def build_je_report_tables(results, benford_results, run_events, settings, calendar):
    """Build traceable JE tables; never aggregate findings into a total monetary loss."""
    codes = [f"JE{number:02d}" for number in range(1, 11)]
    if set(results) != set(codes) or len(run_events) != 10:
        raise ValueError("A complete ten-procedure run is required.")
    if (
        len({event["run_id"] for event in run_events}) != 1
        or {event["test_code"] for event in run_events} != set(codes)
    ):
        raise ValueError("Audit events must belong to one complete run.")
    for event in run_events:
        if (
            event["status"] != "completed"
            or event["output_rows"] != len(results[event["test_code"]]["findings"])
        ):
            raise ValueError("Audit events do not confirm successful finding counts.")

    units = (
        "voucher_group", "duplicate_group", "sequence_gap_interval", "normalized_record",
        "normalized_record", "cash_voucher_group", "voucher_or_unassigned_record",
        "normalized_record", "normalized_record", "population",
    )
    finding_columns = (
        ["finding_id"]
        + [field.name for field in fields(Finding) if field.name != "source_refs"]
        + ["reference_count"]
    )
    key_columns = ["source_file", "source_sheet", "source_row", "source_side"]
    summary, references, coverage_rows, tables = [], [], [], {}

    for code, unit in zip(codes, units):
        findings = sorted(results[code]["findings"], reverse=True)
        if len(findings) != results[code]["coverage"].get("finding_count"):
            raise SchemaError("Coverage does not match the finding count.")

        role = {
            "JE03": "gap_boundary",
            "JE10": "population_example",
        }.get(code, "flagged_record")
        rows, source_keys = [], set()

        for number, finding in enumerate(findings, start=1):
            row = asdict(finding)
            refs = row.pop("source_refs")
            finding_id = f"{code}-{number:06d}"
            row.update(finding_id=finding_id, risk_level=finding.risk_level.name,
                       reference_count=len(refs))
            rows.append(row)

            for source_row, side in refs:
                key = (finding.source_file, finding.source_sheet, source_row, side)
                source_keys.add(key)
                references.append(dict(
                    zip(key_columns, key),
                    finding_id=finding_id, test_code=code, reference_role=role,
                ))

        counts = Counter(finding.risk_level.name for finding in findings)
        amounts = [finding.amount for finding in findings if finding.amount is not None]
        summary.append({
            "test_code": code, "finding_unit": unit, "finding_count": len(findings),
            "HIGH": counts["HIGH"], "MEDIUM": counts["MEDIUM"], "LOW": counts["LOW"],
            "known_amount_sum": sum(amounts, Decimal("0")) if amounts or not findings else None,
            "unknown_amount_findings": len(findings) - len(amounts),
            "amount_bases": tuple(sorted({finding.amount_basis for finding in findings})),
            "flagged_source_records": len(source_keys) if role == "flagged_record" else None,
        })
        tables[code] = pd.DataFrame(rows, columns=finding_columns, dtype=object)
        coverage_rows.extend(
            {"test_code": code, "metric": key, "value": value}
            for key, value in results[code]["coverage"].items()
        )

    references = pd.DataFrame(references, columns=key_columns + [
        "finding_id", "test_code", "reference_role",
    ])
    flagged = references.loc[references["reference_role"].eq("flagged_record")]
    flagged = flagged.groupby(key_columns, dropna=False, sort=False).agg(
        test_codes=("test_code", lambda values: tuple(sorted(set(values)))),
        finding_count=("finding_id", "nunique"),
    ).reset_index()

    statistics_rows, frequencies, observations = [], [], []
    for (source_file, source_sheet, name), result in benford_results.items():
        metadata = {
            "source_file": source_file, "source_sheet": source_sheet, "population": name,
        }
        count = result["statistics"]["observation_count"]
        if (
            count != len(result["observations"])
            or count != result["frequency"]["observed_count"].sum()
        ):
            raise SchemaError("Benford observation coverage does not reconcile.")
        statistics_rows.append({**metadata, **result["coverage"], **result["statistics"]})
        frequencies.append(result["frequency"].assign(**metadata))
        observations.append(result["observations"].assign(population=name))

    notes = [
        ("report_status", "provisional"),
        ("run_id", run_events[0]["run_id"]),
        ("unique_flagged_normalized_records", len(flagged)),
        ("source_units", "Unconfirmed; all amounts retain source units."),
        ("finding_counts", "Different units by procedure; findings are not confirmed errors."),
        ("amount_totals", "Within-procedure screening measures only. Do not sum across procedures or interpret as losses."),
        ("unknown_amounts", "Blank means unavailable, not zero. known_amount_sum excludes unavailable amounts."),
        ("source_identity", "File + sheet + row + side identifies a normalized record; format-B sides can share one physical source row."),
        ("reference_roles", "Only flagged_record references enter Flagged_Records; gap boundaries and population examples are excluded."),
        ("benford_scope", "Ledger debits and credits are analyzed separately, alongside revenue credits and expense debits. Populations overlap; examples are not confirmed errors. Independence and suitability remain unverified."),
        ("risk_levels", "Declared review priorities; not fraud probabilities or engagement risk weights."),
        ("evidence_limits", "Retain integrity exceptions, provisional statements/materiality, calendar assumptions and partial year-end coverage."),
    ]
    parameters = [
        {"parameter": name, "value": settings[name]}
        for name in ("period_start", "period_end", "balance_tolerance", "voucher_key", "je_tests")
        if name in settings
    ]
    parameters.append({"parameter": "performance_materiality", "value": run_events[0].get("performance_materiality")})
    parameters.append({"parameter": "calendar_reference", "value": calendar})

    return {
        "Read_Me": pd.DataFrame(notes, columns=["metric", "value"]),
        "Summary": pd.DataFrame(summary, dtype=object),
        **tables,
        "Coverage": pd.DataFrame(coverage_rows, columns=["test_code", "metric", "value"]),
        "Source_References": references,
        "Flagged_Records": flagged,
        "Benford_Statistics": pd.DataFrame(
            statistics_rows, columns=None if statistics_rows else ["population"],
        ),
        "Benford_Frequency": (
            pd.concat(frequencies, ignore_index=True)
            if frequencies else pd.DataFrame(columns=["population", "digit"])
        ),
        "Benford_Observations": (
            pd.concat(observations, ignore_index=True)
            if observations else pd.DataFrame(columns=key_columns + ["population"])
        ),
        "Parameters": pd.DataFrame(parameters),
        "Run_Log": pd.DataFrame(run_events),
    }
