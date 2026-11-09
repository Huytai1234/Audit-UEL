"""Classify accounting records and reconcile source and ledger evidence."""

from datetime import date
from decimal import Decimal

import pandas as pd

from auditpy.utils import SchemaError, is_missing, compare_amounts


def inspect_trial_balance(table):
    """Flag non-account rows, code hierarchy candidates, and repeated balances."""
    result = table.copy()
    account_rows = result["account"].notna()
    result["row_type"] = [
        "account" if code is not None else "non_account"
        for code in result["account"]
    ]

    codes = set(result["account"].dropna())
    parent_codes = set()
    for code in codes:
        for other in codes:
            if other != code and other.startswith(code):
                parent_codes.add(code)
                break
    result["parent_candidate"] = result["account"].isin(parent_codes)
    result["repeated_account"] = account_rows & result.duplicated(
        subset=["account"], keep=False,
    )

    amount_columns = [
        "opening_debit", "opening_credit", "debit", "credit",
        "closing_debit", "closing_credit",
    ]
    status_columns = [column + "_status" for column in amount_columns]
    amounts_available = result[status_columns].eq("ok").all(axis=1)
    result["repeated_account_amounts"] = (
        account_rows & amounts_available
        & result.duplicated(subset=["account"] + amount_columns, keep=False)
    )
    return result

def classify_ledger_rows(table, row_labels):
    """Classify normalized records without discarding incomplete journal rows."""
    labels = {}
    for row_type, descriptions in row_labels.items():
        for description in descriptions:
            labels[description.strip().casefold()] = row_type

    result = table.copy()
    row_types = []
    for row in result.to_dict("records"):
        description = "" if is_missing(row["description"]) else str(row["description"]).strip()
        no_voucher = is_missing(row["voucher_no"])
        no_dates = is_missing(row["posting_date"]) and is_missing(row["document_date"])
        has_account = not is_missing(row["account"])
        row_type = "journal_candidate" if has_account else "unclassified"

        if row.get("source_side") == "unresolved":
            row_type = "unresolved_pair"
        elif no_voucher and no_dates and row.get("source_side") not in ("debit", "credit"):
            label = labels.get(description.casefold())
            zero_movement = (
                row["debit_status"] == "ok" and row["credit_status"] == "ok"
                and row["debit"] == 0 and row["credit"] == 0
            )
            if label == "subtotal":
                row_type = "subtotal"
            elif label in ("opening_balance", "closing_balance") and zero_movement:
                row_type = label
            elif not has_account and not description and all(
                isinstance(row.get(column), str)
                and row[column].lstrip().upper().startswith("=SUBTOTAL(")
                for column in ("debit_raw", "credit_raw")
            ):
                row_type = "formula_control_candidate"
        row_types.append(row_type)

    result["row_type"] = row_types
    return result


def prepare_trial_balance_reference(table):
    """Prepare one reference per account without adding hierarchy levels."""
    if len(table[["source_file", "source_sheet"]].drop_duplicates()) > 1:
        raise SchemaError("Prepare one trial balance source at a time.")

    amount_columns = [
        "opening_debit", "opening_credit", "debit", "credit",
        "closing_debit", "closing_credit",
    ]
    fields = ["source_file", "source_sheet", "account", "account_name"] + amount_columns
    status_columns = [column + "_status" for column in amount_columns]
    rows = []
    account_rows = table.loc[table["account"].notna()]

    for account, group in account_rows.groupby("account", sort=False):
        row = group.iloc[0][fields].to_dict()
        row["source_rows"] = tuple(group["source_row"])
        if not group[status_columns].eq("ok").all().all():
            status = "unusable_amounts"
        elif len(group.drop_duplicates(subset=["account_name"] + amount_columns)) != 1:
            status = "conflict"
        else:
            status = "unique" if len(group) == 1 else "identical_repeat"

        row["reference_status"] = status
        if status in ("unusable_amounts", "conflict"):
            row["account_name"] = None
            for column in amount_columns:
                row[column] = None
        rows.append(row)

    return pd.DataFrame(
        rows, columns=fields + ["source_rows", "reference_status"], dtype=object,
    )

def check_basic_integrity(
    classified, period_start, period_end,
    max_date_gap_days=30, balance_tolerance=Decimal("0"),
):
    """Check record fields and observed totals without replacing missing amounts."""
    if (
        period_start > period_end or max_date_gap_days < 0
        or not balance_tolerance.is_finite() or balance_tolerance < 0
    ):
        raise ValueError("Invalid period or integrity-check thresholds.")

    checked = classified.loc[classified["row_type"].isin([
        "journal_candidate", "unclassified", "unresolved_pair",
    ])].copy()
    flags = {
        "missing_voucher": checked["voucher_no"].isna(),
        "missing_account": checked["account"].isna(),
    }
    for field in ("posting_date", "document_date", "debit", "credit"):
        for status in ("missing", "invalid"):
            flags[status + "_" + field] = checked[field + "_status"].eq(status)

    amounts_ok = checked[["debit_status", "credit_status"]].eq("ok").all(axis=1)
    flags["negative_amount"] = [
        (debit is not None and debit < 0) or (credit is not None and credit < 0)
        for debit, credit in zip(checked["debit"], checked["credit"])
    ]
    flags["zero_amount"] = amounts_ok & checked["debit"].eq(0) & checked["credit"].eq(0)
    flags["posting_outside_period"] = [
        value is not None and not period_start <= value <= period_end
        for value in checked["posting_date"]
    ]
    flags["date_gap_exceeded"] = [
        posting is not None and document is not None
        and abs((posting - document).days) > max_date_gap_days
        for posting, document in zip(checked["posting_date"], checked["document_date"])
    ]
    for name, values in flags.items():
        checked[name] = values
    checked["has_issue"] = checked[list(flags)].any(axis=1)
    counts = {name: int(checked[name].sum()) for name in flags}

    dates = [value for value in checked["posting_date"] if value is not None]
    observed_months = {
        value.strftime("%Y-%m") for value in dates
        if period_start <= value <= period_end
    }
    expected_months = set(pd.period_range(period_start, period_end, freq="M").astype(str))
    amounts_complete = bool(len(checked) > 0 and amounts_ok.all())
    total_debit = sum(checked["debit"], Decimal("0")) if amounts_complete else None
    total_credit = sum(checked["credit"], Decimal("0")) if amounts_complete else None
    difference = total_debit - total_credit if amounts_complete else None

    summary = {
        "review_rows": len(checked),
        "journal_candidate_rows": int(checked["row_type"].eq("journal_candidate").sum()),
        "unclassified_rows": int(checked["row_type"].eq("unclassified").sum()),
        "unresolved_pair_rows": int(checked["row_type"].eq("unresolved_pair").sum()),
        "rows_with_issues": int(checked["has_issue"].sum()),
        "first_posting_date": min(dates) if dates else None,
        "last_posting_date": max(dates) if dates else None,
        "observed_months": tuple(sorted(observed_months)),
        "missing_months": tuple(sorted(expected_months - observed_months)),
        "amounts_complete": amounts_complete,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "difference": difference,
        "balance_tolerance": balance_tolerance,
        "within_tolerance": abs(difference) <= balance_tolerance if amounts_complete else None,
    }
    return checked, counts, summary

def check_voucher_balances(classified, key_columns, balance_tolerance=Decimal("0")):
    """Check declared voucher groups and retain records with missing key fields."""
    required_key = {"source_file", "source_sheet", "voucher_no"}
    if not required_key.issubset(key_columns) or len(key_columns) != len(set(key_columns)):
        raise SchemaError(
            "Voucher key must contain unique columns including "
            "source_file, source_sheet, and voucher_no."
        )
    required_columns = set(key_columns) | {
        "row_type", "account", "debit", "credit",
        "debit_status", "credit_status", "source_row",
    }
    if not required_columns.issubset(classified.columns):
        raise SchemaError("Required voucher-check columns are missing.")
    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    scope = classified.loc[classified["row_type"].isin([
        "journal_candidate", "unclassified", "unresolved_pair",
    ])].copy()
    required_values = [column for column in key_columns if column != "source_sheet"]
    key_ready = pd.Series(True, index=scope.index)
    for column in required_values:
        key_ready &= ~scope[column].map(is_missing)

    unassigned = scope.loc[~key_ready].copy()
    unassigned["grouping_issue"] = [
        ", ".join(column for column in required_values if is_missing(row[column]))
        for row in unassigned.to_dict("records")
    ]
    rows = []
    for _, group in scope.loc[key_ready].groupby(
        key_columns, sort=False, dropna=False,
    ):
        record = group.iloc[0][key_columns].to_dict()
        complete = (
            all(value == "ok" for value in group["debit_status"])
            and all(value == "ok" for value in group["credit_status"])
        )
        missing_accounts = int(group["account"].isna().sum())
        classification_issues = int(group["row_type"].ne("journal_candidate").sum())
        debit = sum(group["debit"], Decimal("0")) if complete else None
        credit = sum(group["credit"], Decimal("0")) if complete else None
        difference = debit - credit if complete else None
        if not complete:
            status = "unavailable_amounts"
        elif abs(difference) > balance_tolerance:
            status = "unbalanced"
        elif missing_accounts or classification_issues:
            status = "balanced_requires_review"
        else:
            status = "balanced"

        record.update(
            line_count=len(group), total_debit=debit, total_credit=credit,
            difference=difference, balance_status=status,
            missing_account_rows=missing_accounts,
            classification_issue_rows=classification_issues,
            source_rows=tuple(group["source_row"]),
        )
        rows.append(record)

    columns = key_columns + [
        "line_count", "total_debit", "total_credit", "difference",
        "balance_status", "missing_account_rows", "classification_issue_rows",
        "source_rows",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object), unassigned

def reconcile_account_movements(
    classified, reference, balance_tolerance=Decimal("0"),
):
    """Compare account scopes using the source-code prefix hierarchy."""
    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    if reference["account"].isna().any() or reference["account"].duplicated().any():
        raise SchemaError("The trial balance reference must have one row per account.")

    scope = classified.loc[classified["row_type"].isin([
        "journal_candidate", "unclassified", "unresolved_pair",
    ])].copy()

    if len(scope[["source_file", "source_sheet"]].drop_duplicates()) > 1:
        raise SchemaError("Reconcile one ledger source at a time.")

    unallocated = scope.loc[scope["account"].isna()].copy()
    lookup = reference.set_index("account").to_dict("index")
    accounts = sorted(set(scope["account"].dropna()) | set(lookup))
    rows = []

    for account in accounts:
        group = scope.loc[
            scope["account"].str.startswith(account, na=False)
        ]
        members = tuple(sorted(set(group["account"])))

        if not members:
            basis = "no_records"
        elif members == (account,):
            basis = "direct"
        elif account in members:
            basis = "direct_and_descendants"
        else:
            basis = "descendants"

        complete = bool(
            len(group)
            and group[["debit_status", "credit_status"]].eq("ok").all().all()
        )
        debit = sum(group["debit"], Decimal("0")) if complete else None
        credit = sum(group["credit"], Decimal("0")) if complete else None

        client = lookup.get(account, {})
        usable = client.get("reference_status") in (
            "unique", "identical_repeat",
        )
        debit_difference = debit - client["debit"] if complete and usable else None
        credit_difference = credit - client["credit"] if complete and usable else None

        if not client:
            status = "missing_tb_reference"
        elif not usable:
            status = "unusable_tb_reference"
        elif not len(group):
            status = "no_postings"
        elif not complete:
            status = "unavailable_ledger_amounts"
        elif max(abs(debit_difference), abs(credit_difference)) > balance_tolerance:
            status = "difference"
        else:
            status = "matched"

        rows.append({
            "account": account,
            "ledger_accounts": members,
            "ledger_row_count": len(group),
            "match_basis": basis,
            "ledger_debit": debit,
            "ledger_credit": credit,
            "tb_debit": client.get("debit"),
            "tb_credit": client.get("credit"),
            "debit_difference": debit_difference,
            "credit_difference": credit_difference,
            "reconciliation_status": status,
            "reference_status": client.get("reference_status"),
            "ledger_source_file": group["source_file"].iloc[0] if len(group) else None,
            "ledger_source_sheet": group["source_sheet"].iloc[0] if len(group) else None,
            "tb_source_file": client.get("source_file"),
            "tb_source_sheet": client.get("source_sheet"),
            "tb_source_rows": client.get("source_rows"),
        })

    columns = [
        "account", "ledger_accounts", "ledger_row_count", "match_basis",
        "ledger_debit", "ledger_credit", "tb_debit", "tb_credit",
        "debit_difference", "credit_difference", "reconciliation_status",
        "reference_status", "ledger_source_file", "ledger_source_sheet",
        "tb_source_file", "tb_source_sheet", "tb_source_rows",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object), unallocated

def check_source_controls(
    classified, reference, balance_tolerance=Decimal("0"),
):
    """Check source subtotals, balance references, and net roll-forwards."""
    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    for table in (classified, reference):
        if len(table[["source_file", "source_sheet"]].drop_duplicates()) > 1:
            raise SchemaError(
                "Check one ledger and one trial balance source at a time."
            )

    if reference["account"].isna().any() or reference["account"].duplicated().any():
        raise SchemaError("The trial balance reference must have one row per account.")

    rows = []

    def add_check(account, name, actual, expected, ledger_rows=(), tb_rows=()):
        rows.append({
            "account": account,
            **compare_amounts(name, actual, expected, balance_tolerance),
            "ledger_control_rows": ledger_rows, "tb_source_rows": tb_rows,
        })

    lookup = reference.set_index("account").to_dict("index")
    account_rows = classified.loc[classified["account"].notna()]

    for account, group in account_rows.groupby("account", sort=False):
        controls = {}
        for kind in ("opening_balance", "subtotal", "closing_balance"):
            block = group.loc[group["row_type"].eq(kind)]
            values = {"source_rows": tuple(block["source_row"])}

            for field in ("debit", "credit", "balance_debit", "balance_credit"):
                values[field] = (
                    block.iloc[0][field]
                    if len(block) == 1 and block.iloc[0][field + "_status"] == "ok"
                    else None
                )
            controls[kind] = values

        postings = group.loc[group["row_type"].isin([
            "journal_candidate", "unclassified", "unresolved_pair",
        ])]
        movements = {}
        client = lookup.get(account, {})
        usable = client.get("reference_status") in ("unique", "identical_repeat")

        for side in ("debit", "credit"):
            movements[side] = (
                sum(postings[side], Decimal("0"))
                if len(postings) and postings[side + "_status"].eq("ok").all()
                else None
            )
            add_check(
                account, "subtotal_" + side,
                movements[side], controls["subtotal"][side],
                controls["subtotal"]["source_rows"],
            )

            for kind, prefix in (
                ("opening_balance", "opening"),
                ("closing_balance", "closing"),
            ):
                add_check(
                    account, prefix + "_" + side + "_vs_tb",
                    controls[kind]["balance_" + side],
                    client[prefix + "_" + side] if usable else None,
                    controls[kind]["source_rows"],
                    client.get("source_rows", ()),
                )

        opening = controls["opening_balance"]
        closing = controls["closing_balance"]
        values = [
            opening["balance_debit"], opening["balance_credit"],
            movements["debit"], movements["credit"],
        ]
        expected = (
            values[0] - values[1] + values[2] - values[3]
            if all(value is not None for value in values) else None
        )
        actual = (
            closing["balance_debit"] - closing["balance_credit"]
            if closing["balance_debit"] is not None
            and closing["balance_credit"] is not None
            else None
        )
        add_check(
            account, "ledger_net_rollforward", actual, expected,
            opening["source_rows"] + closing["source_rows"],
        )

    for account, client in lookup.items():
        usable = client["reference_status"] in ("unique", "identical_repeat")
        actual = (
            client["closing_debit"] - client["closing_credit"]
            if usable else None
        )
        expected = (
            client["opening_debit"] - client["opening_credit"]
            + client["debit"] - client["credit"]
            if usable else None
        )
        add_check(
            account, "tb_net_rollforward", actual, expected,
            tb_rows=client["source_rows"],
        )

    columns = [
        "account", "check_name", "actual", "expected", "difference",
        "check_status", "ledger_control_rows", "tb_source_rows",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object)

def reconcile_control_totals(
    classified, reference, trial_balance, total_source_row,
    balance_tolerance=Decimal("0"),
):
    """Compare account roots and journal totals with a declared source total."""
    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        raise ValueError("Balance tolerance must be finite and non-negative.")

    for table in (classified, reference, trial_balance):
        if len(table[["source_file", "source_sheet"]].drop_duplicates()) > 1:
            raise SchemaError(
                "Reconcile one ledger and one trial balance source at a time."
            )

    if reference["account"].isna().any() or reference["account"].duplicated().any():
        raise SchemaError("The trial balance reference must have one row per account.")

    source_keys = ["source_file", "source_sheet"]
    reference_source = reference[source_keys].drop_duplicates().to_dict("records")
    total_source = trial_balance[source_keys].drop_duplicates().to_dict("records")
    if reference_source != total_source:
        raise SchemaError(
            "The reference and declared total must belong to the same source."
        )

    selected = trial_balance.loc[
        trial_balance["source_row"].eq(total_source_row)
    ]
    if len(selected) != 1 or not is_missing(selected.iloc[0]["account"]):
        raise SchemaError(
            "The declared total must identify one non-account source row."
        )
    total = selected.iloc[0]

    codes = set(reference["account"])
    root_codes = tuple(sorted(
        code for code in codes
        if not any(
            code != other and code.startswith(other)
            for other in codes
        )
    ))
    roots = reference.loc[reference["account"].isin(root_codes)]
    roots_usable = bool(
        len(roots)
        and roots["reference_status"].isin(["unique", "identical_repeat"]).all()
    )
    scope = classified.loc[classified["row_type"].isin([
        "journal_candidate", "unclassified", "unresolved_pair",
    ])]
    rows = []

    def add_check(name, actual, field):
        expected = total[field] if total[field + "_status"] == "ok" else None
        rows.append({
            **compare_amounts(name, actual, expected, balance_tolerance),
            "tb_source_file": total["source_file"],
            "tb_source_sheet": total["source_sheet"], "tb_total_row": total_source_row,
        })

    for field in (
        "opening_debit", "opening_credit", "debit", "credit",
        "closing_debit", "closing_credit",
    ):
        actual = sum(roots[field], Decimal("0")) if roots_usable else None
        add_check("tb_roots_" + field, actual, field)

    for side in ("debit", "credit"):
        actual = (
            sum(scope[side], Decimal("0"))
            if len(scope) and scope[side + "_status"].eq("ok").all()
            else None
        )
        add_check("ledger_" + side + "_vs_tb_total", actual, side)

    columns = [
        "check_name", "actual", "expected", "difference", "check_status",
        "tb_source_file", "tb_source_sheet", "tb_total_row",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object), root_codes

def check_account_catalogue(
    classified, catalogue_codes, catalogue_name,
    fields=("account", "counter_account"),
):
    """Check exact code membership without treating unmatched codes as invalid."""
    codes = set(catalogue_codes)
    if not codes or any(
        not isinstance(code, str) or not code or code != code.strip()
        for code in codes
    ):
        raise SchemaError("Catalogue codes must be non-empty normalized strings.")

    if not fields or len(fields) != len(set(fields)):
        raise SchemaError("Account fields must be non-empty and unique.")

    if not (set(fields) | {"row_type"}).issubset(classified.columns):
        raise SchemaError("Required account-check columns are missing.")

    if not isinstance(catalogue_name, str) or not catalogue_name.strip():
        raise SchemaError("The account catalogue source must be named.")

    checked = classified.loc[classified["row_type"].isin([
        "journal_candidate", "unclassified", "unresolved_pair",
    ])].copy()
    summary_rows = []
    status_columns = []

    for field in fields:
        status_column = field + "_catalogue_status"
        checked[status_column] = [
            "missing" if is_missing(code) else (
                "in_catalogue" if code in codes else "not_in_catalogue"
            )
            for code in checked[field]
        ]
        status_columns.append(status_column)

        statuses = checked[status_column]
        unmatched = checked.loc[
            statuses.eq("not_in_catalogue"), field
        ]
        summary_rows.append({
            "field": field,
            "catalogue_source": catalogue_name,
            "catalogue_code_count": len(codes),
            "in_catalogue": int(statuses.eq("in_catalogue").sum()),
            "not_in_catalogue": int(statuses.eq("not_in_catalogue").sum()),
            "missing": int(statuses.eq("missing").sum()),
            "unmatched_codes": tuple(sorted(set(unmatched))),
        })

    checked["has_unmatched_account"] = (
        checked[status_columns].eq("not_in_catalogue").any(axis=1)
    )
    return checked, pd.DataFrame(summary_rows, dtype=object)

def prepare_opening_balance_reference(classified, reference):
    """Supplement absent client accounts with explicit ledger opening evidence."""
    fields = [
        "account", "opening_debit", "opening_credit",
        "source_file", "source_sheet", "source_rows",
    ]
    ledger_fields = {
        "account", "row_type", "source_file", "source_sheet", "source_row",
    }
    control_fields = {
        "balance_debit", "balance_credit",
        "balance_debit_status", "balance_credit_status",
    }

    if not ledger_fields.issubset(classified.columns):
        raise SchemaError("Ledger opening-balance columns are missing.")
    if not (set(fields) | {"reference_status"}).issubset(reference.columns):
        raise SchemaError("Client opening-reference columns are missing.")
    if reference["account"].isna().any() or reference["account"].duplicated().any():
        raise SchemaError("Client opening references require unique account codes.")

    for table in (classified, reference):
        if len(table[["source_file", "source_sheet"]].drop_duplicates()) > 1:
            raise SchemaError("Prepare one ledger and one client source at a time.")

    result = reference[fields].copy()
    result["opening_basis"] = [
        "client_trial_balance"
        if status in ("unique", "identical_repeat") else "unavailable"
        for status in reference["reference_status"]
    ]

    unavailable = result["opening_basis"].eq("unavailable")
    result.loc[unavailable, ["opening_debit", "opening_credit"]] = None
    rows = result.to_dict("records")

    client_codes = set(reference["account"])
    controls = classified.loc[
        classified["row_type"].eq("opening_balance")
        & classified["account"].notna()
        & ~classified["account"].isin(client_codes)
    ]

    for account, group in controls.groupby("account", sort=False):
        row = group.iloc[0]
        usable = control_fields.issubset(classified.columns) and len(group) == 1 and all(
            row[field + "_status"] == "ok" and row[field] is not None
            for field in ("balance_debit", "balance_credit")
        )

        rows.append({
            "account": account,
            "opening_debit": row["balance_debit"] if usable else None,
            "opening_credit": row["balance_credit"] if usable else None,
            "source_file": row["source_file"],
            "source_sheet": row["source_sheet"],
            "source_rows": tuple(group["source_row"]),
            "opening_basis": "ledger_opening" if usable else "unavailable",
        })

    return pd.DataFrame(
        rows, columns=fields + ["opening_basis"], dtype=object,
    )


def make_reconciliation_check(
    name, actual, expected, *, group, account=None, tolerance=Decimal("0"),
):
    """Add group and account context to the common numeric comparison."""
    check = compare_amounts(name, actual, expected, tolerance)
    return dict(check_group=group, check_name=name, account=account,
                actual=actual, expected=expected, difference=check["difference"],
                tolerance=tolerance, check_status=check["check_status"])


def build_integrity_tables(classified, client, reference, config, settings, step):
    """Run integrity checks and assemble their report using the supplied audit step."""
    tolerance = Decimal(settings["balance_tolerance"])
    records, counts, summary = step(
        "CHECK_BASIC_INTEGRITY", check_basic_integrity, classified,
        date.fromisoformat(settings["period_start"]),
        date.fromisoformat(settings["period_end"]),
        settings["date_gap_days"],
        tolerance,
        input_counts={"classified_records": len(classified)},
        output_counts=lambda result: {"review_records": len(result[0])},
    )
    vouchers, unassigned = step(
        "CHECK_VOUCHER_BALANCES", check_voucher_balances, classified, settings["voucher_key"], tolerance,
        input_counts={"classified_records": len(classified)},
        output_counts=lambda result: {"voucher_groups": len(result[0]), "unassigned_records": len(result[1])},
    )
    fields = tuple(
        field for field in ("account", "counter_account")
        if field in classified.columns
    )
    catalogue, catalogue_summary = step(
        "CHECK_ACCOUNT_CATALOGUE", check_account_catalogue, classified, reference["account"], "client_trial_balance_reference",
        fields=fields,
        input_counts={"classified_records": len(classified), "account_references": len(reference)},
        output_counts=lambda result: {"review_records": len(result[0]), "catalogue_summary_rows": len(result[1])},
    )
    account_checks, _ = step(
        "RECONCILE_ACCOUNT_MOVEMENTS", reconcile_account_movements, classified, reference, tolerance,
        input_counts={"classified_records": len(classified), "account_references": len(reference)},
        output_counts=lambda result: {"account_checks": len(result[0]), "unallocated_records": len(result[1])},
    )
    source_checks = step(
        "CHECK_SOURCE_CONTROLS", check_source_controls, classified, reference, tolerance,
        input_counts={"classified_records": len(classified), "account_references": len(reference)},
        output_counts=lambda table: {"source_checks": len(table)},
    )
    total_checks, root_codes = step(
        "RECONCILE_CONTROL_TOTALS", reconcile_control_totals, classified, reference, client,
        config["trial_balance"]["total_row"], tolerance,
        input_counts={"classified_records": len(classified), "account_references": len(reference), "client_rows": len(client)},
        output_counts=lambda result: {"control_total_checks": len(result[0])},
    )

    if int(vouchers["line_count"].sum()) + len(unassigned) != len(records):
        raise SchemaError("Voucher grouping does not preserve record coverage.")

    if summary["amounts_complete"]:
        for side in ("debit", "credit"):
            grouped_total = sum(vouchers["total_" + side], Decimal("0"))
            unassigned_total = sum(unassigned[side], Decimal("0"))
            if grouped_total + unassigned_total != summary["total_" + side]:
                raise SchemaError(
                    "Voucher grouping does not preserve " + side + " amounts."
                )

    catalogue_columns = [
        field + "_catalogue_status" for field in fields
    ] + ["has_unmatched_account"]
    if not records.index.equals(catalogue.index):
        raise SchemaError("Record and catalogue checks do not cover the same rows.")

    records = records.join(catalogue[catalogue_columns])
    issue_mask = records["has_issue"] | records["has_unmatched_account"]
    context_columns = [
        column for column in (
            "source_file", "source_sheet", "source_row", "voucher_no",
            "posting_date", "document_date", "account", "counter_account",
            "description", "debit", "credit", "debit_raw", "credit_raw",
        ) if column in records.columns
    ]
    issues = records.loc[
        issue_mask,
        context_columns + list(counts) + ["has_issue"] + catalogue_columns,
    ]

    summary_values = summary.copy()
    summary_values.update({
        "normalized_rows": len(classified),
        "observed_month_count": len(summary["observed_months"]),
        "voucher_key": settings["voucher_key"],
        "date_gap_days": settings["date_gap_days"],
        "unassigned_records": len(unassigned),
        "rows_with_record_or_catalogue_issues": len(issues),
        "non_overlapping_root_accounts": root_codes,
        "ledger_source": classified["source_file"].iloc[0],
        "trial_balance_source": client["source_file"].iloc[0],
        "unit_status": "Not confirmed; amounts shown in source units.",
        "catalogue_basis": "Client trial balance codes; approved client COA not provided.",
        "amount_storage": "Decimal and float values stored as text; calculations performed in Python.",
        "formula_storage": "Source formulas stored as literal text and not evaluated.",
        "interpretation": "Flags overlap and do not establish errors or fraud. Month presence does not prove completeness.",
        "reliability": "Use remains subject to exception review and confirmation of source assumptions.",
    })
    summary_values.update({
        "record_check_" + key: value for key, value in counts.items()
    })
    for prefix, values in (
        ("row_type_", classified["row_type"].value_counts()),
        ("voucher_", vouchers["balance_status"].value_counts()),
        ("source_control_", source_checks["check_status"].value_counts()),
    ):
        summary_values.update({
            prefix + key: int(value) for key, value in values.items()
        })

    tables = {
        "Summary": pd.DataFrame(
            list(summary_values.items()), columns=["metric", "value"],
        ),
        "Record_Issues": issues,
        "Voucher_Balances": vouchers,
        "Unassigned_Records": unassigned,
        "Account_Reconciliation": account_checks,
        "Source_Controls": source_checks,
        "Control_Totals": total_checks,
        "Account_Catalogue": catalogue_summary,
        "Normalized_Data": classified,
    }
    return tables
