"""Score client acceptance using configured weights and review rules."""

from decimal import Decimal
import pandas as pd

from auditpy.models import Decision
from auditpy.benford import BENFORD_POPULATION_SIDES
from auditpy.utils import (
    normalize_amount, read_json_config, table_row_counts,
)


class AcceptanceEngine:
    GROUPS = (
        "data_quality", "journal_entries", "financial_health",
        "reporting_risk", "qualitative",
    )

    def __init__(self, config):
        weights = config.get("weights", {})
        if not isinstance(weights, dict) or set(weights) != set(self.GROUPS):
            raise ValueError("Provide weights for exactly the five risk groups.")
        self._weights = {name: normalize_amount(weights[name])[0] for name in self.GROUPS}
        if any(value is None or value <= 0 for value in self._weights.values()):
            raise ValueError("Weights must be positive finite numbers.")
        if sum(self._weights.values(), Decimal("0")) != 1:
            raise ValueError("Weights must sum to 1 (100%).")
        limits = config.get("decision_thresholds", {})
        self._conditional = normalize_amount(limits.get("conditional"))[0]
        self._reject = normalize_amount(limits.get("reject"))[0]
        if (self._conditional is None or self._reject is None
                or not 0 < self._conditional < self._reject <= 100):
            raise ValueError("Require 0 < conditional cutoff < rejection cutoff <= 100.")

    @classmethod
    def from_config(cls, path):
        return cls(read_json_config(path))

    @property
    def weights(self):
        return self._weights.copy()

    def preliminary_band(self, score):
        """Classify a complete weighted score before any knock-out review."""
        value, status = normalize_amount(score)
        if status != "ok" or not 0 <= value <= 100:
            raise ValueError("A preliminary score must be between 0 and 100.")
        if value >= self._reject:
            return Decision.REJECT
        if value >= self._conditional:
            return Decision.CONDITIONAL
        return Decision.ACCEPT

    def score_groups(self, group_scores):
        """Keep a missing group unassessed, without changing its weight."""
        if set(group_scores) != set(self.GROUPS):
            raise ValueError("Provide all five groups; use None for an unassessed score.")
        details, missing = [], []
        total, assessed_weight = Decimal("0"), Decimal("0")
        for name, weight in self._weights.items():
            record = group_scores[name]
            if not all(isinstance(record.get(field), str) and record[field].strip()
                       for field in ("reason", "source")):
                raise ValueError(f"A reason and source are required for {name}.")
            raw_score = record["score"]
            score, status = normalize_amount(raw_score)
            if raw_score is not None and (status != "ok" or not 0 <= score <= 100):
                raise ValueError(f"Score for {name} must be between 0 and 100, or None.")
            contribution = None if score is None else score * weight
            if score is None:
                missing.append(name)
            else:
                total += contribution
                assessed_weight += weight
            details.append(dict(group=name, score=score, weight=weight,
                                contribution=contribution,
                                reason=record["reason"], source=record["source"]))
        return {
            "details": tuple(details),
            "weighted_score": total if not missing else None,
            "assessed_weight": assessed_weight,
            "missing_groups": tuple(missing),
            "preliminary_decision": self.preliminary_band(total) if not missing else None,
            "decision_stage": "before_knockout_rules",
        }

def highest_observed_score(points):
    """Preserve uncertainty unless the maximum possible score is observed."""
    if 100 in points:
        return Decimal("100")
    if None in points:
        return None
    return max(points)


# Group scoring

def score_data_quality(facts, policy):
    """Score observed integrity indicators without adding overlapping flags."""
    source = facts["source"]
    rows = facts["review_rows"]
    if rows <= 0:
        return dict(
            score=None,
            reason="No review records are available.",
            source=source,
            details=(),
        )

    affected = facts["required_issue_rows"]
    if not 0 <= affected <= rows:
        raise ValueError("Affected rows must be between zero and review rows.")

    limit, _ = normalize_amount(policy["high_field_issue_rate"])
    tolerance, _ = normalize_amount(facts["balance_tolerance"])
    materiality, _ = normalize_amount(facts["performance_materiality"])
    difference, _ = normalize_amount(facts["difference"])
    if limit is None or not 0 < limit <= 1:
        raise ValueError("The field-issue rate cutoff must be between zero and one.")
    if tolerance is None or tolerance < 0:
        raise ValueError("Balance tolerance must be nonnegative.")

    rate = Decimal(affected) / Decimal(rows)
    completeness_points = (
        100 if facts["missing_months"] or not facts["amounts_complete"] else 0
    )
    field_points = 0
    if rate > 0:
        field_points = 50
    if rate >= limit:
        field_points = 100

    balance_points = None
    if difference is not None:
        if abs(difference) <= tolerance:
            balance_points = 0
        elif materiality is not None and materiality > 0:
            balance_points = 100 if abs(difference) > materiality else 50

    catalogue_points = 50 if facts["unmatched_account_rows"] > 0 else 0
    details = (
        dict(criterion="period_and_amounts", score=completeness_points,
             evidence=f"Missing months={facts['missing_months']}; amounts complete={facts['amounts_complete']}"),
        dict(criterion="required_fields", score=field_points,
             evidence=f"Affected rows={affected}/{rows}; rate={rate:.4%}; high cutoff={limit:.2%}"),
        dict(criterion="ledger_balance", score=balance_points,
             evidence=f"Difference={difference}; tolerance={tolerance}; performance materiality={materiality}"),
        dict(criterion="catalogue_reference", score=catalogue_points,
             evidence=f"Unmatched rows={facts['unmatched_account_rows']}; verify reference membership."),
    )
    points = [row["score"] for row in details]
    score = highest_observed_score(points)
    reason = (
        "Assessment incomplete: the balance check cannot be rated."
        if score is None else
        "Highest observed criterion; overlapping flags are not added."
    )
    return dict(score=score, reason=reason, source=source, details=details)

def extract_je_priorities(tables, performance_materiality):
    """Extract priorities from assessable JE results."""
    codes = [f"JE{number:02d}" for number in range(1, 11)]
    priorities = dict.fromkeys(codes)
    materiality, _ = normalize_amount(performance_materiality)
    if tables is None or materiality is None or materiality <= 0:
        return priorities

    summary = tables["Summary"].set_index("test_code")
    logs = tables["Run_Log"].set_index("test_code")
    if not summary.index.is_unique or not logs.index.is_unique:
        raise ValueError("JE summary and run log require one row per procedure.")

    coverage = {
        (row.test_code, row.metric): row.value
        for row in tables["Coverage"].itertuples(index=False)
    }
    eligible_metrics = {
        "JE02": "eligible_records", "JE03": "parsed_identifiers",
        "JE04": "eligible_records", "JE05": "eligible_records",
        "JE06": "positive_outflow_groups", "JE08": "eligible_records",
        "JE09": "review_records",
    }
    statistics = tables["Benford_Statistics"]
    benford_ready = (
        not statistics.empty
        and set(statistics["population"]) == set(BENFORD_POPULATION_SIDES)
        and statistics["chi_square_screen"].isin(
            ("above_critical", "not_above_critical")
        ).all()
    )

    for code in codes:
        if code not in summary.index or code not in logs.index:
            continue
        logged_materiality, _ = normalize_amount(
            logs.loc[code, "performance_materiality"]
        )
        if logs.loc[code, "status"] != "completed" or logged_materiality != materiality:
            continue

        if code == "JE01":
            eligible = (
                coverage.get((code, "voucher_groups"), 0)
                - coverage.get((code, "unavailable_amount_groups"), 0)
            )
        elif code == "JE07":
            eligible = (
                coverage.get((code, "window_vouchers"), 0)
                + coverage.get((code, "window_unassigned_records"), 0)
                - coverage.get((code, "unavailable_amount_units"), 0)
            )
        elif code == "JE10":
            eligible = 1 if benford_ready else 0
        else:
            eligible = coverage.get((code, eligible_metrics[code]))

        if eligible is None or eligible <= 0:
            continue
        row = summary.loc[code]
        priorities[code] = "NO_FLAGS"
        for level in ("HIGH", "MEDIUM", "LOW"):
            if row[level] > 0:
                priorities[code] = level
                break
    return priorities


def score_journal_entries(priorities, policy, source):
    """Average procedure scores without treating unassessed tests as zero."""
    codes = [f"JE{number:02d}" for number in range(1, 11)]
    if set(priorities) != set(codes):
        raise ValueError("Provide JE01 through JE10 exactly once.")

    levels = ("NO_FLAGS", "LOW", "MEDIUM", "HIGH")
    points = {
        level: normalize_amount(policy["priority_points"][level])[0]
        for level in levels
    }
    if any(value is None or not 0 <= value <= 100 for value in points.values()):
        raise ValueError("Priority points must be finite numbers from 0 to 100.")
    if points["NO_FLAGS"] != 0 or not (
        points["NO_FLAGS"] < points["LOW"] < points["MEDIUM"] < points["HIGH"]
    ):
        raise ValueError("Points must increase from NO_FLAGS=0 to HIGH.")

    details = []
    for code in codes:
        priority = priorities[code]
        if priority is not None and priority not in points:
            raise ValueError(f"Unknown priority for {code}.")
        details.append(dict(
            criterion=code, priority=priority,
            score=None if priority is None else points[priority],
        ))
    scores = [row["score"] for row in details]
    score = None if None in scores else sum(scores, Decimal("0")) / Decimal(len(codes))
    reason = (
        "At least one required procedure has no assessable result."
        if score is None else
        "Equal mean of ten procedure scores; priorities are screening signals."
    )
    return dict(score=score, reason=reason, source=source, details=tuple(details))


def score_financial_health(tables, policy):
    """Score five financial dimensions and retain input limitations."""
    labels = ("Good", "Needs attention", "Risk")
    points = {
        label: normalize_amount(policy["assessment_points"][label])[0]
        for label in labels
    }
    if any(value is None for value in points.values()):
        raise ValueError("Financial assessment points must be finite numbers.")
    if not (
        points["Good"] == 0
        and 0 < points["Needs attention"] < 100
        and points["Risk"] == 100
    ):
        raise ValueError("Require Good=0, Risk=100 and a review score between them.")

    dimensions = policy["dimensions"]
    if len(dimensions) != 5 or any(not codes for codes in dimensions.values()):
        raise ValueError("Provide five nonempty financial dimensions.")
    codes = [code for members in dimensions.values() for code in members]
    if len(codes) != len(set(codes)):
        raise ValueError("Each ratio must belong to only one financial dimension.")

    ratios = tables["Ratios"].set_index("code")
    if not ratios.index.is_unique:
        raise ValueError("Each ratio must appear once.")
    if set(ratios.index) - set(codes):
        raise ValueError("Map every supplied ratio to a financial dimension.")

    details, indicators = [], []
    for name, members in dimensions.items():
        dimension_points = []
        for code in members:
            row = ratios.loc[code] if code in ratios.index else None
            assessed = (
                row is not None
                and row["calculation_status"] == "calculated"
                and row["assessment_status"] == "assessed_policy"
            )
            label = row["assessment"] if row is not None else None
            if assessed and label not in points:
                raise ValueError(f"Unknown financial assessment for {code}.")
            score = points[label] if assessed else None
            dimension_points.append(score)
            indicators.append(dict(
                criterion=code, dimension=name,
                value=None if row is None else row["value"],
                assessment=label, score=score,
                assumption_dependent=(
                    bool(row.get("maturity_assumption_applied", False)
                         or row.get("cash_flow_assumption_applied", False))
                    if row is not None else False
                ),
            ))
        details.append(dict(
            criterion=name,
            score=highest_observed_score(dimension_points),
        ))

    dimension_scores = [row["score"] for row in details]
    score = (
        None if None in dimension_scores
        else sum(dimension_scores, Decimal("0")) / Decimal("5")
    )
    context = dict(tables["Summary"].itertuples(index=False, name=None))
    return dict(
        score=score,
        reason="Mean of five risk dimensions; statement and benchmark assumptions apply.",
        source="Financial ratios and configured risk dimensions",
        details=tuple(details),
        indicators=tuple(indicators),
        unassessed_ratios=tuple(
            row["criterion"] for row in indicators if row["score"] is None
        ),
        assumed_ratios=tuple(
            row["criterion"] for row in indicators if row["assumption_dependent"]
        ),
        model_applicability=context.get("altman_applicability"),
    )

def score_reporting_risk(tables, policy):
    """Score monthly reporting signals, without assuming a confirmed error."""
    clear, _ = normalize_amount(policy["no_flags"])
    review, _ = normalize_amount(policy["review_required"])
    if clear != 0 or review is None or not 0 < review <= 100:
        raise ValueError("Require no_flags=0 and a review score from 1 to 100.")

    baselines = tables["Monthly_Baselines"].set_index("metric")
    if not baselines.index.is_unique:
        raise ValueError("Provide one monthly baseline per metric.")

    details = []
    for metric in ("net_revenue", "gross_margin"):
        score = None
        months = ()
        evidence = "Monthly baseline is unavailable."

        if metric in baselines.index:
            row = baselines.loc[metric]
            ready = (
                row["baseline_status"] in ("calculated", "constant_series")
                and row["expected_month_count"] > 0
                and row["valid_months"] == row["expected_month_count"]
                and bool(row["activity_complete"])
            )
            if ready:
                months = tuple(row["flagged_months"])
                score = review if months else clear
                evidence = (
                    f"Flagged months={months}; "
                    f"SD multiplier={row['sd_multiplier']}; "
                    f"ddof={row['ddof']}."
                )
            else:
                evidence = (
                    f"Baseline status={row['baseline_status']}; "
                    "monthly assessment is incomplete."
                )

        details.append(dict(
            criterion=metric,
            score=score,
            flagged_months=months,
            evidence=evidence,
        ))

    score = highest_observed_score([row["score"] for row in details])
    return dict(
        score=score,
        reason=(
            "Highest monthly revenue or gross-margin screening score; "
            "flags require investigation, not an assumed misstatement."
        ),
        source="Monthly baselines and review observations",
        details=tuple(details),
        limitations=(
            "No prior-year DSO comparison is available.",
            "No statistical flag does not establish correct cut-off.",
            "Year-end transaction screening is assessed separately under JE07.",
        ),
    )

def score_qualitative(policy):
    """Score documented qualitative judgments under the project policy."""
    statuses = ("low_risk", "unverified", "high_risk")
    points = {
        status: normalize_amount(policy["status_points"][status])[0]
        for status in statuses
    }
    if (
        points["low_risk"] != 0
        or points["high_risk"] != 100
        or points["unverified"] is None
        or not 0 < points["unverified"] < 100
    ):
        raise ValueError("Require low=0, high=100 and an uncertainty score between them.")

    criteria = policy["criteria"]
    expected = {"industry", "governance", "management_integrity"}
    if set(criteria) != expected:
        raise ValueError("Provide industry, governance and management integrity.")

    details = []
    for name, assessment in criteria.items():
        status = assessment["status"]
        reason = assessment.get("reason")
        evidence = assessment.get("evidence")

        if status not in points:
            raise ValueError(f"Unknown qualitative status for {name}.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"A reason is required for {name}.")
        if status != "unverified":
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"Evidence is required to rate {name} as {status}.")

        details.append(dict(criterion=name, status=status, score=points[status],
                            reason=reason, evidence=evidence))

    pending = tuple(
        row["criterion"] for row in details
        if row["status"] == "unverified"
    )
    return dict(
        score=max(row["score"] for row in details),
        reason=(
            "Highest qualitative criterion; unverified items receive "
            "the configured uncertainty score, not a confirmed adverse finding."
        ),
        source="Documented group assessment in config/risk_weights.json",
        details=tuple(details),
        pending_verification=pending,
    )

# Knock-out rules and sensitivity

def apply_knockout_rules(scoring_result, facts):
    """Apply minimum decision requirements without changing risk scores."""
    materiality, _ = normalize_amount(facts["overall_materiality"])
    if materiality is None or materiality <= 0:
        raise ValueError("Overall materiality must be a positive finite number.")

    difference, _ = normalize_amount(facts.get("ledger_difference"))
    raw_revenue = facts.get("confirmed_fictitious_revenue")
    revenue, _ = normalize_amount(raw_revenue)
    evidence = facts.get("revenue_evidence")

    if raw_revenue is not None:
        if revenue is None or revenue < 0:
            raise ValueError("Confirmed fictitious revenue must be nonnegative.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("A confirmed revenue amount requires supporting evidence.")

    observations = (
        (
            "material_ledger_imbalance",
            None if difference is None else abs(difference),
            "Ledger debit-credit difference",
        ),
        (
            "confirmed_fictitious_revenue",
            revenue,
            evidence,
        ),
    )

    details = []
    for rule, amount, source in observations:
        status = "unavailable"
        if amount is not None:
            status = "triggered" if amount > materiality else "not_triggered"
        details.append(dict(
            rule=rule,
            amount=amount,
            threshold=materiality,
            status=status,
            source=source,
        ))

    triggered = tuple(
        row["rule"] for row in details if row["status"] == "triggered"
    )
    pending = tuple(
        row["rule"] for row in details if row["status"] == "unavailable"
    )
    before = scoring_result["preliminary_decision"]
    after = before

    if triggered and before is not None:
        if before.value < Decision.CONDITIONAL.value:
            after = Decision.CONDITIONAL

    return dict(
        weighted_score=scoring_result["weighted_score"],
        preliminary_decision=before,
        decision=after,
        minimum_required_decision=Decision.CONDITIONAL if triggered else None,
        triggered_rules=triggered,
        pending_rules=pending,
        review_complete=not pending,
        report_status="provisional",
        decision_stage="after_knockout_rules",
        details=tuple(details),
    )

def analyze_acceptance_sensitivity(config, group_scores, knockout_facts):
    """Change one weight or decision cutoff at a time by +/-20%."""
    weights = AcceptanceEngine(config).weights
    factors = (Decimal("0.8"), Decimal("1.2"))
    rows = []

    def record(scenario, candidate):
        engine = AcceptanceEngine(candidate)
        scoring = engine.score_groups(group_scores)
        outcome = apply_knockout_rules(scoring, knockout_facts)
        decision = outcome["decision"]

        rows.append(dict(
            scenario=scenario,
            weighted_score=scoring["weighted_score"],
            decision=None if decision is None else decision.name,
            weights=engine.weights,
            weight_total=sum(engine.weights.values(), Decimal("0")),
            conditional_cutoff=normalize_amount(
                candidate["decision_thresholds"]["conditional"]
            )[0],
            reject_cutoff=normalize_amount(
                candidate["decision_thresholds"]["reject"]
            )[0],
            triggered_rules=outcome["triggered_rules"],
            pending_rules=outcome["pending_rules"],
        ))

    record("baseline", config)

    for name in weights:
        for factor in factors:
            adjusted = dict(weights)
            adjusted[name] *= factor
            total = sum(adjusted.values(), Decimal("0"))
            normalized = {
                group: weight / total
                for group, weight in adjusted.items()
            }

            last = AcceptanceEngine.GROUPS[-1]
            normalized[last] = Decimal("1") - sum(
                (
                    weight for group, weight in normalized.items()
                    if group != last
                ),
                Decimal("0"),
            )

            record(
                f"weight_{name}_{factor}",
                dict(config, weights=normalized),
            )

    for name in ("conditional", "reject"):
        for factor in factors:
            limits = dict(config["decision_thresholds"])
            limits[name] = normalize_amount(limits[name])[0] * factor

            record(
                f"cutoff_{name}_{factor}",
                dict(config, decision_thresholds=limits),
            )

    baseline = rows[0]["decision"]
    for row in rows:
        row["decision_changed"] = (
            None if baseline is None or row["decision"] is None
            else row["decision"] != baseline
        )

    return tuple(rows)

def build_acceptance_actions(scoring_result, knockout_result, sensitivity):
    """Translate assessed risks and pending evidence into follow-up actions."""
    rows = []

    def add(category, reason, action, source):
        rows.append(dict(
            action_id=f"A{len(rows) + 1:02d}",
            category=category,
            reason=reason,
            action=action,
            source=source,
            status="open",
        ))

    responses = {
        Decision.ACCEPT:
            "Complete acceptance checks and prepare the audit plan.",
        Decision.CONDITIONAL:
            "Document and resolve acceptance conditions; obtain engagement approval.",
        Decision.REJECT:
            "Document the rejection reasons and refer the assessment for review.",
        None:
            "Complete the missing assessments before proposing a decision.",
    }
    decision = knockout_result["decision"]
    add(
        "decision",
        f"Current decision={None if decision is None else decision.name}; "
        f"report status={knockout_result['report_status']}.",
        responses[decision],
        "Weighted scoring and knock-out review",
    )

    plans = {
        "data_quality":
            "Resolve missing identifiers and investigate source differences; "
            "document the agreed data scope.",
        "journal_entries":
            "Prioritize high-risk JE findings; inspect supporting documents "
            "and corroborate explanations before concluding.",
        "financial_health":
            "Verify maturity assumptions, missing cash-flow inputs and "
            "model applicability; reassess affected ratios.",
        "reporting_risk":
            "Investigate flagged-month revenue and cost recognition; "
            "test year-end cut-off using subsequent-period evidence.",
        "qualitative":
            "Verify the principal industry, governance arrangements "
            "and management integrity before final acceptance.",
    }
    for row in scoring_result["details"]:
        if row["score"] == 0:
            continue
        add(
            row["group"],
            f"Group score={row['score']}; {row['reason']}",
            plans[row["group"]],
            row["source"],
        )

    for rule in knockout_result["triggered_rules"]:
        add(
            "knockout",
            f"Triggered rule={rule}.",
            "Investigate the underlying evidence and document the required "
            "minimum decision before engagement approval.",
            "Knock-out review",
        )

    for rule in knockout_result["pending_rules"]:
        add(
            "pending_evidence",
            f"Unassessed rule={rule}.",
            "Obtain corroborating evidence and complete the rule assessment; "
            "retain unresolved matters explicitly.",
            "Knock-out review",
        )

    changed = tuple(
        row["scenario"] for row in sensitivity
        if row["decision_changed"] is True
    )
    if changed:
        add(
            "sensitivity",
            f"Decision-changing scenarios={changed}.",
            "Explain threshold dependence, justify the selected cutoffs "
            "and retain the alternative decisions.",
            "Acceptance sensitivity scenarios",
        )

    return tuple(rows)

# Report tables

def build_acceptance_report_tables(engagement, group_scores, risk_config):
    """Build report tables from the scored engagement."""
    stored = engagement.results["acceptance"]
    scoring, knockout = stored["scoring"], stored["knockout"]
    scenarios, actions = stored["sensitivity"], stored["actions"]
    financial = group_scores["financial_health"]
    period = dict(
        engagement.results["statements"]["Summary"].itertuples(index=False, name=None)
    )
    before, after = scoring["preliminary_decision"], knockout["decision"]
    changed = tuple(
        row["scenario"] for row in scenarios if row["decision_changed"] is True
    )

    summary = dict(
        client_reference=engagement.client.reference,
        client_name=engagement.client.name,
        client_industry=engagement.client.industry,
        period_start=period["period_start"],
        period_end=period["period_end"],
        report_status=knockout["report_status"],
        weighted_score=scoring["weighted_score"],
        preliminary_decision=None if before is None else before.name,
        decision_after_knockouts=None if after is None else after.name,
        assessed_weight=scoring["assessed_weight"],
        missing_groups=scoring["missing_groups"],
        knockout_review_complete=knockout["review_complete"],
        pending_knockout_rules=knockout["pending_rules"],
        pending_qualitative_reviews=group_scores["qualitative"]["pending_verification"],
        unassessed_ratios=financial["unassessed_ratios"],
        assumption_dependent_ratios=financial["assumed_ratios"],
        altman_applicability=financial["model_applicability"],
        reporting_limitations=group_scores["reporting_risk"]["limitations"],
        sensitivity_scope="One parameter at a time; fixed scores and materiality; relative +/-20%.",
        sensitivity_scenarios=len(scenarios),
        decision_changed_scenarios=changed,
        open_actions=sum(row["status"] == "open" for row in actions),
        policy_configuration=risk_config,
        interpretation="Provisional academic assessment; scores are not fraud probabilities or engagement approval.",
    )

    criteria = [
        dict(row, group=name, source=group_scores[name]["source"])
        for name in AcceptanceEngine.GROUPS
        for row in group_scores[name]["details"]
    ]
    sensitivity_rows = []
    for row in scenarios:
        flat = {key: value for key, value in row.items() if key != "weights"}
        flat.update({
            f"weight_{name}": weight for name, weight in row["weights"].items()
        })
        sensitivity_rows.append(flat)

    total = sum(
        (row["contribution"] for row in scoring["details"]
         if row["contribution"] is not None),
        Decimal("0"),
    )
    baseline = scenarios[0]
    checks = [
        ("five_groups", len(scoring["details"]) == 5),
        ("group_contribution_total",
         None if scoring["weighted_score"] is None else total == scoring["weighted_score"]),
        ("baseline_score",
         None if scoring["weighted_score"] is None
         else baseline["weighted_score"] == scoring["weighted_score"]),
        ("baseline_decision",
         None if after is None else baseline["decision"] == after.name),
        ("scenario_count", len(scenarios) == 15),
        ("unique_scenarios",
         len({row["scenario"] for row in scenarios}) == len(scenarios)),
        ("normalized_scenario_weights",
         all(sum(row["weights"].values(), Decimal("0")) == 1 for row in scenarios)),
    ]
    return {
        "Summary": pd.DataFrame(
            list(summary.items()), columns=["metric", "value"], dtype=object,
        ),
        "Group_Scores": pd.DataFrame(scoring["details"], dtype=object),
        "Criteria": pd.DataFrame(criteria, dtype=object),
        "Ratio_Evidence": pd.DataFrame(financial["indicators"], dtype=object),
        "Knockout_Rules": pd.DataFrame(knockout["details"], dtype=object),
        "Sensitivity": pd.DataFrame(sensitivity_rows, dtype=object),
        "Actions": pd.DataFrame(actions, dtype=object),
        "Checks": pd.DataFrame([
            dict(
                check_name=name,
                check_status=(
                    "unavailable" if passed is None
                    else "matched" if passed else "difference"
                ),
            )
            for name, passed in checks
        ], dtype=object),
    }

# Pipeline

def build_acceptance_tables(engagement, integrity_tables, risk_config, selection, step):
    """Score the engagement, apply knock-outs and assess sensitivity."""
    summary = dict(integrity_tables["Summary"].itertuples(index=False, name=None))
    issues = integrity_tables["Record_Issues"]
    fields = [
        name for name in issues.columns
        if name.startswith(("missing_", "invalid_")) or name == "posting_outside_period"
    ]
    facts = dict(
        summary,
        required_issue_rows=int(issues[fields].any(axis=1).sum()),
        unmatched_account_rows=int(issues["has_unmatched_account"].sum()),
        performance_materiality=selection["performance_materiality"],
        source="Data-integrity checks and preliminary materiality",
    )
    je = engagement.results["journal_entries"]
    financial = engagement.results["financial_analysis"]
    priorities = step(
        "EXTRACT_JE_PRIORITIES", extract_je_priorities,
        je, selection["performance_materiality"],
        input_counts={"je_procedures": len(je["Summary"])},
        output_counts=lambda value: {"je_priorities": len(value)},
    )

    tasks = (
        ("data_quality", score_data_quality,
         (facts, risk_config["data_quality_policy"]),
         {"review_records": int(summary["review_rows"])}),
        ("journal_entries", score_journal_entries,
         (priorities, risk_config["je_policy"],
          "JE summary, coverage, audit events and Benford statistics"),
         {"je_priorities": len(priorities)}),
        ("financial_health", score_financial_health,
         (financial, risk_config["financial_policy"]),
         {"ratios": len(financial["Ratios"])}),
        ("reporting_risk", score_reporting_risk,
         (financial, risk_config["reporting_policy"]),
         {"monthly_metrics": len(financial["Monthly_Baselines"])}),
        ("qualitative", score_qualitative,
         (risk_config["qualitative_policy"],),
         {"criteria": len(risk_config["qualitative_policy"]["criteria"])}),
    )
    groups = {}
    for name, function, arguments, counts in tasks:
        groups[name] = step(
            "SCORE_" + name.upper(), function, *arguments,
            input_counts=counts,
            output_counts=lambda value: {"criteria": len(value["details"])},
        )

    scoring = step(
        "COMBINE_RISK_GROUPS", AcceptanceEngine(risk_config).score_groups, groups,
        input_counts={"risk_groups": len(groups)},
        output_counts=lambda value: {"risk_groups": len(value["details"])},
    )
    knockout_facts = dict(
        overall_materiality=selection["overall_materiality"],
        ledger_difference=summary["difference"],
        confirmed_fictitious_revenue=None,
        revenue_evidence=None,
    )
    knockout = step(
        "APPLY_KNOCKOUT_RULES", apply_knockout_rules, scoring, knockout_facts,
        input_counts={"risk_groups": len(groups)},
        output_counts=lambda value: {"rules": len(value["details"])},
    )
    scenarios = step(
        "ANALYZE_ACCEPTANCE_SENSITIVITY", analyze_acceptance_sensitivity,
        risk_config, groups, knockout_facts,
        input_counts={"risk_groups": len(groups)},
        output_counts=lambda value: {"scenarios": len(value)},
    )
    actions = step(
        "BUILD_ACCEPTANCE_ACTIONS", build_acceptance_actions,
        scoring, knockout, scenarios,
        input_counts={"risk_groups": len(groups), "scenarios": len(scenarios)},
        output_counts=lambda value: {"actions": len(value)},
    )
    engagement.results["acceptance"] = dict(
        scoring=scoring, knockout=knockout, sensitivity=scenarios, actions=actions,
    )
    tables = step(
        "BUILD_ACCEPTANCE_REPORT", build_acceptance_report_tables,
        engagement, groups, risk_config,
        input_counts={"risk_groups": len(groups), "scenarios": len(scenarios)},
        output_counts=table_row_counts,
    )
    if tables["Checks"]["check_status"].eq("difference").any():
        raise ValueError("Acceptance report consistency checks failed.")
    return tables
