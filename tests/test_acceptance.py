"""Check weighting, incomplete evidence, knock-out rules and sensitivity."""

from copy import deepcopy
from decimal import Decimal as D
from pathlib import Path

import pytest
import pandas as pd

from auditpy.acceptance import (
    AcceptanceEngine, highest_observed_score, score_financial_health,
    score_qualitative, apply_knockout_rules, analyze_acceptance_sensitivity,
    extract_je_priorities,
)
from auditpy.benford import BENFORD_POPULATION_SIDES
from auditpy.models import Decision
from auditpy.utils import read_json_config


@pytest.fixture
def risk_config():
    return read_json_config(Path(__file__).resolve().parents[1] / "config/risk_weights.json")


def sample_scores():
    return {name: dict(score=score, reason="Synthetic example.", source="Hand calculation")
            for name, score in zip(AcceptanceEngine.GROUPS, (20, 60, 50, 40, 30))}


def test_weighted_score_matches_hand_calculation_and_preserves_inputs(risk_config):
    scores = sample_scores()
    before = deepcopy(scores)
    engine = AcceptanceEngine(risk_config)
    result = engine.score_groups(scores)
    assert result["weighted_score"] == D("45")
    assert result["assessed_weight"] == D("1")
    assert result["preliminary_decision"] is Decision.CONDITIONAL
    assert result["decision_stage"] == "before_knockout_rules"
    assert scores == before
    weights = engine.weights
    weights["qualitative"] = D("0")
    assert engine.weights["qualitative"] == D("0.10")


def test_missing_group_is_not_zero_and_does_not_redistribute_weights(risk_config):
    scores = sample_scores()
    scores["qualitative"]["score"] = None
    result = AcceptanceEngine(risk_config).score_groups(scores)
    assert result["weighted_score"] is None
    assert result["assessed_weight"] == D("0.90")
    assert result["missing_groups"] == ("qualitative",)
    assert result["preliminary_decision"] is None
    assert result["details"][-1]["contribution"] is None


@pytest.mark.parametrize("value,band", [
    ("0", Decision.ACCEPT), ("39.99", Decision.ACCEPT),
    ("40", Decision.CONDITIONAL), ("69.99", Decision.CONDITIONAL),
    ("70", Decision.REJECT), ("100", Decision.REJECT),
])
def test_exact_decision_boundaries(risk_config, value, band):
    assert AcceptanceEngine(risk_config).preliminary_band(D(value)) is band


@pytest.mark.parametrize("score", [True, -1, 101, D("NaN"), "bad"])
def test_invalid_scores_are_rejected(risk_config, score):
    scores = sample_scores()
    scores["qualitative"]["score"] = score
    with pytest.raises(ValueError):
        AcceptanceEngine(risk_config).score_groups(scores)


def test_weight_total_and_score_evidence_are_required(risk_config):
    invalid = deepcopy(risk_config)
    invalid["weights"]["qualitative"] = "0.20"
    with pytest.raises(ValueError, match="sum"):
        AcceptanceEngine(invalid)
    scores = sample_scores()
    scores["qualitative"]["source"] = ""
    with pytest.raises(ValueError, match="reason and source"):
        AcceptanceEngine(risk_config).score_groups(scores)


@pytest.fixture
def knockout_facts():
    return dict(
        overall_materiality=D("100"), ledger_difference=D("0"),
        confirmed_fictitious_revenue=D("0"),
        revenue_evidence="Synthetic evidence for the hand example only.",
    )


@pytest.mark.parametrize("points,expected", [
    ([D("0"), D("100"), None], D("100")),
    ([D("0"), D("50"), None], None),
    ([D("0"), D("50"), D("0")], D("50")),
])
def test_unknown_criterion_is_not_assumed_zero(points, expected):
    assert highest_observed_score(points) == expected


@pytest.mark.parametrize("field,amount,expected", [
    ("ledger_difference", "-101", Decision.CONDITIONAL),
    ("ledger_difference", "100", Decision.ACCEPT),
    ("ledger_difference", "101", Decision.CONDITIONAL),
    ("confirmed_fictitious_revenue", "101", Decision.CONDITIONAL),
])
def test_knockout_amount_boundary_and_unchanged_score(knockout_facts, field, amount, expected):
    scoring = dict(weighted_score=D("25"), preliminary_decision=Decision.ACCEPT)
    outcome = apply_knockout_rules(scoring, dict(knockout_facts, **{field: D(amount)}))
    assert outcome["decision"] is expected
    assert outcome["weighted_score"] == D("25")


def test_unverified_revenue_remains_pending(knockout_facts):
    facts = dict(knockout_facts, confirmed_fictitious_revenue=None, revenue_evidence=None)
    scoring = dict(weighted_score=D("25"), preliminary_decision=Decision.ACCEPT)
    outcome = apply_knockout_rules(scoring, facts)
    assert not outcome["review_complete"]
    assert outcome["pending_rules"] == ("confirmed_fictitious_revenue",)
    assert outcome["details"][1]["amount"] is None
    assert outcome["details"][1]["status"] == "unavailable"


def test_knockout_never_downgrades_rejection(knockout_facts):
    scoring = dict(weighted_score=D("80"), preliminary_decision=Decision.REJECT)
    outcome = apply_knockout_rules(scoring, dict(knockout_facts, ledger_difference=D("120")))
    assert outcome["triggered_rules"] == ("material_ledger_imbalance",)
    assert outcome["decision"] is Decision.REJECT


def test_confirmed_revenue_requires_evidence(knockout_facts):
    scoring = dict(weighted_score=D("25"), preliminary_decision=Decision.ACCEPT)
    facts = dict(knockout_facts, confirmed_fictitious_revenue=D("120"), revenue_evidence=None)
    with pytest.raises(ValueError, match="supporting evidence"):
        apply_knockout_rules(scoring, facts)


def test_sensitivity_normalization_boundary_and_input_preservation(risk_config, knockout_facts):
    scores = sample_scores()
    for row in scores.values():
        row["score"] = D("56")
    original = deepcopy((risk_config, scores, knockout_facts))
    scenarios = analyze_acceptance_sensitivity(risk_config, scores, knockout_facts)
    assert len(scenarios) == 15
    assert all(sum(row["weights"].values(), D("0")) == 1 for row in scenarios)
    assert scenarios[0]["weighted_score"] == D("56")
    changed = [row for row in scenarios if row["decision_changed"]]
    assert [row["scenario"] for row in changed] == ["cutoff_reject_0.8"]
    assert changed[0]["reject_cutoff"] == D("56")
    assert changed[0]["decision"] == "REJECT"
    assert (risk_config, scores, knockout_facts) == original


def test_financial_dimension_preserves_unavailable_ratio(risk_config):
    policy = risk_config["financial_policy"]
    rows = [
        dict(code=code, value=D("1"), calculation_status="calculated",
             assessment_status="assessed_policy", assessment="Good")
        for codes in policy["dimensions"].values() for code in codes
    ]
    ratios = pd.DataFrame(rows, dtype=object).set_index("code")
    ratios.loc["accrual_ratio", ["value", "assessment"]] = None
    ratios.loc["accrual_ratio", "calculation_status"] = "unavailable"
    ratios.loc["accrual_ratio", "assessment_status"] = "input_unavailable"
    ratios.loc["net_margin", "assessment"] = "Risk"
    tables = {
        "Ratios": ratios.reset_index(),
        "Summary": pd.DataFrame([("altman_applicability", "Synthetic example")],
                                columns=["metric", "value"]),
    }
    result = score_financial_health(tables, policy)
    assert result["score"] == D("20")
    assert result["unassessed_ratios"] == ("accrual_ratio",)
    tables["Ratios"].loc[tables["Ratios"]["code"].eq("net_margin"), "assessment"] = "Needs attention"
    assert score_financial_health(tables, policy)["score"] is None


def test_qualitative_uncertainty_is_disclosed_and_not_low_risk(risk_config):
    result = score_qualitative(risk_config["qualitative_policy"])
    assert result["score"] == D("50")
    assert set(result["pending_verification"]) == {
        "industry", "governance", "management_integrity",
    }
    assert all(row["evidence"] is None for row in result["details"])


@pytest.mark.parametrize("high,medium,expected", [(0, 4, "MEDIUM"), (1, 3, "HIGH")])
def test_je10_priority_uses_all_four_populations(high, medium, expected):
    tables = {
        "Summary": pd.DataFrame([dict(test_code="JE10", HIGH=high, MEDIUM=medium, LOW=0)]),
        "Run_Log": pd.DataFrame([dict(test_code="JE10", status="completed", performance_materiality=D("100"))]),
        "Coverage": pd.DataFrame(columns=["test_code", "metric", "value"]),
        "Benford_Statistics": pd.DataFrame([
            dict(population=name, chi_square_screen="above_critical")
            for name in BENFORD_POPULATION_SIDES
        ]),
    }
    assert extract_je_priorities(tables, D("100"))["JE10"] == expected
    tables["Benford_Statistics"] = tables["Benford_Statistics"].loc[
        tables["Benford_Statistics"]["population"].ne("all_ledger_credit")
    ]
    assert extract_je_priorities(tables, D("100"))["JE10"] is None
