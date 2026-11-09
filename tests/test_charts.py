"""Check chart units, data coverage, missing inputs and offline output generation."""

from copy import deepcopy
from datetime import date
from decimal import Decimal as D
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from auditpy.benford import benford_frequency
from auditpy.charts import (
    ChartBuilder, build_chart_outputs, prepare_account_groups,
    prepare_line_amounts, prepare_risk_groups, prepare_ratio_dashboard, prepare_balance_structure,
    prepare_weekday_counts,
    supplementary_dashboard_figures,
)
from auditpy.ratios import gross_margin, screen_monthly_series
from auditpy.utils import SchemaError


@pytest.fixture
def visual_tables():
    monthly = pd.DataFrame({
        "month": [f"2024-{month:02d}" for month in range(1, 7)],
        "net_revenue": [D("1000000000")] * 6,
        "cost_of_sales": [D("800000000")] * 5 + [D("300000000")],
    })
    monthly["gross_profit"] = monthly["net_revenue"] - monthly["cost_of_sales"]
    monthly["gross_margin"] = [
        gross_margin(g, r).value for g, r in
        zip(monthly["gross_profit"], monthly["net_revenue"])
    ]
    trend, baseline = screen_monthly_series(monthly, "gross_margin", tuple(monthly["month"]))
    balance = {
        "current_assets": "70000000000", "noncurrent_assets": "30000000000",
        "unclassified_assets": "0", "current_liabilities": "40000000000",
        "noncurrent_liabilities": "0", "unclassified_liabilities": "0",
        "equity": "60000000000", "total_assets": "100000000000",
        "total_funding": "100000000000", "balance_difference": "0",
    }
    statements = {
        "General_Ledger": pd.DataFrame({
            "account": ["1111", "1121", "6421", None],
            "debit": [1000000, 0, -3000000, 0], "credit": [0, 2000000, 0, 0],
            "posting_date": [date(2024, 1, 1), date(2024, 1, 1),
                             date(2024, 1, 7), date(2024, 1, 2)],
        }),
        "Balance_Sheet": pd.DataFrame([
            dict(item=item, amount=D(value), calculation_status="calculated")
            for item, value in balance.items()
        ]),
    }
    frequency = benford_frequency([D("100"), D("120"), D("250"), D("320"), D("900")])
    je = {
        "Benford_Frequency": pd.concat([
            frequency.assign(population=population) for population in
            ("all_ledger_debit", "all_ledger_credit", "revenue", "expenses")
        ], ignore_index=True),
        "Summary": pd.DataFrame([
            dict(test_code=f"JE{i:02d}", finding_count=int(i == 1),
                 HIGH=int(i == 1), MEDIUM=0, LOW=0) for i in range(1, 11)
        ]),
    }
    financial = {
        "Monthly": monthly, "Monthly_Trend": trend,
        "Monthly_Baselines": pd.DataFrame([baseline]),
        "Ratios": pd.DataFrame([
            ("current_ratio", "1.6", "times", "calculated", "Good", "assessed_policy", False),
            ("net_margin", "0.03", "fraction", "calculated", "Risk", "assessed_policy", False),
            ("accrual_ratio", None, "fraction", "unavailable", "", "input_unavailable", False),
        ], columns=["code", "value", "unit", "calculation_status", "assessment",
                    "assessment_status", "maturity_assumption_applied"]),
        "Benchmarks": pd.DataFrame([
            ("current_ratio", "times", "higher", "1.5", "1", "0"),
            ("net_margin", "fraction", "higher", "0.10", "0.05", None),
            ("accrual_ratio", "fraction", "absolute_lower", "0.05", "0.10", None),
        ], columns=["ratio_code", "unit", "direction", "good_cutoff", "risk_cutoff", "valid_minimum"]),
    }
    acceptance = {
        "Group_Scores": pd.DataFrame([
            ("data_quality", 20, "0.15", "3"), ("journal_entries", 60, "0.30", "18"),
            ("financial_health", 50, "0.30", "15"), ("reporting_risk", 40, "0.15", "6"),
            ("qualitative", 30, "0.10", "3"),
        ], columns=["group", "score", "weight", "contribution"]),
        "Summary": pd.DataFrame([
            ("weighted_score", D("45")), ("decision_after_knockouts", "CONDITIONAL"),
            ("policy_configuration", {"decision_thresholds": {"conditional": "40", "reject": "70"}}),
        ], columns=["metric", "value"]),
    }
    return statements, je, financial, acceptance


def test_line_amounts_preserve_signs_missing_values_and_source():
    source = pd.DataFrame({
        "account": ["1111"] * 7,
        "debit": [1000000, 0, -10000000, 0, None, "bad", 3000000],
        "credit": [0, 2000000, 0, 0, 0, 0, 4000000],
    })
    original = source.copy(deep=True)
    selected, coverage = prepare_line_amounts(source)
    assert selected["line_amount"].tolist() == [D("1000000"), D("2000000"), D("10000000"), D("4000000")]
    assert coverage == dict(input_records=7, plotted_records=4, zero_records=1,
                            unavailable_amount_records=2, negative_amount_records=1,
                            both_sides_nonzero_records=1)
    pd.testing.assert_frame_equal(source, original)


def test_account_median_and_unknown_group_do_not_drop_records():
    source = pd.DataFrame({"account": ["1111", "1121", "1311", None],
                           "line_amount": [D("1"), D("2"), D("100"), D("5")]})
    grouped, summary = prepare_account_groups(source)
    assert summary.set_index("account_group").at["1", "median_amount"] == D("2")
    assert summary["line_count"].sum() == len(grouped) == 4
    assert "unclassified" in set(grouped["account_group"])


def test_monthly_plot_converts_only_display_units_and_keeps_flag(visual_tables, tmp_path, monkeypatch):
    monthly = visual_tables[2]["Monthly"]
    original = monthly.copy(deep=True)
    baseline = visual_tables[2]["Monthly_Baselines"].iloc[0].to_dict()
    assert baseline["flagged_months"] == ("2024-06",)
    builder = ChartBuilder(tmp_path)
    def inspect(draw, filename, title, ylabel, **kwargs):
        fig, ax = plt.subplots()
        try:
            draw(ax)
            assert ax.patches[0].get_height() == 1.0
            assert ax.lines[0].get_ydata()[0] == 0.2
            assert ylabel == "Tỷ đồng (VND)"
        finally:
            plt.close(fig)
    monkeypatch.setattr(builder, "_save", inspect)
    builder.monthly_performance(monthly)
    pd.testing.assert_frame_equal(monthly, original)


def test_risk_inputs_match_hand_total_without_mutation(visual_tables):
    groups = visual_tables[3]["Group_Scores"]
    original = groups.copy(deep=True)
    result = prepare_risk_groups(groups.iloc[::-1], "45")
    assert result["contribution"].tolist() == list(map(D, ["3", "18", "15", "6", "3"]))
    pd.testing.assert_frame_equal(groups, original)


@pytest.mark.parametrize("change", ["missing", "duplicate", "weight", "score", "contribution", "total"])
def test_risk_inputs_reject_inconsistent_evidence(visual_tables, change):
    groups = visual_tables[3]["Group_Scores"].copy()
    total = "45"
    if change == "missing": groups = groups.iloc[:-1]
    elif change == "duplicate": groups.loc[1, "group"] = groups.loc[0, "group"]
    elif change == "weight": groups.loc[0, "weight"] = "0"
    elif change == "score": groups.loc[0, "score"] = 101
    elif change == "contribution": groups.loc[0, "contribution"] = "99"
    else: total = "46"
    with pytest.raises(SchemaError): prepare_risk_groups(groups, total)


def test_ratio_table_keeps_unavailable_results_and_exact_thresholds(visual_tables):
    financial = visual_tables[2]
    original = financial["Ratios"].copy(deep=True)
    dashboard = prepare_ratio_dashboard(financial["Ratios"], financial["Benchmarks"]).set_index("code")
    assert dashboard.at["net_margin", "value"] == "3,00%"
    assert dashboard.at["current_ratio", "good_reference"] == "≥ 1,5 lần"
    assert dashboard.at["accrual_ratio", "rating"] == "Chưa tính"
    pd.testing.assert_frame_equal(financial["Ratios"], original)


def test_build_creates_nine_pngs_and_offline_html_from_tables(visual_tables, tmp_path):
    result = build_chart_outputs(*visual_tables, tmp_path)
    assert result["skipped"] == []
    assert len(result["paths"]) == 10
    pngs = [p for p in result["paths"] if p.suffix == ".png"]
    assert len(pngs) == 9
    assert result["weekday_coverage"] == dict(input_records=4, dated_records=4,
                                              unavailable_date_records=0)
    assert (tmp_path / "charts/09_posting_weekday_heatmap.png").is_file()
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in pngs)
    html = (tmp_path / "dashboard.html").read_text()
    assert set(re.findall(r'data-chart="([^"]+)"', html)) == {f"{i:02d}" for i in range(1, 10)} | {"JE"}
    assert len(re.findall(r'<button[^>]+role="tab"', html)) == 3
    assert len(re.findall(r'<div[^>]+role="tabpanel"', html)) == 3
    assert 'plotly.js v' in html and html.count('class="plotly-graph-div"') == 9
    assert not re.search(r'<script[^>]+src=[\'"]https?://', html)
    assert "45,00 / 100" in html and "Chấp nhận có điều kiện" in html
    assert "Chưa tính" in html


@pytest.fixture
def dashboard_details(visual_tables):
    statements, je, financial, acceptance = visual_tables
    amounts, _ = prepare_line_amounts(statements["General_Ledger"])
    grouped, _ = prepare_account_groups(amounts)
    components, totals = prepare_balance_structure(statements["Balance_Sheet"])
    weekday, coverage = prepare_weekday_counts(statements["General_Ledger"])
    return dict(
        frequency=je["Benford_Frequency"], amounts=amounts, grouped=grouped,
        components=components, totals=totals, weekday=weekday, weekday_coverage=coverage,
        trend=financial["Monthly_Trend"], baseline=financial["Monthly_Baselines"].iloc[0].to_dict(),
        thresholds={"conditional": "40", "reject": "70"},
    )


def test_interactive_views_preserve_counts_quantiles_and_signed_balances(dashboard_details):
    details = dashboard_details
    details["components"].loc[details["components"]["item"].eq("equity"), "amount"] = D("-10000000000")
    plots = {code: fig for code, fig, _, _ in supplementary_dashboard_figures(details)}
    assert len(plots["03"].data) == 5
    assert all(sum(trace.y) == pytest.approx(100) for trace in plots["03"].data)
    assert sum(plots["04"].data[0].y) == 3
    assert plots["04"].layout.xaxis.type == "log"
    assert plots["04"].layout.xaxis.exponentformat == "power"
    assert all(center - width / 2 > 0 for center, width in
               zip(plots["04"].data[0].x, plots["04"].data[0].width))
    box = plots["05"].data[0]
    assert box.q1 == pytest.approx((0.00125,))
    assert box.median == pytest.approx((0.0015,))
    assert box.q3 == pytest.approx((0.00175,))
    assert plots["06"].data[-1].y == (-10.0,)
    assert list(plots["09"].data[0].z[0]) == [2, 1, 0, 0, 0, 0, 1]
    assert plots["09"].data[0].zmin == 0 and plots["09"].data[0].zmax == 2
    assert plots["09"].data[0].zauto is False


def test_interactive_margin_flags_and_risk_regions_use_current_evidence(
    visual_tables, dashboard_details, tmp_path, monkeypatch,
):
    import plotly.graph_objects as go
    figures = []
    def capture(fig, **kwargs):
        figures.append(fig)
        return "<div></div>"
    monkeypatch.setattr(go.Figure, "to_html", capture)
    ChartBuilder(tmp_path / "charts").interactive_dashboard(
        visual_tables[2]["Monthly"], visual_tables[1]["Summary"],
        acceptance=visual_tables[3], chart_data=dashboard_details,
    )
    margin, risk = figures[1], figures[3]
    assert list(margin.data[1].x) == ["2024-06"]
    assert list(margin.data[1].y) == [70.0]
    assert margin.layout.shapes[0].y1 == pytest.approx(float(dashboard_details["baseline"]["upper_bound"] * 100))
    assert sum(trace.x[0] for trace in risk.data) == 45
    assert [(shape.x0, shape.x1) for shape in risk.layout.shapes[:3]] == [(0, 40), (40, 70), (70, 100)]
    assert risk.layout.shapes[-1].x0 == 45


def test_missing_chart_inputs_are_skipped_without_filling_zero(visual_tables, tmp_path):
    statements, je, financial, acceptance = deepcopy(visual_tables)
    financial["Monthly"].loc[0, "gross_margin"] = None
    financial["Monthly_Trend"].loc[0, "value"] = None
    acceptance["Summary"].loc[acceptance["Summary"]["metric"].eq("weighted_score"), "value"] = None
    result = build_chart_outputs(statements, je, financial, acceptance, tmp_path)
    assert set(result["skipped"]) == {"monthly_gross_margin", "acceptance_risk", "interactive_dashboard"}
    assert financial["Monthly"].loc[0, "gross_margin"] is None
    assert not (tmp_path / "dashboard.html").exists()


def test_save_closes_figure_after_render_error(tmp_path):
    before = plt.get_fignums()
    def broken(ax): raise RuntimeError("Synthetic render failure")
    with pytest.raises(RuntimeError, match="Synthetic render failure"):
        ChartBuilder(tmp_path)._save(broken, "broken.png", "Example", "Example")
    assert plt.get_fignums() == before
    assert not (tmp_path / "broken.png").exists()


def test_balance_structure_preserves_signed_components_and_source_difference(visual_tables):
    balance = visual_tables[0]["Balance_Sheet"].copy(deep=True)
    for item, value in {"equity": "-10000000000", "total_funding": "30000000000",
                        "balance_difference": "70000000000"}.items():
        balance.loc[balance["item"].eq(item), "amount"] = D(value)
    original = balance.copy(deep=True)
    components, totals = prepare_balance_structure(balance)
    assert components.set_index("item").at["equity", "amount"] == D("-10000000000")
    assert totals["assets"] - totals["funding"] == totals["balance_difference"] == D("70000000000")
    pd.testing.assert_frame_equal(balance, original)
    with pytest.raises(SchemaError):
        prepare_balance_structure(pd.concat([balance, balance.iloc[:1]]))


def test_weekday_counts_count_lines_and_retain_unavailable_date_coverage():
    table = pd.DataFrame({
        "posting_date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2),
                         date(2024, 1, 7), None, "bad date", pd.NaT],
        "voucher_no": ["same"] * 7,
        "debit": [0] * 7,
    })
    original = table.copy(deep=True)
    counts, coverage = prepare_weekday_counts(table)
    assert counts["weekday"].tolist() == list(range(7))
    assert counts["line_count"].tolist() == [2, 1, 0, 0, 0, 0, 1]
    assert counts["share"].tolist() == list(map(D, ["0.5", "0.25", "0", "0", "0", "0", "0.25"]))
    assert coverage == dict(input_records=7, dated_records=4, unavailable_date_records=3)
    pd.testing.assert_frame_equal(table, original)


def test_weekday_heatmap_matches_hand_counts(visual_tables, tmp_path, monkeypatch):
    counts, coverage = prepare_weekday_counts(visual_tables[0]["General_Ledger"])
    builder = ChartBuilder(tmp_path)
    def inspect(draw, filename, title, ylabel, **kwargs):
        fig, ax = plt.subplots()
        try:
            draw(ax)
            assert ax.images[0].get_array().tolist() == [[2, 1, 0, 0, 0, 0, 1]]
            assert ax.get_xticklabels()[-1].get_text() == "Chủ nhật"
            assert filename == "09_posting_weekday_heatmap.png"
            assert "giờ nhập liệu" in kwargs["note"]
        finally:
            plt.close(fig)
    monkeypatch.setattr(builder, "_save", inspect)
    builder.weekday_heatmap(counts, coverage)


def test_no_usable_dates_skip_weekday_chart_without_inventing_activity(visual_tables, tmp_path):
    statements, je, financial, acceptance = deepcopy(visual_tables)
    statements["General_Ledger"]["posting_date"] = None
    result = build_chart_outputs(statements, je, financial, acceptance, tmp_path)
    assert result["skipped"] == ["weekday_heatmap"]
    assert result["weekday_coverage"] == dict(input_records=4, dated_records=0,
                                              unavailable_date_records=4)
    assert not (tmp_path / "charts/09_posting_weekday_heatmap.png").exists()
    with pytest.raises(SchemaError):
        prepare_weekday_counts(pd.DataFrame({"debit": [0]}))
