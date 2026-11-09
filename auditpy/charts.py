"""Draw audit charts from prepared tables; retain Decimal until rendering."""

from decimal import Decimal
from datetime import date
from pathlib import Path
from math import floor, log10
from statistics import median
import pandas as pd
import matplotlib.pyplot as plt
from auditpy.utils import SchemaError, normalize_amount, normalize_code, is_finite_amount
from auditpy.ratios import prepare_benchmark_policies

def prepare_weekday_counts(table):
    """Count normalized ledger lines by posting weekday, retaining date coverage."""
    if "posting_date" not in table:
        raise SchemaError("Weekday analysis requires posting_date.")
    days = table["posting_date"].map(
        lambda value: value.weekday()
        if isinstance(value, date) and not pd.isna(value) else None
    )
    counts = days.value_counts().reindex(range(7), fill_value=0)
    dated = int(counts.sum())
    summary = pd.DataFrame({
        "weekday": range(7), "line_count": counts.to_numpy(),
        "share": [Decimal(int(count)) / Decimal(dated) if dated else None for count in counts],
    })
    coverage = dict(input_records=len(table), dated_records=dated,
                    unavailable_date_records=len(table) - dated)
    return summary, coverage

def require_chart_values(table, columns):
    required = {"month"}.union(columns)
    if table.empty or not required.issubset(table.columns):
        raise SchemaError("Monthly chart columns are missing or empty.")
    if table["month"].isna().any() or table["month"].duplicated().any():
        raise SchemaError("Chart months must be present and unique.")
    for column in columns:
        if not all(is_finite_amount(v) for v in table[column]):
            raise SchemaError(f"Chart column {column} requires finite Decimal values.")
    return table.sort_values("month").copy()

def prepare_line_amounts(table):
    """Measure each ledger line once without netting its debit and credit."""
    if not {"account", "debit", "credit"}.issubset(table.columns):
        raise SchemaError("Line amount analysis requires account, debit and credit.")

    data = table.copy()
    amounts = []
    coverage = {
        "input_records": len(data),
        "plotted_records": 0,
        "zero_records": 0,
        "unavailable_amount_records": 0,
        "negative_amount_records": 0,
        "both_sides_nonzero_records": 0,
    }

    for debit, credit in zip(data["debit"], data["credit"]):
        debit, debit_status = normalize_amount(debit)
        credit, credit_status = normalize_amount(credit)

        if debit_status != "ok" or credit_status != "ok":
            coverage["unavailable_amount_records"] += 1
            amounts.append(None)
            continue

        amount = max(abs(debit), abs(credit))
        coverage["negative_amount_records"] += int(debit < 0 or credit < 0)
        coverage["both_sides_nonzero_records"] += int(debit != 0 and credit != 0)
        coverage["zero_records"] += int(amount == 0)
        amounts.append(amount)

    data["line_amount"] = amounts
    data = data.loc[data["line_amount"].map(
        lambda amount: amount is not None and amount > 0,
    )].copy()

    coverage["plotted_records"] = len(data)
    return data, coverage

def prepare_account_groups(line_amounts):
    """Group observed line sizes by account class without changing their amounts."""
    if not {"account", "line_amount"}.issubset(line_amounts.columns):
        raise SchemaError("Account grouping requires account and line_amount.")

    data = line_amounts.copy()
    if not all(is_finite_amount(v) and v > 0
               for v in data["line_amount"]):
        raise SchemaError("Account grouping requires positive finite Decimals.")

    codes = data["account"].map(normalize_code)
    data["account_group"] = codes.map(
        lambda code: code[0]
        if code and code.isdigit() and code[0] in "123456789"
        else "unclassified"
    )

    rows = []
    for group, records in data.groupby("account_group", sort=True):
        rows.append({
            "account_group": group,
            "line_count": len(records),
            "median_amount": median(records["line_amount"]),
        })

    summary = pd.DataFrame(
        rows, columns=["account_group", "line_count", "median_amount"],
    )
    return data, summary

def prepare_balance_structure(table):
    """Select non-overlapping balance-sheet components and verify their totals."""
    if not {"item", "amount", "calculation_status"}.issubset(table.columns):
        raise SchemaError(
            "Balance structure requires item, amount and calculation_status."
        )

    source = table.set_index("item")
    groups = {
        "assets": (
            "current_assets", "noncurrent_assets", "unclassified_assets",
        ),
        "funding": (
            "current_liabilities", "noncurrent_liabilities",
            "unclassified_liabilities", "equity",
        ),
    }
    required = [item for items in groups.values() for item in items]
    required += ["total_assets", "total_funding", "balance_difference"]

    if source.index.has_duplicates or not set(required).issubset(source.index):
        raise SchemaError("Balance structure items must be present and unique.")

    amounts = {}
    for item in required:
        value, status = normalize_amount(source.at[item, "amount"])
        if status != "ok" or source.at[item, "calculation_status"] != "calculated":
            raise SchemaError(f"Balance structure input is unavailable: {item}")
        amounts[item] = value

    totals = {
        group: sum((amounts[item] for item in items), Decimal("0"))
        for group, items in groups.items()
    }
    totals["balance_difference"] = totals["assets"] - totals["funding"]

    if (totals["assets"] != amounts["total_assets"]
            or totals["funding"] != amounts["total_funding"]
            or totals["balance_difference"] != amounts["balance_difference"]):
        raise SchemaError("Chart components do not reconcile to the balance sheet.")

    components = pd.DataFrame([
        {"group": group, "item": item, "amount": amounts[item]}
        for group, items in groups.items() for item in items
    ])
    return components, totals

def prepare_ratio_dashboard(ratios, benchmarks):
    """Format ratio values, cutoffs and assessments for display."""
    names = {
        "current_ratio": ("Thanh khoản", "Thanh toán hiện hành"),
        "quick_ratio": ("Thanh khoản", "Thanh toán nhanh"),
        "cash_ratio": ("Thanh khoản", "Thanh toán tiền mặt"),
        "debt_to_assets": ("Cơ cấu vốn", "Nợ / Tổng tài sản"),
        "debt_to_equity": ("Cơ cấu vốn", "Nợ / Vốn chủ sở hữu"),
        "interest_coverage": ("Cơ cấu vốn", "Khả năng trả lãi vay"),
        "receivables_turnover": ("Hoạt động", "Vòng quay phải thu"),
        "days_sales_outstanding": ("Hoạt động", "Số ngày thu tiền (DSO)"),
        "inventory_turnover": ("Hoạt động", "Vòng quay hàng tồn kho"),
        "days_inventory_outstanding": ("Hoạt động", "Số ngày tồn kho (DIO)"),
        "days_payables_outstanding": ("Hoạt động", "Số ngày phải trả (DPO)"),
        "cash_conversion_cycle": ("Hoạt động", "Chu kỳ tiền mặt (CCC)"),
        "asset_turnover": ("Hoạt động", "Vòng quay tổng tài sản"),
        "gross_margin": ("Sinh lời", "Biên lợi nhuận gộp"),
        "net_margin": ("Sinh lời", "Biên lợi nhuận ròng"),
        "return_on_assets": ("Sinh lời", "ROA"),
        "return_on_equity": ("Sinh lời", "ROE"),
        "altman_z_double_prime": ("Cảnh báo", "Altman Z''"),
        "accrual_ratio": ("Cảnh báo", "Tỷ lệ dồn tích"),
    }
    operators = {
        "higher": ("≥", "<"),
        "higher_exclusive": (">", "<"),
        "lower": ("≤", "≥"),
        "lower_exclusive": ("≤", ">"),
        "absolute_lower": ("|x| ≤", "|x| ≥"),
        "context": ("", ""),
    }
    required = {
        "code", "value", "unit", "calculation_status",
        "assessment", "assessment_status", "maturity_assumption_applied",
    }
    if ratios.empty or not required.issubset(ratios.columns):
        raise SchemaError("Ratio dashboard columns are missing or empty.")
    if ratios["code"].duplicated().any() or not set(ratios["code"]).issubset(names):
        raise SchemaError("Dashboard ratio codes must be known and unique.")

    policies = prepare_benchmark_policies(benchmarks)
    source = ratios.set_index("code").to_dict("index")

    def display(value, unit, exact=False):
        number, status = normalize_amount(value)
        if status == "missing":
            return "—"
        if status != "ok":
            raise SchemaError("Dashboard values must be finite numbers.")
        suffix = {
            "fraction": "%", "times": " lần",
            "days": " ngày", "score": " điểm",
        }[unit]
        if unit == "fraction":
            number *= 100
        text = format(number, "f") if exact else f"{number:.2f}"
        if exact and "." in text:
            text = text.rstrip("0").rstrip(".")
        return text.replace(".", ",") + suffix

    rows = []
    for code, (group, name) in names.items():
        if code not in source:
            continue
        row, policy = source[code], policies.get(code)
        good = risk = "—"

        if policy is not None:
            if policy["unit"] != row["unit"]:
                raise SchemaError("Ratio and reference units must match.")
            good_op, risk_op = operators[policy["direction"]]
            good = display(policy["good_cutoff"], row["unit"], True)
            risk = display(policy["risk_cutoff"], row["unit"], True)
            good = f"{good_op} {good}".strip() if good != "—" else good
            risk = f"{risk_op} {risk}".strip() if risk != "—" else risk

        value = display(row["value"], row["unit"])
        if row["calculation_status"] == "unavailable":
            value, rating = "Chưa tính được", "Chưa tính"
        elif row["calculation_status"] != "calculated" or value == "—":
            raise SchemaError("Calculated ratios require a numeric value.")
        elif row["assessment_status"] == "assessed_policy":
            if policy is None:
                raise SchemaError("An assessed ratio requires its benchmark.")
            rating = {
                "Good": "Tốt", "Needs attention": "Cần chú ý", "Risk": "Rủi ro",
            }[row["assessment"]]
        else:
            rating = "Cần xác minh"

        name += " *" if row["maturity_assumption_applied"] == True else ""
        rows.append(dict(
            code=code, group=group, indicator=name, value=value,
            good_reference=good, risk_reference=risk, rating=rating,
        ))

    return pd.DataFrame(rows)

def prepare_risk_groups(groups, total_score):
    """Validate weighted chart inputs without changing the reported scores."""
    required = {"group", "score", "weight", "contribution"}
    order = ("data_quality", "journal_entries", "financial_health",
             "reporting_risk", "qualitative")
    if not required.issubset(groups.columns):
        raise SchemaError("Risk chart columns are missing.")
    data = groups.set_index("group").copy()
    if data.index.has_duplicates or set(data.index) != set(order):
        raise SchemaError("Risk chart requires exactly five unique groups.")
    for column in ("score", "weight", "contribution"):
        data[column] = data[column].map(lambda value: normalize_amount(value)[0])
        if data[column].isna().any():
            raise SchemaError(f"Risk chart values are unavailable: {column}")
    if not all(0 <= value <= 100 for value in data["score"]):
        raise SchemaError("Group scores must be between 0 and 100.")
    if not all(value > 0 for value in data["weight"]) or sum(data["weight"]) != 1:
        raise SchemaError("Positive group weights must sum to 1.")
    if not (data["score"] * data["weight"] == data["contribution"]).all():
        raise SchemaError("Contributions must equal score times weight.")
    total = normalize_amount(total_score)[0]
    if total is None or sum(data["contribution"]) != total:
        raise SchemaError("Contributions do not match the reported total.")
    return data.loc[list(order)]


def supplementary_dashboard_figures(details):
    
    import numpy as np
    import plotly.graph_objects as go
    from matplotlib.cbook import boxplot_stats

    figures = []
    scale = Decimal("1000000000")
    frequency = details["frequency"]
    fig = go.Figure()
    populations = {
        "all_ledger_debit": ("Nợ toàn sổ", "#2563EB"),
        "all_ledger_credit": ("Có toàn sổ", "#7C3AED"),
        "revenue": ("Có doanh thu/thu nhập", "#16A34A"),
        "expenses": ("Nợ chi phí", "#F59E0B"),
    }
    for population, (label, color) in populations.items():
        rows = frequency.loc[frequency["population"].eq(population)].sort_values("digit")
        count = int(rows["observed_count"].sum())
        if not count:
            continue
        fig.add_trace(go.Bar(
            x=rows["digit"].tolist(), y=(rows["observed_rate"] * 100).tolist(),
            name=f"{label} ({count:,} quan sát)", marker_color=color,
            hovertemplate="Chữ số %{x}: %{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    expected = frequency.drop_duplicates("digit").sort_values("digit")
    if fig.data:
        fig.add_trace(go.Scatter(
            x=expected["digit"].tolist(), y=(expected["expected_rate"] * 100).tolist(),
            name="Kỳ vọng Benford", mode="lines+markers", line={"color": "#DC2626"},
        ))
    fig.update_layout(barmode="group", xaxis_title="Chữ số có nghĩa đầu tiên",
                      yaxis_title="Tần suất (%)", xaxis={"dtick": 1})
    figures.append(("03", fig, "Benford: thực tế và kỳ vọng",
                    "Các tập có thể giao nhau; độ lệch chưa xác nhận gian lận."))

    amounts = details["amounts"]
    values = [float(v / scale) for v in amounts["line_amount"]]
    fig = go.Figure()
    if values:
        edges = [10.0 ** power for power in range(
            floor(log10(min(values))), floor(log10(max(values))) + 2,
        )]
        counts, _ = np.histogram(values, bins=edges)
        fig.add_trace(go.Bar(
            x=[(left + right) / 2 for left, right in zip(edges[:-1], edges[1:])],
            y=counts.tolist(), width=np.diff(edges).tolist(), marker_color="#2563EB",
            customdata=list(zip(edges[:-1], edges[1:])), showlegend=False,
            hovertemplate="%{customdata[0]:.3g}–%{customdata[1]:.3g} tỷ đồng"
                          "<br>%{y:,} dòng<extra></extra>",
        ))
    fig.update_layout(xaxis={"type": "log", "exponentformat": "power", "dtick": 2,
                             "title": "Quy mô dòng (tỷ đồng VND, thang log 10)"},
                      yaxis_title="Số dòng hạch toán", bargap=0)
    figures.append(("04", fig, "Phân phối quy mô dòng hạch toán",
                    "Quy mô = max(|Nợ|, |Có|); bỏ số 0/thiếu, giữ độ lớn số âm và kết chuyển. Đếm dòng."))

    fig = go.Figure()
    for group, rows in details["grouped"].groupby("account_group", sort=True):
        values = [float(v / scale) for v in rows["line_amount"]]
        stats = boxplot_stats(values, whis=1.5)[0]
        label = (f"Loại {group}" if group != "unclassified" else "Chưa phân loại")
        label += f"<br>{len(rows):,} dòng"
        fig.add_trace(go.Box(
            x=[label], q1=[stats["q1"]], median=[stats["med"]], q3=[stats["q3"]],
            lowerfence=[stats["whislo"]], upperfence=[stats["whishi"]],
            name=label, showlegend=False, fillcolor="#DBEAFE", line={"color": "#2563EB"},
        ))
        fig.add_trace(go.Scattergl(
            x=[label] * len(stats["fliers"]), y=stats["fliers"].tolist(), mode="markers",
            marker={"size": 3, "color": "#64748B", "opacity": 0.4}, showlegend=False,
            hovertemplate="%{x}<br>%{y:.6g} tỷ đồng<extra>Điểm ngoài râu</extra>",
        ))
    fig.update_layout(yaxis={"type": "log", "exponentformat": "power", "dtick": 2,
                             "title": "Quy mô dòng (tỷ đồng VND, thang log 10)"},
                      xaxis_title="Loại tài khoản theo chữ số đầu")
    figures.append(("05", fig, "Boxplot theo nhóm tài khoản",
                    "Hộp Q1–Q3, đường giữa là trung vị; râu 1,5 × IQR trên tiền gốc, như PNG. Giữ kết chuyển."))

    fig = go.Figure()
    labels = {
        "current_assets": "Tài sản ngắn hạn", "noncurrent_assets": "Tài sản dài hạn",
        "unclassified_assets": "Tài sản chưa phân loại", "current_liabilities": "Nợ ngắn hạn",
        "noncurrent_liabilities": "Nợ dài hạn", "unclassified_liabilities": "Nợ chưa phân loại",
        "equity": "Vốn chủ sở hữu",
    }
    for row in details["components"].to_dict("records"):
        fig.add_trace(go.Bar(
            x=["Tài sản" if row["group"] == "assets" else "Nguồn vốn"],
            y=[float(row["amount"] / scale)], name=labels[row["item"]],
            hovertemplate="%{y:.3f} tỷ đồng<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(barmode="relative", yaxis_title="Tỷ đồng (VND)")
    difference = details["totals"].get("balance_difference")
    figures.append(("06", fig, "Cơ cấu tài sản và nguồn vốn",
                    f"Tài sản − nguồn vốn = {difference} đơn vị nguồn."))

    weekdays, coverage = details["weekday"], details["weekday_coverage"]
    labels = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
    fig = go.Figure()
    if coverage["dated_records"]:
        counts = weekdays["line_count"].tolist()
        text = [f"{count:,} dòng<br>{count / coverage['dated_records']:.1%}" for count in counts]
        fig.add_trace(go.Heatmap(
            x=labels, y=["Dòng sổ"], z=[counts], text=[text], texttemplate="%{text}",
            zmin=0, zmax=max(counts), zauto=False,
            colorscale="Blues", colorbar={"title": "Số dòng"},
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Thứ của ngày hạch toán", yaxis={"showticklabels": False})
    figures.append(("09", fig, "Số dòng hạch toán theo thứ",
                    f"Ngày hợp lệ: {coverage['dated_records']:,}; thiếu/không hợp lệ: "
                    f"{coverage['unavailable_date_records']:,}. Giữ kết chuyển và dòng 0; không phải giờ nhập liệu."))
    return figures


class ChartBuilder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, draw, filename, title, ylabel, note="", xlabel="Tháng"):
        fig, ax = plt.subplots(figsize=(11, 5))
        try:
            draw(ax)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

            bottom = 0.08 if note else 0
            fig.tight_layout(rect=(0, bottom, 1, 1))
            if note:
                fig.text(0.5, 0.015, note, ha="center", fontsize=9)

            path = self.output_dir / filename
            fig.savefig(path, dpi=160, bbox_inches="tight")
        finally:
            plt.close(fig)
        return path

    def monthly_performance(self, monthly):
        data = require_chart_values(
            monthly, ("net_revenue", "cost_of_sales", "gross_profit"),
        )
        scale = Decimal("1000000000")

        def draw(ax):
            x = list(range(len(data)))
            revenue = [float(v / scale) for v in data["net_revenue"]]
            costs = [float(v / scale) for v in data["cost_of_sales"]]
            profit = [float(v / scale) for v in data["gross_profit"]]
            ax.bar([i - 0.2 for i in x], revenue, width=0.4,
                   label="Doanh thu thuần", color="#2563EB")
            ax.bar([i + 0.2 for i in x], costs, width=0.4,
                   label="Giá vốn", color="#94A3B8")
            ax.plot(x, profit, marker="o", color="#16A34A", label="Lợi nhuận gộp")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x, data["month"].tolist(), rotation=45)

        return self._save(
            draw, "01_monthly_performance.png",
            "Doanh thu – giá vốn – lợi nhuận gộp theo tháng",
            "Tỷ đồng (VND)",
            )

    def monthly_gross_margin(self, trend, baseline):
        data = require_chart_values(trend, ("value",))
        if baseline["baseline_status"] not in ("calculated", "constant_series"):
            raise SchemaError("Monthly screening is incomplete; review its inputs.")
        if "outside_band" not in data or data["outside_band"].isna().any():
            raise SchemaError("Monthly screening flags are unavailable.")

        def draw(ax):
            x = list(range(len(data)))
            values = [float(v * 100) for v in data["value"]]
            lower = float(baseline["lower_bound"] * 100)
            upper = float(baseline["upper_bound"] * 100)
            mean = float(baseline["mean"] * 100)

            ax.axhspan(
                lower, upper, color="#CBD5E1", alpha=0.5,
                label="Trung bình ± 2 độ lệch chuẩn",
            )
            ax.axhline(
                mean, color="#64748B", linestyle="--",
                label=f"Trung bình tháng: {mean:.2f}%",
            )
            ax.plot(
                x, values, marker="o", color="#2563EB",
                label="Biên gộp",
            )
            flagged = [i for i, flag in enumerate(data["outside_band"]) if flag]
            if flagged:
                ax.scatter(
                    flagged, [values[i] for i in flagged],
                    color="#DC2626", s=85, zorder=5,
                    label="Tháng cần xem xét",
                )
            ax.set_xticks(x, data["month"].tolist(), rotation=45)

        return self._save(
            draw, "02_monthly_gross_margin.png",
            "Biên lợi nhuận gộp theo tháng",
            "Biên lợi nhuận gộp (%)",
            note=(
                f"Vùng tô: {baseline['lower_bound']:.2%}–{baseline['upper_bound']:.2%}; "
                "điểm đỏ là dấu hiệu cần kiểm tra."
            ),
        )

    def benford_comparison(self, frequency):
        required = {
            "population", "digit", "observed_count", "observed_rate", "expected_rate",
        }
        if not required.issubset(frequency.columns):
            raise SchemaError("Benford frequency columns are missing.")

        labels = {
            "all_ledger_debit": ("PS Nợ dương – toàn sổ", "#2563EB"),
            "all_ledger_credit": ("PS Có dương – toàn sổ", "#7C3AED"),
            "revenue": ("PS Có dương – doanh thu/thu nhập", "#16A34A"),
            "expenses": ("PS Nợ dương – chi phí", "#F59E0B"),
        }
        groups = []
        for population in labels:
            data = frequency.loc[frequency["population"].eq(population)].copy()
            if data.empty:
                continue
            data = data.sort_values("digit")
            if data["digit"].tolist() != list(range(1, 10)):
                raise SchemaError("Each Benford population requires digits 1 through 9.")
            if data["observed_count"].isna().any() or data["observed_count"].lt(0).any():
                raise SchemaError("Benford counts must be present and nonnegative.")

            size = int(data["observed_count"].sum())
            if size == 0:
                continue
            for column in ("observed_rate", "expected_rate"):
                if (not data[column].between(0, 1).all()
                        or abs(data[column].sum() - 1) > 1e-8):
                    raise SchemaError("Benford rates must be valid proportions summing to one.")
            groups.append((population, data, size))

        if not groups:
            raise SchemaError("No Benford observations are available.")

        def draw(ax):
            digits = list(range(1, 10))
            width = 0.8 / len(groups)
            for index, (population, data, size) in enumerate(groups):
                label, color = labels[population]
                offset = (index - (len(groups) - 1) / 2) * width
                count_label = f"{size:,}".replace(",", ".")
                ax.bar(
                    [digit + offset for digit in digits],
                    data["observed_rate"].astype(float) * 100,
                    width=width, color=color, label=f"{label} ({count_label} quan sát)",
                )
            ax.plot(
                digits, groups[0][1]["expected_rate"].astype(float) * 100,
                color="#DC2626", marker="o", label="Kỳ vọng Benford",
            )
            ax.set_xticks(digits)

        return self._save(
            draw, "03_benford_comparison.png",
            "Chữ số đầu tiên: thực tế và kỳ vọng Benford",
            "Tần suất (%)",
            xlabel="Chữ số có nghĩa đầu tiên",
            note="Các tập có thể chồng lấn; độ lệch Benford chưa xác nhận gian lận.",
        )

    def amount_histogram(self, data):
        if data.empty or "line_amount" not in data:
            raise SchemaError("No positive ledger line amounts are available.")

        if not all(is_finite_amount(v) and v > 0
                   for v in data["line_amount"]):
            raise SchemaError("Histogram amounts require positive finite Decimals.")

        values = [
            float(v / Decimal("1000000000"))
            for v in data["line_amount"]
        ]
        first = floor(log10(min(values)))
        last = floor(log10(max(values))) + 1
        edges = [10.0 ** power for power in range(first, last + 1)]

        def draw(ax):
            counts, _, _ = ax.hist(
                values, bins=edges,
                color="#2563EB", edgecolor="white",
                label="Quy mô dòng = max(|Nợ|, |Có|)",
            )
            if int(sum(counts)) != len(values):
                raise SchemaError(
                    "Histogram bin counts do not cover every selected line."
                )

            ax.set_xscale("log")
            ax.set_xlim(edges[0], edges[-1])
            ax.text(
                0.98, 0.82,
                f"{len(values):,} dòng".replace(",", "."),
                ha="right", transform=ax.transAxes,
            )

        return self._save(
            draw, "04_amount_histogram.png",
            "Phân phối quy mô dòng hạch toán",
            "Số dòng hạch toán",
            xlabel="Quy mô dòng (tỷ đồng VND, thang log cơ số 10)",
        )

    def account_boxplot(self, data):
        if data.empty or not {"account_group", "line_amount"}.issubset(data.columns):
            raise SchemaError("Boxplot requires grouped positive line amounts.")

        groups = list(data.groupby("account_group", sort=True))
        scale = Decimal("1000000000")
        values = [
            [float(v / scale) for v in records["line_amount"]]
            for _, records in groups
        ]
        labels = [
            (f"Loại {group}" if group != "unclassified" else "Chưa phân loại")
            + "\n" + f"{len(records):,} dòng".replace(",", ".")
            for group, records in groups
        ]

        def draw(ax):
            artists = ax.boxplot(
                values, tick_labels=labels, whis=1.5, patch_artist=True,
                boxprops={"facecolor": "#DBEAFE", "edgecolor": "#2563EB"},
                medianprops={"color": "#DC2626", "linewidth": 1.6},
                flierprops={
                    "marker": ".", "markersize": 2,
                    "markeredgecolor": "#64748B", "alpha": 0.4,
                },
            )
            artists["boxes"][0].set_label("Hộp: từ Q1 đến Q3")
            artists["medians"][0].set_label("Trung vị")
            artists["fliers"][0].set_label("Điểm ngoài râu")
            ax.set_yscale("log")

        return self._save(
            draw, "05_account_boxplot.png",
            "Phân phối quy mô dòng hạch toán theo loại tài khoản",
            "Quy mô dòng (tỷ đồng VND, thang log cơ số 10)",
            xlabel="Loại tài khoản theo chữ số đầu",
            note="Râu theo 1,5 × IQR trên số tiền gốc; điểm ngoài râu chưa xác nhận sai sót. Giữ kết chuyển.",
        )

    def balance_structure(self, components, totals):
        labels = {
            "current_assets": ("Tài sản ngắn hạn", "#2563EB"),
            "noncurrent_assets": ("Tài sản dài hạn", "#16A34A"),
            "unclassified_assets": ("Tài sản chưa phân loại", "#64748B"),
            "current_liabilities": ("Nợ ngắn hạn", "#D97706"),
            "noncurrent_liabilities": ("Nợ dài hạn", "#DC2626"),
            "unclassified_liabilities": ("Nợ chưa phân loại", "#94A3B8"),
            "equity": ("Vốn chủ sở hữu", "#7C3AED"),
        }
        scale = Decimal("1000000000")

        def draw(ax):
            positive = {"assets": 0.0, "funding": 0.0}
            negative = {"assets": 0.0, "funding": 0.0}
            positions = {"assets": 0, "funding": 1}

            for row in components.to_dict("records"):
                group, item = row["group"], row["item"]
                value = float(row["amount"] / scale)
                if value == 0:
                    continue

                base = positive if value > 0 else negative
                label, color = labels[item]
                ax.bar(
                    positions[group], value, bottom=base[group],
                    width=0.5, color=color, label=label,
                )
                ax.text(
                    positions[group], base[group] + value / 2,
                    f"{value:.2f}".replace(".", ","),
                    ha="center", va="center", color="white",
                )
                base[group] += value

            for group, position in positions.items():
                ax.annotate(
                    f"Tổng thuần: {totals[group] / scale:.2f}".replace(".", ","),
                    xy=(position, positive[group]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9,
                )

            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xticks([0, 1], ["Tài sản", "Nguồn vốn"])
            ax.set_xlim(-0.6, 2.4)

        return self._save(
            draw, "06_balance_structure.png",
            "Cơ cấu tài sản và nguồn vốn theo BCTC",
            "Tỷ đồng (VND)",
            xlabel="Số dư cuối kỳ",
            note=(f"Tài sản − nguồn vốn = {totals['balance_difference']} đơn vị nguồn."),
        )

    def ratio_dashboard(self, dashboard):
        columns = [
            "group", "indicator", "value",
            "good_reference", "risk_reference", "rating",
        ]
        colors = {
            "Tốt": "#DCFCE7",
            "Cần chú ý": "#FEF3C7",
            "Rủi ro": "#FEE2E2",
            "Chưa tính": "#E2E8F0",
            "Cần xác minh": "#E2E8F0",
        }
        accrual_missing = dashboard.loc[
            dashboard["code"].eq("accrual_ratio"), "rating"
        ].eq("Chưa tính").any()
        cfo_note = (
            "Dồn tích chưa tính: CFO ước tính chưa đủ căn cứ."
            if accrual_missing else "Dồn tích dùng CFO ước tính theo quy tắc đã công bố."
        )

        def draw(ax):
            ax.figure.set_size_inches(13, 8)
            ax.axis("off")
            table = ax.table(
                cellText=dashboard[columns].values.tolist(),
                colLabels=[
                    "Nhóm", "Chỉ số", "Kết quả",
                    "Ngưỡng Tốt", "Ngưỡng Rủi ro", "Đánh giá",
                ],
                colWidths=[0.13, 0.27, 0.14, 0.15, 0.15, 0.16],
                cellLoc="left", bbox=(0, 0.02, 1, 0.94),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor("#CBD5E1")
                if row == 0:
                    cell.set_facecolor("#1E3A8A")
                    cell.set_text_props(color="white", weight="bold")
                elif column == 5:
                    cell.set_facecolor(
                        colors[dashboard.iloc[row - 1]["rating"]]
                    )

        return self._save(
            draw, "07_ratio_dashboard.png",
            "Bảng các nhóm chỉ số tài chính",
            "", xlabel="",
        )

    def acceptance_risk(self, groups, thresholds, total_score):
        """Show weighted contributions against the reported decision thresholds."""
        labels = {
            "data_quality": ("Chất lượng dữ liệu", "#2563EB"),
            "journal_entries": ("Kiểm tra bút toán", "#7C3AED"),
            "financial_health": ("Sức khỏe tài chính", "#0891B2"),
            "reporting_risk": ("Rủi ro báo cáo", "#D97706"),
            "qualitative": ("Định tính", "#64748B"),
        }
        data = prepare_risk_groups(groups, total_score)
        total = normalize_amount(total_score)[0]

        conditional = normalize_amount(thresholds.get("conditional"))[0]
        reject = normalize_amount(thresholds.get("reject"))[0]
        if conditional is None or reject is None or not 0 < conditional < reject <= 100:
            raise SchemaError("Require 0 < conditional < reject <= 100.")

        def draw(ax):
            bounds = (0, float(conditional), float(reject), 100)
            zones = ("Chấp nhận", "Chấp nhận có điều kiện", "Từ chối")

            for left, right, label, color in zip(
                bounds[:-1], bounds[1:], zones,
                ("#16A34A", "#EAB308", "#DC2626"),
            ):
                ax.axvspan(left, right, color=color, alpha=0.12)
                ax.text(
                    (left + right) / 2, -0.43, label,
                    ha="center", fontsize=9,
                )

            left = 0.0
            for group, (label, color) in labels.items():
                row = data.loc[group]
                width = float(row["contribution"])
                ax.barh(
                    0, width, left=left, height=0.35, color=color,
                    label=f"{label} ({row['weight']:.0%})",
                )
                if width > 0:
                    ax.text(
                        left + width / 2, 0,
                        f"{width:.2f}".replace(".", ","),
                        ha="center", va="center", color="white", fontsize=9,
                    )
                left += width

            ax.vlines(
                float(total), -0.2, 0.45,
                color="black", linestyles="--",
            )
            ax.text(
                float(total), 0.5,
                f"Tổng: {total:.2f}".replace(".", ","),
                ha="center", weight="bold",
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.6, 1.8)
            ax.set_yticks([])
            ax.set_xticks(
                sorted(set([0, float(conditional), float(reject), 100]))
            )

        return self._save(
            draw, "08_acceptance_risk.png",
            "Điểm rủi ro và vùng quyết định", "",
            xlabel="Điểm rủi ro / 100 — điểm cao hơn thể hiện rủi ro cao hơn",
        )

    def weekday_heatmap(self, summary, coverage):
        """Show posting weekdays only; do not infer entry times or independent vouchers."""
        counts = summary["line_count"].tolist()
        total = coverage["dated_records"]
        if summary["weekday"].tolist() != list(range(7)) or total <= 0 or sum(counts) != total:
            raise SchemaError("Weekday counts must cover seven days and reconcile to dated records.")
        labels = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật")
        total_label = f"{total:,}".replace(",", ".")
        unavailable_label = f"{coverage['unavailable_date_records']:,}".replace(",", ".")

        def draw(ax):
            ax.figure.set_size_inches(11, 3.5)
            image = ax.imshow([counts], cmap="Blues", aspect="auto", vmin=0,
                              vmax=max(counts), interpolation="nearest")
            ax.set_xticks(range(7), labels)
            ax.set_yticks([])
            for index, count in enumerate(counts):
                text = f"{int(count):,}".replace(",", ".") + f"\n{count / total:.1%}"
                ax.text(index, 0, text, ha="center", va="center", fontsize=11,
                        color="white" if count > max(counts) * 0.55 else "#0F172A")
            ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Số dòng hạch toán")

        return self._save(
            draw, "09_posting_weekday_heatmap.png",
            "Số dòng hạch toán theo thứ của ngày hạch toán", "",
            xlabel="Thứ trong tuần",
            note=(f"Có ngày hợp lệ: {total_label}; thiếu/ngày không hợp lệ: {unavailable_label}.\n"
                  "Đếm dòng chuẩn hóa, giữ kết chuyển."),
        )

    def interactive_dashboard(
        self, monthly, je_summary, ratios=None, acceptance=None, total_assets=None,
        chart_data=None,
    ):
        """Build an offline dashboard from prepared report tables."""
        from html import escape
        import plotly.graph_objects as go

        data = require_chart_values(
            monthly, ("net_revenue", "cost_of_sales", "gross_profit", "gross_margin"),
        )
        required = {"test_code", "finding_count", "HIGH", "MEDIUM", "LOW"}
        if not required.issubset(je_summary.columns):
            raise SchemaError("Dashboard JE columns are missing.")
        je = je_summary.set_index("test_code").sort_index().copy()
        if je.index.has_duplicates or set(je.index) != {f"JE{i:02d}" for i in range(1, 11)}:
            raise SchemaError("Provide exactly one result for each of JE01-JE10.")
        for column in ("finding_count", "HIGH", "MEDIUM", "LOW"):
            je[column] = je[column].map(lambda v: normalize_amount(v)[0])
            if not all(v is not None and v >= 0 and v == int(v) for v in je[column]):
                raise SchemaError("Finding counts must be nonnegative integers.")
        if not (je[["HIGH", "MEDIUM", "LOW"]].sum(axis=1) == je["finding_count"]).all():
            raise SchemaError("Risk counts do not match finding counts.")

        scale = Decimal("1000000000")
        figures = [go.Figure() for _ in range(4)]
        for column, label, color in (
            ("net_revenue", "Doanh thu", "#2563EB"),
            ("cost_of_sales", "Giá vốn", "#CBD5E1"),
            ("gross_profit", "Lợi nhuận gộp", "#059669"),
        ):
            values = [float(v / scale) for v in data[column]]
            options = dict(x=data["month"].tolist(), y=values, name=label)
            options["hovertemplate"] = "%{x}: %{y:.3f} tỷ đồng<extra>%{fullData.name}</extra>"
            trace = (
                go.Scatter(**options, mode="lines+markers", line={"color": color})
                if column == "gross_profit" else go.Bar(**options, marker_color=color)
            )
            figures[0].add_trace(trace)
        figures[0].update_layout(barmode="group", yaxis_title="Tỷ đồng (VND)")
        figures[1].add_trace(go.Scatter(
            x=data["month"].tolist(), y=[float(v * 100) for v in data["gross_margin"]],
            mode="lines+markers", line={"color": "#7C3AED", "width": 3},
            fill="tozeroy", fillcolor="rgba(124,58,237,0.06)",
            hovertemplate="%{x}: %{y:.2f}%<extra>Biên lợi nhuận gộp</extra>",
        ))
        figures[1].update_layout(showlegend=False, yaxis_title="Biên lợi nhuận gộp (%)")
        if chart_data is not None:
            baseline, trend = chart_data["baseline"], chart_data["trend"]
            if baseline["baseline_status"] in ("calculated", "constant_series"):
                figures[1].add_hrect(
                    y0=float(baseline["lower_bound"] * 100),
                    y1=float(baseline["upper_bound"] * 100),
                    fillcolor="#CBD5E1", opacity=0.4, line_width=0, layer="below",
                )
                figures[1].add_hline(y=float(baseline["mean"] * 100), line_dash="dash")
                flagged = trend.loc[trend["outside_band"].eq(True)]
                figures[1].add_trace(go.Scatter(
                    x=flagged["month"].tolist(), y=[float(v * 100) for v in flagged["value"]],
                    mode="markers", marker={"color": "#DC2626", "size": 11},
                    hovertemplate="%{x}: %{y:.2f}%<extra>Tháng cần xem xét</extra>",
                ))
        for column, label, color in (
            ("HIGH", "Cao", "#DC2626"),
            ("MEDIUM", "Trung bình", "#D97706"),
            ("LOW", "Thấp", "#16A34A"),
        ):
            figures[2].add_trace(go.Bar(
                x=[int(v) for v in je[column]], y=je.index.tolist(),
                orientation="h", name=label, marker_color=color,
                hovertemplate="%{y}: %{x:,} phát hiện<extra>Ưu tiên %{fullData.name}</extra>",
            ))
        figures[2].update_layout(
            barmode="stack", xaxis_title="Số phát hiện", yaxis={"autorange": "reversed"},
            updatemenus=[dict(x=1, y=1.14, xanchor="right", buttons=[
                dict(label="Đủ 3 mức", method="restyle", args=["visible", [True, True, True]]),
                dict(label="Chỉ ưu tiên cao", method="restyle", args=["visible", [True, False, False]]),
            ])],
        )

        total, decision = None, "Chưa có đánh giá"
        if acceptance is not None:
            summary = dict(acceptance["Summary"].itertuples(index=False, name=None))
            groups = prepare_risk_groups(
                acceptance["Group_Scores"], summary["weighted_score"],
            ).reset_index()
            total = normalize_amount(summary["weighted_score"])[0]
            names = {
                "data_quality": "Chất lượng dữ liệu", "journal_entries": "Bút toán",
                "financial_health": "Tài chính", "reporting_risk": "Rủi ro báo cáo",
                "qualitative": "Định tính",
            }
            for row, color in zip(groups.to_dict("records"),
                                  ("#2563EB", "#7C3AED", "#0891B2", "#D97706", "#64748B")):
                figures[3].add_trace(go.Bar(
                    x=[float(row["contribution"])], y=["Điểm rủi ro"], orientation="h",
                    name=names[row["group"]], text=[f"{row['contribution']:.2f}"],
                    textposition="inside", marker_color=color,
                    hovertemplate="%{x:.2f} điểm đóng góp<extra>%{fullData.name}</extra>",
                ))
            if chart_data is not None:
                thresholds = chart_data["thresholds"]
                bounds = [0, float(thresholds["conditional"]), float(thresholds["reject"]), 100]
                for left, right, color, label in zip(
                    bounds[:-1], bounds[1:], ("#16A34A", "#EAB308", "#DC2626"),
                    ("Chấp nhận", "Có điều kiện", "Từ chối"),
                ):
                    figures[3].add_vrect(x0=left, x1=right, fillcolor=color,
                                        opacity=0.12, line_width=0, layer="below")
                    figures[3].add_annotation(x=(left + right) / 2, y=1.12,
                                              yref="paper", text=label, showarrow=False)
            figures[3].add_vline(x=float(total), line_dash="dash")
            decisions = {"ACCEPT": "Chấp nhận", "CONDITIONAL": "Chấp nhận có điều kiện", "REJECT": "Từ chối"}
            decision = decisions.get(summary["decision_after_knockouts"], "Chưa có đánh giá")
        figures[3].update_layout(
            barmode="stack", xaxis={"range": [0, 100], "title": "Điểm nhóm × trọng số; tổng / 100"},
            yaxis={"showticklabels": False},
        )

        for fig in figures[:2]:
            fig.update_xaxes(
                type="category", tickvals=data["month"].tolist(),
                ticktext=[f"{m[-2:]}/{m[:4]}" for m in data["month"]], tickangle=-30,
            )
        figures[2].update_xaxes(tickformat=",d")

        plots = [
            ("01", figures[0], "Kết quả kinh doanh theo tháng", "Đã loại kết chuyển nhận diện; đơn vị tỷ đồng VND."),
            ("02", figures[1], "Biên lợi nhuận gộp theo tháng", "Vùng tô: trung bình ± 2 độ lệch chuẩn; điểm đỏ cần xem xét, chưa xác nhận sai sót."),
            ("JE", figures[2], "Phát hiện bút toán JE01–JE10", "Giữ thêm bảng JE tương tác; phát hiện có thể chồng lấn và khác đơn vị đếm."),
            ("08", figures[3], "Điểm rủi ro và vùng quyết định", "Vùng theo ngưỡng trước chặn; ô tổng quan thể hiện kết luận sau chặn. Còn hành động cần xác minh."),
        ]
        if chart_data is not None:
            plots.extend(supplementary_dashboard_figures(chart_data))
        panels = {}
        for index, (code, fig, title, note) in enumerate(plots):
            if not fig.data:
                fig.add_annotation(text="Chưa đủ dữ liệu để hiển thị", showarrow=False)
            fig.update_layout(
                template="plotly_white", height=400, font={"family": "Arial", "size": 11},
                margin={"l": 60, "r": 25, "t": 40, "b": 135},
                legend={"orientation": "h", "y": -0.32, "x": 0},
                hovermode="closest", bargap=0.25,
            )
            fig.update_xaxes(zeroline=False)
            fig.update_yaxes(zeroline=False)
            plot = fig.to_html(
                full_html=False, include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True, "scrollZoom": False},
            )
            panels[code] = (f'<section class="panel" data-chart="{code}"><h2>{code} · {title}</h2>'
                            f'{plot}<p class="chart-note">{escape(note)}</p></section>')

        ratio_html = "<p>Chưa có bảng tỷ số.</p>"
        if ratios is not None:
            columns = ["group", "indicator", "value", "good_reference", "risk_reference", "rating"]
            headers = ["Nhóm", "Chỉ số", "Kết quả", "Ngưỡng Tốt", "Ngưỡng Rủi ro", "Đánh giá"]
            colors = {"Tốt": "good", "Cần chú ý": "review", "Rủi ro": "risk"}
            rows = []
            for row in ratios[columns].to_dict("records"):
                cells = "".join(f"<td>{escape(str(row[c]))}</td>" for c in columns[:-1])
                rating = escape(str(row["rating"]))
                cells += f'<td><span class="badge {colors.get(row["rating"], "")}">{rating}</span></td>'
                rows.append(f"<tr>{cells}</tr>")
            ratio_html = '<table><thead><tr>' + "".join(f"<th>{h}</th>" for h in headers)
            ratio_html += "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        panels["07"] = ('<section class="panel table-wrap" data-chart="07">'
                        '<h2>07 · Chỉ số tài chính và ngưỡng tham chiếu</h2>' + ratio_html + '</section>')
        tabs = (
            ("performance", "Kết quả kinh doanh", ("01", "02")),
            ("ledger", "Sổ và bút toán", ("03", "04", "05", "09", "JE")),
            ("finance", "Tài chính và rủi ro", ("06", "08", "07")),
        )
        tab_buttons, tab_content = [], []
        for index, (key, title, codes) in enumerate(tabs):
            tab_buttons.append(
                f'<button type="button" role="tab" id="tab-{key}" aria-controls="{key}" '
                f'aria-selected="{str(index == 0).lower()}" tabindex="{0 if index == 0 else -1}">{title}</button>'
            )
            tab_content.append(
                f'<div id="{key}" role="tabpanel" aria-labelledby="tab-{key}"'
                + (' hidden' if index else '') + '><div class="grid">'
                + ''.join(panels[code] for code in codes if code in panels) + '</div></div>'
            )

        def money(value):
            value = normalize_amount(value)[0]
            return "—" if value is None else f"{value / scale:.2f}".replace(".", ",")

        cards = "".join(
            f'<article class="card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>'
            for label, value, detail in (
                ("DOANH THU THUẦN", money(sum(data["net_revenue"])), "Tỷ đồng VND · cả kỳ"),
                ("LỢI NHUẬN GỘP", money(sum(data["gross_profit"])), "Tỷ đồng VND · cả kỳ"),
                ("TỔNG TÀI SẢN", money(total_assets), "Tỷ đồng VND · cuối kỳ"),
                ("ĐIỂM RỦI RO", "—" if total is None else f"{total:.2f} / 100".replace(".", ","), escape(decision)),
            )
        )
        html = """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audit.py — Dashboard</title><style>
*{box-sizing:border-box}body{margin:0;background:#F1F5F9;color:#172554;font:14px Arial,sans-serif}
main{max-width:1540px;margin:auto;padding:26px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}
h1{font-size:25px;margin:5px 0}h2{font-size:16px;margin:0;padding:18px 20px 0}p,small{color:#64748B;line-height:1.6}
.tag{padding:7px 12px;border-radius:20px;background:#FEF3C7;color:#92400E;font-size:12px;white-space:nowrap}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}.card,.panel{background:white;border:1px solid #E2E8F0;border-radius:12px;overflow:hidden}
.card{padding:18px 20px}.card span{display:block;font-size:11px;letter-spacing:1px;color:#64748B}.card strong{display:block;font-size:28px;margin:10px 0}.card small{font-size:12px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.table-wrap{overflow:auto;margin:20px 0;padding-bottom:12px}
.grid>.table-wrap{grid-column:1/-1;margin:0}.panel{min-width:0}.chart-note{padding:0 20px 14px;margin:0;font-size:12px;min-height:48px}
[hidden]{display:none!important}.tabs{display:flex;gap:8px;margin:0 0 20px;flex-wrap:wrap}.tabs button{border:1px solid #CBD5E1;background:white;color:#334155;border-radius:8px;padding:11px 18px;font:inherit;cursor:pointer}.tabs button[aria-selected="true"]{background:#1D4ED8;color:white;border-color:#1D4ED8}.tabs button:focus-visible{outline:3px solid #93C5FD;outline-offset:2px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}th,td{text-align:left;padding:11px 18px;border-bottom:1px solid #E2E8F0;white-space:nowrap}th{background:#F8FAFC;color:#64748B}tr:hover{background:#F8FAFC}.badge{display:inline-block;border-radius:12px;padding:4px 9px;background:#E2E8F0}.good{background:#DCFCE7;color:#166534}.review{background:#FEF3C7;color:#92400E}.risk{background:#FEE2E2;color:#991B1B}
footer{font-size:12px;color:#64748B;line-height:1.7}.hint{margin:0 0 18px;font-size:12px}
@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}main{padding:16px}header{gap:12px}h1{font-size:20px}}
</style><main><header><div><small>AUDIT.PY / TỔNG QUAN</small>
<h1>Phân tích kế toán – kiểm toán</h1><small>__PERIOD__ · Đơn vị tiền hiển thị: tỷ đồng VND</small></div>
<span class="tag">Đánh giá sơ bộ</span></header>
<div class="cards">__CARDS__</div>
<p class="hint">Đủ 9 nội dung tương ứng hình PNG và thêm tổng hợp JE · Rê chuột xem số · Bấm chú giải ẩn/hiện · Kéo phóng to · Bấm đúp đặt lại</p>
<nav class="tabs" role="tablist" aria-label="Nội dung phân tích">__TABS__</nav>
__PANELS__
<footer>Phát hiện JE có thể chồng lấn và khác đơn vị đếm; không cộng thành số sai phạm hoặc tổng tổn thất.<br>
* Giả định kỳ hạn; Altman cần xác nhận phù hợp doanh nghiệp. Dồn tích sử dụng CFO ước tính theo quy tắc công bố.<br>
Điểm rủi ro không phải xác suất gian lận. Cần đọc quy tắc chặn và hành động còn mở trong báo cáo 06.<br>
Ngưỡng theo đề/chính sách dự án; giả định tiền nguồn là đồng VND. Số chi tiết và giới hạn: báo cáo 02, 03, 05, 06.</footer></main>
<script>
const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
function activateTab(button) {
  tabs.forEach(tab => {
    const selected = tab === button;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  });
  requestAnimationFrame(() => {
    document.querySelectorAll('[role="tabpanel"]:not([hidden]) .plotly-graph-div')
      .forEach(plot => Plotly.Plots.resize(plot));
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {event.preventDefault(); tabs[next].focus(); activateTab(tabs[next]);}
  });
});
</script></html>"""
        for token, value in {
            "__PERIOD__": escape(f'{data["month"].iloc[0]} → {data["month"].iloc[-1]}'),
            "__CARDS__": cards, "__PANELS__": "".join(tab_content), "__TABS__": "".join(tab_buttons),
        }.items():
            html = html.replace(token, value)
        path = self.output_dir.parent / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        return path

def build_chart_outputs(statement_tables, je_tables, financial_tables,
                        acceptance_tables, output_dir):
    """Export charts and a dashboard; report unavailable inputs."""
    import json

    builder = ChartBuilder(Path(output_dir) / "charts")
    paths, skipped = [], []

    def add(function, *args, ready=True, **kwargs):
        if not ready:
            skipped.append(function.__name__)
            return
        path = function(*args, **kwargs)
        if not isinstance(path, Path) or not path.is_file():
            raise RuntimeError("Chart rendering did not create its output file.")
        paths.append(path)

    def complete(table, columns):
        return not table.empty and all(
            is_finite_amount(value) for column in columns for value in table[column]
        )

    monthly = financial_tables["Monthly"]
    money_ready = complete(monthly, ("net_revenue", "cost_of_sales", "gross_profit"))
    margin_ready = complete(monthly, ("gross_margin",))
    trend = financial_tables["Monthly_Trend"]
    trend = trend.loc[trend["metric"].eq("gross_margin")].copy()
    baseline = financial_tables["Monthly_Baselines"].set_index("metric").loc["gross_margin"].to_dict()
    add(builder.monthly_performance, monthly, ready=money_ready)
    add(builder.monthly_gross_margin, trend, baseline,
        ready=complete(trend, ("value",)) and baseline["baseline_status"] in
        ("calculated", "constant_series"))

    frequency = je_tables["Benford_Frequency"]
    add(builder.benford_comparison, frequency,
        ready=not frequency.empty and frequency["observed_count"].sum() > 0)
    amounts, coverage = prepare_line_amounts(statement_tables["General_Ledger"])
    add(builder.amount_histogram, amounts, ready=not amounts.empty)
    grouped, _ = prepare_account_groups(amounts)
    add(builder.account_boxplot, grouped, ready=not grouped.empty)

    balance_sheet = statement_tables["Balance_Sheet"]
    if balance_sheet["calculation_status"].eq("calculated").all():
        components, totals = prepare_balance_structure(balance_sheet)
        add(builder.balance_structure, components, totals)
    else:
        skipped.append("balance_structure")
        totals = {"assets": None}
        components = pd.DataFrame(columns=["group", "item", "amount"])

    ratios = prepare_ratio_dashboard(financial_tables["Ratios"], financial_tables["Benchmarks"])
    add(builder.ratio_dashboard, ratios)
    summary = dict(acceptance_tables["Summary"].itertuples(index=False, name=None))
    policy = summary["policy_configuration"]
    if isinstance(policy, str):
        policy = json.loads(policy)
    total = normalize_amount(summary["weighted_score"])[0]
    score_ready = total is not None
    add(builder.acceptance_risk, acceptance_tables["Group_Scores"],
        policy["decision_thresholds"], total, ready=score_ready)
    weekday_counts, weekday_coverage = prepare_weekday_counts(statement_tables["General_Ledger"])
    add(builder.weekday_heatmap, weekday_counts, weekday_coverage,
        ready=weekday_coverage["dated_records"] > 0)
    add(builder.interactive_dashboard, monthly, je_tables["Summary"],
        ratios=ratios, acceptance=acceptance_tables if score_ready else None,
        total_assets=totals["assets"], ready=money_ready and margin_ready,
        chart_data=dict(frequency=frequency, amounts=amounts, grouped=grouped,
                        components=components, totals=totals, trend=trend, baseline=baseline,
                        weekday=weekday_counts, weekday_coverage=weekday_coverage,
                        thresholds=policy["decision_thresholds"]))
    return {"paths": paths, "skipped": skipped, "line_amount_coverage": coverage,
            "weekday_coverage": weekday_coverage}
