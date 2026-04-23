import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Peer Comparison", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

# COMPANY NAMES
TICKER_NAME_MAP = {
    "LLY": "Eli Lilly and Company",
    "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "AMGN": "Amgen Inc.",
    "ABT": "Abbott Laboratories",
    "TMO": "Thermo Fisher Scientific Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "CVS": "CVS Health Corporation",
    "BMY": "Bristol-Myers Squibb Company",
    "MDT": "Medtronic plc",
    "CI": "Cigna Group",
    "ZTS": "Zoetis Inc.",
    "SYK": "Stryker Corporation",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "HCA": "HCA Healthcare, Inc.",
    "DHR": "Danaher Corporation",
    "HUM": "Humana Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "MRNA": "Moderna, Inc.",
    "PFE": "Pfizer Inc.",
    "BIIB": "Biogen Inc.",
    "ILMN": "Illumina, Inc.",
    "EW": "Edwards Lifesciences Corporation",
    "A": "Agilent Technologies, Inc.",
    "DXCM": "DexCom, Inc.",
    "IDXX": "IDEXX Laboratories, Inc.",
    "ALGN": "Align Technology, Inc.",
}

# SIDEBAR
st.sidebar.markdown(f"### Current ticker: `{ticker}`")


# FILE HELPERS
def find_file(filename):
    possible_paths = [
        Path(filename),
        Path(".") / filename,
        Path("data") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


@st.cache_data
def load_input_data():
    csv_path = find_file("ticker_history_input.csv")
    if csv_path is None:
        raise FileNotFoundError("ticker_history_input.csv not found")
    return pd.read_csv(csv_path)


# HELPERS
def company_display_name(ticker_value):
    company_name = TICKER_NAME_MAP.get(str(ticker_value).upper(), "")
    if company_name:
        return f"{ticker_value} ({company_name})"
    return str(ticker_value)


def format_metric_label(metric_name):
    mapping = {
        "roa": "ROA",
        "operating_margin": "Operating Margin",
        "revenue_growth": "Revenue Growth",
        "current_ratio": "Current Ratio",
        "debt_to_assets": "Debt to Assets",
        "price_to_sales": "Price to Sales",
        "price_to_book": "Price to Book",
    }
    return mapping.get(metric_name, metric_name.replace("_", " ").title())


def style_difference(metric_name, company_value, peer_value):
    higher_is_better = {
        "roa": True,
        "operating_margin": True,
        "revenue_growth": True,
        "current_ratio": True,
        "price_to_sales": False,
        "price_to_book": False,
        "debt_to_assets": False,
    }

    if pd.isna(company_value) or pd.isna(peer_value):
        return "neutral"

    better_when_higher = higher_is_better.get(metric_name, True)

    if company_value == peer_value:
        return "neutral"

    if better_when_higher:
        return "favorable" if company_value > peer_value else "unfavorable"
    return "favorable" if company_value < peer_value else "unfavorable"


def build_peer_comparison_table(company_latest, peer_latest):
    compare_metrics = [
        "roa",
        "operating_margin",
        "revenue_growth",
        "current_ratio",
        "debt_to_assets",
        "price_to_sales",
        "price_to_book",
    ]

    rows = []
    for metric in compare_metrics:
        if metric in company_latest.index and metric in peer_latest.index:
            company_val = company_latest[metric]
            peer_val = peer_latest[metric]
            rows.append({
                "metric_key": metric,
                "Metric": format_metric_label(metric),
                "Company": company_val,
                "Peer Average": peer_val,
                "Difference": company_val - peer_val,
                "Signal": style_difference(metric, company_val, peer_val),
            })

    return pd.DataFrame(rows)


def style_peer_table(df):
    def highlight_difference(row):
        signal = row["Signal"]
        styles = ["", "", "", "", ""]

        if signal == "favorable":
            styles[3] = "color: #2ca02c; font-weight: bold;"
        elif signal == "unfavorable":
            styles[3] = "color: #d62728; font-weight: bold;"
        else:
            styles[3] = "font-weight: bold;"

        return styles

    styled_df = df[["Metric", "Company", "Peer Average", "Difference", "Signal"]].copy()

    return styled_df.style.apply(highlight_difference, axis=1).format({
        "Company": "{:,.4f}",
        "Peer Average": "{:,.4f}",
        "Difference": "{:+,.4f}",
    })


def build_chart_df(company_latest, peer_avg, selected_metrics):
    rows = []
    for metric in selected_metrics:
        if metric in company_latest.index and metric in peer_avg.index:
            rows.append({
                "Metric": format_metric_label(metric),
                "metric_key": metric,
                "Company": company_latest[metric],
                "Peer Average": peer_avg[metric],
            })

    if not rows:
        return pd.DataFrame()

    chart_df = pd.DataFrame(rows)
    chart_long = chart_df.melt(
        id_vars=["Metric", "metric_key"],
        value_vars=["Company", "Peer Average"],
        var_name="Group",
        value_name="Value"
    )
    return chart_long


def build_takeaways(company_latest, peer_avg):
    takeaways = []

    roa = company_latest.get("roa", None)
    roa_peer = peer_avg.get("roa", None)

    margin = company_latest.get("operating_margin", None)
    margin_peer = peer_avg.get("operating_margin", None)

    growth = company_latest.get("revenue_growth", None)
    growth_peer = peer_avg.get("revenue_growth", None)

    current_ratio = company_latest.get("current_ratio", None)
    current_ratio_peer = peer_avg.get("current_ratio", None)

    debt = company_latest.get("debt_to_assets", None)
    debt_peer = peer_avg.get("debt_to_assets", None)

    ps = company_latest.get("price_to_sales", None)
    ps_peer = peer_avg.get("price_to_sales", None)

    pb = company_latest.get("price_to_book", None)
    pb_peer = peer_avg.get("price_to_book", None)

    if pd.notna(roa) and pd.notna(roa_peer) and pd.notna(margin) and pd.notna(margin_peer):
        if roa > roa_peer and margin > margin_peer:
            takeaways.append(
                "The company looks stronger than peers on core profitability measures, with both ROA and operating margin above peer averages. "
                "That suggests better operating efficiency and earnings generation relative to the comparison group."
            )
        elif roa < roa_peer and margin < margin_peer:
            takeaways.append(
                "The company looks weaker than peers on core profitability measures, with both ROA and operating margin below peer averages. "
                "That means the firm may need stronger growth or strategic advantages to justify premium valuation."
            )
        else:
            takeaways.append(
                "Profitability signals are mixed relative to peers. "
                "That suggests the company has some operating strengths, but not a uniformly superior profitability profile."
            )

    if pd.notna(current_ratio) and pd.notna(current_ratio_peer) and pd.notna(debt) and pd.notna(debt_peer):
        if current_ratio > current_ratio_peer and debt < debt_peer:
            takeaways.append(
                "The balance-sheet profile looks stronger than peers, with better liquidity and lower leverage. "
                "That can give the company more financial flexibility and resilience."
            )
        elif current_ratio < current_ratio_peer and debt > debt_peer:
            takeaways.append(
                "The balance-sheet profile looks weaker than peers, with lower liquidity and higher leverage. "
                "That may reduce flexibility if operating conditions become more challenging."
            )
        else:
            takeaways.append(
                "Balance-sheet signals are mixed. "
                "The company is not clearly more conservative or more aggressive than peers overall."
            )

    if pd.notna(ps) and pd.notna(ps_peer) and pd.notna(pb) and pd.notna(pb_peer):
        if ps < ps_peer and pb < pb_peer:
            takeaways.append(
                "The company appears cheaper than peers on both price-to-sales and price-to-book. "
                "That could indicate relative valuation support, although it may also reflect more cautious market expectations."
            )
        elif ps > ps_peer and pb > pb_peer:
            takeaways.append(
                "The company trades at richer valuation multiples than peers. "
                "That suggests investors may already be pricing in stronger quality, growth, or market expectations."
            )
        else:
            takeaways.append(
                "Valuation signals are mixed relative to peers. "
                "The stock does not look uniformly cheap or expensive across the main valuation multiples."
            )

    if pd.notna(growth) and pd.notna(growth_peer):
        if growth > growth_peer:
            takeaways.append(
                "Revenue growth is above the peer average, which helps support the company’s investment story and may justify stronger market interest."
            )
        elif growth < growth_peer:
            takeaways.append(
                "Revenue growth is below the peer average, which may make it harder for the company to sustain premium valuation unless other strengths offset it."
            )
        else:
            takeaways.append(
                "Revenue growth is roughly in line with peers, suggesting no major growth advantage or disadvantage at the moment."
            )

    return takeaways[:4]


# PAGE
st.title("Peer Comparison")
st.markdown(f"### Current ticker: `{company_display_name(ticker)}`")

try:
    df = load_input_data()
except Exception as e:
    st.error("ticker_history_input.csv is missing.")
    st.exception(e)
    st.stop()

company_data = df[df["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()

if company_data.empty:
    st.warning("Ticker not found in dataset.")
    st.stop()

latest_by_ticker = df.sort_values("year").groupby("ticker", as_index=False).tail(1)

company_latest = latest_by_ticker[latest_by_ticker["ticker"].astype(str).str.upper() == str(ticker).upper()]
peer_latest = latest_by_ticker[latest_by_ticker["ticker"].astype(str).str.upper() != str(ticker).upper()]

if company_latest.empty:
    st.warning("No peer comparison data found for this ticker.")
    st.stop()

company_latest_row = company_latest.iloc[0]
peer_avg = peer_latest.mean(numeric_only=True)

year_used = int(company_latest_row["year"]) if "year" in company_latest_row.index else None

peer_compare_df = build_peer_comparison_table(company_latest_row, peer_avg)

st.markdown("## 1) Selected Company")
if year_used is not None:
    st.write(
        f"This page benchmarks the selected company against peer averages using the **most recent available observation: {year_used}**."
    )
else:
    st.write(
        "This page benchmarks the selected company against peer averages using the most recent available observation."
    )

st.markdown("## 2) Peer Comparison Summary")
st.caption(
    "The table below compares the selected company’s latest-year values with peer averages."
)

st.dataframe(
    style_peer_table(peer_compare_df),
    use_container_width=True,
    hide_index=True,
)

st.markdown("## 3) Company vs Peer Visuals")
st.caption(
    "Choose which metrics to display below. The grouped bars show the selected company versus the peer average for each metric."
)

chart_metrics = [
    "roa",
    "operating_margin",
    "revenue_growth",
    "current_ratio",
    "debt_to_assets",
    "price_to_sales",
    "price_to_book",
]

default_metrics = [
    "roa",
    "operating_margin",
    "current_ratio",
    "price_to_sales",
    "price_to_book",
]

selected_metrics = st.multiselect(
    "Choose metrics to visualize",
    options=chart_metrics,
    default=default_metrics,
    format_func=format_metric_label,
)

chart_long = build_chart_df(company_latest_row, peer_avg, selected_metrics)

if chart_long.empty:
    st.info("Select at least one metric to display the comparison chart.")
else:
    grouped_bar = (
        alt.Chart(chart_long)
        .mark_bar(size=26)
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            xOffset=alt.XOffset("Group:N"),
            y=alt.Y("Value:Q", title="Metric Value"),
            color=alt.Color(
                "Group:N",
                scale=alt.Scale(
                    domain=["Company", "Peer Average"],
                    range=["#1f77b4", "#9ecae1"]
                ),
                legend=alt.Legend(title="Group")
            ),
            tooltip=[
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Group:N", title="Group"),
                alt.Tooltip("Value:Q", title="Value", format=".4f"),
            ]
        )
        .properties(height=430)
    )

    value_labels = (
        alt.Chart(chart_long)
        .mark_text(dy=-8, fontSize=11)
        .encode(
            x=alt.X("Metric:N"),
            xOffset=alt.XOffset("Group:N"),
            y=alt.Y("Value:Q"),
            text=alt.Text("Value:Q", format=".2f"),
            color=alt.value("white")
        )
    )

    st.altair_chart(grouped_bar + value_labels, use_container_width=True)

st.markdown("## 4) Key Takeaways")

takeaways = build_takeaways(company_latest_row, peer_avg)

if takeaways:
    for takeaway in takeaways:
        st.write(f"- {takeaway}")
else:
    st.write("Not enough peer-comparison signals were available to generate takeaways.")

# EXPLORE MORE
st.markdown("## Explore More")
st.markdown(
    """
- **Valuation Assessment:** Review the model’s latest-year valuation result and historical valuation context
- **Risk Analysis:** Examine beta, alpha, and R-squared for the selected company
- **Methodology:** See the project framing, data sources, and modeling context
"""
)
