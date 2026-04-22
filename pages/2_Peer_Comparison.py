import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Peer Comparison", layout="wide")

# =========================================================
# Shared ticker
# =========================================================
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

# =========================================================
# Helper
# =========================================================
def find_file(filename):
    possible_paths = [
        Path(filename),
        Path(".") / filename,
        Path("data") / filename,
        Path("models") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path(__file__).parent.parent / "models" / filename,
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

def compare_direction(metric_name, company_value, peer_value):
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
        return "No comparison available"

    better_when_higher = higher_is_better.get(metric_name, True)

    if company_value > peer_value:
        return "Better than peers" if better_when_higher else "More expensive / riskier than peers"
    elif company_value < peer_value:
        return "Worse than peers" if better_when_higher else "Cheaper / more conservative than peers"
    else:
        return "In line with peers"

def build_peer_comparison_table(company_latest, peer_latest):
    compare_metrics = [
        "roa",
        "operating_margin",
        "revenue_growth",
        "current_ratio",
        "debt_to_assets",
        "price_to_sales",
        "price_to_book"
    ]

    rows = []
    for metric in compare_metrics:
        if metric in company_latest.index and metric in peer_latest.index:
            company_val = company_latest[metric]
            peer_val = peer_latest[metric]
            diff = company_val - peer_val
            rows.append({
                "Metric": metric,
                "Company": company_val,
                "Peer Average": peer_val,
                "Difference": diff,
                "Interpretation": compare_direction(metric, company_val, peer_val)
            })

    return pd.DataFrame(rows)

st.title("Peer Comparison")
st.markdown(f"### Current ticker: `{ticker}`")

try:
    df = load_input_data()
except Exception as e:
    st.error("ticker_history_input.csv is missing.")
    st.exception(e)
    st.stop()

company_data = df[df["ticker"].astype(str).str.upper() == ticker].copy()

if company_data.empty:
    st.warning("Ticker not found in dataset.")
    st.stop()

latest_by_ticker = df.sort_values("year").groupby("ticker", as_index=False).tail(1)

company_latest = latest_by_ticker[latest_by_ticker["ticker"].astype(str).str.upper() == ticker]
peer_latest = latest_by_ticker[latest_by_ticker["ticker"].astype(str).str.upper() != ticker]

if company_latest.empty:
    st.warning("No peer comparison data found for this ticker.")
    st.stop()

company_latest_row = company_latest.iloc[0]
peer_avg = peer_latest.mean(numeric_only=True)

peer_compare_df = build_peer_comparison_table(company_latest_row, peer_avg)

st.subheader("Latest-year company vs peer average")
st.dataframe(peer_compare_df, use_container_width=True)

chart_metrics = [
    "roa",
    "operating_margin",
    "revenue_growth",
    "current_ratio",
    "debt_to_assets",
    "price_to_sales",
    "price_to_book"
]

chart_rows = []
for metric in chart_metrics:
    if metric in company_latest_row.index and metric in peer_avg.index:
        chart_rows.append({
            "Metric": metric,
            "Company": company_latest_row[metric],
            "Peer Average": peer_avg[metric]
        })

chart_df = pd.DataFrame(chart_rows)

if not chart_df.empty:
    st.subheader("Peer comparison chart")

    chart_long = chart_df.melt(
        id_vars="Metric",
        value_vars=["Company", "Peer Average"],
        var_name="Group",
        value_name="Value"
    )

    grouped_bar = (
        alt.Chart(chart_long)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            xOffset=alt.XOffset("Group:N"),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Group:N", title="Group"),
            tooltip=["Metric", "Group", alt.Tooltip("Value:Q", format=".4f")]
        )
        .properties(height=400)
    )

    st.altair_chart(grouped_bar, use_container_width=True)