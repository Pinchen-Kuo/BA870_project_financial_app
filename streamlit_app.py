import streamlit as st
import pandas as pd
import pickle
import altair as alt
from pathlib import Path

st.set_page_config(page_title="ValueLens", layout="wide")

# =========================================================
# Shared state
# =========================================================
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

# =========================================================
# Helpers
# =========================================================
def find_file(filename):
    possible_paths = [
        Path(filename),
        Path(".") / filename,
        Path("data") / filename,
        Path("models") / filename,
        Path(__file__).parent / filename,
        Path(__file__).parent / "data" / filename,
        Path(__file__).parent / "models" / filename,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

@st.cache_resource
def load_model():
    model_path = find_file("final_model.pkl")
    if model_path is None:
        raise FileNotFoundError("final_model.pkl not found")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_cols():
    feature_cols_path = find_file("feature_cols.pkl")
    if feature_cols_path is None:
        raise FileNotFoundError("feature_cols.pkl not found")
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_input_data():
    csv_path = find_file("ticker_history_input.csv")
    if csv_path is None:
        raise FileNotFoundError("ticker_history_input.csv not found")
    return pd.read_csv(csv_path)

@st.cache_data
def load_capm_data():
    csv_path = find_file("capm_risk_metrics.csv")
    if csv_path is None:
        raise FileNotFoundError("capm_risk_metrics.csv not found")
    return pd.read_csv(csv_path)

label_map = {
    0: "Overvalued",
    1: "Fairly valued",
    2: "Undervalued"
}

def summarize_company_result(pred_series):
    avg_class = pred_series.mean()
    if avg_class < 0.67:
        return "Overvalued"
    elif avg_class < 1.33:
        return "Fairly valued"
    return "Undervalued"

def get_majority_label(pred_series):
    majority_class = pred_series.mode().iloc[0]
    return label_map[int(majority_class)]

def safe_num(x, digits=3):
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"

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
            rows.append({
                "Metric": metric,
                "Company": company_val,
                "Peer Average": peer_val,
                "Difference": company_val - peer_val,
                "Interpretation": compare_direction(metric, company_val, peer_val)
            })

    return pd.DataFrame(rows)

# =========================================================
# Sidebar shared block
# =========================================================
def render_sidebar_header():
    st.sidebar.markdown(f"### Current ticker: `{st.session_state['ticker']}`")

# =========================================================
# Pages
# =========================================================
def home_page():
    st.title("Welcome to ValueLens!")
    st.markdown("""
Hey there! We know you’re passionate about the market, and we're here to level up your game! 🚀

Introducing **ValueLens — Stock Valuation Analyzer**.

This app evaluates healthcare stocks from three perspectives:

- **Valuation Assessment**
- **Peer Comparison**
- **Risk Analysis (CAPM)**

Use the sidebar to navigate across pages.

Happy investing! 📈💰
""")

    st.subheader("Enter ticker once here")
    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state["ticker"],
        key="home_ticker_input"
    ).upper().strip()

    if st.button("Set Ticker"):
        st.session_state["ticker"] = ticker_input
        st.success(f"Ticker set to: {ticker_input}")

    st.markdown(f"### Current selected ticker: `{st.session_state['ticker']}`")
    st.info("Now use the sidebar on the left to open other pages.")

def valuation_page():
    st.title("Valuation Assessment")
    ticker = st.session_state["ticker"]
    st.markdown(f"### Current ticker: `{ticker}`")

    try:
        model = load_model()
        feature_cols = load_feature_cols()
        df = load_input_data()
    except Exception as e:
        st.error("Required project files are missing.")
        st.exception(e)
        st.stop()

    company_data = df[df["ticker"].astype(str).str.upper() == ticker].copy()

    if company_data.empty:
        st.warning("Ticker not found in dataset.")
        st.stop()

    if "year" in company_data.columns:
        company_data = company_data.sort_values("year")

    try:
        X = company_data[feature_cols]
        pred_class = model.predict(X)
        pred_prob = model.predict_proba(X)
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    result_df = company_data[["ticker", "year"]].copy()
    result_df["predicted_class"] = pred_class
    result_df["predicted_label"] = result_df["predicted_class"].map(label_map)
    result_df["prob_overvalued"] = pred_prob[:, 0]
    result_df["prob_fairly_valued"] = pred_prob[:, 1]
    result_df["prob_undervalued"] = pred_prob[:, 2]

    final_label = summarize_company_result(result_df["predicted_class"])
    majority_label = get_majority_label(result_df["predicted_class"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Years Used", len(result_df))
    c2.metric("Final Valuation", final_label)
    c3.metric("Majority-Year Label", majority_label)

    st.subheader("Year-by-year model outputs")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("Prediction Distribution")
    st.bar_chart(result_df["predicted_label"].value_counts())

    with st.expander("See historical input data used for prediction"):
        st.dataframe(company_data, use_container_width=True)

def peer_page():
    st.title("Peer Comparison")
    ticker = st.session_state["ticker"]
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

def risk_page():
    st.title("Risk Analysis (CAPM)")
    ticker = st.session_state["ticker"]
    st.markdown(f"### Current ticker: `{ticker}`")

    try:
        capm_df = load_capm_data()
    except Exception as e:
        st.error("capm_risk_metrics.csv is missing.")
        st.exception(e)
        st.stop()

    capm_row = capm_df[capm_df["ticker"].astype(str).str.upper() == ticker]

    if capm_row.empty:
        st.warning("No CAPM risk metrics found for this ticker.")
        st.stop()

    capm_row = capm_row.iloc[0]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Alpha", safe_num(capm_row["alpha"], 4))
    r2.metric("Beta", safe_num(capm_row["beta"], 3))
    r3.metric("R-squared", safe_num(capm_row["r_squared"], 3))
    r4.metric("Beta Risk Label", str(capm_row["beta_risk_label"]))

    st.subheader("CAPM interpretation")

    alpha_text = (
        "positive alpha, suggesting excess return relative to CAPM expectations."
        if capm_row["alpha"] > 0
        else "negative alpha, suggesting underperformance relative to CAPM expectations."
    )

    beta_text = (
        "higher market sensitivity than the benchmark."
        if capm_row["beta"] > 1
        else "lower market sensitivity than the benchmark."
    )

    rsq_text = (
        "a relatively strong CAPM fit."
        if capm_row["r_squared"] >= 0.4
        else "a weaker CAPM fit, meaning market movement explains a limited share of the stock’s return variation."
    )

    st.write(
        f"For **{ticker}**, CAPM shows **{alpha_text}** "
        f"The stock has **beta = {capm_row['beta']:.3f}**, which implies **{beta_text}** "
        f"Its **R-squared = {capm_row['r_squared']:.3f}**, indicating **{rsq_text}**"
    )

    st.subheader("Risk takeaway")
    if capm_row["beta"] > 1.2:
        st.write("- This stock appears relatively aggressive and may be more sensitive to market swings.")
    elif capm_row["beta"] < 0.8:
        st.write("- This stock appears relatively defensive and may fluctuate less than the market.")
    else:
        st.write("- This stock appears to have moderate systematic risk relative to the market.")

    if capm_row["alpha"] > 0:
        st.write("- Positive alpha is a favorable signal from a risk-adjusted return perspective.")
    else:
        st.write("- Negative alpha suggests weaker risk-adjusted performance.")

    if capm_row["r_squared"] < 0.2:
        st.write("- Low R-squared means CAPM should be interpreted cautiously because market movement explains only a small part of returns.")

def about_page():
    st.title("About this project")
    st.markdown("""
This app is built for our BA870: Financial & Accounting Analytics project.

### Model Overview
We trained a **Logistic Regression classification model** to classify healthcare stocks into:

- **Overvalued**
- **Fairly valued**
- **Undervalued**

### Main Inputs
The model uses multi-year firm-level financial and valuation indicators, including:

- ROA
- Operating margin
- Debt to assets
- Revenue growth
- Current ratio
- Log assets
- Price-to-sales
- Price-to-book
- Relative and change-based features

### Additional Analysis
This app also includes:

- **Peer Comparison** using latest-year company metrics versus peer averages
- **CAPM Risk Analysis** using alpha, beta, and R-squared
""")

# =========================================================
# Navigation
# =========================================================
pg = st.navigation(
    [
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page(valuation_page, title="Valuation Assessment", icon="📊"),
        st.Page(peer_page, title="Peer Comparison", icon="📈"),
        st.Page(risk_page, title="Risk Analysis", icon="⚠️"),
        st.Page(about_page, title="About", icon="ℹ️"),
    ],
    position="sidebar"
)

render_sidebar_header()
pg.run()