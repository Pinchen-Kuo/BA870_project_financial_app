import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Risk Analysis", layout="wide")

if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

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
def load_capm_data():
    csv_path = find_file("capm_risk_metrics.csv")
    if csv_path is None:
        raise FileNotFoundError("capm_risk_metrics.csv not found")
    return pd.read_csv(csv_path)

def safe_num(x, digits=3):
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"

st.title("Risk Analysis (CAPM)")
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