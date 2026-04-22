import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Valuation Assessment", layout="wide")

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

# =========================================================
# Load files
# =========================================================
@st.cache_resource
def load_model():
    model_path = find_file("final_model.pkl")
    if model_path is None:
        raise FileNotFoundError("final_model.pkl not found")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_feature_cols():
    feature_cols_path = find_file("feature_cols.pkl")
    if feature_cols_path is None:
        raise FileNotFoundError("feature_cols.pkl not found")
    with open(feature_cols_path, "rb") as f:
        feature_cols = pickle.load(f)
    return feature_cols

@st.cache_data
def load_input_data():
    csv_path = find_file("ticker_history_input.csv")
    if csv_path is None:
        raise FileNotFoundError("ticker_history_input.csv not found")
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
    else:
        return "Undervalued"

def get_majority_label(pred_series):
    majority_class = pred_series.mode().iloc[0]
    return label_map[int(majority_class)]

st.title("Valuation Assessment")
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