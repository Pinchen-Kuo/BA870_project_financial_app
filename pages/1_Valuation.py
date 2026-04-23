# pages/1_Valuation.py

import streamlit as st
import pandas as pd
import pickle
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Valuation Assessment", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

# LABELS
CLASS_LABELS = {
    0: "Overvalued",
    1: "Fairly valued",
    2: "Undervalued",
}

CLASS_COLORS = {
    "Overvalued": "red",
    "Fairly valued": "orange",
    "Undervalued": "green",
}

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
        Path("models") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path(__file__).parent.parent / "models" / filename,
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
def load_coefficients():
    possible_names = [
        "valuation_final_model_coefficients.csv",
        "valuation_final_model_coefficients (1).csv",
    ]
    for name in possible_names:
        coeff_path = find_file(name)
        if coeff_path is not None:
            return pd.read_csv(coeff_path)
    raise FileNotFoundError("valuation_final_model_coefficients.csv not found")


# HELPERS
def company_display_name(ticker_value):
    company_name = TICKER_NAME_MAP.get(str(ticker_value).upper(), "")
    if company_name:
        return f"{ticker_value} ({company_name})"
    return str(ticker_value)


def label_with_color(label):
    color = CLASS_COLORS.get(label, "blue")
    return f":{color}[**{label}**]"


def format_prob_table(proba_df):
    display_df = proba_df.copy()
    display_df["probability"] = display_df["probability"].map(lambda x: f"{x:.1%}")
    return display_df[["label", "probability"]]


def prettify_feature_name(feature):
    return (
        feature.replace("_", " ")
        .replace("roa", "ROA")
        .title()
        .replace("Roa", "ROA")
    )


def driver_theme(feature):
    mapping = {
        "roa": "profitability",
        "operating_margin": "operating efficiency",
        "debt_to_assets": "leverage",
        "revenue_growth": "growth profile",
        "current_ratio": "liquidity",
        "log_assets": "company scale",
        "price_to_sales": "valuation multiple",
        "price_to_book": "valuation multiple",
        "roa_change": "profitability trend",
        "operating_margin_change": "margin trend",
        "debt_to_assets_change": "leverage trend",
        "roa_rel": "peer-relative profitability",
        "operating_margin_rel": "peer-relative margins",
        "debt_to_assets_rel": "peer-relative leverage",
        "price_to_sales_rel": "peer-relative pricing",
        "price_to_book_rel": "peer-relative pricing",
    }
    return mapping.get(feature, prettify_feature_name(feature).lower())


def build_top_driver_summary(top_drivers):
    if top_drivers.empty:
        return "No major model drivers were available."
    themes = []
    for feature in top_drivers["feature"].tolist():
        theme = driver_theme(feature)
        if theme not in themes:
            themes.append(theme)
    if len(themes) == 1:
        return themes[0]
    if len(themes) == 2:
        return f"{themes[0]} and {themes[1]}"
    return f"{themes[0]}, {themes[1]}, and {themes[2]}"


def make_interpretation(label, driver_df):
    top_driver_text = build_top_driver_summary(driver_df.head(3))

    if label == "Overvalued":
        return (
            f"**Latest-year interpretation:** The model classifies the company as **Overvalued** for the most recent full year. "
            f"This means the latest financial and peer-relative profile looks expensive compared with the patterns the model learned from prior data. "
            f"The strongest influences came from **{top_driver_text}**, which pushed the prediction toward a richer valuation signal. "
            f"From an investor perspective, this does not automatically mean the company is weak, but it does suggest that strong expectations may already be priced in. "
            f"A practical takeaway is to ask whether current profitability, growth, and relative valuation are strong enough to justify that pricing."
        )

    if label == "Fairly valued":
        return (
            f"**Latest-year interpretation:** The model classifies the company as **Fairly valued** for the most recent full year. "
            f"This suggests the latest financial profile is broadly aligned with the model’s expected valuation patterns, without a strong signal of clear overvaluation or undervaluation. "
            f"The most important influences came from **{top_driver_text}**, but not strongly enough to push the company into a more extreme category. "
            f"From an investor perspective, this points to a more balanced setup where the market appears to be valuing the company reasonably in line with its latest fundamentals and peer-relative profile."
        )

    if label == "Undervalued":
        return (
            f"**Latest-year interpretation:** The model classifies the company as **Undervalued** for the most recent full year. "
            f"This suggests the latest financial profile looks relatively attractive compared with the valuation patterns learned by the model. "
            f"The result was driven mainly by **{top_driver_text}**, which provided support from fundamentals and/or more favorable relative valuation signals. "
            f"From an investor perspective, this may indicate potential upside if the market is underpricing the company relative to its current profile. "
            f"At the same time, it is worth checking whether the apparent discount reflects genuine opportunity or company-specific risks that the model does not fully capture."
        )

    return "The model produced a valuation result for the latest full year."


def summarize_historical_signal(pred_series):
    avg_class = pred_series.mean()
    if avg_class < 0.67:
        return "Overvalued"
    elif avg_class < 1.33:
        return "Fairly valued"
    return "Undervalued"


def get_majority_label(pred_series):
    majority_class = pred_series.mode().iloc[0]
    return CLASS_LABELS[int(majority_class)]


def build_probability_df(model, probabilities):
    try:
        classes = model.named_steps["model"].classes_
    except Exception:
        classes = [0, 1, 2]
    proba_df = pd.DataFrame({
        "class_id": classes,
        "probability": probabilities
    })
    proba_df["label"] = proba_df["class_id"].map(CLASS_LABELS)
    proba_df = proba_df.sort_values("probability", ascending=False).reset_index(drop=True)
    return proba_df


def compute_driver_chart_data(model, coef_df, feature_cols, selected_row, pred_class):
    scaler = model.named_steps["scaler"]

    feature_values = selected_row[feature_cols].astype(float).values
    standardized = (feature_values - scaler.mean_) / scaler.scale_

    coef_col = f"coefficient_class_{pred_class}"
    aligned_coef_df = coef_df[coef_df["feature"].isin(feature_cols)].copy()
    aligned_coef_df = aligned_coef_df.set_index("feature").reindex(feature_cols).reset_index()
    coefs = aligned_coef_df[coef_col].values

    contribution = standardized * coefs

    driver_df = pd.DataFrame({
        "feature": feature_cols,
        "raw_value": feature_values,
        "standardized_value": standardized,
        "coefficient": coefs,
        "contribution": contribution,
    })
    driver_df["abs_contribution"] = driver_df["contribution"].abs()
    driver_df = driver_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return driver_df


# LOAD DATA
try:
    model = load_model()
    feature_cols = load_feature_cols()
    valuation_df = load_input_data()
    coef_df = load_coefficients()
except Exception as e:
    st.error("Required valuation files are missing.")
    st.exception(e)
    st.stop()

required_cols = {"ticker", "year", *feature_cols}
missing_data_cols = required_cols - set(valuation_df.columns)
if missing_data_cols:
    st.error(f"ticker_history_input.csv is missing required columns: {sorted(missing_data_cols)}")
    st.stop()

required_coef_cols = {"feature", "coefficient_class_0", "coefficient_class_1", "coefficient_class_2"}
missing_coef_cols = required_coef_cols - set(coef_df.columns)
if missing_coef_cols:
    st.error(f"valuation_final_model_coefficients.csv is missing required columns: {sorted(missing_coef_cols)}")
    st.stop()

company_data = valuation_df[
    valuation_df["ticker"].astype(str).str.upper() == str(ticker).upper()
].copy()

if company_data.empty:
    st.warning("Ticker not found in dataset.")
    st.stop()

company_data = company_data.sort_values("year").reset_index(drop=True)

latest_row = company_data.iloc[-1].copy()
latest_year = int(latest_row["year"])

try:
    latest_X = pd.DataFrame([latest_row[feature_cols].astype(float).to_dict()])
    latest_pred_class = int(model.predict(latest_X)[0])
    latest_pred_prob = model.predict_proba(latest_X)[0]
except Exception as e:
    st.error("Latest-year prediction failed.")
    st.exception(e)
    st.stop()

latest_label = CLASS_LABELS.get(latest_pred_class, f"Class {latest_pred_class}")
latest_proba_df = build_probability_df(model, latest_pred_prob)
latest_top_prob = float(latest_proba_df.iloc[0]["probability"])

driver_df = compute_driver_chart_data(model, coef_df, feature_cols, latest_row, latest_pred_class)
top_drivers = driver_df.head(3).copy()
top_driver_summary = build_top_driver_summary(top_drivers)
latest_interpretation = make_interpretation(latest_label, driver_df)

try:
    historical_X = company_data[feature_cols].astype(float).copy()
    historical_pred_class = model.predict(historical_X)
    historical_pred_prob = model.predict_proba(historical_X)
except Exception as e:
    st.error("Historical prediction generation failed.")
    st.exception(e)
    st.stop()

result_df = company_data[["ticker", "year"]].copy()
result_df["predicted_class"] = historical_pred_class
result_df["predicted_label"] = pd.Series(historical_pred_class).map(CLASS_LABELS).values
result_df["prob_overvalued"] = historical_pred_prob[:, 0]
result_df["prob_fairly_valued"] = historical_pred_prob[:, 1]
result_df["prob_undervalued"] = historical_pred_prob[:, 2]

historical_summary_label = summarize_historical_signal(result_df["predicted_class"])
historical_majority_label = get_majority_label(result_df["predicted_class"])

# PAGE
st.title("Valuation Assessment")
st.markdown(f"### Current ticker: `{company_display_name(ticker)}`")
st.caption(
    "This page highlights the model result for the latest full year first, then shows how the valuation signal has looked across the company’s historical observations."
)

st.markdown("## 1) Latest-Year Valuation Result")
st.write(f"The highlighted result below is based on the **most recent full year in the app dataset: {latest_year}**.")

c1, c2, c3 = st.columns([1.4, 1, 1])

with c1:
    st.markdown("**Latest-Year Classification**")
    st.markdown(f"### {label_with_color(latest_label)}")

with c2:
    st.metric("Model Confidence", f"{latest_top_prob:.1%}")

with c3:
    st.metric("Year Used", str(latest_year))

st.markdown("**Class probabilities for the latest full year**")
st.dataframe(
    format_prob_table(latest_proba_df),
    use_container_width=True,
    hide_index=True,
)

st.markdown("## 2) Interpretation")
st.info(latest_interpretation)

st.markdown("## 3) What Drove This Latest-Year Result?")
st.markdown(f"**Top Drivers:** {top_driver_summary}")
st.caption(
    "The chart below shows the strongest model drivers for the latest full-year prediction. "
    "Bars to the right pushed the prediction more positively for the predicted class, while bars to the left pushed against it."
)

driver_chart_df = driver_df.head(6).copy()
driver_chart_df["feature_label"] = driver_chart_df["feature"].map(prettify_feature_name)
driver_chart_df = driver_chart_df.sort_values("contribution", ascending=True)
driver_chart_df["direction"] = driver_chart_df["contribution"].apply(
    lambda x: "Positive contribution" if x >= 0 else "Negative contribution"
)

zero_rule = pd.DataFrame({"x": [0]})

driver_chart = (
    alt.Chart(driver_chart_df)
    .mark_bar()
    .encode(
        x=alt.X("contribution:Q", title="Contribution to latest-year prediction"),
        y=alt.Y("feature_label:N", title=None, sort=list(driver_chart_df["feature_label"])),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["Negative contribution", "Positive contribution"],
                range=["#d9534f", "#2e86de"]
            ),
            legend=alt.Legend(title=None)
        ),
        tooltip=[
            alt.Tooltip("feature_label:N", title="Feature"),
            alt.Tooltip("contribution:Q", title="Contribution", format=".4f"),
            alt.Tooltip("raw_value:Q", title="Raw Value", format=".4f"),
            alt.Tooltip("standardized_value:Q", title="Standardized Value", format=".4f"),
            alt.Tooltip("coefficient:Q", title="Coefficient", format=".4f"),
        ],
    )
    .properties(height=320)
)

zero_line = (
    alt.Chart(zero_rule)
    .mark_rule(color="gray", strokeDash=[5, 5])
    .encode(x="x:Q")
)

st.altair_chart(driver_chart + zero_line, use_container_width=True)

with st.expander("See latest-year driver details"):
    detail_df = driver_chart_df[[
        "feature_label",
        "raw_value",
        "standardized_value",
        "coefficient",
        "contribution",
    ]].copy()
    detail_df.columns = [
        "Feature",
        "Raw Value",
        "Standardized Value",
        "Coefficient",
        "Contribution",
    ]
    st.dataframe(
        detail_df.style.format({
            "Raw Value": "{:,.4f}",
            "Standardized Value": "{:,.4f}",
            "Coefficient": "{:,.4f}",
            "Contribution": "{:,.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("## 4) Historical Valuation Context")

h1, h2, h3 = st.columns(3)
with h1:
    st.metric("Years Used", len(result_df))
with h2:
    st.metric("Historical Valuation Summary", historical_summary_label)
with h3:
    st.metric("Most Common Historical Label", historical_majority_label)

st.caption(
    "This section summarizes how the model classified the company across all available years, rather than only the latest full year."
)

st.markdown("**Year-by-year model outputs**")
historical_display_df = result_df.copy()
historical_display_df["prob_overvalued"] = historical_display_df["prob_overvalued"].map(lambda x: f"{x:.1%}")
historical_display_df["prob_fairly_valued"] = historical_display_df["prob_fairly_valued"].map(lambda x: f"{x:.1%}")
historical_display_df["prob_undervalued"] = historical_display_df["prob_undervalued"].map(lambda x: f"{x:.1%}")
st.dataframe(historical_display_df, use_container_width=True, hide_index=True)

st.markdown("**Valuation distribution across years**")
distribution_df = (
    result_df["predicted_label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="count")
)

distribution_chart = (
    alt.Chart(distribution_df)
    .mark_bar()
    .encode(
        x=alt.X("label:N", title="Predicted label"),
        y=alt.Y("count:Q", title="Number of years"),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["Overvalued", "Fairly valued", "Undervalued"],
                range=["#d9534f", "#f0ad4e", "#5cb85c"]
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("count:Q", title="Years"),
        ],
    )
    .properties(height=260)
)

st.altair_chart(distribution_chart, use_container_width=True)

st.markdown("## 5) Historical Inputs")
with st.expander("See historical input data used for the year-by-year predictions"):
    st.dataframe(company_data, use_container_width=True, hide_index=True)

# EXPLORE MORE
st.markdown("## Explore More")
st.markdown(
    """
- **Peer Comparison:** Benchmark the selected company against peer averages across key financial metrics  
- **Risk Analysis:** Review beta, alpha, and R-squared to understand the company’s CAPM risk profile  
- **Methodology:** See how the valuation model, data pipeline, and supporting files were built  
"""
)

st.divider()
st.caption(
    "Note: The latest-year valuation result is the main headline signal on this page. "
    "The historical section is included as added context, not as a replacement for the latest-year result."
)
