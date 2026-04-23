# pages/4_Methodology.py

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Methodology", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

st.sidebar.markdown(f"### Current ticker: `{ticker}`")

# LABEL MAP
CLASS_LABELS = {
    0: "Overvalued",
    1: "Fairly Valued",
    2: "Undervalued",
    "0": "Overvalued",
    "1": "Fairly Valued",
    "2": "Undervalued",
}

# FILE HELPERS
def find_file(filename):
    paths = [
        Path(filename),
        Path("data") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path(__file__).parent.parent / "models" / filename,
    ]
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data
def load_csv(filename):
    path = find_file(filename)
    if path is None:
        return None
    return pd.read_csv(path)


comparison_df = load_csv("valuation_model_comparison_results.csv")
pred_df = load_csv("valuation_test_predictions.csv")
coef_df = load_csv("valuation_final_model_coefficients.csv")

# HELPERS
def normalize_label(value):
    if pd.isna(value):
        return value
    return CLASS_LABELS.get(value, CLASS_LABELS.get(str(value), value))


def prettify_feature_name(feature):
    return (
        feature.replace("_", " ")
        .replace("roa", "ROA")
        .title()
        .replace("Roa", "ROA")
    )


def build_confusion_matrix(df):
    if df is None:
        return None

    required_cols = {"actual_valuation_label", "predicted_valuation_label"}
    if not required_cols.issubset(df.columns):
        return None

    actual = df["actual_valuation_label"].map(normalize_label)
    predicted = df["predicted_valuation_label"].map(normalize_label)

    ordered_labels = ["Overvalued", "Fairly Valued", "Undervalued"]

    cm = pd.crosstab(
        pd.Categorical(actual, categories=ordered_labels),
        pd.Categorical(predicted, categories=ordered_labels),
        dropna=False,
    )

    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    return cm
    
def build_class_summary(df):
    if df is None:
        return None

    if "actual_valuation_label" not in df.columns:
        return None

    summary = (
        df["actual_valuation_label"]
        .map(normalize_label)
        .value_counts()
        .reindex(["Overvalued", "Fairly Valued", "Undervalued"], fill_value=0)
        .rename_axis("Class")
        .reset_index(name="Support")
    )

    return summary


def coefficient_class_options(df):
    if df is None:
        return []

    options = []
    if "coefficient_class_0" in df.columns:
        options.append(("Overvalued", "coefficient_class_0"))
    if "coefficient_class_1" in df.columns:
        options.append(("Fairly Valued", "coefficient_class_1"))
    if "coefficient_class_2" in df.columns:
        options.append(("Undervalued", "coefficient_class_2"))

    return options


def build_top_driver_features(df):
    if df is None:
        return None

    required_cols = {"feature", "coefficient_class_0", "coefficient_class_1", "coefficient_class_2"}
    if not required_cols.issubset(df.columns):
        return None

    top_df = df.copy()
    top_df["max_abs_coef"] = top_df[
        ["coefficient_class_0", "coefficient_class_1", "coefficient_class_2"]
    ].abs().max(axis=1)
    top_df = top_df.sort_values("max_abs_coef", ascending=False).head(12).copy()
    top_df["Feature"] = top_df["feature"].map(prettify_feature_name)

    return top_df[["Feature", "max_abs_coef"]]


def format_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "N/A"


def format_num(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


logistic_row = None
if comparison_df is not None and "model" in comparison_df.columns:
    logistic_match = comparison_df[
        comparison_df["model"].astype(str).str.lower() == "logistic regression"
    ]
    if not logistic_match.empty:
        logistic_row = logistic_match.iloc[0]

test_observations = len(pred_df) if pred_df is not None else None
class_summary_df = build_class_summary(pred_df)
cm = build_confusion_matrix(pred_df)
top_driver_features_df = build_top_driver_features(coef_df)

st.title("ℹ️ Methodology")
st.caption("How the project moves from raw financial data to valuation outputs and app deployment.")

# 1) PROJECT OVERVIEW
st.markdown("## 1) Project Overview")

overview_left, overview_right = st.columns([2.2, 1])

with overview_left:
    st.write(
        "This project builds a financial analytics app that estimates whether a stock appears "
        "**Overvalued, Fairly Valued, or Undervalued** based on structured financial, market, and peer-relative features."
    )
    st.write(
        "The goal is not to produce a price target. Instead, the model generates a **relative valuation classification** "
        "based on a firm’s financial profile and how that profile compares with peers."
    )

with overview_right:
    st.markdown("**Final Selected Model**")
    st.markdown(f"### {str(logistic_row['model']) if logistic_row is not None else 'Logistic Regression'}")

    st.markdown("**Classes**")
    st.markdown("### 3")

    st.markdown("**Test Observations**")
    st.markdown(f"### {test_observations if test_observations is not None else 'N/A'}")

# 2) MODELING APPROACH
st.markdown("## 2) Modeling Approach")

approach_left, approach_right = st.columns([2.2, 1])

with approach_left:
    st.write(
        "The final valuation model is a **multiclass logistic regression** model trained on company-year observations. "
        "The classes are encoded as:"
    )
    st.markdown(
        """
- 0 = Overvalued  
- 1 = Fairly Valued  
- 2 = Undervalued  
"""
    )
    st.write(
        "A time-based train/test split was used so that the model was evaluated on later periods rather than randomly mixed observations. "
        "This makes the setup more realistic for applied forecasting and deployment."
    )

with approach_right:
    st.info(
        "**Why logistic regression?**\n\n"
        "- Strong out-of-sample performance\n"
        "- Interpretable coefficients\n"
        "- Well suited for structured financial data"
    )

# 3) FEATURES / INPUTS
st.markdown("## 3) Features / Inputs")

st.write(
    "The model uses a mix of core financial metrics, change-based features, and peer-relative features. "
    "Each group contributes different information about valuation, quality, financial health, and business momentum."
)

features_table = pd.DataFrame({
    "Feature Group": [
        "Profitability",
        "Leverage / Liquidity",
        "Valuation Multiples",
        "Growth",
        "Change Features",
        "Peer-Relative Features",
    ],
    "Example Features": [
        "ROA, Operating Margin",
        "Debt to Assets, Current Ratio",
        "Price to Sales, Price to Book",
        "Revenue Growth",
        "ROA Change, Operating Margin Change, Debt to Assets Change",
        "ROA Rel, Operating Margin Rel, Debt to Assets Rel, Price to Sales Rel, Price to Book Rel",
    ],
})

st.dataframe(features_table, use_container_width=True, hide_index=True)

with st.expander("Top coefficient-based driver features"):
    if top_driver_features_df is not None:
        st.dataframe(
            top_driver_features_df.style.format({"max_abs_coef": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("Coefficient-based driver features are not available.")

# 4) MODEL OUTPUT & INTERPRETATION
st.markdown("## 4) Model Output & Interpretation")

output_left, output_right = st.columns([1.4, 1])

with output_left:
    st.write(
        "The output of the valuation model is a **class prediction**, not a direct valuation multiple or price target."
    )
    st.markdown(
        """
- **Overvalued:** the company’s feature profile looks expensive relative to the model’s learned patterns  
- **Fairly Valued:** the company appears broadly in line with expected valuation conditions  
- **Undervalued:** the company’s feature profile looks relatively attractive given the model inputs  
"""
    )

with output_right:
    st.info(
        "**On the Valuation page, the app shows:**\n\n"
        "- the predicted valuation class\n"
        "- model confidence / class probabilities\n"
        "- key drivers of the decision"
    )

# 5) MODEL EVALUATION
st.markdown("## 5) Model Evaluation")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

if logistic_row is not None:
    metric_1.metric("Final Test Accuracy", format_pct(logistic_row.get("test_accuracy")))
    metric_2.metric("Final Test Macro F1", format_num(logistic_row.get("test_f1"), 3))
    metric_3.metric("Final Train Accuracy", format_pct(logistic_row.get("train_accuracy")))
    metric_4.metric("Final Train Macro F1", format_num(logistic_row.get("train_f1"), 3))
else:
    metric_1.metric("Final Test Accuracy", "N/A")
    metric_2.metric("Final Test Macro F1", "N/A")
    metric_3.metric("Final Train Accuracy", "N/A")
    metric_4.metric("Final Train Macro F1", "N/A")

st.write(
    "The final model was selected based primarily on **out-of-sample macro F1**, which is appropriate for a multiclass setup "
    "because it gives balanced weight across the three valuation classes."
)

st.markdown("### Class-level summary")
if class_summary_df is not None:
    st.dataframe(class_summary_df, use_container_width=True, hide_index=True)
else:
    st.write("Class-level summary is not available.")

with st.expander("Confusion matrix"):
    if cm is not None:
        cm_long = (
            cm.reset_index()
            .melt(id_vars="Actual", var_name="Predicted", value_name="Count")
        )

        heatmap = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(
                x=alt.X("Predicted:N", title="Predicted Label"),
                y=alt.Y("Actual:N", title="Actual Label"),
                color=alt.Color("Count:Q", title="Count"),
                tooltip=["Actual", "Predicted", "Count"],
            )
            .properties(height=320)
        )

        text = (
            alt.Chart(cm_long)
            .mark_text(fontSize=13)
            .encode(
                x="Predicted:N",
                y="Actual:N",
                text="Count:Q",
                color=alt.condition(
                    alt.datum.Count > cm_long["Count"].mean(),
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        st.altair_chart(heatmap + text, use_container_width=True)
    else:
        st.write("Confusion matrix is not available from the current prediction file.")

# 6) MODEL EXPLANATION / COEFFICIENT-BASED INSIGHT
st.markdown("## 6) Model Explanation / Coefficient-Based Insight")

if coef_df is not None and "feature" in coef_df.columns:
    class_options = coefficient_class_options(coef_df)

    if class_options:
        class_label_lookup = {label: col for label, col in class_options}
        selected_label = st.selectbox(
            "Choose a valuation class to inspect",
            options=[label for label, _ in class_options]
        )
        selected_col = class_label_lookup[selected_label]

        coef_display = coef_df[["feature", selected_col]].copy()
        coef_display["abs_coef"] = coef_display[selected_col].abs()
        coef_display = coef_display.sort_values("abs_coef", ascending=False).head(12)
        coef_display["feature_label"] = coef_display["feature"].map(prettify_feature_name)

        coef_chart = (
            alt.Chart(coef_display)
            .mark_bar()
            .encode(
                x=alt.X(f"{selected_col}:Q", title="Coefficient"),
                y=alt.Y("feature_label:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("feature_label:N", title="Feature"),
                    alt.Tooltip(f"{selected_col}:Q", title="Coefficient", format=".4f")
                ],
            )
            .properties(height=360)
        )

        st.altair_chart(coef_chart, use_container_width=True)
        st.caption(
            "These coefficients show which features most strongly influence the logistic regression model for the selected class. "
            "Larger absolute values indicate stronger directional influence."
        )
    else:
        st.write("The coefficient file does not contain the expected class coefficient columns.")
else:
    st.write("Coefficient data is not available.")

# 7) ALTERNATIVE MODELS TESTED
st.markdown("## 7) Alternative Models Tested")

if comparison_df is not None:
    expected_cols = ["model", "train_accuracy", "test_accuracy", "train_f1", "test_f1"]
    available_cols = [col for col in expected_cols if col in comparison_df.columns]

    comparison_display = comparison_df[available_cols].copy()

    st.dataframe(
        comparison_display.style.format({
            "train_accuracy": "{:.1%}",
            "test_accuracy": "{:.1%}",
            "train_f1": "{:.3f}",
            "test_f1": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Several classification models were evaluated before selecting the final model. "
        "This comparison shows that the chosen model was intentional rather than arbitrary."
    )
else:
    st.warning("`valuation_model_comparison_results.csv` could not be found.")

# 8) DATA SOURCES
st.markdown("## 8) Data Sources")

data_sources_df = pd.DataFrame({
    "Source": [
        "WRDS / Wharton (Compustat)",
        "yfinance",
        "Kenneth R. French Data Library",
    ],
    "What It Provided": [
        "Firm-level financial statement data and accounting fundamentals.",
        "Market-based inputs such as price history and related market fields.",
        "Factor data used for the CAPM/risk side of the app.",
    ],
})

st.dataframe(data_sources_df, use_container_width=True, hide_index=True)

# 9) WORKFLOW
st.markdown("## 9) Workflow Summary")

with st.expander("Show pipeline summary"):
    st.markdown(
        """
1. **Collect historical financial and market data** using course-approved sources  
2. **Construct cleaned modeling datasets** and engineer ratios, change features, and peer-relative features  
3. **Train and compare multiple models** offline using the prepared training data  
4. **Export deployment-safe artifacts** such as CSV files and saved model objects  
5. **Deploy the app** using saved files plus public/free sources where appropriate  
"""
    )

# 10) LIMITATIONS
st.markdown("## 10) Limitations")

with st.expander("Show limitations"):
    st.markdown(
        """
- The valuation model is a **classification tool**, not a direct intrinsic value calculator  
- Model outputs depend on the selected features, training period, and class definitions  
- Peer-based comparisons are only as strong as the chosen comparison group  
- CAPM-style risk metrics simplify reality and do not capture all sources of uncertainty  
- Public/live add-ons should be treated as **supplemental snapshots**, not replacements for the main offline analysis  
"""
    )

# EXPLORE MORE
st.markdown("## Explore More")
st.markdown(
    """
- **Valuation Assessment:** Review the model’s latest-year valuation result and historical valuation context
- **Peer Comparison:** Benchmark the selected company against peer averages across key financial metrics  
- **Risk Analysis:** Examine beta, alpha, and R-squared for the selected company
"""
)
