# streamlit_app.py

import streamlit as st

st.set_page_config(page_title="ValueLens", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

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

AVAILABLE_TICKERS = list(TICKER_NAME_MAP.keys())


def format_ticker_option(ticker_value):
    return f"{ticker_value} ({TICKER_NAME_MAP.get(ticker_value, '')})"


# SIDEBAR
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Current ticker: `{st.session_state['ticker']}`")

# PAGE
st.title("ValueLens")
st.subheader("Integrated Healthcare Stock Analysis for Valuation, Peer Comparison, and Risk")

st.write(
    "ValueLens is an integrated equity-analysis app designed to support better decision-making by combining "
    "**machine learning-based valuation**, **peer benchmarking**, and **CAPM-based market risk analysis** "
    "in one workflow. Instead of looking at a company from only one angle, the app helps users evaluate "
    "how the stock appears on valuation, how it compares with peers, and what kind of market-risk profile it carries."
)

top_left, top_right = st.columns([2, 1])

with top_left:
    st.markdown("## Why ValueLens?")
    st.write(
        "A single metric rarely tells the full story. ValueLens brings together three complementary perspectives "
        "so the user can move from a raw ticker selection to a more complete view of the company’s financial profile."
    )
    st.markdown(
        """
- **Valuation Assessment** asks what the model suggests about the company’s valuation status  
- **Peer Comparison** asks how the company stacks up against similar firms on key financial metrics  
- **Risk Analysis (CAPM)** asks what kind of market sensitivity and risk-adjusted performance profile the stock has  
"""
    )
    st.write(
        "Together, these components provide a stronger analytical narrative for class discussion, presentation, and practical investment interpretation."
    )

with top_right:
    st.markdown("## Quick Facts")
    q1, q2 = st.columns(2)
    with q1:
        st.metric("Core Pages", "4")
        st.metric("Sector Focus", "Healthcare")
    with q2:
        st.metric("Final Model", "Logistic Regression")
        st.metric("Classes", "3")

st.divider()

st.markdown("## Select a Company")
st.write(
    "Choose a ticker here once, then click **Analyze**. The selected ticker will carry across the app."
)

selected_option = st.selectbox(
    "Ticker",
    options=AVAILABLE_TICKERS,
    index=AVAILABLE_TICKERS.index(st.session_state["ticker"]) if st.session_state["ticker"] in AVAILABLE_TICKERS else 0,
    format_func=format_ticker_option,
)

button_col1, button_col2 = st.columns([1, 4])
with button_col1:
    analyze_clicked = st.button("Analyze", use_container_width=True)

if analyze_clicked:
    st.session_state["ticker"] = selected_option
    st.success(f"Ticker updated to {selected_option}. You can now open the Valuation, Peer Comparison, Risk, or Methodology pages.")

st.markdown(f"### Current selected ticker: `{st.session_state['ticker']}`")

st.divider()

st.markdown("## What the App Does")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 📈 Valuation Assessment")
    st.write(
        "What does the model suggest about the company’s valuation? "
        "This page classifies the selected company as **Overvalued**, **Fairly Valued**, or **Undervalued**."
    )

with c2:
    st.markdown("### 📊 Peer Comparison")
    st.write(
        "How does the company compare with similar firms? "
        "This page benchmarks the selected company against peer averages across key financial metrics."
    )

with c3:
    st.markdown("### ⚠️ Risk Analysis (CAPM)")
    st.write(
        "What kind of market-risk profile does the stock have? "
        "This page evaluates **beta**, **alpha**, and **R-squared** using a CAPM-based perspective."
    )

st.divider()

st.markdown("## How to Use ValueLens")
st.markdown(
    """
1. Select a company ticker above and click **Analyze**  
2. Open **Valuation Assessment** to see the model’s classification and key drivers  
3. Open **Peer Comparison** to benchmark the company against peers  
4. Open **Risk Analysis (CAPM)** to review market sensitivity and risk-adjusted performance  
5. Open **Methodology** for the model, data, evaluation, and workflow details  
"""
)

st.success("Use the sidebar to navigate across pages.")

st.markdown("## Start Exploring")
link_col1, link_col2, link_col3, link_col4 = st.columns(4)
with link_col1:
    st.page_link("pages/1_Valuation.py", label="Go to Valuation", icon="📈")
with link_col2:
    st.page_link("pages/2_Peer_Comparison.py", label="Go to Peer Comparison", icon="📊")
with link_col3:
    st.page_link("pages/3_Risk.py", label="Go to Risk Analysis", icon="⚠️")
with link_col4:
    st.page_link("pages/4_Methodology.py", label="Go to Methodology", icon="📘")

with st.expander("Data Sources"):
    st.markdown(
        """
- **WRDS / Compustat** — firm-level accounting and financial statement data used offline for dataset construction and model training  
- **yfinance** — public/free market data used for deployment-safe market inputs and live/public updates  
- **Kenneth R. French Data Library** — factor data used for CAPM-style market-risk analysis  
"""
    )
