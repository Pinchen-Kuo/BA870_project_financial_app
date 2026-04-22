import streamlit as st

st.set_page_config(page_title="ValueLens", layout="wide")

# Initialize shared ticker
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

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
ticker_input = st.text_input("Ticker", value=st.session_state["ticker"]).upper().strip()

if st.button("Set Ticker"):
    st.session_state["ticker"] = ticker_input
    st.success(f"Ticker set to: {ticker_input}")

st.markdown(f"### Current selected ticker: `{st.session_state['ticker']}`")

st.info("Now use the left sidebar to open Valuation Assessment, Peer Comparison, or Risk Analysis.")