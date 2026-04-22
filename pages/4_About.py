import streamlit as st

st.set_page_config(page_title="About", layout="wide")

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

### Workflow
1. User enters a ticker once on the Home page
2. The selected ticker is saved in session state
3. Each page reads the same ticker automatically
4. Users can move across pages without retyping
""")