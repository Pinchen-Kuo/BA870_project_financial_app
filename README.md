# BA870: Financial & Accounting Analytics - project
contributors: Pin-Chen Kuo, Kefei Zhang, Bill Odiase

App link:
https://blank-app-vcq62u99kxg.streamlit.app/

## ValueLens: Stock Valuation Analyzer

ValueLens is a data-driven financial analytics application developed as part of the BA870 course. The project focuses on evaluating the valuation and risk profile of healthcare companies using machine learning and financial modeling techniques.

The application allows users to input a stock ticker and receive an integrated analysis based on multiple dimensions:

- Valuation Assessment
A machine learning classification model (Logistic Regression) predicts whether a stock is Overvalued, Fairly Valued, or Undervalued using multi-year financial and valuation indicators.
- Peer Comparison
The app compares the selected company’s latest financial metrics (e.g., profitability, growth, leverage, valuation ratios) against the average of its industry peers, providing context for relative performance.
- Risk Analysis (CAPM)
Using CAPM-based metrics such as alpha, beta, and R-squared, the app evaluates the stock’s systematic risk and risk-adjusted return characteristics.

🔍 Methodology

The model is trained on historical financial data of healthcare companies, incorporating features such as:

- Profitability (ROA, operating margin)
- Growth (revenue growth)
- Leverage (debt-to-assets)
- Liquidity (current ratio)
- Valuation multiples (price-to-sales, price-to-book)
- Relative and change-based features over time

To better reflect real-world decision-making, predictions are aggregated across multiple years for each company, rather than relying on a single-period snapshot.

🚀 Application Features

- Interactive Streamlit web interface
- Multi-page navigation (valuation, peers, risk)
- Multi-year prediction aggregation
- Visual comparison with peer averages
- Integrated financial + statistical interpretation

🛠 Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- Machine Learning (Logistic Regression)
- Visualization (Altair)
- Web App (Streamlit)
- Deployment (Streamlit Community Cloud)

📌 Project Objective

This project aims to bridge financial theory and practical analytics by building an end-to-end system that transforms raw financial data into actionable investment insights.
