import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Predictor", page_icon="📈")

st.title("📈 Stock Prediction App")
st.write("Welcome to my Stock Market Prediction App!")

st.subheader("Search for a stock")

symbol = st.text_input(
    "Enter stock symbol",
    "RELIANCE.NS"
)

if st.button("Predict Stock Price"):

    with st.spinner("Loading stock data..."):

        data = yf.download(
            symbol,
            period="1y",
            progress=False
        )

        if data.empty:
            st.error("Stock data not found. Check the symbol.")
        else:
            prices = data["Close"]

            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            prices = prices.dropna()

            st.subheader("Historical Stock Prices")
            st.line_chart(prices)

            X = np.arange(len(prices)).reshape(-1, 1)
            y = prices.values

            model = LinearRegression()
            model.fit(X, y)

            next_day = np.array([[len(prices)]])
            prediction = model.predict(next_day)[0]

            st.subheader("Next-Day Estimate")

            st.metric(
                "Estimated Price",
                f"₹{prediction:.2f}"
            )

            st.info(
                "This is an experimental educational estimate, "
                "not a reliable forecast or investment advice."
            )
