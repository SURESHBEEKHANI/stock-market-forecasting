import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

# Custom CSS for professional look
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #fafafa;
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #388e3c;
        }
        h1, h2, h3 {
            color: #333;
        }
        .stDataFrame {
            background-color: #ffffff;
        }
        .stTable {
            color: #333;
        }
        .stPlotlyChart {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📈 Stock Market Forecasting")

# Sidebar Inputs
st.sidebar.header("User Input")

# Inputs for stock ticker, date range, and forecast days
ticker = st.sidebar.text_input("Enter stock ticker:", value="GOOGL")
end_date = st.sidebar.date_input("End Date", value=date.today())
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
forecast_days = st.sidebar.slider("Forecast days", min_value=10, max_value=30, value=15)

# Ensure start_date is not later than end_date
if start_date > end_date:
    st.sidebar.error("❌ Start date cannot be later than end date.")
    st.stop()

# Feature selection for stock attributes (e.g., Close, High, Low, Open)
feature_select = st.sidebar.multiselect(
    label="Select Stock Features to Forecast",
    options=["Close", "High", "Low", "Open"],
    default=["Close"]
)

# Feature descriptions (explanation for each stock feature)
feature_descriptions = {
    "Close": "The closing price of the stock at the end of the trading day.",
    "High": "The highest price at which the stock traded during the day.",
    "Low": "The lowest price at which the stock traded during the day.",
    "Open": "The price at which the stock opened at the beginning of the trading day."
}

# Fetch Stock Data based on user input
if st.sidebar.button("Fetch Data"):
    st.subheader(f"Stock Data for {ticker.upper()} ({start_date} to {end_date})")

    try:
        # Download stock data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Check if data is available
        if df.empty:
            st.error("❌ No data found for the selected ticker or date range.")
            st.stop()

        st.success("✅ Data fetched successfully!")

        # Reset index and insert 'Date' column for better readability
        df.insert(0, "Date", df.index)
        df.reset_index(drop=True, inplace=True)

        st.dataframe(df.head())  # Display first few rows of the stock data

        # Initialize an empty DataFrame to store forecasted data for all features
        combined_forecast_df = pd.DataFrame()

        # Iterate over each selected feature and generate forecasts
        for feature in feature_select:
            st.write(f"### 🔍 Forecasting: {feature}")
            st.write(feature_descriptions[feature])

            # Prepare the time series for forecasting (drop NaN values)
            series = df[feature].dropna()

            # Fit the ARIMA model using the selected feature's time series
            model = auto_arima(series,
                               start_p=1, start_q=1,
                               max_p=2, max_q=2,
                               m=12,  # Monthly seasonal cycle
                               start_P=0,
                               seasonal=True,
                               d=1, D=1,
                               trace=True,  # Display fitting progress
                               error_action='ignore',
                               suppress_warnings=True)

            # Forecast future values for the specified number of days
            forecast = model.predict(n_periods=forecast_days)
            future_dates = pd.date_range(start=df["Date"].iloc[-1] + timedelta(days=1), periods=forecast_days)

            # Create a DataFrame to hold the forecasted values
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                feature: forecast
            })

            # Combine the forecasted data for each feature into a single DataFrame
            combined_forecast_df = pd.merge(combined_forecast_df, forecast_df, on="Date", how="outer") if not combined_forecast_df.empty else forecast_df

            # Plot historical data and forecasted data
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df["Date"], series, label="Historical", color='blue')  # Historical data
            ax.plot(forecast_df["Date"], forecast_df[feature], label="Forecast", color='orange')  # Forecasted data
            ax.set_title(f"{feature} Forecast for {ticker.upper()}")
            ax.set_xlabel("Date")
            ax.set_ylabel(feature)
            ax.legend()

            # Display the plot
            st.pyplot(fig)

        # Display the combined forecasted data for all selected features
        st.subheader("Combined Forecasted Data")
        st.dataframe(combined_forecast_df.head())

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.stop()
