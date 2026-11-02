import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Evaluate Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 RetailPulse - Forecast Model Evaluation")
st.write("Evaluate ARIMA predictions against actual historical sales.")

BASE_DIR = Path(__file__).resolve().parent.parent

evaluation_path = (
    BASE_DIR / "data" / "forecast_evaluation_results.csv"
)

if not evaluation_path.exists():
    st.error("Evaluation results file not found.")
    st.info("Run: python forecast_next_30_days.py")
    st.stop()

# Load evaluation results
df = pd.read_csv(evaluation_path)

required_columns = [
    "Date",
    "ActualSales",
    "PredictedSales"
]

if not all(col in df.columns for col in required_columns):
    st.error("Required evaluation columns are missing.")
    st.write("Available columns:", list(df.columns))
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["ActualSales"] = pd.to_numeric(
    df["ActualSales"], errors="coerce"
)
df["PredictedSales"] = pd.to_numeric(
    df["PredictedSales"], errors="coerce"
)

df = df.dropna(subset=required_columns)
df = df.sort_values("Date")

if df.empty:
    st.warning("No valid evaluation data available.")
    st.stop()

# Calculate metrics
actual = df["ActualSales"].to_numpy()
predicted = df["PredictedSales"].to_numpy()

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))

nonzero_mask = actual != 0

if nonzero_mask.any():
    mape = np.mean(
        np.abs(
            (actual[nonzero_mask] - predicted[nonzero_mask])
            / actual[nonzero_mask]
        )
    ) * 100
else:
    mape = np.nan

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric(
    "MAPE",
    f"{mape:.2f}%" if np.isfinite(mape) else "N/A"
)
c4.metric("Test Days", f"{len(df)}")

# Comparison chart
st.subheader("Actual vs Predicted Sales")

fig = px.line(
    df,
    x="Date",
    y=["ActualSales", "PredictedSales"],
    markers=True,
    title="ARIMA: Actual vs Predicted Sales",
    labels={
        "Date": "Date",
        "value": "Sales Amount",
        "variable": "Sales Type"
    }
)

st.plotly_chart(fig, use_container_width=True)

# Error analysis
st.subheader("Prediction Error Analysis")

df["Error"] = df["ActualSales"] - df["PredictedSales"]
df["AbsoluteError"] = df["Error"].abs()

fig_error = px.bar(
    df,
    x="Date",
    y="AbsoluteError",
    title="Absolute Prediction Error by Day",
    labels={
        "Date": "Date",
        "AbsoluteError": "Absolute Error"
    }
)

st.plotly_chart(fig_error, use_container_width=True)

# Data table
st.subheader("Evaluation Details")

st.dataframe(
    df,
    use_container_width=True
)

# Download results
st.download_button(
    label="Download Evaluation Results",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="forecast_evaluation_results.csv",
    mime="text/csv"
)