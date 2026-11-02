import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retailpulse_combined.csv"
FORECAST_PATH = BASE_DIR / "data" / "sales_forecast.csv"

# 1. Load dataset
print("Loading RetailPulse dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Clean data
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "Price"]
)

# Exclude returns and invalid prices
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# 3. Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# 4. Calculate daily sales
daily_sales = (
    df.set_index("InvoiceDate")["Sales"]
    .resample("D")
    .sum()
)

daily_sales = daily_sales.asfreq("D", fill_value=0)

print("First date:", daily_sales.index.min())
print("Last date:", daily_sales.index.max())
print("Total days:", len(daily_sales))

# 5. Split data into training and test sets
# Use the final 30 historical days as test data.
if len(daily_sales) <= 60:
    raise ValueError("Not enough daily data for evaluation.")

train = daily_sales.iloc[:-30]
test = daily_sales.iloc[-30:]

print("\nTraining ARIMA model for evaluation...")

# 6. Fit model using training data only
evaluation_model = ARIMA(
    train,
    order=(5, 1, 0)
)

evaluation_fit = evaluation_model.fit()

# 7. Forecast the held-out 30 days
test_predictions = evaluation_fit.forecast(steps=len(test))

# 8. Compare actual and predicted values
evaluation_df = pd.DataFrame({
    "Date": test.index,
    "ActualSales": test.values,
    "PredictedSales": np.asarray(test_predictions)
})

evaluation_df.to_csv(
    BASE_DIR / "data" / "forecast_evaluation_results.csv",
    index=False
)

# 9. Calculate evaluation metrics
actual = evaluation_df["ActualSales"].to_numpy()
predicted = evaluation_df["PredictedSales"].to_numpy()

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

print("\n--- ARIMA Evaluation Results ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

if np.isfinite(mape):
    print(f"MAPE: {mape:.2f}%")
else:
    print("MAPE: Not available")

print("\nEvaluation results saved to:")
print("data/forecast_evaluation_results.csv")

# 10. Train a new model on all historical data
print("\nTraining final model for future forecasting...")

final_model = ARIMA(
    daily_sales,
    order=(5, 1, 0)
)

final_fit = final_model.fit()

# 11. Forecast the next 30 days
forecast_values = final_fit.forecast(steps=30)

future_dates = pd.date_range(
    start=daily_sales.index[-1].normalize()
          + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "PredictedSales": np.asarray(forecast_values)
})

forecast_df.to_csv(
    FORECAST_PATH,
    index=False
)

print("\n--- Next 30 Days Forecast ---")
print(forecast_df.to_string(index=False))
print("\nSaved:", FORECAST_PATH)

# 12. Plot actual vs predicted test data
plt.figure(figsize=(12, 6))

plt.plot(
    evaluation_df["Date"],
    evaluation_df["ActualSales"],
    label="Actual Sales",
    marker="o"
)

plt.plot(
    evaluation_df["Date"],
    evaluation_df["PredictedSales"],
    label="Predicted Sales",
    marker="o"
)

plt.title("RetailPulse - ARIMA Model Evaluation")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "data" / "arima_evaluation_chart.png"
)

plt.close()

print("Evaluation chart saved.")

# 13. Plot future forecast
plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales.index[-60:],
    daily_sales.values[-60:],
    label="Historical Sales"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["PredictedSales"],
    label="Next 30 Days Forecast",
    marker="o"
)

plt.title("RetailPulse - Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "data" / "arima_next_30_days_chart.png"
)

plt.close()

print("Forecast chart saved.")
print("\nForecast starts:", forecast_df["Date"].min())
print("Forecast ends:", forecast_df["Date"].max())
