import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

# =====================================
# 1. LOAD DATASET
# =====================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_2009_2010.csv"

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

# =====================================
# 2. DATA CLEANING
# =====================================

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "Price"]
)

df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

df["TotalSales"] = df["Quantity"] * df["Price"]

# =====================================
# 3. DAILY SALES
# =====================================

daily_sales = (
    df.groupby(df["InvoiceDate"].dt.normalize())["TotalSales"]
    .sum()
    .sort_index()
)

# Include dates with zero sales
daily_sales = daily_sales.asfreq("D", fill_value=0)

print("Historical daily sales:")
print(daily_sales.tail())

# =====================================
# 4. SARIMA MODEL
# =====================================

print("\nTraining SARIMA model...")

model = SARIMAX(
    daily_sales,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

# =====================================
# 5. NEXT 30 DAYS FORECAST
# =====================================

forecast_result = model_fit.get_forecast(
    steps=30
)

forecast_values = forecast_result.predicted_mean

forecast_dates = pd.date_range(
    start=daily_sales.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "ForecastSales": forecast_values.to_numpy()
})

forecast_df["ForecastSales"] = (
    forecast_df["ForecastSales"].clip(lower=0)
)

# Save forecast results
output_path = BASE_DIR / "sarima_next_30_days.csv"

forecast_df.to_csv(
    output_path,
    index=False
)

print("\nNext 30 days SARIMA forecast:")
print(forecast_df.to_string(index=False))

print(f"\nForecast saved to: {output_path}")

# =====================================
# 6. PLOT FORECAST
# =====================================

plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales.index[-60:],
    daily_sales.iloc[-60:],
    label="Historical Sales"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["ForecastSales"],
    color="red",
    marker="o",
    markersize=3,
    label="SARIMA Forecast"
)

plt.title("RetailPulse - SARIMA Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "sarima_forecast.png",
    dpi=150
)

plt.show()