import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load cleaned dataset
df = pd.read_csv(
    "retail_cleaned_final.csv",
    low_memory=False
)

# Step 2: Convert date column
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

# Step 3: Calculate total sales
df["TotalSales"] = df["Quantity"] * df["Price"]

# Step 4: Remove invalid dates
df = df.dropna(subset=["InvoiceDate"])

# Step 5: Keep valid sales only
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
]

# Step 6: Create daily sales
daily_sales = (
    df.set_index("InvoiceDate")["TotalSales"]
    .resample("D")
    .sum()
)

# Step 7: Fill dates without sales
daily_sales = daily_sales.asfreq("D", fill_value=0)

print("Daily Sales:")
print(daily_sales.head())

# Step 8: Build ARIMA model
model = ARIMA(
    daily_sales,
    order=(1, 1, 1)
)

model_fit = model.fit()

# Step 9: Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Step 10: Save forecast results
forecast_df = forecast.reset_index()
forecast_df.columns = ["Date", "PredictedSales"]

forecast_df.to_csv(
    "data/sales_forecast.csv",
    index=False
)

print("\nNext 30 Days Sales Forecast:")
print(forecast_df)

print("\nForecast saved successfully!")

# Step 11: Plot graph
plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales.index[-60:],
    daily_sales.tail(60),
    label="Actual Sales"
)

plt.plot(
    forecast.index,
    forecast,
    label="Predicted Sales"
)

plt.title("RetailPulse - Sales Forecasting")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()