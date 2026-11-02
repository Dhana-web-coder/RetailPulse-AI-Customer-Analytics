import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load dataset
df = pd.read_csv("data/retailpulse_combined.csv")

# Step 2: Convert date column
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

# Step 3: Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# Step 4: Remove invalid data
df = df.dropna(subset=["InvoiceDate", "Sales"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Step 5: Calculate daily sales
daily_sales = (
    df.set_index("InvoiceDate")["Sales"]
    .resample("D")
    .sum()
)

# Step 6: Fill days without sales
daily_sales = daily_sales.asfreq("D", fill_value=0)

print("Total Days:", len(daily_sales))
print("\nLast 5 Days Sales:")
print(daily_sales.tail())

# Step 7: Split data into training and testing
test_days = 30

train_data = daily_sales.iloc[:-test_days]
test_data = daily_sales.iloc[-test_days:]

print("\nTraining Days:", len(train_data))
print("Testing Days:", len(test_data))

# Step 8: Train ARIMA model using training data only
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# Step 9: Predict the testing period
test_forecast = model_fit.forecast(steps=test_days)
test_forecast.index = test_data.index

# Step 10: Evaluate model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test_data, test_forecast)
rmse = np.sqrt(mean_squared_error(test_data, test_forecast))

print("\n--- ARIMA Model Evaluation ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Step 11: Plot actual vs predicted sales
plt.figure(figsize=(12, 6))

plt.plot(
    test_data.index,
    test_data.values,
    label="Actual Sales"
)

plt.plot(
    test_forecast.index,
    test_forecast.values,
    label="Predicted Sales"
)

plt.title("RetailPulse - ARIMA Model Evaluation")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 12: Train final model using all available data
final_model = ARIMA(daily_sales, order=(1, 1, 1))
final_model_fit = final_model.fit()

# Step 13: Forecast next 30 days
forecast = final_model_fit.forecast(steps=30)

print("\n--- Next 30 Days Sales Forecast ---")
print(forecast.round(2))

# Step 14: Save forecast results
forecast_df = forecast.rename("ForecastSales").reset_index()
forecast_df.to_csv("arima_forecast_30_days.csv", index=False)

print("\nForecast saved to arima_forecast_30_days.csv")