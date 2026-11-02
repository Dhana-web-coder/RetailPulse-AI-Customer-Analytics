import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load dataset
df = pd.read_csv("data/retailpulse_combined.csv", low_memory=False)

# 2. Clean data
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# 3. Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# 4. Calculate daily sales
daily_sales = (
    df.set_index("InvoiceDate")["Sales"]
    .resample("D")
    .sum()
)

# 5. Split training and testing data
train = daily_sales.iloc[:-30]
test = daily_sales.iloc[-30:]

print("Training days:", len(train))
print("Testing days:", len(test))

# 6. Train ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# 7. Predict the last 30 days
predictions = model_fit.forecast(steps=len(test))
predictions.index = test.index

# 8. Evaluate model
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

# Calculate MAPE (ignore days with zero actual sales)
nonzero = test != 0

if nonzero.any():
    mape = np.mean(
        np.abs(
            (test[nonzero] - predictions[nonzero])
            / test[nonzero]
        )
    ) * 100
else:
    mape = np.nan

print("\n--- ARIMA Evaluation Results ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%" if not np.isnan(mape) else "(not available)")
# 9. Save results
results = pd.DataFrame({
    "Date": test.index,
    "Actual Sales": test.values,
    "Predicted Sales": predictions.values
})

results.to_csv("arima_evaluation_results.csv", index=False)
print("\nResults saved successfully!")

# 10. Plot chart
plt.figure(figsize=(12, 6))

plt.plot(test.index, test.values, label="Actual Sales", marker="o")
plt.plot(predictions.index, predictions.values,
         label="Predicted Sales", marker="o")

plt.title("RetailPulse - Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("arima_evaluation_chart.png")
plt.show()

print("Chart saved: arima_evaluation_chart.png")
