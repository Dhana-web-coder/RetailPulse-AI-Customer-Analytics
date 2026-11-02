import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# 4. Daily sales
daily_sales = df.set_index("InvoiceDate")["Sales"].resample("D").sum()

# 5. Split data
train = daily_sales.iloc[:-30]
test = daily_sales.iloc[-30:]

# 6. Train SARIMA model
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

# 7. Predict
predictions = model_fit.forecast(steps=len(test))
predictions.index = test.index

# 8. Evaluate
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("SARIMA Evaluation Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# 9. Save results
results = pd.DataFrame({
    "Date": test.index,
    "Actual Sales": test.values,
    "Predicted Sales": predictions.values
})

results.to_csv("sarima_evaluation_results.csv", index=False)
print("Results saved: sarima_evaluation_results.csv")

# 10. Plot chart
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label="Actual Sales")
plt.plot(predictions.index, predictions.values, label="SARIMA Predicted Sales")

plt.title("RetailPulse - SARIMA Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("sarima_evaluation_chart.png")
plt.show()

print("Chart saved: sarima_evaluation_chart.png")
