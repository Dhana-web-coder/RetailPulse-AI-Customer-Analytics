import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
df = pd.read_csv("data/retail_2009_2010.csv", low_memory=False)

# 2. Clean data
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)
df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "Price"]
)
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
]

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

# 6. Test different ARIMA parameters
orders = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (3, 1, 0),
    (3, 1, 1),
    (5, 1, 0)
]

results = []

for order in orders:
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()

        predictions = model_fit.forecast(
            steps=len(test)
        )

        mae = mean_absolute_error(
            test, predictions
        )

        results.append({
            "ARIMA_Order": str(order),
            "MAE": round(mae, 2)
        })

        print(
            f"ARIMA {order} | MAE: {mae:.2f}"
        )

    except Exception as e:
        print(f"ARIMA {order} failed: {e}")

# 7. Save comparison results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE")

    results_df.to_csv(
        "arima_model_comparison.csv",
        index=False
    )

    print("\n--- ARIMA Model Comparison ---")
    print(results_df.to_string(index=False))
    print("\nBest model based on MAE:")
    print(results_df.iloc[0])
    print("\nSaved: arima_model_comparison.csv")
else:
    print("No ARIMA model completed successfully.")