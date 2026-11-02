import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load forecast files
arima_path = BASE_DIR / "arima_next_30_days.csv"
sarima_path = BASE_DIR / "sarima_next_30_days.csv"

if not arima_path.exists():
    raise FileNotFoundError(
        f"ARIMA forecast file not found: {arima_path}"
    )

if not sarima_path.exists():
    raise FileNotFoundError(
        f"SARIMA forecast file not found: {sarima_path}"
    )

arima = pd.read_csv(arima_path)
sarima = pd.read_csv(sarima_path)

arima["Date"] = pd.to_datetime(arima["Date"])
sarima["Date"] = pd.to_datetime(sarima["Date"])

arima = arima.rename(
    columns={"PredictedSales": "ARIMA_Forecast"}
)

sarima = sarima.rename(
    columns={"ForecastSales": "SARIMA_Forecast"}
)

sarima = sarima.rename(
    columns={"ForecastSales": "SARIMA_Forecast"}
)

# Match forecasts by date
comparison = pd.merge(
    arima[["Date", "ARIMA_Forecast"]],
    sarima[["Date", "SARIMA_Forecast"]],
    on="Date",
    how="inner"
)

if comparison.empty:
    raise ValueError(
        "No matching forecast dates found."
    )

# Display comparison
print("\nARIMA vs SARIMA Forecast Comparison")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    BASE_DIR / "arima_sarima_forecast_comparison.csv",
    index=False
)

#Plot comparison

plt.close("all")

plt.figure(figsize=(12, 6))

plt.plot(
comparison["Date"],
comparison["ARIMA_Forecast"],
marker="o",
markersize=3,
label="ARIMA Forecast"
)

plt.plot(
comparison["Date"],
comparison["SARIMA_Forecast"],
marker="o",
markersize=3,
label="SARIMA Forecast"
)

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig(
BASE_DIR / "arima_sarima_forecast_comparison.png",
dpi=150
)

plt.show()