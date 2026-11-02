import pandas as pd
import matplotlib.pyplot as plt

forecast = pd.read_csv("arima_forecast_30_days.csv")

forecast["index"] = pd.to_datetime(forecast["index"])

plt.figure(figsize=(12, 6))
plt.plot(
forecast["index"],
forecast["ForecastSales"],
marker="o",
label="ARIMA Forecast"
)

plt.title("RetailPulse - 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Forecast Sales")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("arima_forecast_graph.png")
plt.show()

print("Forecast graph saved: arima_forecast_graph.png")