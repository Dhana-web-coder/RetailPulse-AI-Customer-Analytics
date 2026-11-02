import pandas as pd

comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA"],
    "MAE": [30339.33, 34945.81],
    "RMSE": [32736.82, 37582.52]
})

comparison = comparison.sort_values(by="MAE")

comparison.to_csv("model_comparison.csv", index=False)

print("Model Comparison:")
print(comparison.to_string(index=False))
print("\nSaved: model_comparison.csv")
print("\nBest model for this test: ARIMA")
import pandas as pd
import matplotlib.pyplot as plt

# Model evaluation results
comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA"],
    "MAE": [30339.33, 34945.81],
    "RMSE": [32736.82, 37582.52]
})

# Display comparison
print(comparison)

# Create comparison chart
comparison.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("RetailPulse - ARIMA vs SARIMA Model Comparison")
plt.xlabel("Forecasting Model")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(title="Evaluation Metrics")
plt.tight_layout()

# Save chart
plt.savefig("model_comparison_chart.png")
plt.show()

print("Chart saved: model_comparison_chart.png")