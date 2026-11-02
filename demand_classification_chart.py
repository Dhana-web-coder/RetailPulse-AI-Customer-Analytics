import pandas as pd
import matplotlib.pyplot as plt

# Load updated dataset
df = pd.read_csv(
    "data/retailpulse_combined_with_stock_status.csv"
)

# Count unique products in each category
summary = (
    df[["StockCode", "Stock_Status"]]
    .drop_duplicates("StockCode")
    ["Stock_Status"]
    .value_counts()
    .reindex(
        ["High Demand", "Medium Demand", "Low Demand"],
        fill_value=0
    )
)

# Create bar chart
summary.plot(
    kind="bar",
    title="Product Demand Classification"
)

plt.xlabel("Demand Category")
plt.ylabel("Number of Products")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("demand_classification_chart.png")
plt.show()

print("\nDemand Classification Summary:")
print(summary)
print("\nChart saved successfully!")