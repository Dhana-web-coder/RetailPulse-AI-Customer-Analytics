import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "data/retailpulse_combined_with_stock_status.csv"
)

# Remove cancelled/return transactions and invalid quantities
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df = df.dropna(subset=["StockCode", "Quantity"])
df = df[df["Quantity"] > 0]

# Calculate total quantity sold for each product
product_sales = (
    df.groupby(["StockCode", "Description"])["Quantity"]
    .sum()
    .reset_index()
)

# Find 10 products with the lowest total sales
slow_products = product_sales.sort_values(
    "Quantity", ascending=True
).head(10)

print("Top 10 Slow-Moving Products:")
print(slow_products.to_string(index=False))

# Create chart
plt.figure(figsize=(12, 7))

plt.barh(
    slow_products["Description"].fillna("Unknown"),
    slow_products["Quantity"]
)

plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Name")
plt.title("Top 10 Slow-Moving Products")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save chart
plt.savefig("slow_moving_products.png", dpi=300)

plt.show()

print("\nChart saved as slow_moving_products.png")