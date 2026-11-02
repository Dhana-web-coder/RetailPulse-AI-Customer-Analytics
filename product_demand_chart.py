import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/retailpulse_combined_with_stock_status.csv"
)

# Product-wise total quantity sold
product_sales = (
    df.groupby(["StockCode", "Description", "Stock_Status"])["Quantity"]
    .sum()
    .reset_index()
)

# Top 10 high-demand products
top_products = (
    product_sales[product_sales["Stock_Status"] == "High Demand"]
    .sort_values("Quantity", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.barh(
    top_products["Description"].fillna("Unknown"),
    top_products["Quantity"]
)

plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Name")
plt.title("Top 10 High-Demand Products")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("top_10_high_demand_products.png")
plt.show()

print(top_products)