import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv(
    "data/retailpulse_combined.csv",
    low_memory=False
)

# Step 2: Calculate sales
df["Sales"] = df["Quantity"] * df["Price"]

# Step 3: Remove invalid records
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# Step 4: Product-wise analysis
product_sales = df.groupby(
    ["StockCode", "Description"],
    dropna=False
).agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Sales=("Sales", "sum")
).reset_index()

# Step 5: Find top 10 products
top_products = product_sales.sort_values(
    "Total_Quantity",
    ascending=False
).head(10)

print("\nTop 10 Products by Quantity Sold:")
print(top_products)

# Step 6: Plot chart
plt.figure(figsize=(12, 6))

plt.bar(
    top_products["StockCode"].astype(str),
    top_products["Total_Quantity"]
)

plt.title("RetailPulse - Top 10 Products by Quantity Sold")
plt.xlabel("Product Code")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 7: Save results
product_sales.to_csv(
    "inventory_analysis.csv",
    index=False
)

print("\nInventory analysis saved successfully!")
# Step 8: Calculate Average Daily Demand

# Convert invoice date
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

# Calculate daily quantity sold for each product
daily_product_sales = df.groupby(
    [
        "StockCode",
        "Description",
        df["InvoiceDate"].dt.date
    ],
    dropna=False
)["Quantity"].sum().reset_index()

daily_product_sales.columns = [
    "StockCode",
    "Description",
    "Date",
    "Daily_Quantity"
]

# Calculate average daily demand
average_demand = daily_product_sales.groupby(
    ["StockCode", "Description"],
    dropna=False
)["Daily_Quantity"].mean().reset_index()

average_demand.rename(
    columns={"Daily_Quantity": "Average_Daily_Demand"},
    inplace=True
)

# Step 9: Set lead time to 7 days
lead_time = 7

# Step 10: Calculate reorder point
average_demand["Lead_Time"] = lead_time

average_demand["Reorder_Point"] = (
    average_demand["Average_Daily_Demand"] * lead_time
).round(2)

# Step 11: Display products with highest reorder points
print("\nInventory Reorder Point Analysis:")

print(
    average_demand.sort_values(
        "Reorder_Point",
        ascending=False
    ).head(10).to_string(index=False)
)

# Step 12: Save results
average_demand.to_csv(
    "inventory_reorder_points.csv",
    index=False
)

print("\nReorder point results saved successfully!")
# Inventory Optimization Chart

import matplotlib.pyplot as plt

top_products = product_sales.sort_values(
    by="Total_Quantity",
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))
plt.bar(
    top_products["StockCode"].astype(str),
    top_products["Total_Quantity"]
)
plt.title("Top 10 Products - Inventory Analysis")
plt.xlabel("Stock Code")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Step 6: Inventory Optimization Chart

top_products = product_sales.sort_values(
    by="Total_Quantity",
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_products["StockCode"].astype(str),
    top_products["Total_Quantity"]
)

plt.title("Top 10 Products - Inventory Analysis")
plt.xlabel("Product Stock Code")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()