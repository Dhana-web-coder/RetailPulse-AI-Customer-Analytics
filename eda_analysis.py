import pandas as pd
import matplotlib.pyplot as plt

# Load combined dataset
df = pd.read_csv("data/retailpulse_combined.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Calculate sales amount
df["Sales"] = df["Quantity"] * df["Price"]

# Remove invalid records for this analysis
df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Extract month
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

print("Total Transactions:", df["Invoice"].nunique())
print("Total Products:", df["StockCode"].nunique())
print("Total Customers:", df["Customer ID"].nunique())
print("Total Sales:", round(df["Sales"].sum(), 2))

# Monthly sales
monthly_sales = df.groupby("Month")["Sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Plot monthly sales
monthly_sales.plot(kind="line", marker="o")
plt.title("RetailPulse - Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 products by quantity sold
top_products = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products by Quantity Sold:")
print(top_products)

top_products.plot(kind="bar")
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Product Code")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.show()