import pandas as pd

# 1. Load combined dataset
df = pd.read_csv(
    "data/retailpulse_combined.csv",
    low_memory=False
)

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Convert Quantity to numeric
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

# 4. Remove missing values
df = df.dropna(subset=["StockCode", "Quantity"])

# 5. Remove cancelled or invalid sales
df = df[df["Quantity"] > 0]

# 6. Calculate total quantity sold per product
product_sales = df.groupby("StockCode")["Quantity"].transform("sum")

# 7. Calculate demand thresholds
low = product_sales.quantile(0.33)
high = product_sales.quantile(0.67)

# 8. Add Stock_Status column
df["Stock_Status"] = product_sales.apply(
    lambda x: "Low Demand" if x <= low
    else "Medium Demand" if x <= high
    else "High Demand"
)

# 9. Save updated dataset
df.to_csv(
    "data/retailpulse_combined_with_stock_status.csv",
    index=False
)

# 10. Display results
print("Stock_Status column added successfully!")
print(df[["StockCode", "Quantity", "Stock_Status"]].head(10))

print("\nStock Status Summary:")
print(df.groupby("Stock_Status")["StockCode"].nunique())

print("\nSaved: data/retailpulse_combined_with_stock_status.csv")