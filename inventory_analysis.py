import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/retail_2009_2010.csv", low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Convert columns
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

# Remove missing and invalid values
df = df.dropna(subset=["InvoiceDate", "StockCode", "Quantity"])
df = df[df["Quantity"] > 0]

# Product-wise inventory movement
inventory = df.groupby("StockCode").agg(
    Total_Quantity_Sold=("Quantity", "sum"),
    Total_Invoices=("Invoice", "nunique"),
    Last_Sale_Date=("InvoiceDate", "max")
).reset_index()

# Calculate days since last sale
analysis_date = df["InvoiceDate"].max()
inventory["Days_Since_Last_Sale"] = (
    analysis_date - inventory["Last_Sale_Date"]
).dt.days

# Classify products based on sales movement
def classify_product(row):
    if row["Days_Since_Last_Sale"] > 90:
        return "Slow-Moving"
    elif row["Total_Quantity_Sold"] < 20:
        return "Low-Demand"
    else:
        return "Active"

inventory["Inventory_Status"] = inventory.apply(
    classify_product, axis=1
)

# Save results
inventory.to_csv("inventory_analysis_results.csv", index=False)

# Plot inventory status
status_counts = inventory["Inventory_Status"].value_counts()

status_counts.plot(kind="bar", title="Product Inventory Movement")
plt.xlabel("Inventory Status")
plt.ylabel("Number of Products")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("inventory_analysis_chart.png")
plt.show()

print("Inventory Analysis Completed!")
print(inventory.head(10).to_string(index=False))
print("\nInventory Status Summary:")
print(status_counts)
print("\nSaved: inventory_analysis_results.csv")