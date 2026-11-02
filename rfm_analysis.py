import pandas as pd

# Step 1: Load dataset
df = pd.read_csv(
    "data/retailpulse_combined.csv",
    low_memory=False
)

# Step 2: Convert columns
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
)
df["Sales"] = df["Quantity"] * df["Price"]

# Step 3: Remove records without customer/date
df = df.dropna(subset=["Customer ID", "InvoiceDate"])

# Step 4: Keep valid purchases
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# Step 5: Set analysis date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Step 6: Calculate RFM
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x:
             (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Sales", "sum")
)

# Step 7: Display results
print("\nRFM Analysis - First 10 Customers")
print(rfm.head(10))

print("\nTotal Customers:", len(rfm))

# Step 8: Save results
rfm.to_csv("rfm_results.csv")

print("\nRFM results saved successfully!")
import matplotlib.pyplot as plt

# Top 10 customers by Monetary Value
top_customers = rfm.sort_values(
    by="Monetary", ascending=False
).head(10)

plt.figure(figsize=(10, 5))
plt.bar(
    top_customers.index.astype(str),
    top_customers["Monetary"]
)

plt.title("Top 10 Customers by Monetary Value")
plt.xlabel("Customer ID")
plt.ylabel("Monetary Value (Total Sales)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Step 9: Customer Segmentation

# Assign scores from 1 to 4
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    4,
    labels=[4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

# Calculate total RFM score
rfm["RFM_Score"] = (
    rfm["R_Score"] +
    rfm["F_Score"] +
    rfm["M_Score"]
)

# Create customer segments
def segment_customer(score):
    if score >= 10:
        return "Best Customers"
    elif score >= 8:
        return "Loyal Customers"
    elif score >= 5:
        return "Regular Customers"
    else:
        return "Low Engagement"

rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)

# Display results
print("\nCustomer Segmentation:")
print(rfm[[
    "Recency",
    "Frequency",
    "Monetary",
    "RFM_Score",
    "Segment"
]].head(10))

print("\nCustomer Count by Segment:")
print(rfm["Segment"].value_counts())

# Save final RFM results
rfm.to_csv("rfm_results.csv")

print("\nRFM Customer Segmentation completed!")
print("Results saved to rfm_results.csv")
import matplotlib.pyplot as plt

# Count customers in each segment
segment_counts = rfm["Segment"].value_counts()

# Create bar chart
plt.figure(figsize=(10, 5))
plt.bar(segment_counts.index, segment_counts.values)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.show()
segment_counts = rfm["Segment"].value_counts()

plt.figure(figsize=(10, 5))
plt.bar(segment_counts.index, segment_counts.values)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
# Customer Segmentation Pie Chart

segment_counts = rfm["Segment"].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Segmentation - Percentage")
plt.tight_layout()
plt.show()