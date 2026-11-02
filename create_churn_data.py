import pandas as pd

# Load dataset
df = pd.read_csv(
    "retail_cleaned_final.csv",
    low_memory=False
)

# Convert columns
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Customer ID"] = pd.to_numeric(
    df["Customer ID"],
    errors="coerce"
)

# Remove invalid records
df = df.dropna(
    subset=["Customer ID", "InvoiceDate"]
)

# Keep valid purchases
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# Check latest date
last_date = df["InvoiceDate"].max()

# Reserve the last 90 days for churn observation
observation_end = last_date - pd.Timedelta(days=90)

# Customer's last purchase during observation period
observation_df = df[
    df["InvoiceDate"] <= observation_end
].copy()

# Customer purchases during the next 90 days
future_df = df[
    (df["InvoiceDate"] > observation_end) &
    (df["InvoiceDate"] <= last_date)
].copy()

# Build customer features from observation period
features = observation_df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x:
             (observation_end - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Quantity", lambda x: 0)
).reset_index()

# Calculate actual spending
sales = observation_df.assign(
    Sales=observation_df["Quantity"] * observation_df["Price"]
)

monetary = sales.groupby("Customer ID")["Sales"].sum()

features["Monetary"] = features["Customer ID"].map(monetary)

# Customers who purchased in the future 90-day period
future_customers = future_df["Customer ID"].unique()

# Create churn label
features["Churn"] = (
    ~features["Customer ID"].isin(future_customers)
).astype(int)

# Save dataset
features.to_csv(
    "customer_churn_data.csv",
    index=False
)

print("Churn dataset created successfully!")
print("Observation period ends:", observation_end)
print("Prediction period starts after:", observation_end)
print("Prediction period ends:", last_date)
print("Customers:", len(features))
print("\nChurn labels:")
print(features["Churn"].value_counts())
print("\nSaved file: customer_churn_data.csv")