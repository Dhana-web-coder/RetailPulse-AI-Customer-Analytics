import pandas as pd

# Load combined dataset
df = pd.read_csv("data/retailpulse_combined.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

print("========== RETAILPULSE DATASET VERIFICATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDate Range:")
print("Start:", df["InvoiceDate"].min())
print("End:", df["InvoiceDate"].max())

print("\nTotal Sales:")
print(round((df["Quantity"] * df["Price"]).sum(), 2))

print("\nVerification completed!")