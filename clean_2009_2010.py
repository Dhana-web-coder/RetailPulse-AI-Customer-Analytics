import pandas as pd

# Load the 2009-2010 dataset
df = pd.read_csv("data/retail_2009_2010.csv")

print("Original Dataset Shape:", df.shape)

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing essential values
df = df.dropna(
    subset=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"]
)

# Remove cancelled invoices
df = df[
    ~df["Invoice"].astype(str).str.startswith("C")
]

# Remove invalid quantity and price values
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
]

# Save cleaned dataset
df.to_csv(
    "data/retail_cleaned_2009_2010.csv",
    index=False
)

print("Cleaned Dataset Shape:", df.shape)
print("Missing Dates:", df["InvoiceDate"].isna().sum())
print("Start Date:", df["InvoiceDate"].min())
print("End Date:", df["InvoiceDate"].max())
print("Cleaned dataset saved successfully!")