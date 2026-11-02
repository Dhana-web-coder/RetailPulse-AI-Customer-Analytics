import pandas as pd
from pathlib import Path

# Load both cleaned datasets
new_df = pd.read_csv("data/retail_cleaned_2009_2010.csv")
old_df = pd.read_csv("retail_cleaned_final.csv")

# Convert dates using the correct formats
new_df["InvoiceDate"] = pd.to_datetime(
    new_df["InvoiceDate"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

old_df["InvoiceDate"] = pd.to_datetime(
    old_df["InvoiceDate"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Remove rows with invalid dates
new_df = new_df.dropna(subset=["InvoiceDate"])
old_df = old_df.dropna(subset=["InvoiceDate"])

# Keep only the required date ranges to avoid overlap
new_df = new_df[
    (new_df["InvoiceDate"] >= "2009-12-01") &
    (new_df["InvoiceDate"] < "2010-12-01")
]

old_df = old_df[
    (old_df["InvoiceDate"] >= "2010-12-01") &
    (old_df["InvoiceDate"] < "2012-01-01")
]

# Combine datasets
combined_df = pd.concat(
    [new_df, old_df],
    ignore_index=True
)

# Sort by date
combined_df = combined_df.sort_values("InvoiceDate")

# Save the combined dataset
Path("data").mkdir(exist_ok=True)

combined_df.to_csv(
    "data/retailpulse_combined.csv",
    index=False
)

# Display results
print("Combined dataset created successfully!")
print("Total rows:", len(combined_df))
print("Total columns:", len(combined_df.columns))
print("Start date:", combined_df["InvoiceDate"].min())
print("End date:", combined_df["InvoiceDate"].max())
print("Missing dates:", combined_df["InvoiceDate"].isna().sum())
print("Saved file: data/retailpulse_combined.csv")