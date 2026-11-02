import pandas as pd

# Load both datasets
new_df = pd.read_csv("data/retail_cleaned_2009_2010.csv")
old_df = pd.read_csv("retail_cleaned_final.csv")

print("NEW RAW DATES:")
print(new_df["InvoiceDate"].head(10).to_list())

print("\nOLD RAW DATES:")
print(old_df["InvoiceDate"].head(10).to_list())
# Convert dates correctly
new_df["InvoiceDate"] = pd.to_datetime(
    new_df["InvoiceDate"], dayfirst=True, errors="coerce"
)

old_df["InvoiceDate"] = pd.to_datetime(
    old_df["InvoiceDate"], dayfirst=True, errors="coerce"
)

# Make Invoice columns the same data type
new_df["Invoice"] = new_df["Invoice"].astype(str).str.strip()
old_df["Invoice"] = old_df["Invoice"].astype(str).str.strip()

# Display dataset information
print("NEW DATASET")
print("Rows:", len(new_df))
print("Start:", new_df["InvoiceDate"].min())
print("End:", new_df["InvoiceDate"].max())

print("\nOLD DATASET")
print("Rows:", len(old_df))
print("Start:", old_df["InvoiceDate"].min())
print("End:", old_df["InvoiceDate"].max())

# Check exact matching rows
print("\nChecking exact duplicate rows...")

new_compare = new_df.drop_duplicates()
old_compare = old_df.drop_duplicates()

duplicates = pd.merge(
    new_compare,
    old_compare,
    how="inner"
)

print("Exact matching rows:", len(duplicates))