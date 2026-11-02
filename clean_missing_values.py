import pandas as pd
from pathlib import Path

# Dataset path
file_path = Path("data/retail_cleaned_2009_2010.csv")

# Check whether the file exists
if not file_path.exists():
    print("ERROR: Dataset file not found:", file_path)
    raise SystemExit(1)

# Load dataset
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Fill missing product descriptions using StockCode
if "Description" in df.columns and "StockCode" in df.columns:
    df["Description"] = df["Description"].fillna(
        df.groupby("StockCode")["Description"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
        )
    )

# Keep missing Customer ID values as missing.
# They may still be useful for sales analysis.
# Customer analysis should exclude rows without Customer ID.

# Save cleaned dataset as a new file
output_path = Path("data/retail_missing_fixed.csv")
df.to_csv(output_path, index=False)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)
print("\nSaved successfully to:", output_path)