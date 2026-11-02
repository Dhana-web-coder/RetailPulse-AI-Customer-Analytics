import pandas as pd

df = pd.read_csv("online_retail.csv")
# Original data size check
print("Original data shape:", df.shape)

# 1. Duplicate rows check
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Duplicate rows remove pannrom
df_clean = df.drop_duplicates().copy()

# 2. Missing values check
print("\nMissing values:")
print(df_clean.isnull().sum())

# 3. Negative quantity check
print("\nNegative quantity rows:")
print((df_clean["Quantity"] < 0).sum())

# 4. Zero price check
print("\nZero price rows:")
print((df_clean["Price"] == 0).sum())

# Negative price check
print("\nNegative price rows:")
print((df_clean["Price"] < 0).sum())

# 5. Cancelled invoice check
print("\nCancelled invoices:")
print(
    df_clean["Invoice"]
    .astype(str)
    .str.startswith("C")
    .sum()
)

# 6. Cleaned dataset separate-a save pannrom
df_clean.to_csv("cleaned_retail_data.csv", index=False)

print("\nCleaned CSV saved successfully!")
print("Final data shape:", df_clean.shape)

# Create a copy of the original dataset
df_clean = df.copy()

# Count duplicate rows
print("Duplicate rows before cleaning:", df_clean.duplicated().sum())

# Remove duplicate rows
df_clean = df_clean.drop_duplicates().copy()

# Check the result
print("Duplicate rows after cleaning:", df_clean.duplicated().sum())

print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)
# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Create a separate dataset for customer analysis
customer_df = df.dropna(subset=["Customer ID"]).copy()

# Clean product descriptions
customer_df["Description"] = customer_df["Description"].str.strip()

# Remove rows with missing descriptions
customer_df = customer_df.dropna(subset=["Description"])

# Display results
print("Original dataset shape:", df.shape)
print("Customer analysis dataset shape:", customer_df.shape)

print("Remaining missing values:")
print(customer_df[["Customer ID", "Description"]].isnull().sum())
print("Column names:")
print(df.columns.tolist())
print("Actual column names:")
print(df.columns.tolist())
print("Actual column names:")
print(df.columns.tolist())

print(df.head())
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nFirst 5 Rows:")
print(df.head())
print("Missing Values:")
print(df.isnull().sum())
# Missing values analysis

missing_values = df.isnull().sum()

missing_percentage = (missing_values / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_report[missing_report["Missing Values"] > 0])

print("Column names:", df.columns.tolist())
print(df.head())

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Create a separate dataset for customer analysis

print("ACTUAL COLUMN NAMES:")
print(df.columns.tolist())

# customer_df = df.dropna(subset=["CustomerID"]).copy()

print("COLUMN NAMES:", [repr(col) for col in df.columns])

exit()

customer_df = df_clean.dropna(subset=["Customer ID"]).copy()
# Clean product descriptions for the customer dataset

print("Available columns:")
print(df.columns.tolist())

customer_df["Description"] = customer_df["Description"].str.strip()

# Remove rows with missing descriptions from customer analysis
customer_df = customer_df.dropna(subset=["Description"])

# Display results
print("Original dataset shape:", df.shape)
print("Customer analysis dataset shape:", customer_df.shape)

print("Remaining missing values:")
print(customer_df[["CustomerID", "Description"]].isnull().sum())

# Check actual column names
print("Dataset columns:", df.columns.tolist())

# Remove extra spaces from column names
df.columns = df.columns.astype(str).str.strip()

# Check again
print("Cleaned columns:", df.columns.tolist())
print("Column Names:")
print(df.columns.tolist())
df = pd.read_excel("data/Online_Retail.xlsx")

print(df.columns.tolist())
