
import pandas as pd

# Step 1: Load original dataset
df = pd.read_csv("online_retail.csv", encoding="utf-8-sig")
print("Original dataset shape:", df.shape)

# Step 2: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Step 3: Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Step 4: Remove duplicate rows
df = df.drop_duplicates()

# Step 5: Save cleaned dataset
df.to_csv("retail_cleaned_final.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("Cleaned dataset saved successfully!")

# Verify cleaned dataset
print("\nMissing values after duplicate removal:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nFirst 5 rows of cleaned dataset:")
print(df.head())

print("\nFinal dataset shape:")
print(df.shape)
