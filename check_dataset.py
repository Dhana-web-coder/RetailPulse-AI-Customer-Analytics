import pandas as pd

df = pd.read_csv("data/retail_2009_2010.csv")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

print("Dataset Shape:", df.shape)
print("Start Date:", df["InvoiceDate"].min())
print("End Date:", df["InvoiceDate"].max())

print("\nColumns:")
print(df.columns.tolist())