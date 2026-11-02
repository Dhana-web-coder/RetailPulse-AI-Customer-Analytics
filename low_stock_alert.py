import pandas as pd
df = pd.read_csv("current_stock_analysis.csv")
low_stock = df[df["Stock_Status"] == "Low Stock"].copy()
low_stock["Recommended_Order_Quantity"] = (
    low_stock["Reorder_Point"] - low_stock["Current_Stock"]
).clip(lower=0).round(2)
print("LOW STOCK ALERT")
if low_stock.empty:
    print("No low-stock products found.")
else:
    print(low_stock[[
        "StockCode",
        "Description",
        "Current_Stock",
        "Reorder_Point",
        "Recommended_Order_Quantity"
    ]].to_string(index=False))
low_stock.to_csv("low_stock_alerts.csv", index=False)
print("Low stock alerts saved successfully!")
