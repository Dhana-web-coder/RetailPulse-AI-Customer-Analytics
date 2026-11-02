import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="Order and Transaction Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Order and Transaction Analysis Dashboard")
st.write("Analyze orders and customer transactions.")

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILES = [
    BASE_DIR / "data" / "retailpulse_combined_with_stock_status.csv",
    BASE_DIR / "data" / "retailpulse_combined.csv",
    BASE_DIR / "retailpulse_combined.csv",
    BASE_DIR / "data" / "retail_2009_2010.csv"
]

df = None

for file in DATA_FILES:
    if file.exists():
        df = pd.read_csv(file)
        break

if df is None:
    st.error("Dataset not found. Please check the data folder.")
    st.stop()

required_columns = [
    "Invoice", "Quantity", "Customer ID",
    "InvoiceDate", "Description", "Country"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

st.success("Dataset loaded successfully!")
# Order and Transaction KPIs

st.subheader("📊 Order Overview")

# Remove cancelled invoices (invoice numbers starting with C)
df["Invoice"] = df["Invoice"].astype(str)

valid_df = df[
    ~df["Invoice"].str.startswith("C")
].copy()

total_orders = valid_df["Invoice"].nunique()

total_transactions = len(valid_df)

total_customers = valid_df["Customer ID"].nunique()

avg_items = (
    valid_df.groupby("Invoice")["Quantity"].sum().mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Transactions", f"{total_transactions:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Average Items per Order", f"{avg_items:.2f}")
st.subheader("📅 Daily Order Trend")

valid_df["InvoiceDate"] = pd.to_datetime(
    valid_df["InvoiceDate"],
    errors="coerce"
)

daily_orders = (
    valid_df.dropna(subset=["InvoiceDate"])
    .groupby(valid_df["InvoiceDate"].dt.date)["Invoice"]
    .nunique()
    .reset_index(name="Total Orders")
)

daily_orders.rename(
    columns={"InvoiceDate": "Date"},
    inplace=True
)

fig = px.line(
    daily_orders,
    x="Date",
    y="Total Orders",
    markers=True,
    title="Daily Order Trend"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("📆 Monthly Order Analysis")

valid_df["Month"] = (
    valid_df["InvoiceDate"].dt.to_period("M").astype(str)
)

monthly_orders = (
    valid_df.groupby("Month")["Invoice"]
    .nunique()
    .reset_index(name="Total Orders")
)

fig = px.bar(
    monthly_orders,
    x="Month",
    y="Total Orders",
    title="Monthly Order Analysis",
    text="Total Orders"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("🏆 Top 10 Products by Orders")

product_orders = (
    valid_df.groupby("Description")["Invoice"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="Total Orders")
)

fig = px.bar(
    product_orders,
    x="Total Orders",
    y="Description",
    orientation="h",
    title="Top 10 Products by Orders"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("🌍 Country-wise Order Analysis")

country_orders = (
    valid_df.groupby("Country")["Invoice"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="Total Orders")
)

fig = px.bar(
    country_orders,
    x="Country",
    y="Total Orders",
    title="Top 10 Countries by Orders",
    text="Total Orders"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("❌ Cancelled Orders Analysis")

cancelled_df = df[
    df["Invoice"].str.startswith("C", na=False)
].copy()

cancelled_orders = cancelled_df["Invoice"].nunique()

col1, col2 = st.columns(2)

col1.metric("Cancelled Orders", f"{cancelled_orders:,}")

col2.metric(
    "Cancellation Rate",
    f"{(cancelled_orders / df['Invoice'].nunique() * 100):.2f}%"
)

if not cancelled_df.empty:
    st.write("Cancelled Orders by Country")

    cancellation_chart = (
        cancelled_df.groupby("Country")["Invoice"]
        .nunique()
        .reset_index(name="Cancelled Orders")
        .sort_values("Cancelled Orders", ascending=False)
        .head(10)
    )

    fig = px.bar(
        cancellation_chart,
        x="Country",
        y="Cancelled Orders",
        title="Top 10 Countries by Cancelled Orders"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No cancelled invoices found.")
    st.subheader("📋 Order Status Summary")

completed_orders = valid_df["Invoice"].nunique()

status_data = pd.DataFrame({
    "Order Status": ["Completed", "Cancelled"],
    "Number of Orders": [
        completed_orders,
        cancelled_orders
    ]
})

fig = px.pie(
    status_data,
    names="Order Status",
    values="Number of Orders",
    title="Completed vs Cancelled Orders",
    hole=0.4
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("📥 Download Order Report")

order_report = valid_df[
    [
        "Invoice",
        "InvoiceDate",
        "Description",
        "Quantity",
        "Customer ID",
        "Country"
    ]
].copy()

csv_data = order_report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Order Report (CSV)",
    data=csv_data,
    file_name="retailpulse_order_report.csv",
    mime="text/csv"
)