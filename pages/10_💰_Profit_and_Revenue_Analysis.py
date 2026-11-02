import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="Profit and Revenue Analysis",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Profit and Revenue Analysis Dashboard")
st.write("Analyze business revenue and sales performance.")

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

df["Revenue"] = df["Quantity"] * df["Price"]

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

total_revenue = df["Revenue"].sum()
total_orders = df["Invoice"].nunique()
total_customers = df["Customer ID"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")

st.subheader("📊 Revenue Analysis")

revenue_chart = px.histogram(
    df,
    x="Country",
    y="Revenue",
    title="Revenue by Country"
)

st.plotly_chart(revenue_chart, use_container_width=True)
st.subheader("📅 Monthly Revenue Analysis")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_revenue,
    x="Month",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("🏆 Top 10 Products by Revenue")

if "Description" in df.columns:
    product_revenue = (
        df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        product_revenue,
        x="Revenue",
        y="Description",
        orientation="h",
        title="Top 10 Products by Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Product Description column not found.")
    st.subheader("🌍 Revenue by Country Filter")

countries = sorted(df["Country"].dropna().unique())

selected_country = st.selectbox(
    "Select Country",
    ["All Countries"] + countries
)

if selected_country == "All Countries":
    country_df = df
else:
    country_df = df[df["Country"] == selected_country]

country_revenue = country_df["Revenue"].sum()

st.metric(
    "Selected Country Revenue",
    f"{country_revenue:,.2f}"
)
st.subheader("📋 Revenue Summary")

st.write("Total Revenue:", f"{df['Revenue'].sum():,.2f}")
st.write("Average Order Revenue:", 
         f"{df.groupby('Invoice')['Revenue'].sum().mean():,.2f}")
st.write("Highest Revenue Product:", 
         df.groupby("Description")["Revenue"].sum().idxmax())
st.subheader("📈 Revenue Performance KPIs")

avg_customer_revenue = (
    df.groupby("Customer ID")["Revenue"].sum().mean()
)

avg_order_revenue = (
    df.groupby("Invoice")["Revenue"].sum().mean()
)

col1, col2 = st.columns(2)

col1.metric(
    "Average Revenue per Customer",
    f"{avg_customer_revenue:,.2f}"
)

col2.metric(
    "Average Revenue per Order",
    f"{avg_order_revenue:,.2f}"
)
st.subheader("📥 Download Revenue Report")

download_df = df[
    ["Invoice", "InvoiceDate", "Description",
     "Quantity", "Price", "Revenue", "Country"]
].copy()

csv_data = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Revenue Report (CSV)",
    data=csv_data,
    file_name="retailpulse_revenue_report.csv",
    mime="text/csv"
)