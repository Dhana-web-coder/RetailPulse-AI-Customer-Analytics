import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Model Monitoring | RetailPulse",
    page_icon="🛡️",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

results_file = BASE_DIR / "model_performance_results.csv"
drift_file = BASE_DIR / "drift_report.html"

# Dashboard title
st.title("🛡️ Model Performance & Drift Monitoring")
st.caption("RetailPulse | AI-Powered Customer Analytics Platform")

st.divider()

# Load performance results
if results_file.exists():

    df = pd.read_csv(results_file)

    metrics = dict(zip(df["Metric"], df["Value"]))

    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2%}")
    col2.metric("Precision", f"{metrics.get('Precision', 0):.2%}")
    col3.metric("Recall", f"{metrics.get('Recall', 0):.2%}")
    col4.metric("F1 Score", f"{metrics.get('F1 Score', 0):.2%}")
    col5.metric("AUC Score", f"{metrics.get('AUC Score', 0):.2%}")

    st.subheader("📈 Performance Comparison")

    chart_df = df.copy()
    chart_df["Value"] = chart_df["Value"] * 100

    st.bar_chart(
        chart_df.set_index("Metric")["Value"]
    )

    st.subheader("📋 Performance Report")
    st.dataframe(df, use_container_width=True)

else:
    st.warning("Model performance results not found.")

st.divider()

# Data drift monitoring
st.subheader("🔍 Data Drift Monitoring")

if drift_file.exists():

    st.success("Data Drift Report Available")

    with st.expander("View Data Drift Report", expanded=False):

        html_report = drift_file.read_text(encoding="utf-8")

        components.html(
            html_report,
            height=800,
            scrolling=True
        )

else:
    st.warning("Drift report not found.")

st.divider()

# Monitoring summary
st.subheader("✅ Monitoring Summary")

col1, col2 = st.columns(2)

with col1:
    st.info("📊 Model Performance Monitoring")

with col2:
    st.info("🔍 Data Drift Detection")

st.caption("RetailPulse | Customer Analytics & Forecasting Platform")
# Download Monitoring Reports

st.divider()

st.subheader("📥 Download Monitoring Reports")

col1, col2 = st.columns(2)

with col1:
    if results_file.exists():
        with open(results_file, "rb") as file:
            st.download_button(
                label="📊 Download Performance Report",
                data=file,
                file_name="model_performance_results.csv",
                mime="text/csv"
            )

with col2:
    if drift_file.exists():
        with open(drift_file, "rb") as file:
            st.download_button(
                label="🔍 Download Data Drift Report",
                data=file,
                file_name="drift_report.html",
                mime="text/html"
            )
# Automated Monitoring Status

st.divider()

st.subheader("🖥️ System Monitoring Status")

col1, col2 = st.columns(2)

with col1:
    if results_file.exists():
        modified = pd.Timestamp(
            results_file.stat().st_mtime,
            unit="s"
        )

        st.success("Model Monitoring: Available")
        st.write("Last Updated:", modified.strftime("%d-%m-%Y %H:%M:%S"))
    else:
        st.error("Model Monitoring: Not Available")

with col2:
    if drift_file.exists():
        modified = pd.Timestamp(
            drift_file.stat().st_mtime,
            unit="s"
        )

        st.success("Data Drift Monitoring: Available")
        st.write("Last Updated:", modified.strftime("%d-%m-%Y %H:%M:%S"))
    else:
        st.error("Data Drift Monitoring: Not Available")
# Run Automated Monitoring

st.divider()

st.subheader("🚀 Run Automated Monitoring")

if st.button("▶️ Run Monitoring Now", type="primary"):

    import subprocess
    import sys

    with st.spinner("Running data drift and model performance monitoring..."):

        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "run_monitoring.py")],
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

    if result.returncode == 0:
        st.success("Monitoring completed successfully!")

        st.code(result.stdout)

        st.info("Refresh the page to view updated reports.")

    else:
        st.error("Monitoring failed!")
        st.code(result.stderr)
# Monitoring Execution History

st.divider()

st.subheader("📜 Monitoring Execution History")

history_file = BASE_DIR / "monitoring_history.csv"

if history_file.exists():

    history_df = pd.read_csv(history_file)

    st.dataframe(
        history_df.sort_values(
            "Date_Time",
            ascending=False
        ),
        use_container_width=True
    )

    total_runs = len(history_df)
    successful_runs = (
        history_df["Status"] == "Success"
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric("Total Monitoring Runs", total_runs)
    col2.metric("Successful Runs", int(successful_runs))

else:
    st.info("No monitoring execution history available yet.")
# Model Performance Trend

st.divider()

st.subheader("📈 Model Performance Trend")

if history_file.exists():

    history_df = pd.read_csv(history_file)

    if len(history_df) >= 2:

        st.info(
            "Performance trend requires historical metric values. "
            "The current history file stores execution status only."
        )

    else:
        st.info(
            "Run monitoring at least twice to build execution history."
        )

else:
    st.info("Monitoring history is not available yet.")
# Historical Model Performance Trend

st.divider()

st.subheader("📈 Historical Model Performance Trend")

history_file = BASE_DIR / "monitoring_history.csv"

if history_file.exists():

    history_df = pd.read_csv(history_file)

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC Score"
    ]

    available_metrics = [
        col for col in metric_columns
        if col in history_df.columns
    ]

    if available_metrics:

        history_df["Date_Time"] = pd.to_datetime(
            history_df["Date_Time"]
        )

        history_df = history_df.sort_values("Date_Time")

        chart_df = history_df.set_index("Date_Time")[
            available_metrics
        ] * 100

        st.line_chart(chart_df)

        st.caption(
            "Historical model performance across monitoring runs."
        )

    else:
        st.info("Run automated monitoring to collect metric history.")

else:
    st.info("Monitoring history is not available yet.")
# Final Monitoring Summary

st.divider()

st.header("🏆 RetailPulse Monitoring Summary")

history_file = BASE_DIR / "monitoring_history.csv"

if history_file.exists():

    history_df = pd.read_csv(history_file)

    total_runs = len(history_df)

    successful_runs = (
        history_df["Status"] == "Success"
    ).sum()

    failed_runs = total_runs - successful_runs

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Runs", total_runs)
    col2.metric("Successful Runs", int(successful_runs))
    col3.metric("Failed Runs", int(failed_runs))

    st.subheader("📊 Latest Model Performance")

    if "Accuracy" in history_df.columns:

        valid_runs = history_df[
            history_df["Status"] == "Success"
        ]

        if not valid_runs.empty:

            latest = valid_runs.iloc[-1]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{latest['Accuracy']:.2%}"
            )

            col2.metric(
                "Precision",
                f"{latest['Precision']:.2%}"
            )

            col3.metric(
                "Recall",
                f"{latest['Recall']:.2%}"
            )

else:
    st.info("Run automated monitoring to generate a summary.")

st.divider()

st.caption(
    "RetailPulse | AI-Powered Customer Analytics "
    "and Forecasting Platform"
)