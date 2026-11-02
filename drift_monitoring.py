import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

# Select numerical columns
reference_data = df[["Recency", "Frequency", "Monetary"]].copy()
current_data = reference_data.sample(
    frac=0.8,
    random_state=42
)

# Create drift report
report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save HTML report
result.save_html("drift_report.html")

print("Data Drift Monitoring Completed!")
print("Report saved as drift_report.html")