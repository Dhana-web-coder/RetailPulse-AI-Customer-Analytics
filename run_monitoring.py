import subprocess
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

history_file = BASE_DIR / "monitoring_history.csv"
metrics_file = BASE_DIR / "model_performance_results.csv"

scripts = [
    "drift_monitoring.py",
    "model_performance_monitoring.py"
]

print("RETAILPULSE AUTOMATED MONITORING")
print("=" * 40)

status = "Success"

for script in scripts:
    print(f"\nRunning {script}...")

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        status = "Failed"
        print(f"{script} failed!")
        break

    print(f"{script} completed successfully!")

# Save monitoring history
record = {
    "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Status": status
}

if status == "Success" and metrics_file.exists():

    metrics = pd.read_csv(metrics_file)

    for _, row in metrics.iterrows():
        record[row["Metric"]] = row["Value"]

history = pd.DataFrame([record])

if history_file.exists():
    old_history = pd.read_csv(history_file)
    history = pd.concat(
        [old_history, history],
        ignore_index=True
    )

history.to_csv(history_file, index=False)

print("\nMonitoring Status:", status)
print("Monitoring history updated!")