import pandas as pd
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load customer churn dataset
data_path = Path("customer_churn_data.csv")

if not data_path.exists():
    data_path = Path("data/customer_churn_data.csv")

df = pd.read_csv(data_path)

# Select features and target
features = ["Recency", "Frequency", "Monetary"]

X = df[features]
y = df["Churn"]

# Latest MLflow model location
model_dir = Path(
    "mlruns/1/models/m-584e87afdce44b328e08224be8545f6f/artifacts"
)

if not (model_dir / "MLmodel").exists():
    raise FileNotFoundError("MLflow model not found!")

# Load existing MLflow model
model = mlflow.sklearn.load_model(str(model_dir))

# Generate predictions
predictions = model.predict(X)

# Calculate performance metrics
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)

print("\nMODEL PERFORMANCE MONITORING")
print("--------------------------------")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
    if y.nunique() == 2:
        auc = roc_auc_score(y, probabilities)
        metrics["AUC Score"] = auc
        print(f"AUC Score: {auc:.3f}")

# Save performance report
results = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value"]
)

results.to_csv(
    "model_performance_results.csv",
    index=False
)

print("--------------------------------")
print("Performance monitoring completed!")
print("Report saved: model_performance_results.csv")