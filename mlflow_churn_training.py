import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. Project folder
BASE_DIR = Path(__file__).resolve().parent

# 2. Find churn dataset
DATA_PATH = BASE_DIR / "customer_churn_data.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "customer_churn_data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "customer_churn_data.csv not found in project root or data folder."
    )

# 3. Load and prepare data
df = pd.read_csv(DATA_PATH)

required_columns = ["Recency", "Frequency", "Monetary", "Churn"]
df = df[required_columns].copy()

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()
df = df[
    (df["Recency"] >= 0)
    & (df["Frequency"] >= 0)
    & (df["Monetary"] >= 0)
    & (df["Churn"].isin([0, 1]))
]

df["Churn"] = df["Churn"].astype(int)

if df.empty or df["Churn"].nunique() != 2:
    raise ValueError("Dataset must contain both churn classes: 0 and 1.")

X = df[["Recency", "Frequency", "Monetary"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Configure MLflow
tracking_folder = BASE_DIR / "mlflow_tracking"
tracking_folder.mkdir(exist_ok=True)

tracking_db = BASE_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
mlflow.set_experiment("RetailPulse_Customer_Churn")

# 5. Train and track model
with mlflow.start_run(run_name="RandomForest_Churn"):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # 6. Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_probability)

    # 7. Save parameters in MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("features", "Recency, Frequency, Monetary")

    # 8. Save performance metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_score", auc)

    # 9. Save trained model
    mlflow.sklearn.log_model(
    model,
    name="random_forest_churn_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
    print("Model training completed!")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC Score: {auc:.3f}")
    print("Model and metrics saved in MLflow.")