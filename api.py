from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

app = FastAPI(title="RetailPulse Churn Prediction API")

BASE_DIR = Path(__file__).resolve().parent

# Connect to MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
)

# Load the latest trained model
experiment = mlflow.get_experiment_by_name(
    "RetailPulse_Customer_Churn"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

latest_run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{latest_run_id}/random_forest_churn_model"

model = mlflow.sklearn.load_model(model_uri)


class CustomerData(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float


@app.get("/")
def home():
    return {"message": "RetailPulse Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([{
        "Recency": data.Recency,
        "Frequency": data.Frequency,
        "Monetary": data.Monetary
    }])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "result": "Churn" if prediction == 1 else "Not Churn"
    }