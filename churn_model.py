print("Churn model started!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

print("Dataset loaded:", df.shape)

# Features and target
X = df[["Recency", "Frequency", "Monetary"]]
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("Churn prediction completed!")

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(y_test, y_pred, zero_division=0)

print("\nConfusion Matrix:")
print(y_test, y_pred)

print("\nChurn prediction completed!")