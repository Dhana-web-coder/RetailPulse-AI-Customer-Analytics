import requests

url = "http://127.0.0.1:8001/predict"

customer_data = {
    "Recency": 30,
    "Frequency": 5,
    "Monetary": 1000
}

response = requests.post(url, json=customer_data)

print("Status Code:", response.status_code)
print("Prediction Result:", response.json())