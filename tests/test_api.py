import requests

url = "http://localhost:8000/predict"
payload = {
    "pickup_longitude": 0,
    "pickup_latitude": 0, 
    "dropoff_longitude": 0, 
    "dropoff_latitude": 0, 
    "passenger_count": 0, 
    "dayofweek": 0,
    "day": 0, 
    "month": 0,
    "year": 0,
    "hour": 0,
    "minute": 0, 
    "second": 0,
    "distance": 0,
    "bearing": 0
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response Body:", response.json())
