import requests

url = "https://master-project-backend.onrender.com/predict/"
data = {
    "scan_id": 1234,
    "scan_url": "https://utfs.io/f/KmmwjBRGMj2FR3HMJUdlOZo85GfuU3hMd6XsjmwWeKF4JLbz",
}

response = requests.post(url, json=data)
print(response.json())
