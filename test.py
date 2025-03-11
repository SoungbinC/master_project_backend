import requests

url = "http://127.0.0.1:8000/predict/"
data = {
    "scan_id": 1234,
    "scan_url": "https://utfs.io/f/KmmwjBRGMj2Fs7MPfRq2mYHBSQbVu635hUAC4d0cMjvEs1ZO",
}

response = requests.post(url, json=data)
print(response.json())
