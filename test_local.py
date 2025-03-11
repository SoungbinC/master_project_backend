import requests

# 🔹 API Endpoint
BASE_URL = "http://127.0.0.1:8000/predict/"

# 🔹 Test Data
test_data = {
    "scan_id": 1234,
    "scan_url": "https://utfs.io/f/KmmwjBRGMj2FR3HMJUdlOZo85GfuU3hMd6XsjmwWeKF4JLbz",
}


def test_post_request():
    """Test POST request with JSON input"""
    response = requests.post(BASE_URL, json=test_data)
    print("\n📢 Testing POST Request:")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


def test_get_request():
    """Test GET request with query parameters"""
    params = {"scan_id": test_data["scan_id"], "scan_url": test_data["scan_url"]}
    response = requests.get(BASE_URL, params=params)
    print("\n📢 Testing GET Request:")
    print("Status Code:", response.status_code)
    print("Response:", response.json())


if __name__ == "__main__":
    print("🚀 Running API Tests...\n")

    # Run both tests
    test_post_request()
    test_get_request()
