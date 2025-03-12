import requests

# 🔹 API Endpoint
BASE_URL = "http://127.0.0.1:8000/predict/"

# 🔹 Test Data (Replace with a real image URL)
test_data = {
    "scan_id": 1234,
    "scan_url": "https://utfs.io/f/KmmwjBRGMj2FE3SJAGEg3rbGHUVA6WMmj9C8Zft4YaI51oik",
}


def test_post_request():
    """Test POST request with JSON input"""
    try:
        response = requests.post(BASE_URL, json=test_data)
        response_json = response.json()
        print("\n📢 Testing POST Request:")
        print("Status Code:", response.status_code)
        print("Response:", response_json)

        if "classification" in response_json:
            print("\n✅ Predicted Label:", response_json["classification"])
            print("📊 Probabilities:")
            for label, prob in response_json["probabilities"].items():
                print(f"  {label}: {prob*100:.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error in POST request: {e}")


def test_get_request():
    """Test GET request with query parameters"""
    try:
        params = {"scan_id": test_data["scan_id"], "scan_url": test_data["scan_url"]}
        response = requests.get(BASE_URL, params=params)
        response_json = response.json()
        print("\n📢 Testing GET Request:")
        print("Status Code:", response.status_code)
        print("Response:", response_json)

        if "classification" in response_json:
            print("\n✅ Predicted Label:", response_json["classification"])
            print("📊 Probabilities:")
            for label, prob in response_json["probabilities"].items():
                print(f"  {label}: {prob*100:.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error in GET request: {e}")


if __name__ == "__main__":
    print("🚀 Running API Tests...\n")

    # Run both tests
    test_post_request()
    test_get_request()
    print("\n🚀 API Tests Completed.")
