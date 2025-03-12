import requests

# 🔹 Replace with your deployed Render API URL
DEPLOYED_URL = "https://master-project-backend.onrender.com/predict"

# 🔹 Test Data (Use a valid image URL)
test_data = {
    "scan_id": 1234,
    "scan_url": "https://utfs.io/f/KmmwjBRGMj2FE3SJAGEg3rbGHUVA6WMmj9C8Zft4YaI51oik",  # Replace with an actual image URL
}


def test_post_request():
    """Test POST request on deployed API"""
    try:
        response = requests.post(DEPLOYED_URL, json=test_data)
        response_json = response.json()
        print("\n📢 Testing POST Request:")
        print("Status Code:", response.status_code)
        print("Response:", response_json)

        if response.status_code == 200:
            print(
                "\n✅ Predicted Label:", response_json.get("classification", "Unknown")
            )
            print("📊 Probabilities:")
            for label, prob in response_json.get("probabilities", {}).items():
                print(f"  {label}: {prob*100:.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error in POST request: {e}")


def test_get_request():
    """Test GET request on deployed API"""
    try:
        params = {"scan_id": test_data["scan_id"], "scan_url": test_data["scan_url"]}
        response = requests.get(DEPLOYED_URL, params=params)
        response_json = response.json()
        print("\n📢 Testing GET Request:")
        print("Status Code:", response.status_code)
        print("Response:", response_json)

        if response.status_code == 200:
            print(
                "\n✅ Predicted Label:", response_json.get("classification", "Unknown")
            )
            print("📊 Probabilities:")
            for label, prob in response_json.get("probabilities", {}).items():
                print(f"  {label}: {prob*100:.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error in GET request: {e}")


if __name__ == "__main__":
    print("🚀 Running Tests on Deployed API...\n")

    # Run both tests
    test_post_request()
    test_get_request()
