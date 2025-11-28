import requests
import json

# Test the API locally
base_url = "http://localhost:8000"

def test_api():
    print("🧪 Testing House Price Prediction API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Health check: {response.json()}")
    except:
        print("❌ Server not running. Start with: uvicorn main:app --reload")
        return
    
    # Test prediction endpoint
    test_data = {
        "data": [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        result = response.json()
        print(f"✅ Prediction: ${result['prediction']:,.2f}")
        print(f"   Features used: {result['input_features']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    test_api()