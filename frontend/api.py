import requests
import os

# Grab the Render URL from Streamlit Secrets. 
# If it can't find it (like when you are testing on your laptop), it defaults to localhost.
BASE_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def train_model():
    try:
        # Increased timeout to 120 seconds to give Render enough time to train the models
        # and account for Render's "cold start" delay on free tiers!
        resp = requests.post(f"{BASE_URL}/train", timeout=120)
        return resp.json()
    except Exception as e:
        print(f"Training API Error: {e}") # This will print exactly why it failed in Streamlit's logs
        return None

def get_forecast():
    try:
        resp = requests.get(f"{BASE_URL}/predict-risk", timeout=10)
        return resp.json()
    except:
        return None
