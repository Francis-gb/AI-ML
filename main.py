from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

app = FastAPI()
model = joblib.load("wbgt_model.pkl")

class ForecastRequest(BaseModel):
    horizon: str  # "Now", "3h", "6h", "12h"

def fetch_latest_value(url: str, key: str) -> float:
    try:
        response = requests.get(url)
        data = response.json()
        readings = data["items"][0]["readings"]
        values = [r["value"] for r in readings if isinstance(r["value"], (int, float))]
        return round(sum(values) / len(values), 2) if values else 0.0
    except Exception as e:
        print(f"Error fetching {key}: {e}")
        return 0.0

def get_weather_data():
    temp = fetch_latest_value("https://api.data.gov.sg/v1/environment/air-temperature", "temp")
    rh = fetch_latest_value("https://api.data.gov.sg/v1/environment/relative-humidity", "rh")
    wind = fetch_latest_value("https://api.data.gov.sg/v1/environment/wind-speed", "wind")
    return temp, rh, wind

def get_time_features(horizon: str):
    shift_map = {"Now": 0, "3h": 3, "6h": 6, "12h": 12}
    now = datetime.now() + timedelta(hours=shift_map.get(horizon, 0))
    hour = now.hour
    month = now.month
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour, month, hour_sin, hour_cos

@app.post("/predict")
def predict(request: ForecastRequest):
    try:
        temp, rh, wind = get_weather_data()
        hour, month, hour_sin, hour_cos = get_time_features(request.horizon)

        input_data = {
            "temp_c": temp,
            "rh_percent": rh,
            "wind_speed_ms": wind,
            "wbgt_primary": 0,
            "hour": hour,
            "month": month,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        return {
            "horizon": request.horizon,
            "inputs": input_data,
            "wbgt_prediction": round(float(prediction), 2)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}
