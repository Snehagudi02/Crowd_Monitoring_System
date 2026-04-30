from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.density import process_image
from backend.app.services.forecast import train_system, get_predictions, get_predictive_risk
from backend.app.services.alert import generate_alert
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class PredictRequest(BaseModel):
    periods: int = 30
    history_counts: Optional[List[int]] = None

@router.post("/live-density")
async def live_density(file: UploadFile = File(...), zone_name: str = "Unknown", max_capacity: int = 25):
    try:
        contents = await file.read()
        results = process_image(contents)
        
        forecast_info = None
        if results["count"] > 0.5 * max_capacity:
            forecast_info = get_predictive_risk(periods=15)
            
        alert_info = generate_alert(results["count"], zone_name, max_capacity, forecast_info)
        results["risk"] = alert_info
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train")
def train():
    return train_system()

@router.post("/predict-zone")
def predict_zone(req: PredictRequest):
    return get_predictive_risk(req.periods, req.history_counts)

@router.get("/predict")
def predict(periods: int = 30):
    return get_predictions(periods)

@router.get("/predict-risk")
def predict_risk(periods: int = 30):
    return get_predictive_risk(periods)

@router.get("/test-email-alert")
def test_email_alert(count: int = 30, zone_name: str = "Test Zone", max_capacity: int = 25):
    return generate_alert(count, zone_name, max_capacity)
