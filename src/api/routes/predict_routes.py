from fastapi import APIRouter, Depends, HTTPException
from api.core.auth import get_current_user, require_role
from api.models.schemas import CustomerData, ChurnPrediction
from api.services.model_service import ModelService
import pandas as pd

router = APIRouter()
model_service = ModelService()

@router.post("/mlp", response_model=ChurnPrediction)
def predict(
    data: CustomerData,
    current_user=Depends(require_role(["admin"]))
):
    try:
        df = pd.DataFrame([data.model_dump()])
        proba = model_service.predict_mlp(df)

        return {
            "churn_prediction": int(proba > 0.5),
            "churn_probability": proba
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/lr", response_model=ChurnPrediction)
def predict_lr(
    data: CustomerData,
    current_user=Depends(get_current_user)
):
    try:
        df = pd.DataFrame([data.model_dump()])
        proba = model_service.predict_lr(df)

        return {
            "churn_prediction": int(proba > 0.5),
            "churn_probability": proba
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))