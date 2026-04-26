from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.services.model_service import ModelService

router = APIRouter()
_model_service = ModelService()


@router.get("/")
def root():
    return {
        "name": "Churn Prediction API",
        "version": "2.1.0",
        "description": "API para previsão de churn com autenticação JWT",
        "how_to_use": {
            "1_login": "POST /auth/login → obter token JWT",
            "2_authorize": "Clique em 'Authorize' no /docs e cole o token",
            "3_predict": "POST /predict/mlp → enviar dados do cliente"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "login": "/auth/login",
            "predict_mlp": "/predict/mlp",
            "predict_lr": "/predict/lr",
            "predict_tree": "/predict/tree",
        }
    }


@router.get("/health")
def health():
    if not _model_service.loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "churn-api",
                "version": "2.1.0",
                "reason": "models not loaded",
            },
        )

    return {
        "status": "ok",
        "service": "churn-api",
        "version": "2.1.0",
        "models_loaded": True,
    }