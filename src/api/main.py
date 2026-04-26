"""
API para previsão de churn usando FastAPI, PyTorch e Pipeline, 
com autenticação JWT e rate limiting. (Moularizada)

"""

# pragma: no cover
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.middleware.latency import latency_middleware
from api.middleware.rate_limit import rate_limiter
from api.routes import auth_routes, predict_routes, system_routes
from src.utils.logging_config import setup_logging

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn com FastAPI, MLflow e autenticação JWT",
    version="2.1.0"
)

# Rotas
app.include_router(system_routes.router, tags=["System"])
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
app.include_router(predict_routes.router, prefix="/predict", tags=["Predict"])

# Middleware
app.middleware("http")(rate_limiter)
app.middleware("http")(latency_middleware)