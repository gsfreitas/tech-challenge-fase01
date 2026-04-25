"""
API para previsão de churn usando FastAPI, PyTorch e Pipeline, 
com autenticação JWT e rate limiting. (Moularizada)

"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.routes import auth_routes, predict_routes, system_routes
from api.middleware.rate_limit import rate_limiter

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn com FastAPI, MLflow e autenticação JWT",
    version="2.0.0"
)

# Rotas
app.include_router(system_routes.router, tags=["System"])
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
app.include_router(predict_routes.router, prefix="/predict", tags=["Predict"])

# Middleware
app.middleware("http")(rate_limiter)