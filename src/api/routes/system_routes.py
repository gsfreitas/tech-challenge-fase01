from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():

    """
    Rota principal que retorna informações sobre a API.
    Util para verificar se API esta no ar e ver os endpoints disponiveis.

    Teste curl http://localhost:8000/

    """
    return {
        "name": "Churn Prediction API",
        "version": "2.0.0",
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
            "predict_mlp": "/predict/mlp"
        }
    }

@router.get("/health")
def health():

    """
    Endpoint para verificar se o modelo está carregado e a API está saudável."""

    return {
        "status": "ok",
        "service": "churn-api",
        "version": "2.0.0"
    }