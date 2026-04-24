from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


import jwt
import os
import pandas as pd
import torch
import joblib
import mlflow.sklearn
import mlflow.pytorch


from src.features.feature_engineering import FeatureEngineer

SECRET_KEY = os.getenv("JWT_SECRET", "minha-chave-jwt-super-secreta")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30
REQUESTS_LIMIT = 100
WINDOW_SECONDS = 3600
request_history = defaultdict(deque)

USERS_DB = {
    "admin": {
        "role": "admin",
        "password": "admin123"
    },
    "user": {
        "role": "user",
        "password": "user123"
    }
}

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = TOKEN_EXPIRE_MINUTES * 60

# =========================
# SCHEMA DE ENTRADA
# =========================
class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=800.2)

# =========================
# SCHEMA DE SAÍDA
# =========================
class ChurnPrediction(BaseModel):
    churn_prediction: int
    churn_probability: float

# =========================
# APP
# =========================
def create_token(username: str, role: str) -> str:
    """
    Cria um token JWT com expiração

    """
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()

def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> dict:
    """
    Decodifica o token JWT e retorna as informações do usuário.
    """
    try:
        payload = jwt.decode(credentials.credentials,
                            SECRET_KEY,
                            algorithms=[ALGORITHM]
                            )
        return {"username": payload["sub"], "role": payload["role"]}
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado. Faça login novamente.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")
    

app = FastAPI(
    title="Churn Prediction API com Autenticação",
    description="API para prever churn usando PyTorch + Pipeline protejida por autenticação JWT",
    version="2.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Endpoints de autenticação"},
        {"name": "prediction", "description": "Endpoints de previsão de churn"}
    ]
)

# =========================
# FUNÇÃO DE FEATURE ENGINEERING (REUTILIZÁVEL)
# =========================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(df)
    df = fe.create_family_status()
    df = fe.create_tenure_bins()
    df = fe.create_family_size_proxy()
    df = fe.create_risk_profile()
    df = fe.create_risk_combo()
    df = fe.create_support_bundle()
    df = fe.create_diff_monthly_charges()
    return df

# =========================
# CARREGAR ARTEFATOS
# =========================
try:
    preprocessor = joblib.load("models/mlp_preprocessor.pkl")

    # dummy para descobrir número de features
    dummy = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 20.0,
        "TotalCharges": 20.0
    }])

    dummy = apply_feature_engineering(dummy)

    n_features = preprocessor.transform(dummy).shape[1]

    lr_model = mlflow.sklearn.load_model(
        "models:/logistic_regression/Production"
        )
    tree_model = mlflow.sklearn.load_model(
        "models:/decision_tree/Production"
        )
    mlp_model = mlflow.pytorch.load_model(
        "models:/mlp_pytorch/Production",
        map_location="cpu"
        )


    

    MODELO_CARREGADO = True
    print("[OK] Modelos e preprocessor carregados.")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERRO] Falha ao carregar modelo: {e}")
    MODELO_CARREGADO = False

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def home():   
    
    """
    Rota principal que retorna informações sobre a API.
    Util para verificar se API esta no ar e ver os endpoints disponiveis.

    teste curl http://localhost:8000/

    """
    return {
        "nome": "Churn Prediction API com Autenticação e Rate Limiting",
        "limite": f"{REQUESTS_LIMIT} requisições por {WINDOW_SECONDS}-s por IP",
        "versao": "2.0.0",
        "descricao": "API para prever churn usando PyTorch + Pipeline protejida por autenticação JWT",
        "endpoints": {
            "GET /": "Publico - Informações sobre a API",
            "GET /docs": "GET - Documentação interativa da API (Swagger UI)",
            "GET /health": "GET - Verifica a saúde da API",
            "POST /login": "POST - Autenticação de usuário e geração de token JWT",
            "GET /me (protegido)": "GET - Endpoint protegido que retorna informações do usuário autenticado",
            "POST /predict/mlp (protegido)": "POST - Prever churn usando modelo MLP treinado",
            "POST /predict/lr (protegido)": "POST - Prever churn usando modelo de regressão logística treinado",
            "POST /predict/tree (protegido)": "POST - Prever churn usando modelo de árvore de decisão treinado"

        }
    }

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    """Endpoint para verificar se o modelo está carregado e a API está saudável."""

    return {"status": "healthy" if MODELO_CARREGADO else "unhealthy",
            "modelo_carregado": MODELO_CARREGADO
            }

# =========================
# MIDLEWARE DE AUTENTICAÇÃO
# =========================

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    """
    Middleware para limitar o número de requisições por IP.
    Permite no máximo 100 requisições por hora por IP.
    """
    client_ip = request.client.host
    now = time.time()
    history = request_history[client_ip]

    # Remove requisições antigas (fora da janela)
    while history and now - history[0] > WINDOW_SECONDS:
        history.popleft()

    # Verifica se passou do limite
    if len(history) >= REQUESTS_LIMIT:
        return JSONResponse(status_code=429,
                            content={
                                "detail": f"Limite excedido: {REQUESTS_LIMIT} requisições por {WINDOW_SECONDS}-s",
                                "retry_after": int(WINDOW_SECONDS - (now - history[0]))
                                }
                            )

    # Reguitra a requisição atual
    history.append(now)

    # Continua com a requisição
    response = await call_next(request)

    # Adiciona header informando o número de requisições restantes
    response.headers["X-RateLimit-Remaining"] = str(REQUESTS_LIMIT - len(history))
    response.headers["X-RateLimit-Reset"] = str(REQUESTS_LIMIT)

    return response

# =========================
# LOGIN E AUTENTICAÇÃO
# =========================

@app.post("/login", response_model=TokenResponse, tags=["auth"])
def login(credentials: LoginRequest):
    """
    Endpoint de login que autentica o usuário e retorna um token JWT.
    """

    user = USERS_DB.get(credentials.username)

    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = create_token(credentials.username, user["role"])

    return TokenResponse(access_token=token)

# =========================
# INFO USUÁRIO AUTENTICADO
# =========================

@app.get("/me", tags=["auth"])
def get_me(current_user: dict = Depends(get_current_user)):
    """
    Retorna informações do usuário logado. Endpoint protegido que requer um token JWT válido.

    """
    return {
        "username": current_user["role"],
        "message": "Você está autenticado!"
    }

# =========================
# PREDICT  MODELO MLP
# =========================
@app.post("/predict/mlp", response_model=ChurnPrediction, tags=["Predict"])
def predict(payload: CustomerData, current_user: dict = Depends(get_current_user)):

    """
    Endpoint protegido por token JWT.

    Esse Endpoint recebe dados do cliente e preve churn usando mlp treinado. Ele faz o seguinte:
    1. Verifica se o modelo está carregado. Se não estiver, retorna um erro 503.
    2. Converte os dados de entrada em um DataFrame do pandas.
    3. Aplica as transformações de feature engineering usando a classe FeatureEngineer.
    4. Aplica o pré-processamento usando o pipeline carregado.
    5. Converte os dados processados em um tensor do PyTorch.
    6. Faz a previsão usando o modelo MLP e calcula a probabilidade de churn.
    7. Retorna a previsão de churn (0 ou 1) e a probabilidade associada.
    """

    if not MODELO_CARREGADO:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se os arquivos .pkl existem."
        )

    df = pd.DataFrame([payload.dict()])

    # Feature Engineering
    df = apply_feature_engineering(df)

    # preprocessing
    X_processed = preprocessor.transform(df)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to("cpu")

    with torch.no_grad():
        logits = mlp_model(X_tensor)
        proba = torch.sigmoid(logits).item()

    return {
        "churn_prediction": int(proba > 0.5),
        "churn_probability": proba,
        "usuario": current_user["username"]
    }

# =========================================
# PREDICT  MODELO LOGISTIC REGRESSION - LR
# =========================================

@app.post("/predict/lr", response_model=ChurnPrediction)
def predict_lr(data: CustomerData):

    """
    Esse Endpoint recebe dados do cliente e preve churn usando modelo de regressão logística treinado. Ele faz o seguinte:
    1. Verifica se o modelo está carregado. Se não estiver, retorna um erro 503.
    2. Converte os dados de entrada em um DataFrame do pandas.
    3. Aplica as transformações de feature engineering usando a classe FeatureEngineer.
    4. Aplica o pré-processamento usando o pipeline carregado.
    5. Faz a previsão usando o modelo de regressão logística e calcula a probabilidade de churn.
    6. Retorna a previsão de churn (0 ou 1) e a probabilidade associada.
    """

    if not MODELO_CARREGADO:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se os arquivos .pkl existem."
        )

    df = pd.DataFrame([data.dict()])
    df = apply_feature_engineering(df) 
    proba = lr_model.predict_proba(df)[0][1]
    
    pred = int(proba > 0.5)

    return {
        "churn_prediction": int(proba > 0.5),
        "churn_probability": float(proba)
    }


# =========================================
# PREDICT  MODELO DECISION TREE - TREE
# =========================================
@app.post("/predict/tree", response_model=ChurnPrediction)
def predict_tree(data: CustomerData):

    """
    Esse Endpoint recebe dados do cliente e preve churn usando modelo de árvore de decisão treinado. Ele faz o seguinte:
    1. Verifica se o modelo está carregado. Se não estiver, retorna um erro 503.
    2. Converte os dados de entrada em um DataFrame do pandas.
    3. Aplica as transformações de feature engineering usando a classe FeatureEngineer.
    4. Aplica o pré-processamento usando o pipeline carregado.
    5. Faz a previsão usando o modelo de árvore de decisão e calcula a probabilidade de churn.
    6. Retorna a previsão de churn (0 ou 1) e a probabilidade associada.
    """

    if not MODELO_CARREGADO:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se os arquivos .pkl existem."
        )

    df = pd.DataFrame([data.dict()])
    df = apply_feature_engineering(df)
    proba = tree_model.predict_proba(df)[0][1]
    pred = int(proba > 0.5)

    return {
        "churn_prediction": int(proba > 0.5),
        "churn_probability": float(proba)
    }