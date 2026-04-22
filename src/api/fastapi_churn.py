from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
import torch
import joblib


from src.features.feature_engineering import FeatureEngineer
from src.utils.config import MLP_HIDDEN_DIMS, MLP_DROPOUT_RATES
from src.models.mlp import ChurnMLP

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
app = FastAPI(
    title="Churn Prediction API",
    description="API para prever churn usando PyTorch + Pipeline",
    version="1.0.0"
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

    model = ChurnMLP(
        n_features=n_features,
        hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rates=MLP_DROPOUT_RATES
    )

    lr_model = joblib.load("models/logistic_regression.pkl")
    tree_model = joblib.load("models/decision_tree.pkl")

    model.load_state_dict(torch.load("models/mlp.pt", map_location="cpu"))
    model.eval()
    

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
        "nome": "Churn Prediction API",
        "versao": "1.0.0",
        "descricao": "API para prever churn usando PyTorch + Pipeline",
        "endpoints": {
            "GET /": "GET - Informações sobre a API",
            "GET /health": "GET - Verifica a saúde da API",
            "GET /docs": "GET - Documentação interativa da API (Swagger UI)",
            "POST /predict": "POST - Prever churn com base nos dados do cliente"

        }
    }

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    """Endpoint para verificar se o modelo está carregado e a API está saudável."""

    return {"status": "healthy" if MODELO_CARREGADO else "unhealthy"
            "moddelo_carregado"> MODELO_CARREGADO   
            }

# =========================
# PREDICT  MODELO MLP
# =========================
@app.post("/predict/mlp", response_model=ChurnPrediction)
def predict(data: CustomerData):

    """
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

    df = pd.DataFrame([data.dict()])

    # Feature Engineering
    df = apply_feature_engineering(df)

    # preprocessing
    X_processed = preprocessor.transform(df)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        proba = torch.sigmoid(logits).item()

    return {
        "churn_prediction": int(proba > 0.5),
        "churn_probability": proba
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