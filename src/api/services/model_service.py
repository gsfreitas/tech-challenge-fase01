"""
ModelService — carrega e serve os 3 modelos de churn.

Estratégia de carga:
- Modelos sklearn (LogReg, Decision Tree): joblib.load do .pkl em disco
- Modelo PyTorch (MLP): torch.load do .pt em disco
- Preprocessor compartilhado: joblib.load do .pkl em disco

Por que carga direta do disco e não do MLflow Registry:
- Portabilidade: paths em disco não dependem de paths absolutos no
  banco SQLite do MLflow (que é problemático cross-platform)
- Performance: ~3x mais rápido que carregar via Registry
- Simplicidade: 1 dependência a menos em runtime
- O Registry continua sendo a fonte da verdade no treino e MLflow UI;
  só não é usado em produção pra runtime
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import torch

from api.services.feature_engineering import apply_feature_engineering
from src.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        try:
            base_path = Path(os.getenv("MODEL_PATH", "models"))

            # Preprocessor (pipeline sklearn com ColumnTransformer)
            self.preprocessor = joblib.load(base_path / "mlp_preprocessor.pkl")

            # Baselines sklearn (pickle direto)
            self.lr_model = joblib.load(base_path / "logistic_regression.pkl")
            self.tree_model = joblib.load(base_path / "decision_tree.pkl")

            # MLP PyTorch (precisa instanciar arquitetura + carregar state_dict)
            # Inferimos n_features do preprocessor (preserva contrato com treino)
            n_features = self._infer_input_dim()
            self.mlp_model = ChurnMLP(n_features=n_features)
            self.mlp_model.load_state_dict(
                torch.load(base_path / "mlp.pt", map_location="cpu", weights_only=True)
            )
            self.mlp_model.eval()

            self.loaded = True
            print("[OK] Modelos carregados")

        except Exception as e:
            self.loaded = False
            print(f"[ERRO] Falha ao carregar modelos: {e}")
            logger.exception("Falha ao carregar modelos")

    def _infer_input_dim(self) -> int:
        """
        Descobre quantas features o preprocessor produz, pra arquitetura do MLP.
        Funciona pra ColumnTransformer + OneHotEncoder (caso do projeto).
        """
        # Tenta via método público do sklearn (>= 1.0)
        try:
            return len(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback: roda transform num exemplo dummy
            dummy = pd.DataFrame([{
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.5, "TotalCharges": 800.2,
            }])
            dummy = apply_feature_engineering(dummy)
            X = self.preprocessor.transform(dummy)
            return X.shape[1]

    # =========================
    # MLP
    # =========================
    def predict_mlp(self, df: pd.DataFrame) -> float:
        if not self.loaded:
            raise Exception("Modelo não carregado")

        df = apply_feature_engineering(df)
        X = self.preprocessor.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = self.mlp_model(X_tensor)
            proba = torch.sigmoid(logits).item()

        return float(proba)

    # =========================
    # LOGISTIC REGRESSION
    # =========================
    def predict_lr(self, df: pd.DataFrame) -> float:
        if not self.loaded:
            raise Exception("Modelo não carregado")

        df = apply_feature_engineering(df)
        proba = self.lr_model.predict_proba(df)[0][1]
        return float(proba)

    # =========================
    # TREE
    # =========================
    def predict_tree(self, df: pd.DataFrame) -> float:
        if not self.loaded:
            raise Exception("Modelo não carregado")

        df = apply_feature_engineering(df)
        proba = self.tree_model.predict_proba(df)[0][1]
        return float(proba)