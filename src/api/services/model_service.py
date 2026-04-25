import joblib
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import torch
import os

from api.services.feature_engineering import apply_feature_engineering


class ModelService:

    def __init__(self):
        try:
            base_path = os.getenv("MODEL_PATH", "models")

            self.preprocessor = joblib.load(f"{base_path}/mlp_preprocessor.pkl")

            self.lr_model = mlflow.sklearn.load_model(
                "models:/logistic_regression/Production"
            )

            self.tree_model = mlflow.sklearn.load_model(
                "models:/decision_tree/Production"
            )

            self.mlp_model = mlflow.pytorch.load_model(
                "models:/mlp_pytorch/Production",
                map_location="cpu"
            )

            self.loaded = True
            print("[OK] Modelos carregados")

        except Exception as e:
            self.loaded = False
            print(f"[ERRO] Falha ao carregar modelos: {e}")

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