"""
Fixtures específicas para testes de API
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session", autouse=True)
def _set_test_env():
    """
    Garante variáveis mínimas de ambiente pros testes
    """
    os.environ.setdefault("JWT_SECRET", "test-secret-only-for-testing")
    os.environ.setdefault("ADMIN_PASSWORD", "admin123")
    os.environ.setdefault("USER_PASSWORD", "user123")
    
    os.environ["REQUESTS_LIMIT"] = "100000"
    os.environ["WINDOW_SECONDS"] = "3600"


@pytest.fixture(scope="session", autouse=True)
def _mock_model_service(_set_test_env):
    """
    Substitui ModelService.__init__ pra usar mocks ao invés de MLflow real
    """

    from api.services import model_service as ms_module

    original_init = ms_module.ModelService.__init__

    def mocked_init(self):
        self.preprocessor = MagicMock()
        self.lr_model = MagicMock()
        self.tree_model = MagicMock()
        self.mlp_model = MagicMock()
        self.loaded = True

    ms_module.ModelService.__init__ = mocked_init

    # mocks para não chamar modelo real
    def _high_risk_or_low_risk(df):
        try:
            row = df.iloc[0]
            high_risk = (
                row.get("Contract", "") == "Month-to-month"
                and row.get("tenure", 0) <= 6
            )
            return 0.85 if high_risk else 0.05
        except Exception:
            return 0.5

    ms_module.ModelService.predict_mlp = lambda self, df: _high_risk_or_low_risk(df)
    ms_module.ModelService.predict_lr = lambda self, df: _high_risk_or_low_risk(df)
    ms_module.ModelService.predict_tree = lambda self, df: _high_risk_or_low_risk(df)

    yield

    ms_module.ModelService.__init__ = original_init


@pytest.fixture(scope="session")
def app(_mock_model_service):
    """App FastAPI carregado uma vez por sessão de testes."""
    from api.main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)

# TOKENS

@pytest.fixture
def admin_token(client) -> str:
    """
    JWT válido com role admin
    """
    resp = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200, f"Login admin falhou: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def user_token(client) -> str:
    """
    JWT válido com role user
    """
    resp = client.post(
        "/auth/login",
        json={"username": "user", "password": "user123"},
    )
    assert resp.status_code == 200, f"Login user falhou: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def admin_headers(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def user_headers(user_token) -> dict:
    return {"Authorization": f"Bearer {user_token}"}


# PAYLOADS

@pytest.fixture
def valid_customer() -> dict:
    """
    Cliente válido
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 800.2,
    }


@pytest.fixture
def high_risk_customer() -> dict:
    """
    Perfil de alto risco
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 70.0,
    }