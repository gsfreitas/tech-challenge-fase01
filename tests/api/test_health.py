"""Testes do endpoint /health e raiz"""

import pytest


class TestHealthEndpoint:
    """
    /health: contrato e comportamento básico
    """

    def test_health_returns_200(self, client):
        """
        Given: API rodando com modelos carregados
        When: GET /health
        Then: retorna 200 com status ok
        """
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_response_schema(self, client):
        """
        Given: API rodando
        When: GET /health
        Then: response inclui status, service, version
        """
        body = client.get("/health").json()

        assert "status" in body
        assert "service" in body
        assert "version" in body
        assert body["service"] == "churn-api"

    def test_health_returns_latency_header(self, client):
        """
        Given: middleware de latência registrado
        When: GET /health
        Then: header X-Process-Time-Ms está presente e é numérico
        """
        response = client.get("/health")

        assert "X-Process-Time-Ms" in response.headers
        latency = float(response.headers["X-Process-Time-Ms"])
        assert latency >= 0


class TestRootEndpoint:
    """
    / : página de boas-vindas com metadados da API.
    """

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_metadata(self, client):
        body = client.get("/").json()

        assert body["name"] == "Churn Prediction API"
        assert "endpoints" in body
        assert "how_to_use" in body