"""
Testes dos endpoints de predição
"""

import pytest


class TestPredictAuthentication:
    """
    Endpoints de predição exigem autenticação
    """

    @pytest.mark.parametrize("endpoint", ["/predict/mlp", "/predict/lr"])
    def test_predict_without_token_is_denied(self, client, valid_customer, endpoint):
        """Sem token, predict retorna 401 ou 403."""
        response = client.post(endpoint, json=valid_customer)
        assert response.status_code in (401, 403)


class TestPredictAuthorization:
    """
    Roles diferentes têm permissões diferentes
    """

    def test_predict_mlp_requires_admin(
        self, client, user_headers, valid_customer
    ):
        """
        Given: token de role 'user'
        When: POST /predict/mlp
        Then: retorna 403 (apenas admin pode usar MLP)
        """
        response = client.post(
            "/predict/mlp", json=valid_customer, headers=user_headers
        )
        assert response.status_code == 403

    def test_predict_lr_accepts_user_role(
        self, client, user_headers, valid_customer
    ):
        """
        Given: token de role 'user'
        When: POST /predict/lr
        Then: retorna 200 (qualquer autenticado pode usar LR)
        """
        response = client.post(
            "/predict/lr", json=valid_customer, headers=user_headers
        )
        assert response.status_code == 200

    def test_predict_mlp_accepts_admin_role(
        self, client, admin_headers, valid_customer
    ):
        """
        Admin acessa MLP
        """
        response = client.post(
            "/predict/mlp", json=valid_customer, headers=admin_headers
        )
        assert response.status_code == 200


class TestPredictResponseSchema:
    """
    Resposta de predição segue Pydantic
    """

    @pytest.mark.parametrize("endpoint", ["/predict/mlp", "/predict/lr"])
    def test_response_contains_required_fields(
        self, client, admin_headers, valid_customer, endpoint
    ):
        """
        Resposta inclui churn_prediction e churn_probability
        """
        response = client.post(endpoint, json=valid_customer, headers=admin_headers)
        assert response.status_code == 200

        body = response.json()
        assert "churn_prediction" in body
        assert "churn_probability" in body

    @pytest.mark.parametrize("endpoint", ["/predict/mlp", "/predict/lr"])
    def test_prediction_is_binary(
        self, client, admin_headers, valid_customer, endpoint
    ):
        """
        Churn é 0 ou 1
        """
        body = client.post(
            endpoint, json=valid_customer, headers=admin_headers
        ).json()

        assert body["churn_prediction"] in (0, 1)

    @pytest.mark.parametrize("endpoint", ["/predict/mlp", "/predict/lr"])
    def test_probability_is_in_valid_range(
        self, client, admin_headers, valid_customer, endpoint
    ):
        """
        churn_probability está em [0, 1]
        """
        body = client.post(
            endpoint, json=valid_customer, headers=admin_headers
        ).json()

        proba = body["churn_probability"]
        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0


class TestPredictBusinessLogic:
    """
    Threshold é aplicado corretamente
    """

    def test_high_risk_customer_classified_as_churn(
        self, client, admin_headers, high_risk_customer
    ):
        """
        Cliente de alto risco (Contract=Month-to-month, tenure=1)
        com mock retornando 0.85 deve ser classificado como churn (1),
        já que 0.85 > 0.11
        """
        response = client.post(
            "/predict/mlp", json=high_risk_customer, headers=admin_headers
        )
        body = response.json()

        assert body["churn_prediction"] == 1, (
            f"Cliente arriscado classificado como não-churn "
            f"(prob={body['churn_probability']})"
        )

    def test_low_risk_customer_classified_as_no_churn(
        self, client, admin_headers, valid_customer
    ):
        """
        Cliente de baixo risco (Contract=Two year) com mock retornando 0.05
        deve ser classificado como não-churn (0), já que 0.05 < 0.11.
        """
        response = client.post(
            "/predict/mlp", json=valid_customer, headers=admin_headers
        )
        body = response.json()

        assert body["churn_prediction"] == 0


class TestPredictValidation:
    """
    Validação Pydantic dos inputs
    """

    def test_missing_required_field_returns_422(self, client, admin_headers):
        """
        Payload sem campos obrigatórios → 422
        """
        incomplete = {"gender": "Female"}
        response = client.post(
            "/predict/mlp", json=incomplete, headers=admin_headers
        )
        assert response.status_code == 422

    def test_wrong_type_returns_422(
        self, client, admin_headers, valid_customer
    ):
        """
        tenure como string em vez de int → 422
        """
        invalid = valid_customer.copy()
        invalid["tenure"] = "doze"

        response = client.post(
            "/predict/mlp", json=invalid, headers=admin_headers
        )
        assert response.status_code == 422


class TestPredictMiddleware:
    """
    Middleware de latência nos endpoints de predição
    """

    def test_latency_header_present(self, client, admin_headers, valid_customer):
        """
        Toda resposta deve ter header X-Process-Time-Ms.
        Sinal de que o middleware de latência está rodando.
        """
        response = client.post(
            "/predict/mlp", json=valid_customer, headers=admin_headers
        )

        assert "X-Process-Time-Ms" in response.headers
        latency = float(response.headers["X-Process-Time-Ms"])
        assert latency >= 0