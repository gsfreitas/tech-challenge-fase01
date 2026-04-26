"""Testes JWT e autorização baseada em RBAC"""

import pytest


class TestLogin:
    """/auth/login: gera tokens JWT válidos."""

    def test_login_with_valid_admin_credentials(self, client):
        """
        Given: credenciais válidas de admin
        When: POST /auth/login
        Then: retorna 200 com token JWT
        """
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )

        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert isinstance(body["expires_in"], int)
        assert body["expires_in"] > 0

    def test_login_with_valid_user_credentials(self, client):
        """User comum também consegue logar """
        response = client.post(
            "/auth/login",
            json={"username": "user", "password": "user123"},
        )

        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_with_wrong_password_returns_401(self, client):
        """
        Given: usuário existe mas senha incorreta
        When: POST /auth/login
        Then: retorna 401 com mensagem clara
        """
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "senha-errada"},
        )

        assert response.status_code == 401

    def test_login_with_unknown_user_returns_401(self, client):
        """Usuário inexistente também retorna 401 (não 404)."""
        response = client.post(
            "/auth/login",
            json={"username": "fantasma", "password": "qualquer"},
        )

        assert response.status_code == 401

    def test_login_with_missing_fields_returns_422(self, client):
        """Payload incompleto: validação Pydantic retorna 422."""
        response = client.post("/auth/login", json={"username": "admin"})

        assert response.status_code == 422


class TestMeEndpoint:
    """/auth/me: retorna dados do usuário autenticado."""

    def test_me_without_token_returns_401_or_403(self, client):
        """Sem header Authorization -> negar acesso"""
        response = client.get("/auth/me")
        assert response.status_code in (401, 403)

    def test_me_with_invalid_token_returns_401(self, client):
        """Token mal formado -> retorna 401."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
        assert response.status_code == 401

    def test_me_with_valid_admin_token(self, client, admin_headers):
        """
        Given: token válido de admin
        When: GET /auth/me
        Then: retorna username e role corretos
        """
        response = client.get("/auth/me", headers=admin_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "admin"
        assert body["role"] == "admin"

    def test_me_with_valid_user_token(self, client, user_headers):
        """User comum também acessa /me."""
        response = client.get("/auth/me", headers=user_headers)

        assert response.status_code == 200
        assert response.json()["role"] == "user"