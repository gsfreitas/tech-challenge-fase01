"""
MODULO de Autenticação JWT
Funções para criar e validar tokens JWT
"""

from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.core.config import JWT_SECRET, TOKEN_EXPIRE_MINUTES, ADMIN_PASSWORD, USER_PASSWORD

ALGORITHM = "HS256"

security = HTTPBearer()

USERS_DB = {
    "admin": {"password": ADMIN_PASSWORD, "role": "admin"},
    "user": {"password": USER_PASSWORD, "role": "user"},
}

def authenticate_user(username: str, password: str) -> dict | None:
    user = USERS_DB.get(username)
    if not user or user["password"] != password:
        return None
    return user

def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Token não fornecido")

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[ALGORITHM]
        )

        return {
            "username": payload["sub"],
            "role": payload["role"]
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")
    
from fastapi import Depends, HTTPException

def require_role(required_role: list[str]):
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in required_role:
            raise HTTPException(
                status_code=403,
                detail="Acesso negado: permissão insuficiente"
            )
        return current_user
    return role_checker