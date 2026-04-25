from fastapi import APIRouter, HTTPException, Depends
from api.core.auth import authenticate_user, create_token, get_current_user
from api.models.schemas import LoginRequest, TokenResponse
from api.core.config import TOKEN_EXPIRE_MINUTES

router = APIRouter()

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    user = authenticate_user(data.username, data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = create_token(data.username, user["role"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_EXPIRE_MINUTES * 60
    }

@router.get("/me")
def get_me(current_user=Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "message": "Usuário autenticado com sucesso"
    }