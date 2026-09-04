from fastapi import APIRouter, Header, HTTPException

from app.schemas import LoginRequest, RegisterRequest
from app.services.auth_service import get_user_from_token, login_user, register_user
from app.utils.response import error_response, success_response


router = APIRouter()


@router.post("/auth/register")
def register(payload: RegisterRequest):
    try:
        return success_response(register_user(payload), "Account created successfully")
    except ValueError as exc:
        return error_response(str(exc), "Registration failed")


@router.post("/auth/login")
def login(payload: LoginRequest):
    try:
        return success_response(login_user(payload), "Login successful")
    except ValueError as exc:
        return error_response(str(exc), "Login failed")


@router.get("/auth/me")
def me(authorization: str | None = Header(default=None)):
    token = authorization.removeprefix("Bearer ").strip() if authorization else ""
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return success_response(user, "Authenticated user")