# routers/auth.py — 인증 엔드포인트

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from config import JWT_EXPIRE_MINUTES
from deps import encode_token, get_current_user, require_admin
from rag_engine import user_manager

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = JWT_EXPIRE_MINUTES * 60
    username:     str
    role:         str


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)
    role:     str = Field("user", pattern="^(user|admin)$")


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """사용자 로그인 — JWT 액세스 토큰 발급"""
    if not user_manager.authenticate(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user  = user_manager.get_user(form_data.username)
    token = encode_token({"sub": form_data.username})
    return TokenResponse(
        access_token=token,
        username=form_data.username,
        role=user.get("role", "user"),
    )


@router.post("/register")
async def register(
    req: UserCreateRequest,
    current_user: dict = Depends(require_admin),
):
    """신규 사용자 등록 (관리자 전용)"""
    ok, msg = user_manager.create_user(req.username, req.password, req.role)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"message": msg, "role": req.role}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """현재 로그인 사용자 정보"""
    user  = user_manager.get_user(current_user["username"])
    usage = user_manager.get_usage(current_user["username"]) or {}
    return {
        "username":        current_user["username"],
        "role":            current_user["role"],
        "display_name":    (user or {}).get("display_name", current_user["username"]),
        "usage_today":     usage.get("queries_24h", 0),
        "usage_this_hour": usage.get("queries_1h", 0),
        "remaining_1h":    usage.get("remaining_1h", 0),
    }
