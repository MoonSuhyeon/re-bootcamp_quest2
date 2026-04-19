# deps.py — v25 공유 의존성
# JWT 유틸리티 + 인덱스 전역 상태
# 모든 router 가 여기서 import

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer

try:
    from jose import jwt
    JOSE_AVAILABLE = True
except ImportError:
    import jwt as pyjwt
    JOSE_AVAILABLE = False

from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES
from rag_engine import user_manager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# =====================================================================
# 인덱스 전역 상태 — docs 라우터에서 쓰고 chat 라우터에서 읽음
# =====================================================================

INDEX_STATE: dict = {
    "index":       None,
    "chunks":      [],
    "sources":     [],
    "mv_index":    None,
    "doc_count":   0,
    "last_upload": None,
}


def index_ready() -> bool:
    return INDEX_STATE["index"] is not None and len(INDEX_STATE["chunks"]) > 0


# =====================================================================
# JWT
# =====================================================================

def encode_token(data: dict) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {**data, "exp": expire}
    if JOSE_AVAILABLE:
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        if JOSE_AVAILABLE:
            return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return pyjwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰이 유효하지 않거나 만료되었습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =====================================================================
# FastAPI 의존성
# =====================================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload  = decode_token(token)
    username = payload.get("sub", "")
    if not username:
        raise HTTPException(status_code=401, detail="사용자 정보 없음")
    user = user_manager.get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다")
    return {"username": username, "role": user.get("role", "user")}


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다")
    return current_user
