# routers/admin.py — 사용자 관리 (관리자 전용)

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from deps import get_current_user, require_admin
from rag_engine import user_manager

router = APIRouter(prefix="/users", tags=["admin"])


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)
    role:     str = Field("user", pattern="^(user|admin)$")


@router.get("")
async def list_users(current_user: dict = Depends(require_admin)):
    """사용자 목록 조회"""
    users = user_manager.list_users()
    return {"total": len(users), "users": users}


@router.post("")
async def create_user(
    req: UserCreateRequest,
    current_user: dict = Depends(require_admin),
):
    ok, msg = user_manager.create_user(req.username, req.password, req.role)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"message": msg}


@router.delete("/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(require_admin),
):
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="자신의 계정은 삭제할 수 없습니다")
    ok, msg = user_manager.delete_user(username)
    if not ok:
        raise HTTPException(status_code=404, detail=msg)
    return {"message": msg}


@router.post("/{username}/reset-password")
async def reset_password(
    username:     str,
    new_password: str,
    current_user: dict = Depends(require_admin),
):
    ok = user_manager.reset_password(username, new_password)
    if not ok:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return {"message": f"'{username}' 비밀번호 초기화 완료"}


@router.get("/{username}/usage")
async def get_user_usage(
    username: str,
    current_user: dict = Depends(require_admin),
):
    usage = user_manager.get_usage(username)
    if usage is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return {"username": username, "usage": usage}
