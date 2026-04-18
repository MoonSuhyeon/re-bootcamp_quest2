# server_api.py — v23 FastAPI 서버
# rag_engine.py 를 불러와 실제 API 엔드포인트로 노출
# 인증(JWT) / Rate Limit / 파일 업로드 / 채팅 / 메트릭 / 관리

from __future__ import annotations

import os
import io
import json
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File,
    status, BackgroundTasks, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    try:
        import jwt as pyjwt
        JOSE_AVAILABLE = False
    except ImportError:
        raise RuntimeError("python-jose 또는 PyJWT 를 설치하세요: pip install python-jose[cryptography]")

# rag_engine 에서 필요한 심볼 임포트
from rag_engine import (
    # 인덱스 / 문서 처리
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    build_faiss_index,
    build_multi_vector_index,
    # 파이프라인
    process_rag_query,
    run_rag_pipeline,
    # 관리 객체
    user_manager,
    metrics_collector,
    failure_dataset,
    eval_log,
    answer_cache,
    # 상수
    LOG_FILE,
    USERS_FILE,
    EMBED_CACHE_FILE,
)

# =====================================================================
# 앱 초기화
# =====================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="RAG API v23",
    description="LLM Compression · ToolRegistry · AsyncRAGEngine",
    version="23.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# JWT 설정
# =====================================================================

SECRET_KEY    = os.getenv("JWT_SECRET_KEY", "rag-v23-secret-change-in-production")
ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    if JOSE_AVAILABLE:
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    else:
        return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        if JOSE_AVAILABLE:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        else:
            return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰이 유효하지 않거나 만료되었습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    username: str = payload.get("sub", "")
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


# =====================================================================
# 인덱스 전역 상태
# =====================================================================

_INDEX_STATE: dict = {
    "index":    None,   # faiss.Index
    "chunks":   [],     # list[str]
    "sources":  [],     # list[str]
    "mv_index": None,   # MultiVectorIndex | None
    "doc_count": 0,
    "last_upload": None,
}


def _index_ready() -> bool:
    return _INDEX_STATE["index"] is not None and len(_INDEX_STATE["chunks"]) > 0


# =====================================================================
# Pydantic 모델
# =====================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    username: str
    role: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field("hybrid", description="hybrid | dense | bm25")
    top_k: int = Field(5, ge=1, le=20)
    prefilter_n: int = Field(30, ge=10, le=100)
    use_rewrite: bool = True
    use_self_refine: bool = True
    use_multidoc: bool = True
    use_compression: bool = True
    use_llm_compress: bool = False       # [NEW v23]
    use_tool_registry: bool = False      # [NEW v23]
    enabled_tools: List[str] = Field(default_factory=lambda: ["calculator", "datetime"])
    use_async_engine: bool = False       # [NEW v23]
    auto_routing: bool = True
    dedup_threshold: float = Field(0.85, ge=0.0, le=1.0)
    auto_save_failure: bool = True
    gen_improvement_hint: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    question: str
    mode_used: str
    latency_ms: float
    evaluation: Optional[dict] = None
    quality_report: Optional[dict] = None
    failure_types: List[str] = Field(default_factory=list)
    failure_saved: bool = False
    tool_calls: List[dict] = Field(default_factory=list)
    async_used: bool = False
    request_id: str


class UploadResponse(BaseModel):
    message: str
    doc_count: int
    chunk_count: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    chunk_count: int
    version: str = "23.0.0"
    timestamp: str


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field("user", pattern="^(user|admin)$")


# =====================================================================
# 미들웨어 — 요청 로깅
# =====================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.0f}ms)")
    return response


# =====================================================================
# 인증 엔드포인트
# =====================================================================

@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """사용자 로그인 — JWT 액세스 토큰 발급"""
    ok = user_manager.authenticate(form_data.username, form_data.password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = user_manager.get_user(form_data.username)
    token = create_access_token({"sub": form_data.username})
    return TokenResponse(
        access_token=token,
        username=form_data.username,
        role=user.get("role", "user"),
    )


@app.post("/auth/register", response_model=dict, tags=["auth"])
async def register(req: UserCreateRequest, current_user: dict = Depends(require_admin)):
    """신규 사용자 등록 (관리자 전용)"""
    success = user_manager.create_user(req.username, req.password, req.role)
    if not success:
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자명입니다")
    return {"message": f"사용자 '{req.username}' 생성 완료", "role": req.role}


@app.get("/auth/me", tags=["auth"])
async def me(current_user: dict = Depends(get_current_user)):
    """현재 로그인 사용자 정보"""
    user = user_manager.get_user(current_user["username"])
    usage = user_manager.get_usage(current_user["username"])
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "usage_today": usage.get("today", 0),
        "usage_this_hour": usage.get("this_hour", 0),
    }


# =====================================================================
# 문서 업로드
# =====================================================================

@app.post("/docs/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 50,
    build_mv: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """PDF 문서 업로드 → 청킹 → 임베딩 → 인덱스 구축"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50 MB
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다")

    try:
        raw_text = extract_text_from_pdf(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF 파싱 실패: {e}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다")

    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="청크 생성 실패")

    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    mv_index = None
    if build_mv:
        try:
            mv_index = build_multi_vector_index(chunks, embeddings)
        except Exception:
            mv_index = None

    sources = [file.filename] * len(chunks)

    _INDEX_STATE["index"]       = index
    _INDEX_STATE["chunks"]      = chunks
    _INDEX_STATE["sources"]     = sources
    _INDEX_STATE["mv_index"]    = mv_index
    _INDEX_STATE["doc_count"]   = _INDEX_STATE["doc_count"] + 1
    _INDEX_STATE["last_upload"] = datetime.now().isoformat()

    logger.info(f"[upload] {file.filename}: {len(chunks)} chunks by {current_user['username']}")

    return UploadResponse(
        message="업로드 및 인덱싱 완료",
        doc_count=_INDEX_STATE["doc_count"],
        chunk_count=len(chunks),
        filename=file.filename,
    )


@app.delete("/docs/reset", tags=["documents"])
async def reset_index(current_user: dict = Depends(require_admin)):
    """인덱스 초기화 (관리자 전용)"""
    _INDEX_STATE.update({"index": None, "chunks": [], "sources": [], "mv_index": None,
                          "doc_count": 0, "last_upload": None})
    answer_cache.clear()
    return {"message": "인덱스 초기화 완료"}


# =====================================================================
# 채팅 (핵심 엔드포인트)
# =====================================================================

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """질문 → RAG 파이프라인 → 답변"""
    username = current_user["username"]

    # Rate limit 확인
    allowed, reason = user_manager.check_rate_limit(username)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    if not _index_ready():
        raise HTTPException(status_code=503, detail="문서가 업로드되지 않았습니다. /docs/upload 먼저 호출하세요")

    request_id = str(uuid.uuid4())[:12]
    t0 = time.time()

    try:
        result = process_rag_query(
            question        = req.question,
            index           = _INDEX_STATE["index"],
            chunks          = _INDEX_STATE["chunks"],
            sources         = _INDEX_STATE["sources"],
            user_id         = username,
            auto_routing    = req.auto_routing,
            mode            = req.mode,
            top_k           = req.top_k,
            prefilter_n     = req.prefilter_n,
            use_rewrite     = req.use_rewrite,
            use_self_refine = req.use_self_refine,
            use_multidoc    = req.use_multidoc,
            use_compression = req.use_compression,
            use_llm_compress    = req.use_llm_compress,
            use_tool_registry   = req.use_tool_registry,
            enabled_tools       = req.enabled_tools,
            use_async_engine    = req.use_async_engine,
            dedup_threshold     = req.dedup_threshold,
            mv_index            = _INDEX_STATE["mv_index"],
            auto_save_failure   = req.auto_save_failure,
            gen_improvement_hint= req.gen_improvement_hint,
        )
    except Exception as e:
        logger.error(f"[chat] pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"파이프라인 오류: {e}")

    latency_ms = (time.time() - t0) * 1000
    user_manager.record_usage(username)

    return ChatResponse(
        answer          = result.get("answer", ""),
        sources         = result.get("sources", []),
        question        = req.question,
        mode_used       = result.get("mode", req.mode),
        latency_ms      = latency_ms,
        evaluation      = result.get("evaluation"),
        quality_report  = result.get("quality_report"),
        failure_types   = result.get("failure_types", []),
        failure_saved   = result.get("failure_saved", False),
        tool_calls      = result.get("tool_calls", []),
        async_used      = req.use_async_engine,
        request_id      = request_id,
    )


# =====================================================================
# 메트릭
# =====================================================================

@app.get("/metrics", tags=["monitoring"])
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """응답 지연 / 평가 통계 / 실패 카운트"""
    report = metrics_collector.get_report()
    failure_summary = {
        "total": failure_dataset.size(),
        "by_type": {
            ft: len(failure_dataset.get_by_type(ft))
            for ft in ["low_accuracy", "hallucination", "incomplete_answer",
                       "retrieval_failure", "low_relevance"]
        },
    }
    logs = eval_log.get_all()
    total_q = len(logs)
    avg_acc = (
        sum(l.get("accuracy_score", 0) for l in logs) / total_q
        if total_q else 0.0
    )
    return {
        "latency": report,
        "failure": failure_summary,
        "total_queries": total_q,
        "avg_accuracy": round(avg_acc, 2),
        "index_ready": _index_ready(),
        "chunk_count": len(_INDEX_STATE["chunks"]),
        "last_upload": _INDEX_STATE["last_upload"],
    }


@app.get("/metrics/latency", tags=["monitoring"])
async def get_latency(current_user: dict = Depends(get_current_user)):
    """P50 / P95 / P99 응답 시간"""
    return metrics_collector.get_report()


# =====================================================================
# 로그
# =====================================================================

@app.get("/logs", tags=["monitoring"])
async def get_logs(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """평가 로그 조회 (최신순)"""
    logs = eval_log.get_all()
    logs_sorted = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
    return {
        "total": len(logs_sorted),
        "items": logs_sorted[offset: offset + limit],
    }


@app.delete("/logs", tags=["monitoring"])
async def clear_logs(current_user: dict = Depends(require_admin)):
    """로그 초기화 (관리자 전용)"""
    eval_log.clear()
    return {"message": "로그 초기화 완료"}


# =====================================================================
# 실패 데이터셋
# =====================================================================

@app.get("/failures", tags=["failures"])
async def get_failures(
    failure_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """실패 케이스 조회"""
    if failure_type:
        items = failure_dataset.get_by_type(failure_type)
    else:
        items = failure_dataset.get_all()
    items_sorted = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)
    return {
        "total": len(items_sorted),
        "items": items_sorted[offset: offset + limit],
    }


@app.get("/failures/export/jsonl", tags=["failures"])
async def export_failures_jsonl(current_user: dict = Depends(get_current_user)):
    """Fine-tune JSONL 내보내기"""
    from fastapi.responses import Response
    data = failure_dataset.export_finetune_jsonl()
    filename = f"failure_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    return Response(
        content=data,
        media_type="application/jsonlines",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/failures/export/json", tags=["failures"])
async def export_failures_json(current_user: dict = Depends(get_current_user)):
    """문제 분석 JSON 내보내기"""
    from fastapi.responses import Response
    data = failure_dataset.export_problems_json()
    filename = f"failure_problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/failures", tags=["failures"])
async def clear_failures(current_user: dict = Depends(require_admin)):
    """실패 데이터셋 초기화 (관리자 전용)"""
    failure_dataset.clear()
    return {"message": "실패 데이터셋 초기화 완료"}


# =====================================================================
# 사용자 관리 (관리자 전용)
# =====================================================================

@app.get("/users", tags=["admin"])
async def list_users(current_user: dict = Depends(require_admin)):
    """사용자 목록 조회"""
    users = user_manager.list_users()
    return {"total": len(users), "users": users}


@app.delete("/users/{username}", tags=["admin"])
async def delete_user(username: str, current_user: dict = Depends(require_admin)):
    """사용자 삭제"""
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="자신의 계정은 삭제할 수 없습니다")
    success = user_manager.delete_user(username)
    if not success:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return {"message": f"사용자 '{username}' 삭제 완료"}


@app.post("/users/{username}/reset-password", tags=["admin"])
async def reset_password(
    username: str,
    new_password: str,
    current_user: dict = Depends(require_admin),
):
    """비밀번호 초기화 (관리자 전용)"""
    success = user_manager.reset_password(username, new_password)
    if not success:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return {"message": f"'{username}' 비밀번호 초기화 완료"}


@app.get("/users/{username}/usage", tags=["admin"])
async def get_user_usage(username: str, current_user: dict = Depends(require_admin)):
    """사용자 사용량 조회"""
    usage = user_manager.get_usage(username)
    if usage is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    return {"username": username, "usage": usage}


# =====================================================================
# 헬스체크
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """서버 상태 확인 (인증 불필요)"""
    return HealthResponse(
        status       = "ok",
        index_ready  = _index_ready(),
        chunk_count  = len(_INDEX_STATE["chunks"]),
        timestamp    = datetime.now().isoformat(),
    )


@app.get("/", tags=["system"])
async def root():
    return {
        "name": "RAG API v23",
        "docs": "/docs",
        "health": "/health",
        "features": ["LLM Compression", "ToolRegistry", "AsyncRAGEngine"],
    }


# =====================================================================
# 엔트리포인트
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server_api:app", host="0.0.0.0", port=port, reload=True)
