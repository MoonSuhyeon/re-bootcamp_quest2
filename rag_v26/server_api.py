# server_api.py — v26 FastAPI 앱 팩토리
# 라우터 조립만 담당 — 비즈니스 로직은 routers/ 에 있음

import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT
from routers import auth, docs, chat, metrics, admin
from deps import INDEX_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

# =====================================================================
# 앱 생성
# =====================================================================

app = FastAPI(
    title="RAG API v26",
    description="Streaming SSE · RAGAS Evaluation · Multi-Hop Reasoning · Self-RAG · Config 분리",
    version="26.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 미들웨어
# =====================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    ms       = (time.time() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.0f}ms)")
    return response

# =====================================================================
# 라우터 등록
# =====================================================================

app.include_router(auth.router)
app.include_router(docs.router)
app.include_router(chat.router)
app.include_router(metrics.router)
app.include_router(admin.router)

# =====================================================================
# 시스템 엔드포인트
# =====================================================================

@app.get("/health", tags=["system"])
async def health():
    from datetime import datetime
    return {
        "status":      "ok",
        "index_ready": INDEX_STATE["index"] is not None,
        "chunk_count": len(INDEX_STATE["chunks"]),
        "version":     "26.0.0",
        "timestamp":   datetime.now().isoformat(),
    }


@app.get("/", tags=["system"])
async def root():
    return {
        "name":     "RAG API v26",
        "docs":     "/docs",
        "health":   "/health",
        "features": ["Streaming SSE", "RAGAS Evaluation", "Multi-Hop Reasoning", "Self-RAG", "Config 분리"],
    }

# =====================================================================
# 엔트리포인트
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_api:app", host=API_HOST, port=API_PORT, reload=True)
