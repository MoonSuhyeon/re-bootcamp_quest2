# config.py — v27 설정 분리
# 모든 상수/모델명/경로/임계값을 한 곳에서 관리
# 환경변수로 오버라이드 가능 (.env 파일 사용)

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# =====================================================================
# 파일 경로
# =====================================================================

_BASE                = os.path.dirname(os.path.abspath(__file__))
LOG_FILE             = os.path.join(_BASE, "rag_eval_logs_v27.json")
EMBED_CACHE_FILE     = os.path.join(_BASE, "embed_cache_v27.pkl")
ANSWER_CACHE_FILE    = os.path.join(_BASE, "answer_cache_v27.json")
FAILURE_DATASET_FILE = os.path.join(_BASE, "failure_dataset_v27.json")
USERS_FILE           = os.path.join(_BASE, "rag_users_v27.json")
USAGE_LOG_FILE       = os.path.join(_BASE, "rag_usage_v27.json")
RAGAS_LOG_FILE       = os.path.join(_BASE, "ragas_log_v27.json")

# =====================================================================
# 모델
# =====================================================================

LLM_MODEL   = os.getenv("LLM_MODEL",   "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL",  "text-embedding-3-small")

# =====================================================================
# 청킹 기본값
# =====================================================================

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE",    "500"))
DEFAULT_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",  "50"))

# =====================================================================
# 캐시 TTL
# =====================================================================

ANSWER_CACHE_TTL_SEC = int(os.getenv("ANSWER_CACHE_TTL", "1800"))
QUERY_CACHE_TTL_SEC  = int(os.getenv("QUERY_CACHE_TTL",  "3600"))

# =====================================================================
# 검색 / 파이프라인
# =====================================================================

PARALLEL_MAX_WORKERS       = int(os.getenv("PARALLEL_WORKERS",   "4"))
DEDUP_THRESHOLD_DEFAULT    = float(os.getenv("DEDUP_THRESHOLD",  "0.85"))
FAILURE_THRESHOLD_ACCURACY = int(os.getenv("FAILURE_THRESHOLD",  "3"))
LLM_COMPRESS_MAX_SENTS     = int(os.getenv("LLM_COMPRESS_SENTS", "12"))
TOOL_WEBSEARCH_TIMEOUT_SEC = int(os.getenv("WEBSEARCH_TIMEOUT",  "5"))

# [NEW v25] Agentic RAG
MULTIHOP_MAX_HOPS = int(os.getenv("MULTIHOP_MAX_HOPS", "4"))
SELF_RAG_MAX_ITER = int(os.getenv("SELF_RAG_MAX_ITER",  "3"))

# [NEW v27] Corrective RAG (CRAG)
CRAG_RELEVANCE_THRESHOLD  = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "5.0"))   # 0~10 점수
CRAG_AMBIGUOUS_THRESHOLD  = float(os.getenv("CRAG_AMBIGUOUS_THRESHOLD", "3.0"))   # 이 미만 = Incorrect

# =====================================================================
# Rate Limit
# =====================================================================

RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))

# =====================================================================
# Alert 임계값
# =====================================================================

ALERT_ACCURACY_MIN   = float(os.getenv("ALERT_ACCURACY_MIN",  "3.0"))
ALERT_HALL_MAX       = float(os.getenv("ALERT_HALL_MAX",      "0.30"))
ALERT_LATENCY_P95_MS = int(os.getenv("ALERT_LATENCY_P95",    "15000"))

# =====================================================================
# JWT
# =====================================================================

JWT_SECRET_KEY     = os.getenv("JWT_SECRET_KEY", "rag-v27-secret-change-in-production")
JWT_ALGORITHM      = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# =====================================================================
# API 서버
# =====================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))
