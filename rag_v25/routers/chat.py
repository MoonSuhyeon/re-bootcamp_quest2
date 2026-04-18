# routers/chat.py — RAG 채팅 엔드포인트  [v25: use_multihop, use_self_rag 추가]

import uuid
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from config import DEDUP_THRESHOLD_DEFAULT
from deps import INDEX_STATE, index_ready, get_current_user
from rag_engine import process_rag_query, user_manager

logger = logging.getLogger("rag_api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str               = Field(..., min_length=1, max_length=2000)
    mode: str                   = Field("hybrid", description="hybrid | dense | bm25")
    top_k: int                  = Field(3, ge=1, le=20)
    prefilter_n: int            = Field(10, ge=5, le=100)
    use_rewrite: bool           = True
    use_reranking: bool         = True
    use_self_refine: bool       = True
    use_multidoc: bool          = True
    use_compression: bool       = False
    use_llm_compress: bool      = False
    use_tool_registry: bool     = False
    enabled_tools: List[str]    = Field(default_factory=lambda: ["calculator", "datetime"])
    use_async_engine: bool      = False
    use_parallel_search: bool   = True
    auto_routing: bool          = True
    dedup_threshold: float      = Field(DEDUP_THRESHOLD_DEFAULT, ge=0.0, le=1.0)
    auto_save_failure: bool     = True
    gen_improvement_hint: bool  = False
    use_session_cache: bool     = True
    use_multihop: bool          = False  # [NEW v25]
    use_self_rag: bool          = False  # [NEW v25]


class ChatResponse(BaseModel):
    answer:         str
    sources:        List[str]
    question:       str
    mode_used:      str
    latency_ms:     float
    evaluation:     Optional[dict] = None
    quality_report: Optional[dict] = None
    failure_types:  List[str]      = Field(default_factory=list)
    failure_saved:  bool           = False
    tool_calls:     List[dict]     = Field(default_factory=list)
    async_used:     bool           = False
    request_id:     str
    trace_id:       str            = ""


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """질문 → RAG 파이프라인 → 답변"""
    username = current_user["username"]

    allowed, remaining = user_manager.check_rate_limit(username)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"시간당 요청 한도 초과 (남은 횟수: {remaining}). 잠시 후 다시 시도하세요.",
        )

    if not index_ready():
        raise HTTPException(
            status_code=503,
            detail="문서가 업로드되지 않았습니다. /docs/upload 를 먼저 호출하세요",
        )

    request_id = str(uuid.uuid4())[:12]
    t0         = time.time()
    use_bm25   = req.mode in ("hybrid", "bm25")

    try:
        result = process_rag_query(
            question             = req.question,
            index                = INDEX_STATE["index"],
            chunks               = INDEX_STATE["chunks"],
            sources              = INDEX_STATE["sources"],
            user_id              = username,
            auto_routing         = req.auto_routing,
            use_bm25             = use_bm25,
            use_reranking        = req.use_reranking,
            top_k                = req.top_k,
            use_query_rewrite    = req.use_rewrite,
            prefilter_n          = req.prefilter_n,
            use_multidoc         = req.use_multidoc,
            mv_index             = INDEX_STATE["mv_index"],
            use_parallel_search  = req.use_parallel_search,
            use_self_refine      = req.use_self_refine,
            use_compression      = req.use_compression,
            use_llm_compress     = req.use_llm_compress,
            use_tool_registry    = req.use_tool_registry,
            enabled_tools        = req.enabled_tools,
            use_async_engine     = req.use_async_engine,
            auto_save_failure    = req.auto_save_failure,
            gen_improvement_hint = req.gen_improvement_hint,
            use_session_cache    = req.use_session_cache,
            use_multihop         = req.use_multihop,   # [NEW v25]
            use_self_rag         = req.use_self_rag,   # [NEW v25]
        )
    except Exception as e:
        logger.error(f"pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"파이프라인 오류: {e}")

    latency_ms = (time.time() - t0) * 1000
    trace_id   = result.get("log_entry", {}).get("trace_id", request_id)

    return ChatResponse(
        answer         = result.get("answer", ""),
        sources        = list(set(result.get("final_sources", []))),
        question       = req.question,
        mode_used      = result.get("mode", req.mode),
        latency_ms     = latency_ms,
        evaluation     = result.get("evaluation"),
        quality_report = result.get("quality_report"),
        failure_types  = result.get("failure_types", []),
        failure_saved  = result.get("failure_saved", False),
        tool_calls     = result.get("tool_registry_calls", []),
        async_used     = req.use_async_engine,
        request_id     = request_id,
        trace_id       = trace_id,
    )
