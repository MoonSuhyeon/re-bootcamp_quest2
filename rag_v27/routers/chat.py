# routers/chat.py — RAG 채팅 엔드포인트  [v27: CRAG 추가]

import json
import uuid
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import DEDUP_THRESHOLD_DEFAULT
from deps import INDEX_STATE, index_ready, get_current_user
from rag_engine import (
    process_rag_query, user_manager,
    retrieve_for_streaming, stream_generate_answer,
    build_ragas_log_entry, save_ragas_log,
    run_crag_pipeline,   # [NEW v27]
)

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
    use_crag: bool              = False  # [NEW v27]


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
            use_multihop         = req.use_multihop,
            use_self_rag         = req.use_self_rag,
            use_crag             = req.use_crag,
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


# =====================================================================
# [NEW v26] 스트리밍 엔드포인트 (SSE)
# =====================================================================

class StreamRequest(BaseModel):
    question:           str   = Field(..., min_length=1, max_length=2000)
    mode:               str   = Field("hybrid")
    top_k:              int   = Field(3, ge=1, le=20)
    prefilter_n:        int   = Field(10, ge=5, le=100)
    use_rewrite:        bool  = True
    use_reranking:      bool  = True
    use_parallel_search: bool = True
    use_crag:           bool  = False   # [NEW v27]


@router.post("/stream", tags=["chat"])
async def chat_stream(
    req: StreamRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    스트리밍 RAG 응답 (SSE).
    [v27] use_crag=true 이면 CRAG 파이프라인(Grade → Branch → WebSearch) 적용.

    SSE 이벤트 형식:
      data: {"type": "token",   "content": "..."}\n\n
      data: {"type": "status",  "content": "..."}\n\n   ← CRAG 중간 상태
      data: {"type": "done",    "trace_id": "...", "latency_ms": 123, "sources": [...], "crag_grade": {...}}\n\n
    """
    username = current_user["username"]

    allowed, remaining = user_manager.check_rate_limit(username)
    if not allowed:
        raise HTTPException(status_code=429,
                            detail=f"시간당 요청 한도 초과 (남은 횟수: {remaining})")

    if not index_ready():
        raise HTTPException(status_code=503,
                            detail="문서가 업로드되지 않았습니다. /docs/upload 를 먼저 호출하세요")

    use_bm25 = req.mode in ("hybrid", "bm25")
    eff = {
        "use_bm25":          use_bm25,
        "use_reranking":     req.use_reranking,
        "top_k":             req.top_k,
        "use_query_rewrite": req.use_rewrite,
    }

    def generate():
        t0       = time.time()
        trace_id = str(uuid.uuid4())[:8]

        # ── [v27] CRAG 분기 ──────────────────────────────────────────
        if req.use_crag:
            yield f"data: {json.dumps({'type': 'status', 'content': '📋 CRAG: 문서 검색 및 채점 중...'}, ensure_ascii=False)}\n\n"
            try:
                crag_result = run_crag_pipeline(
                    req.question, eff,
                    INDEX_STATE["index"], INDEX_STATE["chunks"], INDEX_STATE["sources"],
                    prefilter_n         = req.prefilter_n,
                    mv_index            = INDEX_STATE["mv_index"],
                    use_parallel_search = req.use_parallel_search,
                )
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
                return

            grade          = crag_result.get("crag_grade", {})
            label          = grade.get("label", "Correct")
            web_used       = crag_result.get("web_used", False)
            web_query_used = crag_result.get("web_query_used", "")
            score          = grade.get("score", 0)

            if web_used:
                msg = (f'🌐 CRAG: {label} (점수 {score:.1f}/10) '
                       f'→ 웹 해설 검색: "{web_query_used}"')
            else:
                msg = f'✅ CRAG: {label} (점수 {score:.1f}/10) → 내부 문서로 답변 생성'
            yield f"data: {json.dumps({'type': 'status', 'content': msg}, ensure_ascii=False)}\n\n"

            full_answer = crag_result.get("answer", "")
            use_chunks  = [r[0] for r in crag_result.get("ranked", [])]
            use_src     = crag_result.get("final_sources", [])

            # CRAG는 non-streaming으로 이미 답변 생성 완료
            # → 단어 단위 pseudo-스트리밍으로 타이핑 효과 제공 (LLM 재호출 없음)
            import re as _re
            tokens = _re.split(r'(\s+)', full_answer)
            for tok in tokens:
                if tok:
                    yield f"data: {json.dumps({'type': 'token', 'content': tok}, ensure_ascii=False)}\n\n"

            latency_ms  = (time.time() - t0) * 1000
            ragas_entry = build_ragas_log_entry(
                question=req.question, answer=full_answer, contexts=use_chunks,
                trace_id=trace_id, latency_ms=latency_ms, user_id=username,
            )
            try:
                save_ragas_log(ragas_entry)
            except Exception as e:
                logger.warning(f"RAGAS 로그 저장 실패: {e}")

            user_manager.record_usage(username)
            yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id, 'latency_ms': round(latency_ms, 1), 'sources': list(set(use_src)), 'crag_grade': grade, 'web_query_used': web_query_used}, ensure_ascii=False)}\n\n"
            return

        # ── 기본 스트리밍 (v26 동일) ──────────────────────────────────
        # ── Step 1: 검색 (동기) ──────────────────────────────────────
        try:
            final_chunks, final_sources = retrieve_for_streaming(
                req.question, eff,
                INDEX_STATE["index"], INDEX_STATE["chunks"], INDEX_STATE["sources"],
                prefilter_n    = req.prefilter_n,
                mv_index       = INDEX_STATE["mv_index"],
                use_parallel_search = req.use_parallel_search,
                top_k          = req.top_k,
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            return

        if not final_chunks:
            yield f"data: {json.dumps({'type': 'error', 'content': '관련 문서를 찾을 수 없습니다'}, ensure_ascii=False)}\n\n"
            return

        # ── Step 2: 스트리밍 생성 ────────────────────────────────────
        full_answer = ""
        try:
            for token in stream_generate_answer(req.question, final_chunks):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            return

        # ── Step 3: RAGAS 로그 저장 ──────────────────────────────────
        latency_ms = (time.time() - t0) * 1000
        ragas_entry = build_ragas_log_entry(
            question   = req.question,
            answer     = full_answer,
            contexts   = final_chunks,
            trace_id   = trace_id,
            latency_ms = latency_ms,
            user_id    = username,
        )
        try:
            save_ragas_log(ragas_entry)
        except Exception as e:
            logger.warning(f"RAGAS 로그 저장 실패: {e}")

        user_manager.record_usage(username)

        # ── Step 4: 완료 이벤트 ──────────────────────────────────────
        yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id, 'latency_ms': round(latency_ms, 1), 'sources': list(set(final_sources))}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
