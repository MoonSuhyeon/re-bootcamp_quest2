# routers/metrics.py — 메트릭 / 로그 / 실패 데이터셋 엔드포인트

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from deps import INDEX_STATE, get_current_user, require_admin
from rag_engine import eval_log, metrics_collector, failure_dataset

router = APIRouter(tags=["monitoring"])


# =====================================================================
# 메트릭
# =====================================================================

@router.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """응답 지연 / 평가 통계 / 실패 카운트"""
    report = metrics_collector.get_report()
    logs   = eval_log.get_all()
    total  = len(logs)
    avg_acc = (
        sum(l.get("evaluation", {}).get("정확도", 0) for l in logs) / total
        if total else 0.0
    )
    return {
        "latency":       report,
        "failure":       {
            "total":   failure_dataset.size(),
            "by_type": {
                ft: len(failure_dataset.get_by_type(ft))
                for ft in ["low_accuracy", "hallucination", "incomplete_answer",
                           "retrieval_failure", "low_relevance"]
            },
        },
        "total_queries": total,
        "avg_accuracy":  round(avg_acc, 2),
        "index_ready":   INDEX_STATE["index"] is not None,
        "chunk_count":   len(INDEX_STATE["chunks"]),
        "last_upload":   INDEX_STATE["last_upload"],
    }


@router.get("/metrics/latency")
async def get_latency(current_user: dict = Depends(get_current_user)):
    """P50 / P95 / P99 응답 시간"""
    return metrics_collector.get_report()


# =====================================================================
# 로그
# =====================================================================

@router.get("/logs")
async def get_logs(
    limit:  int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """평가 로그 조회 (최신순)"""
    logs    = eval_log.get_all()
    sorted_ = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"total": len(sorted_), "items": sorted_[offset: offset + limit]}


@router.get("/logs/{trace_id}")
async def get_log_detail(
    trace_id: str,
    current_user: dict = Depends(get_current_user),
):
    """특정 trace_id 의 전체 상세 데이터 (NDCG · spans · fallback · 압축 통계)"""
    for log in eval_log.get_all():
        if log.get("trace_id") == trace_id:
            return log
    raise HTTPException(status_code=404, detail=f"trace_id '{trace_id}' 를 찾을 수 없습니다")


@router.delete("/logs")
async def clear_logs(current_user: dict = Depends(require_admin)):
    """로그 초기화 (관리자 전용)"""
    eval_log.clear()
    return {"message": "로그 초기화 완료"}


# =====================================================================
# 실패 데이터셋
# =====================================================================

@router.get("/failures")
async def get_failures(
    failure_type: Optional[str] = None,
    limit:  int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    items   = (failure_dataset.get_by_type(failure_type)
               if failure_type else failure_dataset.get_all())
    sorted_ = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"total": len(sorted_), "items": sorted_[offset: offset + limit]}


@router.get("/failures/export/jsonl")
async def export_failures_jsonl(current_user: dict = Depends(get_current_user)):
    """Fine-tune JSONL 내보내기"""
    data     = failure_dataset.export_finetune_jsonl()
    filename = f"failure_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    return Response(
        content=data, media_type="application/jsonlines",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/failures/export/json")
async def export_failures_json(current_user: dict = Depends(get_current_user)):
    """문제 분석 JSON 내보내기"""
    data     = failure_dataset.export_problems_json()
    filename = f"failure_problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        content=data, media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/failures")
async def clear_failures(current_user: dict = Depends(require_admin)):
    failure_dataset.clear()
    return {"message": "실패 데이터셋 초기화 완료"}
