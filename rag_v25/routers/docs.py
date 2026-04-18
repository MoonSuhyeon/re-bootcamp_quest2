# routers/docs.py — 문서 업로드 / 인덱스 관리

import io
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from deps import INDEX_STATE, get_current_user, require_admin
from rag_engine import (
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    build_faiss_index,
    build_multi_vector_index,
    answer_cache,
)

logger = APIRouter.__module__ and logging.getLogger("rag_api.docs")
router = APIRouter(prefix="/docs", tags=["documents"])


class UploadResponse(BaseModel):
    message:     str
    doc_count:   int
    chunk_count: int
    filename:    str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file:       UploadFile = File(...),
    chunk_size: int  = DEFAULT_CHUNK_SIZE,
    overlap:    int  = DEFAULT_OVERLAP,
    build_mv:   bool = True,
    current_user: dict = Depends(get_current_user),
):
    """PDF 업로드 → 청킹 → 임베딩 → 인덱스 구축"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 50 MB를 초과합니다")

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
    index      = build_faiss_index(embeddings)
    mv_index   = None
    if build_mv:
        try:
            mv_index = build_multi_vector_index(chunks, embeddings)
        except Exception:
            mv_index = None

    INDEX_STATE["index"]       = index
    INDEX_STATE["chunks"]      = chunks
    INDEX_STATE["sources"]     = [file.filename] * len(chunks)
    INDEX_STATE["mv_index"]    = mv_index
    INDEX_STATE["doc_count"]   = INDEX_STATE["doc_count"] + 1
    INDEX_STATE["last_upload"] = __import__("datetime").datetime.now().isoformat()

    return UploadResponse(
        message="업로드 및 인덱싱 완료",
        doc_count=INDEX_STATE["doc_count"],
        chunk_count=len(chunks),
        filename=file.filename,
    )


@router.delete("/reset")
async def reset_index(current_user: dict = Depends(require_admin)):
    """인덱스 초기화 (관리자 전용)"""
    INDEX_STATE.update({
        "index": None, "chunks": [], "sources": [], "mv_index": None,
        "doc_count": 0, "last_upload": None,
    })
    answer_cache.clear()
    return {"message": "인덱스 초기화 완료"}
