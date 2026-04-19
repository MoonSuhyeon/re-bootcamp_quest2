# v27 변동사항 요약 — Corrective RAG (CRAG)

## 테마: "검색 품질 자동 교정"

v26의 Streaming SSE + RAGAS 평가에 이어, v27은 **검색 결과의 신뢰도를 LLM이 직접 채점**하고
부족하면 자동으로 웹 검색을 보완하는 Corrective RAG(CRAG) 파이프라인을 추가합니다.

---

## 핵심 변경

### 1. `CRAGGrader` 클래스 (rag_engine.py)

검색된 청크가 질문에 얼마나 유용한지 LLM이 **0~10점**으로 채점합니다.

```
score >= CRAG_RELEVANCE_THRESHOLD (5.0)  → Correct   ✅ 내부 문서로 생성
score >= CRAG_AMBIGUOUS_THRESHOLD (3.0)  → Ambiguous 🟡 웹 검색 보완 후 생성
score < CRAG_AMBIGUOUS_THRESHOLD (3.0)   → Incorrect 🔴 웹 검색 결과로만 생성
```

```python
grade_result = crag_grader.grade(question, chunks)
# → {"score": 7.2, "label": "Correct", "reason": "문서에 직접 답변 포함"}
```

### 2. `extract_work_title()` + `build_literary_web_query()` (rag_engine.py) — 문학 특화

소설 등 문학 텍스트를 업로드했을 때 웹 검색 쿼리를 훨씬 정확하게 만드는 두 헬퍼 함수.

```python
title, author = extract_work_title(chunks)
# LLM이 청크 앞 3개를 보고 → {"title": "소나기", "author": "황순원"}
# 문학 작품이 아니면 ("", "") 반환

web_query = build_literary_web_query(question, title, author)
# title 있을 때: "소나기 황순원 해설 소년과 소녀의 관계는"
# title 없을 때: 원문 질문 그대로
```

웹 검색 실패 시 원문 질문으로 자동 재시도(fallback).

### 3. `run_crag_pipeline()` 함수 (rag_engine.py)

```
retrieve_for_streaming()
  ↓
CRAGGrader.grade() → label 판정
  ↓
Correct   → 내부 chunks 그대로 사용
Ambiguous → extract_work_title() → build_literary_web_query()
            → 웹 해설 검색 → 내부 chunks + 웹 결과 병합
Incorrect → 위와 동일 → 웹 결과만 사용
  ↓
LLM 답변 생성
```

반환 딕셔너리:
```python
{
  "answer": "...",
  "crag_grade":     {"score": 4.1, "label": "Ambiguous", "reason": "..."},
  "web_used":       True,
  "web_query_used": "소나기 황순원 해설 소년과 소녀의 관계는",  # [NEW]
}
```

### 3. `process_rag_query()` — `use_crag` 파라미터 추가

```python
process_rag_query(..., use_crag=False)
```

- `use_crag=True` 이면 CRAG 파이프라인으로 위임 (Multi-Hop과 동일한 early-return 패턴)
- 기존 Fallback 루프와 독립적으로 동작

### 4. `routers/chat.py` — ChatRequest + StreamRequest 업데이트

```python
class ChatRequest(BaseModel):
    ...
    use_crag: bool = False   # [NEW v27]

class StreamRequest(BaseModel):
    ...
    use_crag: bool = False   # [NEW v27]
```

스트리밍 엔드포인트(`/chat/stream`)에서 CRAG 분기 시 **status 이벤트** 발송:

```json
{"type": "status", "content": "📋 CRAG: 문서 검색 및 채점 중..."}
{"type": "status", "content": "🌐 CRAG: Ambiguous (점수 4.2/10) → 웹 해설 검색: \"소나기 황순원 해설 ...\""}
{"type": "done",   "crag_grade": {"label": "Ambiguous", "score": 4.1, ...}, "web_query_used": "...", ...}
```

**버그 수정**: 이전 구현에서 `stream_generate_answer()`를 재호출해 LLM을 두 번 사용하던 문제 수정.
CRAG는 non-streaming으로 이미 생성된 답변을 **단어 단위 pseudo-스트리밍**으로 전달 (LLM 1회 호출).

### 5. `client_app.py` — CRAG UI

- 사이드바 **[NEW v27] Corrective RAG** 섹션 추가
- `use_crag` 체크박스 → 스트리밍 요청에 전달
- status 이벤트 → `st.caption()` 으로 실시간 표시
- done 이벤트의 `crag_grade` → 🟢/🟡/🔴 아이콘 + 점수 + 이유 + **웹 검색 쿼리** 표시

```
🌐 CRAG 등급: Ambiguous (점수 4.2/10) — 일부 관련 내용 있으나 불충분
🔍 웹 검색 쿼리: `소나기 황순원 해설 소년과 소녀의 관계는`
```

### 6. `config.py` — CRAG 임계값 상수

```python
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "5.0"))
CRAG_AMBIGUOUS_THRESHOLD = float(os.getenv("CRAG_AMBIGUOUS_THRESHOLD", "3.0"))
```

환경변수로 임계값 조정 가능.

---

## 파일 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `config.py` | `CRAG_RELEVANCE_THRESHOLD`, `CRAG_AMBIGUOUS_THRESHOLD` 추가, 파일명 v27 |
| `rag_engine.py` | `CRAGGrader`, `extract_work_title()`, `build_literary_web_query()`, `run_crag_pipeline()`, `process_rag_query(use_crag=)` |
| `routers/chat.py` | `ChatRequest.use_crag`, `StreamRequest.use_crag`, `/chat/stream` CRAG 분기 (버그 수정 포함), `web_query_used` 전달 |
| `client_app.py` | CRAG 사이드바 토글, status/crag_grade/web_query UI, 버전 v27 |
| `server_api.py` | 버전 v27, description 업데이트 |

---

## 임계값 튜닝 가이드

| 상황 | 권장 설정 |
|------|----------|
| 내부 문서 품질 높음 | `CRAG_RELEVANCE_THRESHOLD=7.0` (웹 검색 덜 사용) |
| 내부 문서 불완전 | `CRAG_RELEVANCE_THRESHOLD=4.0` (웹 검색 더 자주) |
| 웹 검색 비용 절감 | `CRAG_AMBIGUOUS_THRESHOLD=1.0` (완전 실패만 교정) |

---

## RAGAS와의 연계

CRAG 적용 전후 `ragas_log_v27.json` 비교:

```bash
python evaluate_ragas.py --last 20
```

- Faithfulness: 웹 보완 후 환각 감소 여부 확인
- Context Precision: Incorrect→웹 전환 후 관련성 개선 여부 확인
