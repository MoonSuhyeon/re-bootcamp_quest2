# Advanced RAG System (v2 ~ v27)

단순한 RAG 구현을 넘어, 검색 품질 → 추론 구조 → 관측 가능성 → 자동 평가 → 실험 기반 개선 → 자동 실패 학습 → 병렬화 → 인증/모니터링 → 아키텍처 분리 → Agentic 검색 → 실시간 응답 + 정량 검증 → 검색 품질 자동 교정(CRAG)까지 확장된 엔드투엔드 RAG 엔지니어링 시스템입니다.

---

## 목차

- [시스템 진화](#시스템-진화)
- [버전별 해결 문제 요약](#버전별-해결-문제-요약)
- [최종 스택 (v26)](#최종-스택-v26)
- [빠른 시작](#빠른-시작)
- [폴더 구조](#폴더-구조)
- [기술 스택](#기술-스택)

---

## 시스템 진화

### v2: OpenAI API 기반 전환
- Embedding + GPT를 OpenAI API로 통합
- 로컬 모델 → API 기반으로 전환

### v3: 기본 검색 품질 개선
- 코사인 유사도 기반 검색
- 문단/문장 단위 청킹

### v4: 검색 파이프라인 고도화
- Query Rewriting 추가
- LLM 기반 Reranking 도입

### v5: Chunking 품질 개선
- Overlap chunking
- Semantic chunking (문맥 단절 문제 해결)

### v6: 답변 품질 제어
- 근거 인용(Source attribution)
- 구조화된 답변 포맷 (`📌 요약 / 📖 근거 / ✅ 결론`)

### v7: Multi-document Reasoning
- 3단계 추론 파이프라인 (요약 → 관계 분석 → 최종 답변)
- 문서 간 비교 및 종합 reasoning

### v8: Retrieval 안정성 & Recall 개선
- 의도 분해 Query Rewriting
- Embedding Pre-filter
- `(chunk, source)` 튜플 구조로 출처 정확성 보장

### v9: Observability Layer
- LLM 자동 평가 (accuracy / relevance / hallucination)
- 로그 저장 시스템
- 대시보드 UI

### v10: Full Tracing & AI Observability
- Span 기반 Tracing 시스템
- Token usage / latency 측정, Bottleneck detection
- Hallucination root-cause 분석
- Arize Phoenix 수준 관측성 구현

### v11: 안정성 버그 수정
- 평가 점수 클램핑 (1~5 보장)
- `[출처 N]` 클릭 가능한 앵커 링크

### v12: Hybrid Retrieval + Evaluation
- Dense + BM25 + RRF Hybrid Search
- Embedding Cache 시스템
- NDCG@k 기반 검색 품질 평가

### v13: Query Routing Engine
- 질문 의도 자동 분류 (6 types: factual / definition / multi_hop / reasoning / exploratory / ambiguous)
- 검색 전략 동적 선택

### v14: Ablation Study Framework
- 구성 요소별 파이프라인 자동 실험
- Query Rewrite / BM25 / Rerank 기여도 분석

### v15: Self-Refinement Loop
- Draft → Critique → Refine 3단계 답변 생성
- LLM이 자신의 초안을 스스로 비판하고 재작성

### v16: Multi-Vector Index
- 3-tier 검색: chunk index + sentence index + keyword index
- 세밀한 검색 단위로 Recall 개선

### v17: Context Compression (Phase 2)
- 임베딩 유사도 기반 문장 추출
- Context window 절약 + 노이즈 감소

### v18: Caching System
- `QueryResultCache`: 동일 쿼리 검색 결과 TTL 캐시 (3600초)
- `AnswerCache`: 동일 질문 최종 답변 캐시 (1800초)

### v19: Integrated Pipeline + Fallback
- v16~v18 모든 기능 통합
- 평가 기반 자동 재시도 Fallback (낮은 정확도 감지 시 재검색)

### v20: Failure Dataset & Auto Improvement Loop
- `classify_failure_types()`: 평가 결과 기반 실패 자동 분류
  - `low_accuracy` / `hallucination` / `incomplete_answer` / `retrieval_failure` / `low_relevance`
- `FailureDataset`: 실패 케이스 전용 저장
- Fine-tune JSONL 내보내기 (OpenAI fine-tuning API 직접 사용 가능)

### v21: Parallel Search + Retrieval Metrics
- `ThreadPoolExecutor` 기반 4-way 병렬 검색 (Dense / BM25 / Sentence / Keyword 동시 실행)
- NDCG@k 검색 품질 정량 평가
- Rate Limiting, P50 / P95 / P99 응답 지연 측정

### v22: 인증 / 모니터링 / 자동 라우팅
- JWT 기반 로그인 게이트
- `UserManager`: 사용자 생성·삭제·비밀번호 관리
- 자동 쿼리 라우팅 + Alert 시스템

### v23: 아키텍처 분리 + LLM Compression + ToolRegistry + AsyncRAGEngine

**구조 변경: 모놀리식 → 3파일 분리**

| 파일 | 역할 |
|------|------|
| `rag_engine.py` | 순수 RAG 로직 (UI 없음) |
| `server_api.py` | FastAPI 서버 — JWT 인증 + REST API |
| `client_app.py` | Streamlit 프론트엔드 — HTTP 요청만 수행 |

- **LLM Compression Phase 3**: LLM이 전역 청크에서 관련 문장 직접 선별
- **ToolRegistry**: OpenAI function calling 기반 4종 도구 (계산기 / 날짜 / 단위변환 / 웹검색)
- **AsyncRAGEngine**: `asyncio.gather()` + `AsyncOpenAI` 비동기 병렬 실행

---

### v24: Config 분리 + Service Layer 분리

**구조 변경: 단일 server_api.py → 라우터 기반 서비스 계층**

```
rag_v24/
├── config.py          ← 모든 상수·모델명·경로 중앙화 (환경변수 오버라이드)
├── deps.py            ← INDEX_STATE + JWT 공유 의존성
├── rag_engine.py      ← config import로 상수 제거
├── server_api.py      ← 얇은 앱 팩토리
└── routers/
    ├── auth.py        ← POST /auth/login, /register, GET /me
    ├── docs.py        ← POST /docs/upload, DELETE /docs/reset
    ├── chat.py        ← POST /chat
    ├── metrics.py     ← GET /metrics, /logs, /logs/{trace_id}, /failures
    └── admin.py       ← GET/POST/DELETE /users
```

- 모든 하드코딩 상수 → `config.py` 한 곳 관리
- FastAPI 표준 라우터 구조 적용

---

### v25: Multi-Hop Reasoning (Agentic RAG) + Self-RAG

**Multi-Hop Reasoning**

복합 질문("A와 B의 차이점", "원인과 결과")을 엔진이 스스로 분해해 단계별로 검색합니다.

```
질문 → needs_multihop() 판단 → decompose() 분해
  → hop1 검색 + 중간답변
  → hop2 검색 + 중간답변
  → ...
  → synthesize() 최종 종합
```

**Self-RAG (Self-Correction)**

검색 직후 LLM이 "이 결과가 충분한가?" 판단 → 부족하면 쿼리 재작성 후 재검색 (최대 3회).

```
검색 결과 → check_sufficiency() 판단
  → 충분: 그대로 진행
  → 부족: rewrite_hint로 재검색 → 결과 병합
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `use_multihop` | `false` | Multi-Hop 파이프라인 활성화 |
| `use_self_rag` | `false` | Self-RAG 재검색 루프 활성화 |

---

### v26: Streaming SSE + RAGAS Evaluation

**테마: "실시간 응답 + 정량 검증"**

**Streaming 응답 (`POST /chat/stream`)**

```
검색 (동기) → 스트리밍 생성 (yield token) → RAGAS 로그 저장 → done 이벤트
```

SSE 이벤트:
```json
data: {"type": "token",  "content": "..."}
data: {"type": "done",   "trace_id": "abc", "latency_ms": 1200, "sources": [...]}
```

**RAGAS Evaluation (오프라인 평가 도구)**

스트리밍 응답이 쌓인 `ragas_log_v26.json`을 읽어 품질을 정량 평가합니다.

```bash
python evaluate_ragas.py               # 전체 로그 평가
python evaluate_ragas.py --last 20     # 최근 20개만
```

| RAGAS 메트릭 | 설명 |
|-------------|------|
| Faithfulness | 답변이 검색 문서에 근거하는 정도 (환각 탐지) |
| Answer Relevancy | 답변이 질문에 적절한 정도 |
| Context Precision | 검색된 문서가 실제로 유용한 비율 |

---

### v27: Corrective RAG (CRAG) + 문학 특화 웹 검색

**테마: "검색 품질 자동 교정 + 문학·세계관 해설 보완"**

검색 결과를 LLM이 채점하고, 점수에 따라 내부 문서 / 웹 검색 보완 / 웹 검색 대체를 자동 선택합니다.

```
검색 → CRAGGrader.grade() → Correct / Ambiguous / Incorrect
  → Correct   : 내부 문서로 생성
  → Ambiguous : 내부 문서 + 문학 해설 웹 검색 병합 후 생성
  → Incorrect : 문학 해설 웹 검색 결과로만 생성
```

**문학 특화 웹 검색 쿼리 자동 생성**

소설·세계관 문서를 업로드하면 검색이 부족할 때 단순 질문이 아닌 해설 특화 쿼리로 웹을 탐색합니다.

```python
# extract_work_title(): 청크에서 작품 제목·저자 자동 추출
title, author = extract_work_title(chunks)
# → ("소나기", "황순원")

# build_literary_web_query(): 해설 특화 쿼리 생성
query = build_literary_web_query(question, title, author)
# → "소나기 황순원 해설 소년과 소녀의 관계는"
```

SSE 스트리밍에 **status 이벤트** 추가:
```json
{"type": "status",  "content": "📋 CRAG: 문서 검색 및 채점 중..."}
{"type": "status",  "content": "🌐 CRAG: Ambiguous (점수 4.2/10) → 웹 해설 검색: \"소나기 황순원 해설 ...\""}
{"type": "done",    "crag_grade": {"label": "Ambiguous", "score": 4.1, ...}, "web_query_used": "...", ...}
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `use_crag` | `false` | CRAG 파이프라인 활성화 |
| `CRAG_RELEVANCE_THRESHOLD` | `5.0` | 이상이면 Correct (환경변수 조정 가능) |
| `CRAG_AMBIGUOUS_THRESHOLD` | `3.0` | 이상이면 Ambiguous, 미만이면 Incorrect |

---

## 버전별 해결 문제 요약

| 버전 범위 | 테마 | 핵심 해결 |
|-----------|------|----------|
| v2~v5 | 기반 구축 | API 전환, 검색 품질, 청킹 |
| v6~v8 | 답변·검색 고도화 | 출처 인용, Multi-doc, Recall |
| v9~v11 | 관측 가능성 | 자동 평가, Tracing, 버그 수정 |
| v12~v14 | Hybrid + 실험 | BM25 RRF, 라우팅, Ablation |
| v15~v17 | 품질 자동화 | Self-Refine, Multi-Vector, Compression |
| v18~v19 | 운영 안정성 | Cache, Fallback, 통합 평가 |
| v20 | 학습 루프 | 실패 → 데이터셋 → Fine-tune |
| v21~v22 | 프로덕션 준비 | 병렬화, 인증, 모니터링 |
| v23 | 서비스 분리 | API 서버, 비동기 엔진, 도구 호출 |
| v24 | 설정·구조 분리 | Config 중앙화, Service Layer (routers/) |
| v25 | Agentic 검색 | Multi-Hop Reasoning, Self-RAG |
| v26 | 실시간 + 검증 | Streaming SSE, RAGAS 정량 평가 |
| v27 | 검색 품질 교정 | Corrective RAG (CRAG), 웹 검색 자동 보완 |

---

## 최종 스택 (v27)

```
[Client] Streamlit (client_app.py)
    ↓ HTTP / SSE
[Server] FastAPI (server_api.py)
    ├── routers/auth.py      — JWT 인증
    ├── routers/docs.py      — PDF 업로드 · 인덱싱
    ├── routers/chat.py      — RAG 질의응답 · Streaming · CRAG
    ├── routers/metrics.py   — 지표 · 로그 · 실패 데이터셋
    └── routers/admin.py     — 사용자 관리
    ↓
[Engine] rag_engine.py
    ├── CRAGGrader            — 검색 결과 채점 (0~10) · Grade 분류   [NEW v27]
    ├── MultiHopPlanner       — 질문 분해 · hop별 검색 · 종합
    ├── SelfRAGChecker        — 검색 충분성 판단 · 자기 교정
    ├── AsyncRAGEngine        — asyncio 기반 비동기 파이프라인
    ├── ToolRegistry          — Calculator · DateTime · WebSearch
    ├── MetricsCollector      — P50/P95/P99 지연 측정
    ├── FailureDataset        — 실패 케이스 저장 · JSONL export
    └── UserManager           — 사용자 인증 · Rate Limit
    ↓
[Index] FAISS (IndexFlatIP)
    ├── chunk_index    — 청크 단위 벡터
    ├── sent_index     — 문장 단위 벡터
    └── kw_index       — 키워드 단위 벡터
    ↓
[Eval] evaluate_ragas.py  — 오프라인 RAGAS 평가 도구
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 루트에 .env 파일 생성
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
pip install ragas datasets   # RAGAS 평가 시 필요
```

### 3. 서버 실행 (터미널 1)

```bash
cd rag_v27
python server_api.py
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### 4. 클라이언트 실행 (터미널 2)

```bash
cd rag_v27
streamlit run client_app.py
# → http://localhost:8501
```

### 5. 기본 계정

| 계정 | 비밀번호 | 권한 |
|------|----------|------|
| `admin` | `admin123` | 관리자 |
| `demo` | `demo123` | 일반 사용자 |

### 6. RAGAS 평가 실행

```bash
# /chat/stream 으로 질문을 몇 개 보낸 후
cd rag_v27
python evaluate_ragas.py --last 20
```

---

## 폴더 구조

```
rag-app/
├── rag_v23/              ← 3파일 분리 완성 (안정 버전)
│   ├── rag_engine.py
│   ├── server_api.py
│   └── client_app.py
├── rag_v24/              ← Config 분리 + Service Layer
│   ├── config.py
│   ├── deps.py
│   ├── rag_engine.py
│   ├── server_api.py
│   ├── client_app.py
│   └── routers/
├── rag_v25/              ← Multi-Hop + Self-RAG
│   └── (v24 구조 동일)
├── rag_v26/              ← Streaming + RAGAS
│   ├── evaluate_ragas.py
│   └── (v25 구조 동일)
├── rag_v27/              ← Corrective RAG (CRAG) (최신)
│   ├── evaluate_ragas.py
│   └── (v26 구조 동일)
├── rag_versions/         ← v2~v22 단일 파일 히스토리
├── change_logs/          ← 버전별 변동사항 요약
└── requirements.txt
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | OpenAI GPT-4o-mini (환경변수로 교체 가능) |
| Embedding | text-embedding-3-small |
| Vector DB | FAISS (IndexFlatIP) |
| API 서버 | FastAPI + Uvicorn |
| 프론트엔드 | Streamlit |
| 인증 | JWT (python-jose / PyJWT) |
| 검색 | Dense + BM25 (rank-bm25) + RRF Hybrid |
| 평가 | RAGAS (Faithfulness / Answer Relevancy / Context Precision) |
| 스트리밍 | FastAPI StreamingResponse + SSE |
| 검색 교정 | Corrective RAG — LLM Grade + DuckDuckGo 웹 검색 |
