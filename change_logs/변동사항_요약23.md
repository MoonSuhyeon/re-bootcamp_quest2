# rag_app v23 변동사항 요약

## 핵심 변경: 모놀리식 → 3파일 분리 + 3가지 신규 기능

v22 단일 파일(3331줄)을 **rag_engine.py / server_api.py / client_app.py** 3개로 분리하고,  
**LLM Compression(Phase 3) · ToolRegistry · AsyncRAGEngine** 을 추가.

---

## 파일 구조

| 파일 | 역할 | 비고 |
|------|------|------|
| `rag_engine.py` | 순수 RAG 로직 (UI 없음) | `process_rag_query()` 단일 진입점 |
| `server_api.py` | FastAPI 서버 (엔드포인트) | rag_engine 을 import해서 HTTP로 노출 |
| `client_app.py` | Streamlit 프론트엔드 | 서버 HTTP 요청만 수행, 로컬 로직 없음 |
| `변동사항_요약23.md` | 이 문서 | — |

---

## 최종 스택

| 구성 | v22 (monolith) | v23 (3-file split) |
|------|----------------|---------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 + AsyncOpenAI |
| 검색 방식 | Multi-Vector + BM25 RRF | 동일 |
| 병렬 검색 | ThreadPoolExecutor | **[NEW] asyncio.gather() (AsyncRAGEngine)** |
| Context Compression | 임베딩 유사도 문장 추출 | 동일 + **[NEW] LLM Compression (Phase 3)** |
| 도구 사용 | (없음) | **[NEW] ToolRegistry (함수 호출 4종)** |
| API 서버 | (없음 — Streamlit only) | **[NEW] FastAPI server_api.py** |
| 프론트엔드 | Streamlit 직접 처리 | **Streamlit → HTTP only (client_app.py)** |
| 인증 | session_state 로컬 | **[NEW] JWT Bearer Token** |

---

## [NEW v23] 기능 1 — LLM Context Compression (Phase 3)

### `compress_chunks_llm()` in rag_engine.py

```python
def compress_chunks_llm(question: str, chunks: list,
                         max_total_sentences: int = LLM_COMPRESS_MAX_SENTS,
                         tracer=None) -> tuple:
```

| 단계 | 내용 |
|------|------|
| 1 | 각 청크를 문장 분리 → `"ci_si"` 형식 ID 부여 (예: `"0_0"`, `"1_2"`) |
| 2 | 총 문장 수 ≤ max_total_sentences 이면 압축 생략 |
| 3 | LLM에 질문 + 전체 문장 목록 전달 → 관련 문장 ID 선택 요청 |
| 4 | 선택된 ID로 청크 재구성 (원본 청크 구조 유지, 선택 문장만 포함) |
| 5 | (compressed_chunks, stats) 반환 |

- **Phase 1**: BM25 필터 (v21부터)
- **Phase 2**: 임베딩 유사도 문장 추출 (v22부터, `compress_chunks()`)
- **Phase 3**: LLM 직접 선별 (v23 신규) — 전역 관점에서 가장 관련 높은 N문장 선택

```python
LLM_COMPRESS_MAX_SENTS = 12  # [NEW v23 상수]
```

---

## [NEW v23] 기능 2 — ToolRegistry (함수 호출)

### 클래스 계층

```
Tool (추상 베이스)
├── CalculatorTool    — 수식 계산 (ast.parse 안전 eval)
├── DateTimeTool      — 현재시각 / 날짜 차이 / 날짜 덧셈
├── UnitConverterTool — 길이·무게·온도 단위 변환 (_FACTORS dict)
└── WebSearchTool     — DuckDuckGo Instant Answer API (httpx, 키 불필요)
```

### `ToolRegistry.run_with_llm()`

```python
def run_with_llm(self, question, context, enabled_names, tracer) -> tuple:
    # 1. OpenAI function calling (tool_choice="auto")
    # 2. 반환된 tool_calls 모두 실행
    # 3. 도구 결과 포함 두 번째 LLM 호출 → final_answer
    # returns (final_answer, tool_calls_list)
```

#### 상수

```python
TOOL_WEBSEARCH_TIMEOUT_SEC = 5  # [NEW v23]
```

#### 보안

- `CalculatorTool`: `ast.parse(expr, mode='eval')` — 문장(statement) 불가, 표현식만 허용
- 화이트리스트 `safe_globals`: `abs`, `round`, `sum`, `min`, `max`, `pow` 만 허용

---

## [NEW v23] 기능 3 — AsyncRAGEngine

### `AsyncRAGEngine` in rag_engine.py

```python
class AsyncRAGEngine:
    @staticmethod
    def run_sync(coro):
        # Streamlit 호환 — 실행 중인 이벤트 루프 감지
        try:
            asyncio.get_running_loop()
            # 이미 루프 있음 → 별도 스레드에서 asyncio.run()
            with ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    async def retrieve_parallel_async(self, ...):
        results = await asyncio.gather(
            self._dense_async(...),    # FAISS dense 검색
            self._bm25_async(...),     # BM25 검색
            self._sentence_async(...), # 문장 레벨 검색 (Multi-Vector)
            self._keyword_async(...),  # 키워드 레벨 검색 (Multi-Vector)
        )
        # RRF 병합 → (result, parallel_ms) 반환
```

#### AsyncOpenAI

```python
async_client = AsyncOpenAI(api_key=...)  # [NEW v23 전역]
```

- `async def agenerate_answer(...)` — `await async_client.chat.completions.create(...)`
- `async def arun(...)` — 전체 비동기 파이프라인

---

## 3-File 아키텍처 상세

### rag_engine.py — 순수 로직

```
process_rag_query()        ← 최상위 진입점 (NEW v23)
    ├── auto_routing → run_rag_pipeline()
    ├── fallback loop
    └── eval_log.add() + failure_dataset.add()

run_rag_pipeline()
    ├── Phase 1: BM25 pre-filter
    ├── Phase 2: 임베딩 compression (compress_chunks)
    ├── [NEW] Phase 3: LLM compression (compress_chunks_llm)
    ├── [NEW] ToolRegistry.run_with_llm()
    ├── [NEW] AsyncRAGEngine.run_sync(async_engine.arun())
    └── evaluate() → quality_report()
```

### server_api.py — FastAPI 서버

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/auth/login` | POST | OAuth2PasswordRequestForm → JWT |
| `/auth/register` | POST | 신규 사용자 등록 (관리자) |
| `/auth/me` | GET | 현재 사용자 정보 |
| `/docs/upload` | POST | PDF 업로드 → 인덱싱 |
| `/docs/reset` | DELETE | 인덱스 초기화 (관리자) |
| `/chat` | POST | RAG 질의응답 (핵심) |
| `/metrics` | GET | P50/P95/P99 + 실패 통계 |
| `/metrics/latency` | GET | 지연시간 상세 |
| `/logs` | GET/DELETE | 평가 로그 |
| `/failures` | GET/DELETE | 실패 케이스 |
| `/failures/export/jsonl` | GET | Fine-tune JSONL 다운로드 |
| `/failures/export/json` | GET | 문제 분석 JSON 다운로드 |
| `/users` | GET | 사용자 목록 (관리자) |
| `/users/{username}` | DELETE | 사용자 삭제 (관리자) |
| `/users/{username}/usage` | GET | 사용량 조회 |
| `/health` | GET | 서버 상태 (인증 불필요) |

#### 인증

```python
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "rag-v23-secret-change-in-production")
ALGORITHM  = "HS256"
# python-jose 우선, 없으면 PyJWT fallback
```

#### 인덱스 전역 상태

```python
_INDEX_STATE = {
    "index":    None,    # faiss.Index
    "chunks":   [],
    "sources":  [],
    "mv_index": None,    # MultiVectorIndex | None
    "doc_count": 0,
    "last_upload": None,
}
```

### client_app.py — Streamlit 클라이언트

- 로컬 RAG 로직 **전혀 없음** — 모든 작업을 HTTP로 위임
- JWT 토큰: `st.session_state["token"]` 저장 → `Authorization: Bearer <token>` 헤더
- 서버 URL: 환경변수 `RAG_API_BASE` (기본값: `http://localhost:8000`)

| 탭 | 기능 |
|----|------|
| 💬 챗봇 | `POST /chat` → 스트리밍 없이 응답 표시, 메타데이터 metrics |
| 📋 평가 로그 | `GET /logs` + 정확도 추이 차트 |
| 📊 메트릭 | `GET /metrics` → P50/P95/P99 + 실패 분포 bar chart |
| 🚨 실패 데이터셋 | `GET /failures` + JSONL/JSON 내보내기 버튼 |
| 👑 관리자 | 사용자 관리 / 인덱스 초기화 (관리자만) |
| 🔌 API 정보 | 엔드포인트 목록 + Swagger 링크 |

---

## 파일 상수 변경

| 상수 | v22 | v23 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v22.json` | `rag_eval_logs_v23.json` |
| EMBED_CACHE_FILE | `embed_cache_v22.pkl` | `embed_cache_v23.pkl` |
| ANSWER_CACHE_FILE | `answer_cache_v22.json` | `answer_cache_v23.json` |
| FAILURE_DATASET_FILE | `failure_dataset_v22.json` | `failure_dataset_v23.json` |
| USERS_FILE | `rag_users_v22.json` | `rag_users_v23.json` |
| USAGE_LOG_FILE | `rag_usage_v22.json` | `rag_usage_v23.json` |
| LLM_COMPRESS_MAX_SENTS | (없음) | `12` (NEW) |
| TOOL_WEBSEARCH_TIMEOUT_SEC | (없음) | `5` (NEW) |

---

## LLM 호출 횟수

| 상황 | v22 | v23 |
|------|-----|-----|
| 기본 (압축/도구/힌트 OFF) | 동일 | 동일 |
| LLM Compression ON | 동일 | **+1회** (compress_chunks_llm) |
| ToolRegistry ON + 도구 호출 | — | **+1~2회** (함수 호출 + 최종 응답) |
| Async Engine | 동일 횟수 | 동일 (비동기만, 횟수 불변) |
| 실패 힌트 생성 ON | 동일 | **+1회** (improvement_hint) |

---

## 실행 방법

```bash
# 1. 서버 시작
cd ClaudeProjects/rag-app
uvicorn server_api:app --host 0.0.0.0 --port 8000 --reload

# 2. 클라이언트 시작 (별도 터미널)
RAG_API_BASE=http://localhost:8000 streamlit run client_app.py
```

### 의존성 추가 (v23 신규)

```bash
pip install fastapi uvicorn python-jose[cryptography] httpx
# 또는
pip install fastapi uvicorn PyJWT httpx
```

---

## 버전 진화 요약

```
v1~v10:  파이프라인 구축 (임베딩 · FAISS · BM25 · RRF)
v11~v15: 평가 시스템 (8개 필드 · 품질 리포트)
v16~v19: 최적화 (캐시 · Multi-Vector · Self-Refinement)
v20:     실패 → 학습 루프 (FailureDataset · JSONL export)
v21:     병렬 검색 (ThreadPoolExecutor · NDCG · Rate Limit)
v22:     인증 / 모니터링 / 라우팅 자동화
v23:     아키텍처 분리 + LLM Compression + ToolRegistry + AsyncRAGEngine
```
