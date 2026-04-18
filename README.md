# re-bootcamp_quest2
# 📚 Advanced RAG System (v2 ~ v23 Evolution)

이 프로젝트는 단순한 RAG 구현이 아니라  
검색 품질 → 추론 구조 → 관측 가능성 → 자동 평가 → 실험 기반 개선(Ablation) →  
자동 실패 학습 → 병렬화 → 인증/모니터링 → 아키텍처 분리까지 확장된  
엔드투엔드 RAG 엔지니어링 시스템입니다.

---

# System Evolution

각 버전은 "RAG의 한 문제"를 해결하는 방향으로 확장되었습니다.

---

## 🔹 v2: OpenAI API 기반 전환
- Embedding + GPT를 OpenAI API로 통합
- 로컬 모델 → API 기반 확장

---

## 🔹 v3: 기본 검색 품질 개선
- 코사인 유사도 기반 검색
- 문단/문장 단위 청킹

---

## 🔹 v4: 검색 파이프라인 고도화
- Query Rewriting 추가
- LLM 기반 Reranking 도입
- 검색 정확도 개선

---

## 🔹 v5: Chunking 품질 개선
- Overlap chunking
- Semantic chunking
- 문맥 단절 문제 해결

---

## 🔹 v6: 답변 품질 제어
- 근거 인용 (Source attribution)
- 확신도 기반 출력
- 구조화된 답변 포맷

---

## 🔹 v7: Multi-document Reasoning
- 3단계 추론 파이프라인
- 문서 간 비교 및 종합 reasoning 추가

---

## 🔹 v8: Retrieval 안정성 & Recall 개선
- 의도 분해 Query Rewriting
- Embedding Pre-filter
- (chunk, source) 튜플 구조로 출처 정확성 보장

---

## 🔹 v9: Observability Layer
- LLM 자동 평가 (accuracy / relevance / hallucination)
- 로그 저장 시스템
- Arize-style 대시보드 UI

---

## 🔹 v10: Full Tracing & AI Observability
- Span 기반 Tracing 시스템
- Token usage / latency 측정
- Bottleneck detection
- Hallucination root-cause 분석
- Arize Phoenix 수준 관측성 구현

---

## 🔹 v11: 안정성 버그 수정
- 평가 점수 클램핑 (1~5 보장)
- [출처 N] 클릭 가능한 앵커 링크 구현

---

## 🔹 v12: Hybrid Retrieval + Evaluation
- Dense + BM25 + RRF Hybrid Search
- Embedding Cache 시스템
- NDCG@k 기반 검색 품질 평가

---

## 🔹 v13: Query Routing Engine
- 질문 의도 자동 분류 (6 types)
- 검색 전략 동적 선택
- Retrieval policy 자동화

---

## 🔹 v14: Ablation Study Framework
- 구성 요소별 파이프라인 자동 실험
- Query Rewrite / BM25 / Rerank 기여도 분석
- RAG 시스템을 "실험 가능한 구조"로 전환

---

## 🔹 v15: Self-Refinement Loop
- Draft → Critique → Refine 3단계 답변 생성
- LLM이 자신의 초안을 스스로 비판하고 재작성
- 답변 품질 자가 교정 구조 도입

---

## 🔹 v16: Multi-Vector Index
- 청크 단위 임베딩 외에 문장 단위·키워드 단위 벡터 인덱스 추가
- 3-tier 검색: chunk index + sentence index + keyword index
- 세밀한 검색 단위로 Recall 개선

---

## 🔹 v17: Context Compression (Phase 2)
- 임베딩 유사도 기반 문장 추출 (`compress_chunks()`)
- 검색된 청크에서 질문과 무관한 문장 제거
- Context window 절약 + 노이즈 감소

---

## 🔹 v18: Caching System
- `QueryResultCache`: 동일 쿼리 검색 결과 TTL 캐시 (3600초)
- `AnswerCache`: 동일 질문 최종 답변 캐시 (1800초)
- 중복 API 호출 제거 → 비용·지연 절감

---

## 🔹 v19: Integrated Pipeline + Fallback
- v16~v18 모든 기능 통합 (Multi-Vector + Compression + Cache)
- 평가 기반 자동 재시도 Fallback: 낮은 정확도 감지 시 검색 전략 변경 후 재시도
- 구조화 8개 필드 평가 시스템 완성
  - accuracy_score / relevance_score / hallucination / missing_info
  - completeness / reasoning_quality / source_quality / overall

---

## 🔹 v20: Failure Dataset & Auto Improvement Loop
- `classify_failure_types()`: LLM 호출 없이 평가 결과만으로 실패 자동 분류
  - `low_accuracy` / `hallucination` / `incomplete_answer` / `retrieval_failure` / `low_relevance`
- `FailureDataset`: 실패 케이스 전용 파일 저장 (`failure_dataset_v20.json`)
- `generate_improvement_hint()`: 실패 원인 분석 + 청크·쿼리·답변 개선 방향 LLM 생성
- Fine-tune JSONL 내보내기: OpenAI fine-tuning API 직접 사용 가능
- **핵심**: 단순 로그 → 재사용 가능한 학습 데이터로 전환
  ```
  실패 감지 → 유형 분류 → FailureDataset 저장 → JSONL export → fine-tuning
  ```

---

## 🔹 v21: Parallel Search + Retrieval Metrics
- `ThreadPoolExecutor` 기반 4-way 병렬 검색
  - Dense / BM25 / Sentence-level / Keyword-level 동시 실행
- NDCG@k 검색 품질 정량 평가
- Rate Limiting: 사용자별 시간당 호출 횟수 제한
- MetricsCollector: P50 / P95 / P99 응답 지연 측정

---

## 🔹 v22: 인증 / 모니터링 / 자동 라우팅
- JWT 기반 로그인 게이트 (session_state 로컬 처리)
- UserManager: 사용자 생성·삭제·비밀번호 관리 (`rag_users_v22.json`)
- 자동 쿼리 라우팅: 질문 유형에 따라 검색 전략 자동 결정
  - factual / analytical / comparative / procedural / exploratory / ambiguous
- 에이전트 분석 탭: 로그 기반 패턴 분석 + 실패 분포 시각화
- Alert 시스템: 정확도 임계값·환각률·P95 지연 초과 시 경고

---

## 🔹 v23: 아키텍처 분리 + LLM Compression + ToolRegistry + AsyncRAGEngine

### 구조 변경: 모놀리식 → 3파일 분리

| 파일 | 역할 |
|------|------|
| `rag_engine.py` | 순수 RAG 로직 (UI 없음) — `process_rag_query()` 단일 진입점 |
| `server_api.py` | FastAPI 서버 — JWT 인증 + 16개 REST 엔드포인트 |
| `client_app.py` | Streamlit 프론트엔드 — HTTP 요청만 수행, 로컬 로직 없음 |

### [NEW] LLM Context Compression (Phase 3)
- `compress_chunks_llm()`: LLM이 전체 청크에 걸쳐 가장 관련 높은 N문장을 직접 선별
- 문장마다 `"ci_si"` ID 부여 → LLM이 ID 목록 반환 → 청크 재구성
- Phase 1(BM25) → Phase 2(임베딩 유사도) → **Phase 3(LLM 전역 선별)** 완성

### [NEW] ToolRegistry (함수 호출)
- OpenAI function calling (`tool_choice="auto"`) 기반
- 4종 도구: `CalculatorTool` / `DateTimeTool` / `UnitConverterTool` / `WebSearchTool`
- `CalculatorTool`: `ast.parse(mode='eval')` 안전 수식 계산
- `WebSearchTool`: DuckDuckGo Instant Answer API (API 키 불필요)

### [NEW] AsyncRAGEngine
- `asyncio.gather()` 기반 4-way 비동기 병렬 검색 (ThreadPoolExecutor 대체)
- `AsyncOpenAI` — non-blocking LLM 호출
- `run_sync()`: Streamlit 이벤트 루프 충돌 방지 (별도 스레드에서 `asyncio.run()`)

### FastAPI 주요 엔드포인트
- `POST /auth/login` — JWT 발급
- `POST /docs/upload` — PDF 업로드 → 인덱싱
- `POST /chat` — RAG 질의응답 (핵심)
- `GET /metrics` — P50/P95/P99 + 실패 통계
- `GET /failures/export/jsonl` — Fine-tune JSONL 다운로드

---

# 버전별 해결 문제 요약

| 버전 범위 | 주제 | 핵심 문제 해결 |
|-----------|------|---------------|
| v2~v5 | 기반 구축 | API 전환, 검색 품질, 청킹 |
| v6~v8 | 답변·검색 고도화 | 출처 인용, Multi-doc, Recall |
| v9~v11 | 관측 가능성 | 자동 평가, Tracing, 버그 수정 |
| v12~v14 | Hybrid + 실험 | BM25 RRF, 라우팅, Ablation |
| v15~v17 | 품질 자동화 | Self-Refine, Multi-Vector, Compression |
| v18~v19 | 운영 안정성 | Cache, Fallback, 통합 평가 |
| v20 | 학습 루프 | 실패 → 데이터셋 → Fine-tune |
| v21~v22 | 프로덕션 준비 | 병렬화, 인증, 모니터링 |
| v23 | 서비스 분리 | API 서버, 비동기 엔진, 도구 호출 |

---

# 최종 스택 (v23)

```
[Client] Streamlit (client_app.py)
    ↓ HTTP (JWT)
[Server] FastAPI (server_api.py)
    ↓ import
[Engine] rag_engine.py
    ├── AsyncRAGEngine       — asyncio.gather() 4-way 병렬 검색
    ├── ToolRegistry         — Calculator / DateTime / Unit / WebSearch
    ├── compress_chunks_llm  — LLM Phase 3 문장 선별
    ├── compress_chunks      — 임베딩 유사도 Phase 2
    ├── BM25 Pre-filter      — Phase 1
    ├── Self-Refinement      — Draft → Critique → Refine
    ├── Multi-Vector Index   — chunk + sentence + keyword
    ├── FailureDataset       — 실패 자동 분류·저장·JSONL export
    ├── MetricsCollector     — P50 / P95 / P99
    ├── UserManager          — JWT + Rate Limit
    └── QueryResultCache + AnswerCache
```
