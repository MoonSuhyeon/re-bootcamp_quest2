# Advanced RAG System (v2 ~ v26)

단순한 RAG 구현을 넘어, 검색 품질 → 추론 구조 → 관측 가능성 → 자동 평가 → 실험 기반 개선 → 자동 실패 학습 → 병렬화 → 인증/모니터링 → 아키텍처 분리 → Agentic 검색 → 실시간 응답 + 정량 검증까지 확장된 엔드투엔드 RAG 엔지니어링 시스템입니다.

**LangChain / LlamaIndex 없이 전 구성요소를 직접 구현**했으며, 각 버전은 실제로 관찰된 문제에서 출발합니다.

---

## 목차

- [스택](#스택)
- [파이프라인 구성요소 진화](#파이프라인-구성요소-진화)
- [시스템 진화](#시스템-진화)
- [버전별 해결 문제 요약](#버전별-해결-문제-요약)
- [빠른 시작](#빠른-시작)
- [폴더 구조](#폴더-구조)

---

## 스택

### 파이프라인 구성요소

| 구성요소 | 패키지 | 모델 / 종류 | 특징 |
|---------|--------|------------|------|
| **문서 로드** | `pdfplumber` | — (rule-based PDF 파서) | PyMuPDF/PyPDF2 대비 표·레이아웃 추출에 강함 |
| **텍스트 분할** | 직접 구현 (`re` 모듈) | — (Rule-based + Semantic 2종) | `RecursiveCharacterTextSplitter` 없이 직접 구현 |
| **임베딩** | `openai` SDK | `text-embedding-3-small` | API 호출 + MD5 해시 기반 디스크 캐시 |
| **벡터 스토어** | `faiss` | `IndexFlatIP` (내적 전수 탐색) | In-memory, 청크·문장·키워드 3인덱스 구조 |
| **리트리버** | `faiss` + `rank_bm25` | Dense + BM25Okapi Hybrid | Dense/BM25 점수 RRF 병합 |
| **리랭커** | `openai` SDK | `gpt-4o-mini` (0~10 채점) | LLM-as-Reranker. 전용 모델(Cohere/BGE) 없음 |
| **체인** | `openai` SDK | `gpt-4o-mini` | Chat Completions 직접 호출, 3단계 Multi-doc chain |

### 아키텍처

```
[Client] Streamlit (client_app.py)
    ↓ HTTP / SSE
[Server] FastAPI (server_api.py)
    ├── routers/auth.py      — JWT 인증
    ├── routers/docs.py      — PDF 업로드·인덱싱
    ├── routers/chat.py      — RAG 질의응답·Streaming
    ├── routers/metrics.py   — 지표·로그·실패 데이터셋
    └── routers/admin.py     — 사용자 관리
    ↓
[Engine] rag_engine.py
    ├── MultiHopPlanner   — 질문 분해·hop별 검색·종합
    ├── SelfRAGChecker    — 검색 충분성 판단·자기 교정
    ├── AsyncRAGEngine    — asyncio 기반 비동기 파이프라인
    ├── ToolRegistry      — Calculator·DateTime·WebSearch
    ├── MetricsCollector  — P50/P95/P99 지연 측정
    ├── FailureDataset    — 실패 케이스 저장·JSONL export
    └── UserManager       — 사용자 인증·Rate Limit
    ↓
[Index] FAISS (IndexFlatIP) — in-memory
    ├── chunk_index    — 청크 단위 벡터
    ├── sent_index     — 문장 단위 벡터
    └── kw_index       — 키워드 단위 벡터
[Eval] evaluate_ragas.py — Faithfulness · Answer Relevancy · Context Precision
```

---

## 파이프라인 구성요소 진화

각 구성요소에 어떤 버전에서 어떤 기능이 추가됐는지 정리합니다.

```
┌─────────────────┬──────────────────────────────────────────────────────────────────────┐
│  구성 요소       │  추가된 기능 (버전)                                                   │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│  문서 로드       │  v1  PyPDF2 → pdfplumber 교체 (한글 깨짐 해결)                        │
│  pdfplumber     │                                                                      │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v1~4  기본 문단/문장 경계 분할                                       │
│  텍스트 분할     │  v5   Overlap 청킹: 청크 경계 손실 방지 (이전 청크 끝 100자 중복 포함) │
│  직접 구현       │  v5   Semantic 청킹: 문장 임베딩 코사인 유사도로 주제 전환점 자동 탐지 │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v1   HuggingFace SentenceTransformer (로컬)                          │
│  임베딩          │  v2   OpenAI text-embedding-3-small 으로 교체 (API)                   │
│  OpenAI API     │  v7   Multi-Vector: 청크·문장·키워드 3종류 임베딩 생성                │
│                 │  v19  embed_cache: MD5 해시 기반 디스크 캐시 분리 (API 중복 호출 방지) │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│  벡터 스토어     │  v1   FAISS IndexFlatIP 첫 도입                                       │
│  FAISS          │  v7   Multi-Vector Index: 청크/문장/키워드 IndexFlatIP 3개 구조       │
│  IndexFlatIP    │                                                                      │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v1   Dense only (FAISS 단독)                                         │
│                 │  v4   BM25Okapi 추가 → Hybrid (Dense + BM25 RRF 통합)                │
│                 │  v8   Query Routing: 6개 의도 분류 → 검색 전략 자동 선택              │
│  리트리버        │  v9   Pre-filtering + Dynamic Retrieval (의도별 파라미터 자동 조정)   │
│  FAISS +        │  v13  Fallback: 평가 낮으면 파라미터 높여 재검색                       │
│  BM25Okapi      │  v19  Multi-Vector: 청크/문장/키워드 3채널 동시 검색                  │
│  (Hybrid)       │  v21  병렬 검색: ThreadPoolExecutor 4채널 동시 실행 (대기시간 단축)   │
│                 │  v25  Multi-Hop: 복합 질문 → sub-query 분해 → 순차 검색 → 종합        │
│                 │  v25  Self-RAG: 검색 결과 충분성 자동 판단 → 불충분 시 재검색          │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v4   LLM-as-Reranker 첫 도입 (gpt-4o-mini, 0~10 채점)               │
│  리랭커          │  v10  Tracer: 리랭커 latency·토큰 span 추적                           │
│  gpt-4o-mini    │  v21  LongContextReorder: 고점수 청크를 프롬프트 앞/뒤 배치           │
│                 │        (Lost in the Middle 논문 — 추가 LLM 호출 없음)                 │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v1~2  단순 단일 LLM 호출                                              │
│                 │  v3   Query Rewrite: 질문 재작성으로 검색 recall 향상                  │
│                 │  v7   Multi-doc Chain: 요약(step1) → 관계분석(step2) → 최종답변(step3)│
│                 │  v11  QueryResultCache: 동일 쿼리 결과 재사용                          │
│                 │  v12  AnswerCache: 완성된 답변 TTL 캐시                                │
│  체인            │  v12  Self-Refinement: Draft → Critique → Refine (3단계 자기검토)    │
│  gpt-4o-mini    │  v15  LLM 평가: 8개 필드 구조화 평가 (정확도/관련성/환각/신뢰도...)    │
│                 │  v19  Context Compression Phase 1: 임베딩 유사도로 관련 문장만 추출   │
│                 │  v21  Context Compression Phase 2: 청크 간 중복 문장 제거             │
│                 │  v21  Tool-Augmented RAG: 계산 의도 탐지 → Python 실행 → 결과 주입    │
│                 │  v23  LLM Compression Phase 3: LLM이 직접 압축 (프롬프트 최소화)      │
│                 │  v23  ToolRegistry: 함수 호출 4종 (계산기·날짜·검색·단위변환)         │
│                 │  v23  AsyncRAGEngine: asyncio.gather() 비동기 파이프라인               │
│                 │  v25  Multi-Hop synthesize: hop별 중간답변 → 최종 종합                │
│                 │  v26  Streaming SSE: stream=True 토큰 실시간 yield                    │
├─────────────────┼──────────────────────────────────────────────────────────────────────┤
│                 │  v10  Tracer: span 기반 단계별 latency·토큰 추적                       │
│                 │  v20  FailureDataset: 실패 케이스 자동 분류·저장·JSONL 내보내기        │
│  운영/인프라     │  v22  MetricsCollector: P50/P95/P99, 환각률, 알림, Prometheus 내보내기│
│  (파이프라인     │  v22  UserManager: 로그인·역할·시간당 Rate Limit                      │
│   외부)          │  v23  3파일 분리: rag_engine / server_api / client_app                │
│                 │  v24  config.py + routers/ 구조화 (표준 FastAPI 아키텍처)              │
│                 │  v26  RAGAS 평가 로그: 질문·답변·컨텍스트 자동 저장                    │
└─────────────────┴──────────────────────────────────────────────────────────────────────┘
```

---

## 시스템 진화

각 버전은 실제로 관찰된 문제에서 출발합니다. **문제 → 해결 → 효과** 순서로 설계 근거를 기술합니다.

---

### v2~v3: API 전환 + 기초 검색

**문제** 로컬 모델은 품질이 낮고, 텍스트 전체를 그대로 넘기면 관련 없는 내용까지 LLM이 읽어야 합니다.

**해결** OpenAI Embedding + GPT API로 전환, 코사인 유사도 기반 유사 청크만 검색.

**효과** 품질 기반선 확보, LLM이 읽는 컨텍스트 범위를 관련 구간으로 좁힘.

---

### v4~v5: Query Rewriting + Semantic Chunking

**문제** 질문을 그대로 검색하면 의미 파악에 실패하고, 고정 길이 청킹은 문맥을 중간에 끊습니다.

**해결** LLM으로 질문을 재작성해 검색 의도를 명확히 하고, Overlap/Semantic Chunking으로 문맥 단절 제거.

**효과** 검색 관련성 향상, 청크 경계에서 발생하는 정보 손실 감소.

---

### v6~v8: 출처 인용 + Multi-doc + Recall 보강

**문제** 답변에 출처가 없어 신뢰도가 낮고, 단일 청크로는 복합 질문에 답하지 못하며, 키워드 검색을 배제하면 정확한 고유명사·수치가 누락됩니다.

**해결**
- Source attribution (`📌 요약 / 📖 근거 / ✅ 결론` 포맷)
- 3단계 Multi-doc reasoning (요약 → 관계 분석 → 종합)
- Embedding Pre-filter + `(chunk, source)` 튜플로 출처 추적

**효과** 답변 신뢰도·투명성 향상, 복합 질문 처리 가능, Recall 개선.

---

### v9~v11: Observability (블랙박스 탈출)

**문제** 답변이 좋은지 나쁜지 알 방법이 없고, 어디서 지연이 발생하는지도 보이지 않습니다.

**해결** LLM 자동 평가(accuracy/relevance/hallucination), Span 기반 Tracing, Token usage·latency 측정, 대시보드 UI.

**효과** 품질 문제와 병목을 수치로 파악 → 다음 개선 방향 결정 가능.

---

### v12~v14: Hybrid Search + 라우팅 + Ablation

**문제** Dense Search만 사용하면 키워드 누락으로 Recall이 떨어집니다. 어떤 구성 요소가 실제로 도움이 되는지도 알 수 없습니다.

**해결**
- BM25 + Dense 결과를 RRF(Reciprocal Rank Fusion)로 병합 → 키워드 Recall 향상
- Reranking으로 상위 결과 Precision 향상
- Query Routing으로 질문 유형별(factual / reasoning / exploratory …) 최적 전략 자동 선택
- Ablation Study: 구성 요소 ON/OFF 자동 실험으로 기여도 수치화

**효과** Recall + Precision 동시 향상, 과학적 구성 검증.

---

### v15~v17: 답변 품질 자동화 + 세밀한 검색

**문제** 첫 번째 초안이 항상 최선이 아니고, 청크 단위 검색만으로는 짧은 결정적 문장을 놓칩니다. 긴 컨텍스트는 LLM 비용과 노이즈를 높입니다.

**해결**
- Self-Refinement: Draft → Critique → Refine 3단계 자기 비판·재작성
- Multi-Vector Index: chunk / sentence / keyword 3계층 검색으로 세밀한 매칭
- Context Compression: 유사도 낮은 문장 제거로 컨텍스트 축소

**효과** 답변 완성도 향상, Context window 절약, 환각 소재 감소.

---

### v18~v19: 캐싱 + Fallback (운영 안정성)

**문제** 동일 질문을 반복할 때마다 LLM을 재호출하면 비용·지연이 낭비되고, 낮은 품질 답변이 그냥 반환됩니다.

**해결**
- `QueryResultCache` (TTL 3600초) + `AnswerCache` (TTL 1800초): 캐시 히트 시 즉시 반환
- 평가 점수 미달 시 쿼리 재작성 → 재검색 Fallback 자동 실행

**효과** 반복 질문 응답 속도·비용 감소, 저품질 답변 자동 재시도.

---

### v20: 실패 학습 루프

**문제** 실패 케이스가 쌓여도 모델이 개선되지 않습니다. 어떤 유형의 실패가 얼마나 발생하는지도 불분명합니다.

**해결** `classify_failure_types()`로 실패를 5종(`low_accuracy` / `hallucination` / `incomplete_answer` / `retrieval_failure` / `low_relevance`)으로 자동 분류 → `FailureDataset`에 저장 → Fine-tune JSONL 내보내기.

**효과** 실패 패턴 가시화, OpenAI fine-tuning으로 약점 도메인 직접 보완 가능.

---

### v21~v22: 병렬화 + 프로덕션 준비

**문제** Dense / BM25 / Sentence / Keyword 4개 인덱스를 순차 실행하면 지연이 누적되고, 인증·모니터링 없이는 실서비스 배포가 불가능합니다.

**해결**
- `ThreadPoolExecutor` 4-way 병렬 검색 → 전체 검색 지연 최소화
- JWT 인증 게이트, `UserManager`(생성·삭제·비밀번호), Rate Limiting
- P50/P95/P99 응답 지연 측정, Alert 시스템

**효과** 검색 속도 대폭 개선, 사용자 격리 및 과부하 방지.

---

### v23: 모놀리식 → 3파일 분리

**문제** 단일 파일에 UI·API·RAG 로직이 혼재하면 테스트·확장·협업이 불가능합니다.

**해결** 책임을 3파일로 분리:

| 파일 | 역할 |
|------|------|
| `rag_engine.py` | 순수 RAG 로직 (UI 없음) |
| `server_api.py` | FastAPI 서버 — JWT 인증 + REST API |
| `client_app.py` | Streamlit 프론트엔드 — HTTP 요청만 수행 |

추가: **LLM Compression**(전역 청크에서 관련 문장 직접 선별), **ToolRegistry**(계산기/날짜/단위변환/웹검색), **AsyncRAGEngine**(`asyncio.gather()`).

**효과** 각 계층 독립 테스트 가능, 프론트엔드 교체 시 서버 무변경.

---

### v24: Config 중앙화 + Service Layer

**문제** 상수가 여러 파일에 하드코딩되어 환경 변경 시 다수 파일을 수정해야 합니다.

**해결** `config.py`에 모든 상수 집중(환경변수 오버라이드), `routers/` 서비스 계층으로 엔드포인트 분리:

```
routers/auth.py · docs.py · chat.py · metrics.py · admin.py
```

**효과** 환경별 설정 변경이 config.py 1곳에서 끝남, 라우터 단위 독립 유지보수.

---

### v25: Multi-Hop + Self-RAG (Agentic 검색)

**문제** "A와 B의 차이점과 그 원인"처럼 복합 질문은 단일 검색 한 번으로 완전한 답변을 낼 수 없습니다. 검색 결과가 부족해도 시스템이 그냥 진행합니다.

**해결**
- **Multi-Hop**: `decompose()` → hop별 검색 → `synthesize()` 종합
- **Self-RAG**: `check_sufficiency()` 판단 → 부족하면 `rewrite_hint`로 재검색 (최대 3회)

```
질문 → 분해 → hop1 검색+중간답 → hop2 검색+중간답 → 최종 종합
검색 → 충분성 판단 → 부족: 재작성 → 재검색 → 결과 병합
```

**효과** 복합 질문 처리 가능, 검색 품질 자가 검증.

---

### v26: Streaming + RAGAS 정량 평가

**문제** 긴 답변이 완성될 때까지 UI가 아무것도 표시하지 않아 UX가 나쁘고, 품질을 수치로 비교할 수 없습니다.

**해결**
- SSE `POST /chat/stream`: 첫 토큰부터 실시간 표시
- RAGAS 오프라인 평가: Faithfulness(환각 탐지) / Answer Relevancy / Context Precision

```bash
python evaluate_ragas.py --last 20   # 최근 20개 응답 품질 수치화
```

**효과** 체감 응답 속도 개선, 버전 간 품질 수치 비교 가능.

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

---

## 빠른 시작

### 1. 환경 설정

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
pip install ragas datasets   # RAGAS 평가 시 필요
```

### 3. 서버 실행 (터미널 1)

```bash
cd rag_v26
python server_api.py
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### 4. 클라이언트 실행 (터미널 2)

```bash
cd rag_v26
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
cd rag_v26
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
├── rag_v26/              ← Streaming + RAGAS (최신)
│   ├── evaluate_ragas.py
│   └── (v25 구조 동일)
├── rag_versions/         ← v2~v22 단일 파일 히스토리
├── change_logs/          ← 버전별 변동사항 요약
└── requirements.txt
```
