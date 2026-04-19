# Advanced RAG System (v2 ~ v28)

단순한 RAG 구현을 넘어, 검색 품질 → 추론 구조 → 관측 가능성 → 자동 평가 → 실험 기반 개선 → 자동 실패 학습 → 병렬화 → 인증/모니터링 → 아키텍처 분리 → Agentic 검색 → 실시간 응답 + 정량 검증 → 검색 품질 자동 교정(CRAG) → 문학·세계관 지능화(Literary Intelligence)까지 확장된 엔드투엔드 RAG 엔지니어링 시스템입니다.

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

**문제** 답변에 출처가 없어 신뢰도가 낮고, 단일 청크로는 복합 질문에 답하지 못하며, 키워드 검색을 완전히 배제하면 정확한 고유명사·수치가 누락됩니다.

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

### v27: Corrective RAG (CRAG) + 문학 특화 웹 검색

**문제** RAG는 검색 결과가 나빠도 그냥 답변을 생성합니다(garbage in, garbage out). 소설·세계관 문서는 내부 텍스트만으로는 해석·분석 질문에 한계가 있습니다.

**해결**
- `CRAGGrader.grade()` 0~10 채점 → Correct / Ambiguous / Incorrect 분기
- Incorrect 이면 웹 검색으로 대체, Ambiguous 이면 병합 후 생성
- `extract_work_title()` + `build_literary_web_query()` → 단순 질문 대신 **"소나기 황순원 해설 ..."** 형태의 해설 특화 쿼리로 웹 탐색

```
검색 → 채점 → Correct: 내부 문서만 / Ambiguous: 병합 / Incorrect: 웹 대체
```

SSE에 status 이벤트 추가: 실시간으로 채점 결과와 웹 쿼리 표시.

**효과** 검색 품질이 낮을 때 자동으로 웹 해설로 보완 → 잘못된 답변 방지.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `use_crag` | `false` | CRAG 파이프라인 활성화 |
| `CRAG_RELEVANCE_THRESHOLD` | `5.0` | 이상이면 Correct |
| `CRAG_AMBIGUOUS_THRESHOLD` | `3.0` | 이상이면 Ambiguous, 미만이면 Incorrect |

---

### v28: Literary Intelligence Layer (엔티티 인덱스 + 관계도 + 질문 유형 분류)

**문제** "등장인물 목록을 알려줘", "소년과 소녀의 관계는?"처럼 구조적으로 답할 수 있는 질문도 전체 RAG 파이프라인을 태웁니다. 해석 질문과 사실 질문을 동일하게 처리하면 불필요한 웹 검색이 발생하거나 반대로 해석이 필요한 질문에 웹 보완이 생략됩니다.

**해결**
- 업로드 즉시 `LiteraryEntityExtractor`가 인물·지명·세력·아이템 인덱스와 인물 관계도를 구축
- `answer_from_entity_index()`: 구조적 질문은 RAG 없이 인덱스에서 직접 반환
- `LiteraryQueryRouter.classify()`: factual → RAG only, interpretive → CRAG 자동 활성화

```
질문 → 엔티티 직접 답변 시도 → 히트: 즉시 반환 (LLM 불필요)
      → 미스: factual / interpretive 분류 → interpretive 이면 use_crag=True 자동 적용
```

**효과** 구조적 질문 응답 속도 대폭 감소, 해석 질문은 항상 웹 해설 보완 보장.

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
| v28 | 문학·세계관 지능화 | 엔티티 인덱스, 인물 관계도, 질문 유형 자동 분류 |

---

## 최종 스택 (v28)

```
[Client] Streamlit (client_app.py)
    ↓ HTTP / SSE
[Server] FastAPI (server_api.py)
    ├── routers/auth.py      — JWT 인증
    ├── routers/docs.py      — PDF 업로드 · 인덱싱 · 엔티티 추출 · 관계도
    ├── routers/chat.py      — RAG 질의응답 · Streaming · CRAG · 질문 유형 분류
    ├── routers/metrics.py   — 지표 · 로그 · 실패 데이터셋
    └── routers/admin.py     — 사용자 관리
    ↓
[Engine] rag_engine.py
    ├── LiteraryEntityExtractor — 인물·지명·세력·아이템 추출 · 관계도 구축  [NEW v28]
    ├── LiteraryQueryRouter     — 질문 유형 분류 (사실 / 해석)               [NEW v28]
    ├── CRAGGrader              — 검색 결과 채점 (0~10) · Grade 분류          [NEW v27]
    ├── MultiHopPlanner         — 질문 분해 · hop별 검색 · 종합
    ├── SelfRAGChecker          — 검색 충분성 판단 · 자기 교정
    ├── AsyncRAGEngine          — asyncio 기반 비동기 파이프라인
    ├── ToolRegistry            — Calculator · DateTime · WebSearch
    ├── MetricsCollector        — P50/P95/P99 지연 측정
    ├── FailureDataset          — 실패 케이스 저장 · JSONL export
    └── UserManager             — 사용자 인증 · Rate Limit
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
cd rag_v28
python server_api.py
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### 4. 클라이언트 실행 (터미널 2)

```bash
cd rag_v28
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
cd rag_v28
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
├── rag_v27/              ← Corrective RAG (CRAG)
│   ├── evaluate_ragas.py
│   └── (v26 구조 동일)
├── rag_v28/              ← Literary Intelligence Layer (최신)
│   ├── evaluate_ragas.py
│   └── (v27 구조 동일)
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
| 문학 지능화 | 엔티티 인덱스 · 인물 관계도 · 질문 유형 자동 분류 |
