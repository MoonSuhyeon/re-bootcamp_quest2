# rag_app_v19.py 변동사항 요약

## 핵심 변경: 캐시 전략 분리 + Context Compression + Multi-Vector Retrieval

v18의 "Dynamic Retrieval + Self-Refinement" 위에,
**쿼리/답변 캐시로 비용 30~70% 절감**,
**핵심 문장만 추출해 노이즈·토큰 감소**,
**문장·키워드·청크 3중 벡터 인덱스로 검색 정확도 향상**하는
세 가지 실무 핵심 패턴을 추가.

---

## 최종 스택

| 구성 | v18 (rag_app_v18.py) | v19 (rag_app_v19.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | **Multi-Vector(문장/키워드/청크) + BM25 RRF** |
| 평가 시스템 | 구조화 8개 필드 | 동일 |
| 품질 리포트 | overall_score + grade | 동일 |
| 검색 품질 분석 | NDCG + Reranker Gain | 동일 |
| Fallback 시스템 | 평가 기반 자동 재시도 | 동일 |
| 검색 전략 | Dynamic Retrieval 프로필 | 동일 |
| 답변 생성 | Draft → Critique → Refine | 동일 |
| **캐시 전략** | 임베딩 캐시만 | **QueryResultCache + AnswerCache (TTL)** |
| **컨텍스트** | chunk 원문 그대로 | **compress_chunks() — 핵심 문장 추출** |
| **검색 벡터** | 청크 단일 벡터 | **청크 + 문장 + 키워드 3중 인덱스** |

---

## 변경 사항

### 1. 파일 상수 추가

```python
ANSWER_CACHE_FILE    = os.path.join(_BASE, "answer_cache_v19.json")
ANSWER_CACHE_TTL_SEC = 1800   # 30분 — 답변 캐시 TTL
QUERY_CACHE_TTL_SEC  = 3600   # 1시간 — 쿼리 결과 캐시 TTL
```

---

### 2. `QueryResultCache` — 세션 기반 쿼리 결과 캐시 (신규)

```python
class QueryResultCache:
    def _key(self, question, use_bm25, prefilter_n) -> str:
        return hashlib.md5(f"{question}|{use_bm25}|{prefilter_n}".encode()).hexdigest()
    def get(self, question, use_bm25, prefilter_n, ttl=QUERY_CACHE_TTL_SEC) -> list | None
    def set(self, question, use_bm25, prefilter_n, items)
    def clear(self)
```

- `st.session_state.query_result_cache_store` 에 저장 (메모리, 세션 단위)
- TTL 초과 시 자동 만료
- 문서 재처리 시 `query_result_cache.clear()` 호출

---

### 3. `AnswerCache` — 파일 기반 답변 캐시 (신규)

```python
class AnswerCache:
    def get(self, key, ttl=ANSWER_CACHE_TTL_SEC) -> dict | None
        # {"answer", "evaluation", "quality_report"} 반환
    def set(self, key, data)
    def valid_size(self, ttl) -> int
    def clear_all(self)
    def clear_expired(self, ttl)
```

- JSON 파일 (`answer_cache_v19.json`) 영구 저장
- TTL 기반 만료 관리
- 캐시 키 = `hashlib.md5(f"{question}|{top_k}|{use_multidoc}".encode()).hexdigest()`
- **읽기/쓰기 조건**: `use_session_cache=True` 이고 첫 번째 시도(attempt=0)일 때만
  → Fallback 재시도는 항상 신규 생성

---

### 4. `extract_keywords_simple()` — TF 기반 키워드 추출 (신규)

```python
def extract_keywords_simple(text: str, top_k: int = 10) -> str:
```

- 한국어/영어 불용어 제거
- 단어 빈도(TF) 기반 top_k 단어 추출
- LLM 호출 없음 — 순수 텍스트 처리
- Multi-Vector 키워드 인덱스 생성에 사용

---

### 5. `build_multi_vector_index()` — 3중 벡터 인덱스 구축 (신규)

```python
def build_multi_vector_index(chunks: list) -> dict:
```

#### 3개 인덱스 구조

| 인덱스 | 단위 | 내용 |
|--------|------|------|
| `chunk_index` | 청크 전체 | 기존 Dense 검색과 동일 |
| `sent_index` | 개별 문장 | 청크를 문장으로 분리 → 임베딩 |
| `kw_index` | 키워드 텍스트 | 청크별 TF 키워드 → 임베딩 |

```python
return {
    "chunk_index":   FAISS IndexFlatIP,
    "sent_index":    FAISS IndexFlatIP,
    "kw_index":      FAISS IndexFlatIP,
    "sent_to_chunk": list[int],   # 문장 → 부모 청크 인덱스 매핑
    "n_chunks":      int,
    "n_sentences":   int,
}
```

- 문서 처리 탭에서 "Multi-Vector 인덱스 구축" 버튼으로 생성
- `st.session_state.mv_index` 에 저장

---

### 6. `retrieve_multi_vector()` — 3중 벡터 RRF 검색 (신규)

```python
def retrieve_multi_vector(queries, mv_index, chunks, sources,
                          prefilter_n, use_bm25, top_k) -> list:
```

#### 검색 흐름

```
쿼리 임베딩
    ↓
① chunk_index 검색 → chunk 후보
② sent_index 검색 → sent_to_chunk로 청크 매핑 → chunk 후보
③ kw_index 검색 → chunk 후보
④ BM25 검색 (use_bm25=True 시)
    ↓
RRF 통합 (k=60) → 상위 top_k 반환
```

- 3개 벡터 경로가 같은 청크를 가리킬수록 순위 상승 (다중 신호 교차 검증)
- 기존 `retrieve_chunks()` 완전 대체 가능

---

### 7. `compress_chunks()` — Context Compression (신규)

```python
def compress_chunks(question: str, chunks: list,
                    max_sentences: int = 5,
                    min_sim: float = 0.25) -> tuple[list, list]:
```

#### 압축 알고리즘

```
각 청크에 대해:
  1. 문장 분리 (마침표·줄바꿈 기준)
  2. 문장 임베딩 (EmbeddingCache 활용, 추가 API 호출 최소화)
  3. 질문 임베딩과 코사인 유사도 계산
  4. sim >= min_sim 인 문장만 유지
  5. 유사도 순 정렬 후 max_sentences 개 선택
  6. 원래 순서로 재조합
```

```python
# 반환값
compressed_chunks: list[str]            # 압축된 청크 텍스트
compression_stats: list[dict]           # 청크별 통계
# {"original": 원문 문장수, "compressed": 압축 문장수, "ratio": float}
```

- LLM 호출 없음 — 임베딩 유사도만 사용
- 이미 캐시된 임베딩 재활용 → 추가 비용 거의 없음

---

### 8. `run_rag_pipeline()` — `use_compression`, `mv_index` 파라미터 추가

```python
def run_rag_pipeline(...,
    use_compression: bool = False,
    compress_max_sentences: int = 5,
    compress_min_sim: float = 0.25,
    mv_index: dict | None = None,
    use_session_cache: bool = True) -> dict:
```

#### 파이프라인 실행 흐름 (v19)

```
[0] 답변 캐시 확인 (use_session_cache=True & attempt=0)
    ↓ 캐시 히트 → 즉시 반환 (⚡ 답변 캐시 히트)
[1] 쿼리 리라이트
[2] 쿼리 결과 캐시 확인
    ↓ 캐시 히트 → [4]로 스킵 (🔁 쿼리 결과 캐시 히트)
[3] 검색 (Multi-Vector OR Hybrid)
    → 결과를 쿼리 캐시에 저장
[4] Rerank
[5] Context Compression (use_compression=True 시)
    final_chunks → compressed_chunks (생성에 사용)
[6] 답변 생성 (multidoc or simple, gen_chunks 기준)
[7] Self-Refinement (use_self_refine=True 시)
[8] 평가
    → 답변 캐시에 저장 (use_session_cache=True & attempt=0)
```

반환 dict 추가 필드:
```python
{
    "gen_chunks":         list,    # 생성에 실제 사용된 청크 (압축 후)
    "cache_hit":          str,     # "answer" | "query" | None
    "compression_stats":  list,    # 청크별 압축 통계
}
```

---

### 9. `build_log_entry()` — 신규 필드 추가

```python
def build_log_entry(...,
    cache_hit=None,
    compression_stats=None,
    mv_retrieval=False):
```

#### 신규 로그 필드
```python
{
    "cache_hit":         "answer" | "query" | None,
    "compression_stats": [{"original": int, "compressed": int, "ratio": float}, ...],
    "mv_retrieval":      True | False,
}
```

---

### 10. 사이드바 변화

#### Multi-Vector 설정 섹션 (신규)
```
🔀 Multi-Vector 검색 설정
└─ Multi-Vector 검색 사용 토글 (기본값: OFF)
   ※ 문서 처리 탭에서 인덱스를 먼저 구축해야 함
```

#### 캐시 전략 설정 섹션 (신규)
```
⚡ 캐시 전략 설정
├─ 답변 캐시 사용 (기본값: ON) — 유효 캐시 N건 표시
├─ 쿼리 결과 캐시 사용 (기본값: ON)
├─ [캐시 전체 초기화] 버튼
└─ [답변 캐시만 초기화] 버튼
```

#### Context Compression 설정 섹션 (신규)
```
🗜️ Context Compression 설정
├─ Context Compression 사용 토글 (기본값: OFF)
├─ 최대 문장 수 슬라이더 (1~10, 기본값: 5)
└─ 최소 유사도 슬라이더 (0.0~1.0, 기본값: 0.25)
```

---

### 11. 문서 처리 탭 변화

- Multi-Vector 활성화 시 "🔀 Multi-Vector 인덱스 구축" 버튼 표시
- 인덱스 구축 완료: `청크 N개 → 문장 M개 (청크 평균 K.K문장), 키워드 벡터 N개` 표시
- 문서 변경 시 `query_result_cache.clear()` 자동 호출

---

### 12. 챗봇 탭 변화

#### 캐시 히트 배너
```
⚡ 답변 캐시 히트 — 저장된 답변을 즉시 반환했습니다. (TTL: 30분)
🔁 쿼리 결과 캐시 히트 — 검색을 건너뛰고 저장된 결과를 사용했습니다.
```

#### 🗜️ Context Compression 내역 expander (신규, Compression ON 시만)
```
🗜️ Context Compression 내역
  청크 N: 원문 A문장 → 압축 B문장 (압축률 C%)
  청크 N+1: ...
  전체 평균 압축률: X%
```

---

### 13. 에이전트 분석 탭 변화

#### ⚡ 캐시 전략 분석 섹션 (신규)
- 답변 캐시 히트 건수 / 비율
- 쿼리 캐시 히트 건수 / 비율
- 예상 토큰 절감량 (답변 캐시 히트당 ~1,500 토큰 기준)

#### 🗜️ Context Compression 분석 섹션 (신규)
- Compression 적용 건수 / 비율
- 평균 압축률 bar chart (청크별)
- 토큰 절감 추정치

#### 🔀 Multi-Vector 검색 분석 섹션 (신규)
- Multi-Vector 사용 건수 / 비율
- Hybrid vs Multi-Vector 평균 정확도 비교

#### per-log expander 추가 표시
- `⚡ 캐시 히트: 답변/쿼리` 배지
- `🔀 Multi-Vector: ✅` metric
- `🗜️ Compression: 평균 X%` metric

---

### 14. 검색 품질 탭 데이터 테이블

`Multi-Vector` 컬럼 (`🔀` 표시) 및 `캐시 히트` 컬럼 추가

---

## LLM 호출 횟수

| 상황 | v18 | v19 |
|------|-----|-----|
| 캐시 히트 (답변) | 7~10회 | **0회** |
| 캐시 히트 (쿼리) | 7~10회 | **4~5회** (검색 스킵) |
| Self-Refine OFF, Fallback 없음 | 7~8회 | 7~8회 (동일) |
| Self-Refine ON, Fallback 없음 | 9~10회 | 9~10회 (동일) |
| Compression ON | 동일 | 동일 (LLM 추가 없음) |
| Multi-Vector ON | 동일 | 동일 (LLM 추가 없음) |

> Compression, Multi-Vector는 LLM 추가 호출 없음 (임베딩만 사용).
> 답변 캐시 히트 시 LLM 호출 전혀 없음 → 비용 100% 절감.

---

## 파일 변경

| 항목 | v18 | v19 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v18.json` | `rag_eval_logs_v19.json` |
| EMBED_CACHE_FILE | `embed_cache_v18.pkl` | `embed_cache_v19.pkl` |
| ANSWER_CACHE_FILE | (없음) | `answer_cache_v19.json` (신규) |

---

## 세 패턴의 핵심 원리

### 캐시 전략 분리 — "계층형 캐싱"
```
[L1] 답변 캐시 (파일, TTL 30분)
    같은 질문 + 같은 파라미터 → 전체 LLM 호출 스킵
        ↓ 미스
[L2] 쿼리 결과 캐시 (세션, TTL 1시간)
    같은 검색 조건 → 검색 단계만 스킵
        ↓ 미스
[L3] 임베딩 캐시 (파일, 영구)
    같은 텍스트 → 임베딩 API 스킵
```

### Context Compression — "신호 강화 필터링"
```
청크 원문 (노이즈 포함)
    ↓
문장 분리 → 각 문장 임베딩
    ↓
질문과 코사인 유사도 계산
    ↓
관련 문장만 선택 (sim >= min_sim, top max_sentences)
    ↓
압축된 컨텍스트 → 정확도 ↑, 토큰 ↓
```

### Multi-Vector Retrieval — "다중 신호 교차 검증"
```
청크 텍스트
    ├─ ① 청크 벡터  → 전체 의미 포착
    ├─ ② 문장 벡터  → 세밀한 의미 포착 (부모 청크로 매핑)
    └─ ③ 키워드 벡터 → 핵심 개념 포착
        ↓
BM25 + 3개 벡터 신호 RRF 통합
    ↓
여러 경로에서 공통으로 높은 순위 → 신뢰도 ↑
```
