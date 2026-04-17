# rag_app_v12.py 변동사항 요약

## 핵심 변경: Hybrid 검색 (Dense + BM25 RRF) + 임베딩 캐시 + NDCG@k 정량 평가

v11의 기능을 그대로 유지하면서 검색 품질·비용·평가 세 축을 개선.

---

## 최종 스택
| 구성 | v11 (rag_app_v11.py) | v12 (rag_app_v12.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| **검색 방식** | Dense(FAISS)만 | **Dense + BM25 → RRF 융합 (Hybrid)** |
| **임베딩 캐시** | 없음 | **EmbeddingCache 클래스 — 디스크 영속 캐시** |
| **쿼리 리라이팅 캐시** | 없음 | **세션 내 dict 캐시 — 동일 질문 LLM 호출 생략** |
| **정량 평가** | 없음 | **NDCG@k 자동 계산 + 추이 차트** |
| **UI 탭** | 3개 | 3개 (지표 항목 확대) |

---

## 변경 사항

### 1. EmbeddingCache 클래스 — 디스크 영속 임베딩 캐시

```python
class EmbeddingCache:
    def __init__(self, path: str)   # pkl 파일 로드
    def size() -> int               # 캐시 저장 수
    def clear()                     # 캐시 초기화 + 파일 삭제
```

- 텍스트 → md5 해시를 키로 `embed_cache_v12.pkl` 에 저장
- `get_embeddings_cached(texts)`: 캐시 히트는 API 미호출, 미스만 API 호출 후 저장
- `build_index()` 에서도 캐시 적용 → 문서 청크 임베딩이 인덱싱 시점에 저장되어 이후 prefilter에서 재사용

**효과**: 동일 문서 반복 질문 시 prefilter 단계 API 호출 = 0

---

### 2. 쿼리 리라이팅 세션 캐시

```python
# st.session_state.rewrite_cache = {f"{질문}||{n}": [쿼리 리스트]}
```

같은 질문을 같은 세션에서 다시 하면 LLM 호출 없이 캐시에서 반환.
캐시 히트 시 tracer span에 `"캐시 히트: LLM 호출 생략"` 결정 기록.

---

### 3. Hybrid 검색 (`retrieve_hybrid`) — Dense + BM25 + RRF

```
질문
  → Dense 검색 (FAISS, top 20/쿼리)
  → BM25 검색 (BM25Okapi, top 20/쿼리)
  → RRF 융합: score = Σ 1/(k=60 + rank + 1)
  → 최종 후보 (중복 제거, RRF 점수 내림차순)
```

| 방식 | 강점 | 약점 |
|------|------|------|
| Dense | 의미 기반, 동의어 처리 | 정확한 키워드 누락 |
| BM25  | 정확한 용어 검색 | 의미 변형 취약 |
| RRF 융합 | 두 랭킹을 결합해 recall 극대화 | — |

- `rank_bm25` 미설치 시 Dense only로 자동 fallback + 사이드바 경고

---

### 4. NDCG@k 검색 품질 정량 평가

```python
def compute_ndcg(ordered_items, score_dict, k) -> float:
    # DCG: prefilter 출력 순서(임베딩 유사도 기준), relevance = 리랭크 점수
    # IDCG: 리랭크 점수 기준 이상적 순서
    # NDCG = DCG / IDCG
```

| NDCG 값 | 의미 |
|---------|------|
| ≈ 1.0 | 임베딩 검색이 이미 최적 순서, 리랭킹은 확인만 |
| < 0.7 | 리랭킹이 순서를 크게 개선함 |

- `rerank_chunks()` 가 이제 `(top_k 결과, 전체 점수 dict)` 튜플 반환
- NDCG는 리랭킹 활성화 시에만 계산 (OFF면 None)
- 로그에 `ndcg_at_k` 필드로 저장

---

## 로그 데이터 구조 변화

```python
# v12 추가 필드
{
    "ndcg_at_k": 0.8234,           # NDCG@top_k (리랭킹 OFF면 null)
    "embed_cache_size": 142,        # 현재 캐시 저장 수
    "embed_cache_hits": 38,         # 세션 누적 캐시 히트
    "embed_cache_misses": 12,       # 세션 누적 캐시 미스
}
```

---

## UI 변화

| 위치 | v11 | v12 |
|------|-----|-----|
| 사이드바 | BM25 없음 | BM25 Hybrid 토글 + 캐시 히트율 표시 |
| 챗봇 뱃지 | 5개 | **6개 (NDCG@k 추가)** |
| 트레이싱 탭 요약 | 4개 지표 | **6개 (캐시 히트율, 평균 NDCG@k 추가)** |
| 트레이싱 탭 차트 | 2개 | **3개 (NDCG 추이 차트 추가)** |
| 에이전트 분석 탭 요약 | 4개 지표 | **5개 (평균 NDCG@k 추가)** |
| 에이전트 분석 탭 차트 | 2개 | **3개 (NDCG 추이 차트 추가)** |
| CSV | 13개 컬럼 | **14개 컬럼 (ndcg_at_k 추가)** |

---

## LLM 호출 횟수

| 단계 | v12 |
|------|-----|
| 쿼리 리라이팅 | 1 (캐시 히트 시 0) |
| Rerank | 1 |
| Step1 요약 | 1 |
| Step2 분석 | 1 |
| Step3 답변 | 1 |
| 평가 | 1 |
| 환각 원인 분석 | +1 (환각 시에만) |
| **합계** | **6~7회 (캐시 히트 시 5~6회)** |

## 성능 개선 기대 포인트
- Dense 단독 대비 Hybrid 검색으로 recall 향상 (특히 고유명사·숫자 검색)
- 문서 처리 후 반복 질문 시 prefilter 임베딩 비용 = 0 (캐시 재사용)
- NDCG@k 축적으로 "어떤 질문 유형에서 임베딩 검색이 부족한가" 데이터 수집 가능
