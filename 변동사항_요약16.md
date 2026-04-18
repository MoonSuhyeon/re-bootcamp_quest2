# rag_app_v16.py 변동사항 요약

## 핵심 변경: 검색 엔진 성능 측정 시스템

v15가 "답변 품질 진단"이었다면,
v16은 **Dense Retrieval 자체 품질을 NDCG@k로 정량화**하고
리랭커가 그 위에서 얼마나 개선하는지를 분리 측정하는
**검색 엔진 성능 측정 시스템**으로 진화.

---

## 최종 스택
| 구성 | v15 (rag_app_v15.py) | v16 (rag_app_v16.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | 동일 |
| 평가 시스템 | 8개 필드 구조화 | 동일 |
| 품질 리포트 | overall_score + grade | 동일 |
| **NDCG 활용** | 단순 저장 | **Dense 품질 측정 + Reranker Gain 분리** |
| **검색 품질 리포트** | 없음 | **compute_search_quality_report() 신규** |
| **로그 필드** | ndcg_at_k | **+ reranker_gain + search_quality_report** |
| **UI 탭** | 4탭 | **5탭 (🔍 검색 품질 분석 추가)** |

---

## 변경 사항

### 1. `compute_search_quality_report()` — 검색 품질 리포트 (신규)

```python
def compute_search_quality_report(ndcg_prefilter: float, use_bm25: bool,
                                   n_candidates: int) -> dict:
```

#### 핵심 개념

```
ndcg_prefilter   = Dense 검색이 만든 순서의 NDCG (리랭커 점수 기준)
                   → 임베딩 검색이 얼마나 좋은 순서를 만들었는가

reranker_gain    = 1.0 - ndcg_prefilter
                   → 리랭커가 개선할 수 있었던 여지
                   → 0에 가까울수록 임베딩이 이미 최적
                   → 1에 가까울수록 리랭킹이 크게 기여
```

#### 품질 등급 기준

| quality_label | ndcg_prefilter 범위 | 의미 |
|:---:|:---:|------|
| `excellent` | ≥ 0.9 | 임베딩이 이미 최적 순서 → 리랭커는 확인 역할 |
| `good`      | 0.7 ~ 0.9 | 임베딩 품질 양호 → 리랭커 소폭 개선 |
| `fair`      | 0.5 ~ 0.7 | 임베딩 품질 보통 → 리랭커 개선 효과 유의미 |
| `poor`      | < 0.5 | 임베딩 품질 낮음 → 청킹/임베딩 모델 개선 필요 |

#### 반환 구조
```python
{
    "ndcg_prefilter":  0.6234,
    "reranker_gain":   0.3766,
    "quality_label":   "fair",
    "diagnosis":       "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미",
    "use_bm25":        True,
    "n_candidates":    10,
}
```

---

### 2. `build_log_entry()` — 신규 필드 추가

```python
def build_log_entry(..., search_quality_report=None):
    return {
        ...
        "search_quality_report": sqr,            # [NEW v16] 검색 품질 리포트 전체
        "reranker_gain": sqr.get("reranker_gain"), # [NEW v16] 최상위에 바로 접근 가능
        ...
    }
```

`ndcg_at_k`는 기존과 동일하게 유지(하위 호환). `reranker_gain`은 top-level 필드로 추가해 빠른 집계 가능.

---

### 3. `run_single_config()` — Ablation 결과에 검색 품질 지표 추가

Ablation config별로 `ndcg_prefilter`, `reranker_gain`, `search_quality` 필드 추가:

```python
return {
    ...
    "ndcg_prefilter":  ndcg_prefilter,          # [NEW v16]
    "reranker_gain":   sqr["reranker_gain"],     # [NEW v16]
    "search_quality":  sqr["quality_label"],     # [NEW v16]
    ...
}
```

---

### 4. 챗봇 탭 UI 변화

#### 리랭킹 status 메시지 확장
```
✅ 상위 3개 | Dense NDCG@3: 0.723 | Reranker Gain: 0.277 | good
```

#### NDCG 뱃지 → Dense NDCG + Gain delta 표시
```python
cols[6].metric(
    f"🔍 Dense NDCG@{k}",
    f"{ndcg_k:.3f}",
    delta=f"Gain {sqr['reranker_gain']:.3f}"
)
```

#### 🔍 검색 품질 리포트 expander (신규)
```
🔍 검색 품질 리포트 — 🟡 FAIR (Dense NDCG 0.623 / Reranker Gain 0.377)
  Dense NDCG@k   : 0.6234
  Reranker Gain  : 0.3766   (= 1.0 − Dense NDCG)
  후보 수         : 10개
  📋 임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미
```

---

### 5. Ablation 탭 변화

#### 결과 테이블 컬럼 추가
```
... Dense NDCG | Reranker Gain | 검색 품질 | ...
```

#### 자동 분석 5번째 카드 변경
- 기존: 💰 최소 토큰
- 변경: **🔍 최고 Dense NDCG** config (리랭킹 데이터 있을 때)

#### BM25 기여도 분석 강화
```python
# 정확도 차이 + NDCG 차이 동시 표시
"BM25 하이브리드 기여도: 정확도 +0.5 / Dense NDCG +0.045 (유효)"
```

#### 실험 결과 요약에 추가
- 실험 평균 Dense NDCG / 평균 Reranker Gain 출력

#### 오른쪽 시각화 차트: 응답 시간 → **Dense NDCG / Reranker Gain 비교 차트** (리랭킹 데이터 있을 때)

#### CSV 컬럼 추가
```
... ndcg_prefilter, reranker_gain, search_quality, ...
```

---

### 6. [NEW v16] TAB 5 — 🔍 검색 품질 분석

5번째 탭 신규 추가. Dense Retrieval 품질을 전용 대시보드에서 분석.

#### 요약 지표 (5개)
```
평균 Dense NDCG | 평균 Reranker Gain | Excellent 건수 | Poor 건수 | 총 측정 건수
```

#### 추이 차트 (2열)
| 차트 | 내용 |
|------|------|
| Dense NDCG@k 추이 | 시간에 따른 임베딩 검색 품질 변화 |
| Reranker Gain 추이 | 리랭커가 순서를 얼마나 개선했는지 변화 |

#### 분포 차트 (2열)
| 차트 | 내용 |
|------|------|
| NDCG 품질 등급 분포 | Excellent / Good / Fair / Poor 건수 bar chart |
| BM25 ON vs OFF NDCG 비교 | BM25 사용 여부별 평균 NDCG 비교 |

#### 의도별 평균 Dense NDCG
- 라우팅 의도(factual_lookup / reasoning / ...) 별 NDCG 평균 bar chart
- 특정 의도에서 낮은 NDCG → 해당 질문 유형에 맞는 전략 개선 근거

#### 저품질 질문 목록 (NDCG < 0.7)
- NDCG 오름차순 정렬
- 각 질문의 진단 메시지, 후보 수, BM25 사용 여부 표시
- 개선 힌트 제공

#### 전체 데이터 테이블 + CSV 다운로드
- `search_quality_YYYYMMDD_HHMMSS.csv` 파일명 자동 생성
- 컬럼: timestamp, question, ndcg_prefilter, reranker_gain, quality_label, use_bm25, n_candidates, intent

---

## 로그 데이터 구조 변화

```python
# v16 신규 필드
{
    "ndcg_at_k": 0.6234,               # 기존 유지 (= ndcg_prefilter, 하위 호환)
    "reranker_gain": 0.3766,            # [NEW v16] top-level 바로 접근용
    "search_quality_report": {          # [NEW v16] 검색 품질 리포트 전체
        "ndcg_prefilter":  0.6234,
        "reranker_gain":   0.3766,
        "quality_label":   "fair",
        "diagnosis":       "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미",
        "use_bm25":        True,
        "n_candidates":    10
    }
}
```

---

## LLM 호출 횟수

v16은 LLM 호출 추가 없음. `compute_search_quality_report()`는 순수 계산 함수.

| 단계 | v16 |
|------|-----|
| 쿼리 라우팅 | 1 |
| 쿼리 리라이팅 | 0~1 |
| Rerank | 0~1 |
| Step1~3 | 3 |
| 평가 | 1 |
| 환각 분석 | +1 (환각 시) |
| **합계** | **7~8회 (v15 동일)** |

---

## 파일 변경

| 항목 | v15 | v16 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v15.json` | `rag_eval_logs_v16.json` |
| EMBED_CACHE_FILE | `embed_cache_v15.pkl` | `embed_cache_v16.pkl` |

---

## 성능 개선 기대 포인트

| 측정 가능해진 것 | 활용 방법 |
|------|-----------|
| Dense NDCG 추이 | 청킹 크기·전략 변경 시 검색 품질 변화 확인 |
| Reranker Gain 추이 | 리랭커가 불필요한 케이스 파악 → 응답 속도 최적화 |
| BM25 ON vs OFF NDCG 비교 | BM25 비용 대비 효과 데이터 기반 판단 |
| 의도별 NDCG | 특정 의도(factual/multi_hop)에서 검색 부족 → 쿼리 전략 조정 |
| Poor NDCG 질문 목록 | 재현 가능한 개선 대상 질문 추출 → 데이터 기반 개선 루프 |
