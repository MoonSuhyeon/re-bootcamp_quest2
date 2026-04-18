# rag_app_v14.py 변동사항 요약

## 핵심 변경: Ablation Study — 파이프라인 구성 요소별 기여도 자동 실험

v13의 기능을 그대로 유지하면서, 동일 질문에 여러 파이프라인 설정을 자동 실행해 각 구성 요소의 성능 기여도를 측정하는 실험 탭 추가.

---

## 최종 스택
| 구성 | v13 (rag_app_v13.py) | v14 (rag_app_v14.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | 동일 |
| 임베딩 캐시 | EmbeddingCache | 동일 |
| NDCG@k | 자동 계산 | 동일 |
| 쿼리 라우팅 | 6개 의도 자동 분류 | 동일 |
| **Ablation Study** | 없음 | **6개 Config 자동 비교 실험** |
| **실험 결과 분석** | 없음 | **자동 기여도 분석 + 시각화** |
| **UI 탭** | 3탭 | **4탭 (🧬 Ablation Study 추가)** |

---

## 변경 사항

### 1. `ABLATION_CONFIGS` — 실험할 파이프라인 설정 목록

6개의 사전 정의 Config. 각 Config는 세 가지 구성 요소(쿼리 리라이팅 / BM25 / 리랭킹)의 ON/OFF 조합:

| Config ID | 이름 | 쿼리 리라이팅 | BM25 | 리랭킹 |
|-----------|------|:---:|:---:|:---:|
| `full` | ✅ Full Pipeline | ON | ON | ON |
| `no_rewrite` | ❌ No Query Rewrite | OFF | ON | ON |
| `no_bm25` | ❌ No BM25 (Dense만) | ON | OFF | ON |
| `no_rerank` | ❌ No Rerank | ON | ON | OFF |
| `dense_rerank` | ⚡ Dense + Rerank | OFF | OFF | ON |
| `minimal` | 🔥 Minimal (기본만) | OFF | OFF | OFF |

```python
ABLATION_CONFIGS = [
    {"id": "full",         "name": "✅ Full Pipeline",     "query_rewrite": True,  "bm25": True,  "rerank": True},
    {"id": "no_rewrite",   "name": "❌ No Query Rewrite",  "query_rewrite": False, "bm25": True,  "rerank": True},
    {"id": "no_bm25",      "name": "❌ No BM25 (Dense만)", "query_rewrite": True,  "bm25": False, "rerank": True},
    {"id": "no_rerank",    "name": "❌ No Rerank",         "query_rewrite": True,  "bm25": True,  "rerank": False},
    {"id": "dense_rerank", "name": "⚡ Dense + Rerank",    "query_rewrite": False, "bm25": False, "rerank": True},
    {"id": "minimal",      "name": "🔥 Minimal (기본만)",  "query_rewrite": False, "bm25": False, "rerank": False},
]
```

---

### 2. `run_single_config()` — 단일 Config 파이프라인 전체 실행

```python
def run_single_config(question: str, config: dict,
                      index, chunks: list, sources: list,
                      top_k: int = 3, prefilter_n: int = 10) -> dict:
```

Config 한 개로 전체 파이프라인(쿼리 리라이팅 → 검색 → prefilter → rerank → Step1~3 → 평가)을 실행하고 결과 dict 반환.

반환 필드:

| 필드 | 내용 |
|------|------|
| `config_id` | config 식별자 |
| `config_name` | config 표시명 |
| `query_rewrite` / `bm25` / `rerank` | 적용된 설정값 |
| `accuracy` | 정확도 0~5 |
| `relevance` | 관련성 0~5 |
| `hallucination` | 없음/부분적/있음 |
| `latency_ms` | 전체 소요 시간 ms |
| `total_tokens` | 총 토큰 수 |
| `ndcg` | NDCG@k (리랭킹 OFF면 None) |
| `answer` | 생성된 답변 |
| `error` | 예외 발생 시 오류 메시지 |

**핵심**: `use_session_cache=False` 파라미터로 `rewrite_queries()` 호출 → ablation 실행이 챗봇 세션 캐시를 오염시키지 않음.

---

### 3. `rewrite_queries()` — `use_session_cache` 파라미터 추가

```python
def rewrite_queries(original_query: str, n: int = 3, tracer: Tracer = None,
                    use_session_cache: bool = True):
```

- 챗봇 탭 호출 시: `use_session_cache=True` (기존 동작 유지)
- Ablation 실행 시: `use_session_cache=False` → 캐시 읽기/쓰기 모두 생략

---

### 4. 세션 상태 추가

```python
("ablation_results", []),  # [NEW v14]
```

실험 결과를 세션에 보존 → 탭 전환 후에도 결과 유지.

---

### 5. UI — 🧬 Ablation Study 탭 (4번째 탭)

```
tab_chat, tab_trace, tab_agent, tab_ablation = st.tabs(
    ["💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation Study"]
)
```

#### 실험 설정 섹션
- **질문 입력**: 동일 질문으로 모든 Config 실행
- **Config 멀티셀렉트**: 6개 중 원하는 Config만 선택 (기본: 전체)
- **top_k / pre-filter 수 슬라이더**: ablation 전용 (`key="abl_top_k"`, `key="abl_pf"`)
- **예상 API 호출 안내**: `n_configs × 5 = 예상 호출 수` 자동 표시
- **실행 버튼**: 질문 미입력 또는 Config 미선택 시 비활성화

#### 실행 흐름
```
[실행 버튼 클릭]
  → Progress bar + 현재 실행 중인 Config 표시
  → run_single_config() 순차 실행
  → st.session_state.ablation_results 저장
```

#### 결과 표시 섹션

**1. 비교 테이블** (`st.dataframe`)
```
Config | 정확도(/5) | 관련성(/5) | 환각여부 | 응답(ms) | 토큰 | NDCG@k | 오류
```

**2. 자동 분석 결과** (4개 지표 카드)
| 카드 | 내용 |
|------|------|
| 🏆 최고 정확도 | 가장 높은 정확도 Config |
| 🎯 최고 관련성 | 가장 높은 관련성 Config |
| ⚡ 최저 지연 | 가장 빠른 Config |
| 💰 최소 토큰 | 가장 저렴한 Config |

**3. 구성 요소별 기여도 텍스트 분석**
```python
# Full vs No_Rewrite → 쿼리 리라이팅 기여도
diff = full_result["accuracy"] - no_rewrite["accuracy"]
"쿼리 리라이팅 기여도: 정확도 +{diff} (유효/미미)"

# Full vs No_BM25 → BM25 기여도
# Full vs No_Rerank → 리랭킹 기여도
```

**4. 시각화** (2열)
- 왼쪽: 정확도/관련성 막대 차트 (Config별 비교)
- 오른쪽: 응답 시간 막대 차트

**5. Config별 답변 드릴다운** (expander)
- 각 Config의 실제 생성 답변 + 정확도/관련성/환각/응답/토큰/NDCG 지표

**6. CSV 다운로드**
- `ablation_YYYYMMDD_HHMMSS.csv` 파일명 자동 생성

---

## 임베딩 비용 최적화 원리

```
1회차 ablation 실행
  → 첫 번째 Config: 모든 청크 임베딩 API 호출 → 캐시 저장
  → 2번째~6번째 Config: 동일 청크 → 캐시 히트 → API 호출 0

= 6개 Config 실행 시 임베딩 비용은 1개 Config 분과 동일
```

---

## LLM 호출 횟수

| 단계 | Config당 (ablation) |
|------|---------------------|
| 쿼리 리라이팅 | 0~1 (config.query_rewrite) |
| Rerank | 0~1 (config.rerank) |
| Step1 요약 | 1 |
| Step2 분석 | 1 |
| Step3 답변 | 1 |
| 평가 | 1 |
| **1 Config 합계** | **4~6회** |
| **6 Config 전체** | **약 24~36회** |

> 챗봇 탭의 일반 질문은 v13과 동일하게 7~8회.

---

## 파일 변경

| 항목 | v13 | v14 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v13.json` | `rag_eval_logs_v14.json` |
| EMBED_CACHE_FILE | `embed_cache_v13.pkl` | `embed_cache_v14.pkl` |

---

## 성능 개선 기대 포인트
- 구성 요소별 기여도를 실제 수치로 확인 → 불필요한 단계 제거 근거 확보
- BM25 없이도 충분하다면 BM25 OFF → 응답 속도 개선
- 리라이팅 기여도가 작으면 OFF → LLM 호출 1회 절감
- 임베딩 캐시 덕분에 반복 ablation 실험 시 추가 임베딩 비용 없음
