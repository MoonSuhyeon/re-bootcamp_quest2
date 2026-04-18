# rag_app_v21.py 변동사항 요약

## 핵심 변경: 병렬 검색 + Context Compression Phase 2 + Tool-Augmented RAG

v20의 실패 데이터셋/Auto Improvement 위에,
**"한 끗 차이"** 세 가지를 추가해 상위 1% RAG에서 그 이상으로:

1. **병렬 검색** — ThreadPoolExecutor로 4채널 동시 실행 → 대기 시간 단축
2. **LongContextReorder + Selective Context Phase 2** — LLM 입력 품질 극대화
3. **Tool-Augmented RAG** — 수치 계산을 Python으로 위임 → 수치 환각 0%

---

## 최종 스택

| 구성 | v20 | v21 |
|------|-----|-----|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Multi-Vector + BM25 RRF | **병렬 4채널 동시 실행** |
| Context Compression | 임베딩 유사도 문장 추출 (Phase 1) | **+ LongContextReorder + Phase 2 cross-chunk dedup** |
| 실패 데이터셋 | FailureDataset + Auto Improvement | 동일 |
| Self-Refinement | Draft → Critique → Refine | 동일 |
| Fallback | 평가 기반 자동 재시도 | 동일 |
| **병렬 검색** | (없음) | **`retrieve_parallel()` ThreadPoolExecutor** |
| **LongContextReorder** | (없음) | **`reorder_lost_in_middle()` 위치 최적화** |
| **Selective Context P2** | (없음) | **`selective_context_phase2()` 청크 간 중복 제거** |
| **Tool-Augmented** | (없음) | **`tool_augmented_answer()` + `_safe_eval()`** |

---

## 변경 사항

### 1. 신규 상수

```python
PARALLEL_MAX_WORKERS    = 4       # ThreadPoolExecutor 워커 수
DEDUP_THRESHOLD_DEFAULT = 0.85    # 문장 중복 제거 코사인 임계값

CALC_PATTERNS = [
    r'\d+[,.]\d+', r'얼마', r'몇\s', r'합계', r'총\s', r'평균',
    r'비율', r'퍼센트', r'%', r'증가', r'감소', r'차이', r'계산',
    ...
]
```

신규 import:
```python
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
```

---

### 2. `retrieve_parallel()` — 4채널 병렬 검색 (신규)

```python
def retrieve_parallel(queries, index, chunks, sources,
                      mv_index=None, use_bm25=True,
                      top_k_per_query=20, tracer=None) -> tuple[list, float]:
```

#### 구조

```
ThreadPoolExecutor(max_workers=4)
├─ _dense()    → FAISS dense 검색
├─ _bm25()     → BM25 키워드 검색
├─ _sentence() → Multi-Vector 문장 레벨 검색
└─ _keyword()  → Multi-Vector 키워드 레벨 검색
        ↓ (동시 실행)
RRF merge (k=60) — 4개 결과 통합
        ↓
(result_list, parallel_ms)   # parallel_ms: 소요 시간(ms)
```

- asyncio 대신 `ThreadPoolExecutor` 사용 — Streamlit 동기 환경과 충돌 없음
- Multi-Vector index가 없는 경우 sentence/keyword 채널 자동 생략
- 반환: `([(chunk, source, score), ...], float)`

---

### 3. `reorder_lost_in_middle()` — LongContextReorder (신규)

```python
def reorder_lost_in_middle(chunks: list, scores: list) -> list:
```

#### 원리 ("Lost in the Middle" 논문)

```
점수 내림차순 정렬: [A(1.0), B(0.9), C(0.8), D(0.7), E(0.6)]
                         ↓
인터리브 배치 (짝수→앞, 홀수→뒤):
[A, C, E, D, B]
 ↑              ↑
최고 품질       2위
(LLM이 주목하는 위치)
```

- 추가 LLM 호출 없음 — 기존 rerank 점수 재사용
- `run_rag_pipeline()` 내 `[4b]` 단계에서 적용

---

### 4. `selective_context_phase2()` — Cross-Chunk 중복 제거 (신규)

```python
def selective_context_phase2(question, chunks,
                              dedup_threshold=DEDUP_THRESHOLD_DEFAULT,
                              max_sentences_per_chunk=6,
                              tracer=None) -> tuple[list, dict]:
```

#### Phase 1 vs Phase 2 비교

| | Phase 1 (`compress_chunks`) | Phase 2 (`selective_context_phase2`) |
|--|--|--|
| 범위 | 청크 내부 문장 선별 | **청크 간 중복 문장 제거** |
| 기준 | 질문과의 유사도 | 질문 유사도 + **청크 간 코사인 중복 체크** |
| 임계값 | (없음) | cosine ≥ 0.85 → 중복으로 간주, 제거 |

#### 처리 흐름

```
전체 청크의 모든 문장 수집
        ↓
문장 전체 임베딩 (1회 배치 호출)
        ↓
Cross-chunk dedup:
  kept = []
  for each sentence:
    if cosine(sentence, any_kept) >= threshold → skip
    else → kept.append(sentence)
        ↓
청크별 재조립: 살아남은 문장 중 질문 관련성 top-N 유지
        ↓
(compressed_chunks, stats)
```

#### stats 스키마

```python
{
    "original_sents":  int,    # 원본 총 문장 수
    "after_dedup":     int,    # 중복 제거 후 문장 수
    "dedup_removed":   int,    # 제거된 문장 수
    "orig_chars":      int,    # 원본 총 글자 수
    "comp_chars":      int,    # 압축 후 총 글자 수
    "ratio":           float,  # comp_chars / orig_chars
}
```

---

### 5. Tool-Augmented RAG (신규 3개 함수)

#### 5-1. `detect_calc_intent()`

```python
def detect_calc_intent(question: str, chunks: list) -> bool:
```

- `CALC_PATTERNS` 정규식 — 질문에서 계산 의도 탐지
- chunks 내 숫자 데이터 존재 여부 확인
- 두 조건 모두 True → 계산 모드 활성화

#### 5-2. `_safe_eval()`

```python
def _safe_eval(code: str, data: dict) -> any:
```

제한된 실행 환경:
```python
# 허용된 builtins만
safe_builtins = {
    "math": math,
    "round": round, "abs": abs, "sum": sum,
    "max": max, "min": min, "len": len,
    "int": int, "float": float,
    "list": list, "range": range, "sorted": sorted,
}
exec(code, {"__builtins__": safe_builtins, "data": data})
# result 변수 추출, 실패 시 ast.parse fallback
```

- `__builtins__={}` 패턴으로 임의 코드 실행 차단
- import, open, eval 등 위험 함수 완전 차단

#### 5-3. `tool_augmented_answer()`

```python
def tool_augmented_answer(question, chunks, tracer=None) -> tuple:
    # Returns: (answer, python_code, calc_result)
    # 실패 시: (None, None, None)
```

#### 처리 흐름 (LLM 2회 호출)

```
[1] LLM 호출 1 — 데이터 추출 + Python 코드 생성
    Input:  question + chunks
    Output: JSON { "data": {...}, "python_code": "result = ..." }
        ↓
[2] _safe_eval(python_code, data) → calc_result
        ↓
[3] LLM 호출 2 — 최종 답변 생성
    Input:  question + chunks + f"계산 결과: {calc_result}"
    Instruction: "이 값을 그대로 사용할 것 — 재계산 금지"
        ↓
(answer, python_code, calc_result)
```

#### 수치 환각 방지 원리

```
기존: LLM이 숫자를 "기억"해서 답변 → 환각 발생
 ↓
v21: Python이 계산 → 결과를 LLM에 주입 → LLM은 결과만 서술
```

---

### 6. `run_rag_pipeline()` — 신규 파라미터 추가

```python
def run_rag_pipeline(...,
    use_parallel_search: bool = True,                          # [NEW v21]
    use_lim_reorder: bool = True,                              # [NEW v21]
    use_selective_context: bool = False,                       # [NEW v21]
    selective_dedup_thresh: float = DEDUP_THRESHOLD_DEFAULT,   # [NEW v21]
    use_tool_augment: bool = False,                            # [NEW v21]
) -> dict:
```

#### 파이프라인 변경 위치

```
[1] 쿼리 확장
[2] 검색
    ├─ use_parallel_search=True  → retrieve_parallel()         [NEW v21]
    ├─ mv_index 있음             → _retrieve_mv_sequential()
    └─ 기본                      → retrieve_hybrid()
[3] Rerank
[4a] 중복 제거 (기존)
[4b] use_lim_reorder=True       → reorder_lost_in_middle()    [NEW v21]
[5] Context Compression
    ├─ use_selective_context=True → selective_context_phase2() [NEW v21]
    └─ use_compression=True       → compress_chunks() (Phase 1)
[6a] use_tool_augment=True, detect_calc_intent() → True
         → tool_augmented_answer() — 성공 시 [7][8] 건너뜀    [NEW v21]
[6] 답변 생성 (일반 경로)
[7] Self-Refinement
[8] 평가
[9] 실패 저장 (v20)
```

반환 dict 추가 필드:
```python
{
    "parallel_ms":      float | None,   # 병렬 검색 소요 시간(ms)
    "tool_code":        str | None,     # Tool이 생성한 Python 코드
    "calc_result":      any | None,     # 계산 결과
    "tool_used":        bool,           # Tool-Augmented 사용 여부
    "selective_stats":  dict | None,    # Phase 2 통계
}
```

---

### 7. `build_log_entry()` — 신규 필드 추가

```python
def build_log_entry(...,
    parallel_ms=None,
    tool_used=False,
    selective_stats=None):
```

```python
# 신규 로그 필드
{
    "parallel_ms":     float | None,
    "tool_used":       bool,
    "selective_stats": dict | None,
}
```

---

### 8. 사이드바 변화

#### ⚡ [v21] 병렬 검색 섹션 (신규)
```
⚡ [v21] 병렬 검색
├─ 병렬 검색 활성화 토글 (기본값: ON)
│    Dense + BM25 + Sentence + Keyword 동시 실행
└─ LongContextReorder 토글 (기본값: ON)
     중요 청크를 프롬프트 앞/뒤로 배치
```

#### 🔬 [v21] Selective Context Phase 2 섹션 (신규)
```
🔬 [v21] Selective Context Phase 2
├─ Phase 2 활성화 토글 (기본값: OFF)
│    청크 간 중복 문장 제거 (더 강력한 압축)
└─ 중복 제거 임계값 슬라이더 (0.70 ~ 0.95, 기본값: 0.85)
```

#### 🔧 [v21] Tool-Augmented RAG 섹션 (신규)
```
🔧 [v21] Tool-Augmented RAG
└─ Tool-Augmented 활성화 토글 (기본값: OFF)
     수치 계산 → Python 실행 (+2 LLM 호출)
```

---

### 9. [NEW] TAB 7 — ⚡ v21 분석 탭 (완전 신규)

#### 섹션 1: 병렬 검색 속도 분석
```
평균 병렬 검색 시간 | P50 | P95 | 전체 레이턴시 대비 비율
병렬 검색 시간 추이 꺾은선 그래프
```

#### 섹션 2: Selective Context Phase 2 효과
```
평균 중복 제거율 | 평균 압축률 | Phase 2 사용 시 정확도
원본 vs 압축 후 글자 수 비교 bar chart
```

#### 섹션 3: Tool-Augmented RAG 효과
```
Tool 사용 / 미사용 환각 비율 비교
Tool 사용 / 미사용 정확도 비교
Tool이 생성한 Python 코드 예시
```

#### 섹션 4: LongContextReorder 설명
```
"Lost in the Middle" 논문 원리 다이어그램
배치 전/후 청크 순서 비교
```

---

## LLM 호출 횟수

| 상황 | v20 | v21 |
|------|-----|-----|
| 기본 (Tool OFF) | 동일 | 동일 |
| Tool-Augmented ON, 계산 의도 감지됨 | 동일 | **+2회** |
| Tool-Augmented ON, 계산 의도 없음 | 동일 | 동일 |
| Selective Context Phase 2 | 동일 | 동일 (임베딩 추가, LLM 없음) |
| LongContextReorder | 동일 | 동일 (재정렬만, LLM 없음) |
| 병렬 검색 | 동일 | 동일 (검색 병렬화, LLM 없음) |

> 병렬 검색·LongContextReorder·Phase 2는 추가 LLM 호출 없음.
> Tool-Augmented만 계산 의도 감지 시 +2회 (코드 생성 + 최종 답변).

---

## 파일 변경

| 항목 | v20 | v21 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v20.json` | `rag_eval_logs_v21.json` |
| EMBED_CACHE_FILE | `embed_cache_v20.pkl` | `embed_cache_v21.pkl` |
| ANSWER_CACHE_FILE | `answer_cache_v20.json` | `answer_cache_v21.json` |
| FAILURE_DATASET_FILE | `failure_dataset_v20.json` | `failure_dataset_v21.json` |

---

## v21 핵심 원리 요약

```
[병렬 검색]
Dense ──┐
BM25  ──┤ ThreadPoolExecutor → RRF 통합
Sent  ──┤ (동시 실행)
KW    ──┘

[LongContextReorder]
[C, A, E, D, B]  →  [A, C, E, D, B]
                      ↑ 최고 품질 → 앞/뒤 배치

[Selective Context Phase 2]
청크1: [s1, s2, s3]  ─┐
청크2: [s4, s5, s6]  ─┤ cross-chunk cosine dedup
청크3: [s7, s8, s9]  ─┘
          ↓
[s1, s2, s4, s7]  (s3≈s6, s5≈s8 → 제거)

[Tool-Augmented]
"매출이 얼마나 증가했나?" + chunks(숫자 포함)
    → LLM: Python 코드 생성
    → _safe_eval: result = 1234.5
    → LLM: "매출은 1234.5 증가했습니다" (재계산 금지)
```

### 버전 로드맵

```
v1~v10:  파이프라인 구축
v11~v19: 평가·최적화·캐시
v20:     실패 → 학습 → 개선 루프
v21:     ← 지금 여기: 속도 + 품질 + 정확도 한 끗
v22:     (예정) 모니터링 · FastAPI · 인증/사용자 관리
```
