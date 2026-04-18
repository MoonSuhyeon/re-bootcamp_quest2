# rag_app_v17=py 변동사항 요약

## 핵심 변경: Fallback 자동 재시도 시스템 — 진짜 에이전트 동작

v16의 "검색 엔진 성능 측정" 위에,
**자기 평가 → 불만족 시 자율 재시도 → 최선 결과 채택**
하는 Fallback 루프를 추가해 "진짜 에이전트 느낌"을 구현.

---

## 최종 스택
| 구성 | v16 (rag_app_v16.py) | v17 (rag_app_v17.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | 동일 |
| 평가 시스템 | 구조화 8개 필드 | 동일 |
| 품질 리포트 | overall_score + grade | 동일 |
| 검색 품질 분석 | NDCG + Reranker Gain | 동일 |
| **파이프라인 구조** | 단일 실행 | **`run_rag_pipeline()` 분리 + Fallback 루프** |
| **Fallback 시스템** | 없음 | **평가 결과 기반 자동 재시도 (최대 2회)** |
| **파라미터 에스컬레이션** | 없음 | **top_k↑ + prefilter_n↑ + 모든 기능 강제 ON** |
| **최선 결과 선택** | 없음 | **overall_score 기준 자동 채택** |

---

## 변경 사항

### 1. Fallback 상수

```python
MAX_RETRIES         = 2   # 최대 재시도 횟수
FALLBACK_MIN_ACC    = 3   # 정확도 이 미만이면 트리거
FALLBACK_HALL_TYPES = ("부분적", "있음")  # 이 환각 유형이면 트리거
```

---

### 2. `should_fallback(evaluation)` — 재시도 필요 판단 (신규)

```python
def should_fallback(evaluation: dict) -> tuple:
    # Returns: (bool, str) — (재시도 필요 여부, 트리거 사유)
```

트리거 조건 (OR):
| 조건 | 예시 |
|------|------|
| 정확도 < 3 | `"정확도 2/5 < 3"` |
| 환각 감지 (부분적/있음) | `"환각 감지: 부분적"` |
| 두 조건 동시 | `"정확도 2/5 < 3 + 환각 감지: 있음"` |

---

### 3. `escalate_params(base_eff, base_prefilter_n, attempt)` — 파라미터 강화 (신규)

```python
def escalate_params(base_eff: dict, base_prefilter_n: int, attempt: int) -> tuple:
    # Returns: (new_eff, new_prefilter_n)
```

| 재시도 | top_k | prefilter_n | BM25 | Reranking | Rewrite |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1회차 | base+1 (max 5) | base+5 (max 20) | 강제 ON | 강제 ON | 강제 ON |
| 2회차 | base+2 (max 5) | base+10 (max 20) | 강제 ON | 강제 ON | 강제 ON |

---

### 4. `run_rag_pipeline()` — 파이프라인 단일 실행 함수 (신규)

```python
def run_rag_pipeline(question, eff, index, chunks, sources,
                     prefilter_n, use_multidoc,
                     num_rewrites=3, use_session_cache=True) -> dict:
```

기존에 챗봇 탭 UI 코드에 산재되어 있던 파이프라인 전체를 **하나의 함수로 캡슐화**.
Fallback 루프가 같은 파이프라인을 파라미터만 바꿔 반복 호출할 수 있도록 분리.

반환 dict 포함 필드:
```python
{
    "tracer", "queries", "ranked", "final_chunks", "final_sources", "final_scores",
    "summaries", "analysis", "answer", "mode",
    "evaluation", "hall_cause", "quality_report",
    "ndcg_k", "sqr", "eff", "prefilter_n"
}
```

> `run_single_config()` (Ablation) 도 내부적으로 `run_rag_pipeline()` 호출로 리팩터.

---

### 5. 챗봇 탭 — Fallback 루프 구조

```
[Step 0] 쿼리 라우팅 (auto_routing ON 시)

[Attempt 0] run_rag_pipeline(question, eff, ...)
  → evaluate_answer()
  → should_fallback(evaluation)
  
  ┌─ fallback 불필요 → 최종 답변 출력
  │
  └─ fallback 필요 → 
      [Attempt 1] escalate_params(eff, prefilter_n, 1)
                  run_rag_pipeline(question, new_eff, ...)
                  should_fallback() → 여전히 필요?
                  
                  ┌─ No → 종료
                  └─ Yes → 
                      [Attempt 2] escalate_params(eff, prefilter_n, 2)
                                  run_rag_pipeline(...)
                                  종료 (MAX_RETRIES=2)

→ 모든 시도 중 overall_score 최고인 결과를 최종 답변으로 채택
```

**세션 캐시 정책:**
- Attempt 0: `use_session_cache=True` (정상 캐시 활용)
- Attempt 1+: `use_session_cache=False` (캐시 오염 방지, 새로운 rewrite 생성)

---

### 6. 챗봇 탭 UI 변화

#### 진행 상태 표시
```
🤔 답변 생성 중 (시도 1)...
  → ⚠️ 시도 1 완료 | 정확도 2/5 · 환각 부분적 · 등급 D → Fallback 예정: 정확도 2/5 < 3

🔄 Fallback 재시도 1/2 — 정확도 2/5 < 3
  → ✅ 개선됨 재시도 1 | 정확도 4/5 · 환각 없음 · 등급 B
```

#### 뱃지: 7개 → 8개 (재시도 뱃지 추가)
```
⏱ 응답 | 🔤 토큰 | 📐 정확도 | 🎯 관련성 | 환각 | 🧭 신뢰도 | 🔄 재시도 | 🔍 NDCG
```

- Fallback 미발생: `재시도: 없음`
- Fallback 발생: `재시도: N회 | delta: D→B` (등급 변화 delta)

#### 🔄 Fallback 실행 내역 expander (Fallback 발생 시만 표시)
```
🔄 Fallback 실행 내역 (1회 재시도)
  📋 시도 1: 정확도 2/5 · 환각 부분적 · 등급 D(2.5) | top_k=3 prefilter=10 BM25=OFF
  🏆 시도 2: 정확도 4/5 · 환각 없음 · 등급 B(3.75)  | top_k=4 prefilter=15 BM25=ON
```

#### 라우팅 expander: "적용된 파라미터 (최종)" 으로 레이블 변경

---

### 7. 사이드바 변화

```
🔄 Fallback 설정
├─ Fallback 자동 재시도 토글 (기본값: ON)
├─ 트리거: 정확도 < 3 / 환각 감지
└─ 전략: top_k↑, prefilter_n↑, 모든 기능 강제 ON
```

---

### 8. `build_log_entry()` — Fallback 필드 추가

```python
def build_log_entry(...,
    fallback_triggered=False,
    fallback_attempts=0,
    fallback_history=None):
```

#### 신규 로그 필드
```python
{
    "fallback_triggered": True,          # Fallback 발동 여부
    "fallback_attempts":  1,             # 실제 재시도 횟수
    "fallback_history": [                # 시도별 상세 기록
        {
            "attempt": 0,
            "trigger": None,
            "eff": {"use_bm25": False, "use_reranking": True, "top_k": 3, ...},
            "prefilter_n": 10,
            "accuracy": 2, "hallucination": "부분적",
            "overall_score": 2.0, "grade": "D",
            "tokens": 3200, "latency_ms": 8500
        },
        {
            "attempt": 1,
            "trigger": "정확도 2/5 < 3",
            "eff": {"use_bm25": True, "use_reranking": True, "top_k": 4, ...},
            "prefilter_n": 15,
            "accuracy": 4, "hallucination": "없음",
            "overall_score": 3.75, "grade": "B",
            ...
        }
    ]
}
```

---

### 9. 에이전트 분석 탭 — Fallback 통계 섹션 추가

#### 요약 지표 6번째: Fallback 율 추가
```
평균 정확도 | 평균 관련성 | 환각 없음 비율 | 평균 NDCG@k | 평균 종합점수 | 🔄 Fallback 율
```

#### 🔄 Fallback 분석 섹션 (신규)
- Fallback 발생 건수 / 평균 재시도 횟수
- **Fallback 전 정확도 → 후 정확도 개선 효과** (`avg_first → avg_final` metric)
- **트리거 사유 분포** bar chart (정확도 부족 vs 환각 감지)

#### per-log expander: Fallback 내역 표시
- 각 시도별 정확도·환각·등급·파라미터 요약
- 재시도 metric 카드 추가

#### 트레이싱 탭: 6번째 요약 지표 → `🔄 Fallback 발생 N회`

---

### 10. 검색 품질 탭 데이터 테이블

`Fallback` 컬럼 추가 — 해당 질문에서 Fallback이 발생했는지 `🔄` 표시

---

## LLM 호출 횟수

| 상황 | 호출 수 |
|------|---------|
| Fallback 없음 | 7~8회 (v16 동일) |
| Fallback 1회 | 14~16회 |
| Fallback 2회 | 21~24회 |

> Fallback은 비용이 크므로 사이드바에서 ON/OFF 가능.
> 임베딩 캐시 덕분에 재시도 시 embedding API 호출 = 0 (캐시 히트).

---

## 파일 변경

| 항목 | v16 | v17 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v16.json` | `rag_eval_logs_v17.json` |
| EMBED_CACHE_FILE | `embed_cache_v16.pkl` | `embed_cache_v17.pkl` |

---

## 에이전트 동작 원리 요약

```
사용자 질문
    ↓
[1] 파이프라인 실행 (검색 → 생성)
    ↓
[2] 자기 평가 (evaluate_answer)
    ↓
[3] 품질 기준 미달? (정확도 < 3 OR 환각)
    ↓ YES
[4] 파라미터 강화 (top_k↑, 모든 기능 ON)
    ↓
[5] 파이프라인 재실행 → 다시 [2]로
    ↓ NO or MAX_RETRIES 도달
[6] 모든 시도 중 최고 점수 결과 선택
    ↓
[7] 최종 답변 출력 + Fallback 내역 표시
```

이것이 "진짜 에이전트" 느낌의 핵심:
- **자기 인식** (자신의 답변 품질을 스스로 평가)
- **자율 결정** (기준 미달 시 재시도 결정)
- **전략 조정** (더 많은 컨텍스트, 더 강한 검색으로 에스컬레이션)
- **최선 선택** (여러 시도 중 가장 좋은 결과 채택)
