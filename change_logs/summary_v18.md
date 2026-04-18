# rag_app_v18.py 변동사항 요약

## 핵심 변경: Dynamic Retrieval + Answer Self-Refinement

v17의 "Fallback 자동 재시도" 위에,
**질문 유형에 따라 검색 전략을 자동 조정**하고
**Draft → Critique → Refine** 패턴으로 답변을 스스로 개선하는
두 가지 LLM 활용 핵심 패턴을 추가.

---

## 최종 스택
| 구성 | v17 (rag_app_v17.py) | v18 (rag_app_v18.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | 동일 |
| 평가 시스템 | 구조화 8개 필드 | 동일 |
| 품질 리포트 | overall_score + grade | 동일 |
| 검색 품질 분석 | NDCG + Reranker Gain | 동일 |
| Fallback 시스템 | 평가 기반 자동 재시도 | 동일 |
| **검색 전략** | 라우팅 결과 단일 적용 | **`DYNAMIC_RETRIEVAL_PROFILES` 의도별 자동 조정** |
| **답변 생성** | 단일 생성 | **Draft → Critique → Refine (Self-Refinement)** |

---

## 변경 사항

### 1. `DYNAMIC_RETRIEVAL_PROFILES` — 의도별 검색 전략 프로필 (신규)

```python
DYNAMIC_RETRIEVAL_PROFILES = {
    "definition":     {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": False},
    "factual_lookup": {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": None},
    "reasoning":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True},
    "multi_hop":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 8, "top_k_delta": 1, "multidoc_override": True},
    "exploratory":    {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True},
    "ambiguous":      {"bm25_boost": True,  "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 0, "multidoc_override": None},
}
```

| 의도 | 핵심 조정 | 이유 |
|------|-----------|------|
| `definition` | BM25 ↑, MultiDoc OFF | 정의형은 키워드 매칭 + 단순 답변으로 충분 |
| `factual_lookup` | Rerank 강제 ON | 정확한 사실 검색에 정밀 재정렬 필수 |
| `reasoning` | MultiDoc 강제, Rerank ON, top_k+1 | 분석형은 다수 문서 종합 필요 |
| `multi_hop` | prefilter+8, MultiDoc 강제 | 여러 청크 연결 추론 → 넓은 후보 필요 |
| `exploratory` | BM25 ↑, MultiDoc ON, top_k+1 | 탐색형은 넓은 Recall + 다양한 관점 |
| `ambiguous` | BM25 + Rerank 강화 | 의도 불명 → 전략적 안전망 |

---

### 2. `apply_dynamic_retrieval()` — 의도별 프로필 적용 (신규)

```python
def apply_dynamic_retrieval(intent: str, eff: dict, prefilter_n: int,
                             use_multidoc: bool) -> tuple:
    # Returns: (new_eff, new_prefilter_n, new_use_multidoc, profile_label)
```

- 라우팅 `eff` 위에 프로필을 덧씌우는 **2-layer 전략 구조**
- `multidoc_override=None` → 사용자 설정 유지
- `multidoc_override=True/False` → 의도에 따라 강제 변경
- `prefilter_delta` → `min(20, prefilter_n + delta)` 상한 제한

#### 챗봇 탭 적용 시점
```
[Step 0] 쿼리 라우팅 → _apply_routing() → 기본 eff 결정
           ↓
[Step 0-2] apply_dynamic_retrieval(intent, eff) → eff 세부 조정
           ↓
[Attempt 0] run_rag_pipeline(조정된 eff, ...)
```

---

### 3. `critique_answer()` — Draft 비판 분석 (신규)

```python
def critique_answer(question, context_chunks, draft, tracer=None) -> str:
```

Draft 답변을 참고 문서와 대조해 세 가지 관점으로 비판:

```
**문제점**
- (문서와 불일치하거나 부정확한 내용)

**누락**
- (문서에 있지만 답변에서 빠진 중요 정보)

**개선 방향**
- (구체적으로 어떻게 고쳐야 하는지)
```

---

### 4. `refine_answer()` — Critique 반영 개선 답변 (신규)

```python
def refine_answer(question, context_chunks, draft, critique, tracer=None) -> str:
```

- Draft + Critique를 입력으로 최종 답변 생성
- 기존 Step3 포맷 (**📌 요약** / **📖 근거** / **✅ 결론**) 유지
- Critique에서 지적된 문제점·누락 항목 **반드시 반영** 명시

---

### 5. `run_rag_pipeline()` — `use_self_refine` 파라미터 추가

```python
def run_rag_pipeline(..., use_self_refine: bool = False) -> dict:
```

#### Self-Refinement 실행 위치
```
[5] 답변 생성 (multidoc or simple)
    ↓
[6-NEW] Self-Refinement (use_self_refine=True 시)
    draft_answer = answer
    critique     = critique_answer(question, final_chunks, draft_answer)
    answer       = refine_answer(question, final_chunks, draft_answer, critique)
    ↓
[7] 평가 (Refined 답변 기준으로 수행)
```

반환 dict 추가 필드:
```python
{
    "draft_answer": draft_answer,   # [NEW v18] 초안 답변
    "critique":     critique,       # [NEW v18] 비판 내용 (None if OFF)
    ...
}
```

---

### 6. `build_log_entry()` — 신규 필드 추가

```python
def build_log_entry(...,
    self_refinement=None,
    dynamic_retrieval_profile=None):
```

#### 신규 로그 필드
```python
{
    "self_refinement": {
        "enabled":  True,
        "draft":    "<초안 답변 앞 500자>",
        "critique": "<비판 전문>",
    },
    "dynamic_retrieval_profile": "분석형 → MultiDoc ↑, Rerank 강화",
}
```

---

### 7. 챗봇 탭 UI 변화

#### 라우팅 status 확장
```
✅ 의도: reasoning | top_k: 4 | 프로필: 분석형 → MultiDoc ↑, Rerank 강화
```

#### 시도 완료 status에 Self-Refine 표시
```
✅ 시도 1 완료 ✏️ Refined | 정확도 4/5 · 환각 없음 · 등급 B
```

#### ✏️ Self-Refinement 내역 expander (신규, Self-Refine 적용 시만)
```
✏️ Self-Refinement 내역 (Draft → Critique → Refined)
  📝 Draft 답변
    [초안 전문]
  🔎 Critique (자기 비판)
    **문제점** / **누락** / **개선 방향**
  ✅ Refined 최종 답변 ← 위에 표시된 답변
```

#### 라우팅 expander — Dynamic Retrieval 프로필 표시
```
🎯 Dynamic Retrieval 프로필: 분석형 → MultiDoc ↑, Rerank 강화
```

---

### 8. 사이드바 변화

#### Self-Refinement 설정 섹션 (신규)
```
✏️ Self-Refinement 설정
├─ Self-Refinement 토글 (기본값: ON)
└─ 설명: Draft → Critique → Refine
```

#### Dynamic Retrieval 설정 섹션 (신규)
```
🎯 Dynamic Retrieval 설정
├─ 의도별 검색 전략 자동 조정 토글 (기본값: ON, 라우팅 ON 시만 활성)
├─  정의형 → BM25 ↑
├─  분석형/멀티홉 → MultiDoc ↑
└─  Fact형 → Rerank 강화
```

---

### 9. 에이전트 분석 탭 변화

#### ✏️ Self-Refinement 분석 섹션 (신규)
- Self-Refine 적용 건수 / 비율 / 미적용 건수
- Self-Refined 답변 vs 전체 평균 정확도 비교 bar chart

#### 🎯 Dynamic Retrieval 프로필 분포 섹션 (신규)
- 프로필별 적용 건수 bar chart
- "프로필 없음" = 라우팅 OFF 또는 Dynamic Retrieval OFF

#### per-log expander
- `🎯 Dynamic Retrieval: {profile_label}` 표시
- `✏️ Self-Refinement 내역` nested expander (Draft + Critique)
- col_b에 `Self-Refine: ✅` metric 추가

---

### 10. 트레이싱 탭 변화

#### 요약 지표 6번째 → `✏️ Self-Refine 적용 N회`

#### 트레이스 expander 제목에 `✏️ Refined` 태그 추가

#### 새로운 Span 2개 (Self-Refine ON 시)
| Span | 내용 |
|------|------|
| `critique` | Draft 비판 분석 LLM 호출 |
| `refine` | Critique 반영 개선 답변 LLM 호출 |

---

### 11. 검색 품질 탭 데이터 테이블

`Self-Refine` 컬럼 추가 — `✏️` 표시

---

## LLM 호출 횟수

| 상황 | v17 | v18 |
|------|-----|-----|
| Self-Refine OFF, Fallback 없음 | 7~8회 | 7~8회 (동일) |
| Self-Refine ON, Fallback 없음 | - | **9~10회** (+critique +refine) |
| Self-Refine ON, Fallback 1회 | - | **18~20회** |
| Self-Refine ON, Fallback 2회 | - | **27~30회** |

> Self-Refinement와 Fallback은 각각 ON/OFF 가능.
> Dynamic Retrieval은 LLM 추가 호출 없음 (순수 파라미터 조정).

---

## 파일 변경

| 항목 | v17 | v18 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v17.json` | `rag_eval_logs_v18.json` |
| EMBED_CACHE_FILE | `embed_cache_v17.pkl` | `embed_cache_v18.pkl` |

---

## 두 패턴의 핵심 원리

### Dynamic Retrieval — "조건 기반 라우팅"
```
질문 의도 분류
    ↓
의도별 검색 전략 프로필 선택
    ↓
BM25 비중 / Rerank 강도 / MultiDoc 여부 / 후보 수 자동 조정
    ↓
최적화된 파라미터로 검색 실행
```

### Self-Refinement — "Draft → Critique → Refine"
```
[1] Draft 생성   ← 기존 Step3
[2] Critique     ← LLM이 자신의 Draft를 비판
                    문제점 · 누락 · 개선 방향 도출
[3] Refine       ← Critique를 반영해 개선된 최종 답변 생성
[4] 평가         ← Refined 답변 기준으로 수행
```

이 두 패턴이 "진짜 LLM 활용 핵심":
- **Dynamic Retrieval**: 질문을 이해하고 그에 맞는 도구를 선택
- **Self-Refinement**: 자신의 출력을 비판하고 스스로 개선
