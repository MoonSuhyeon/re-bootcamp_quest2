# rag_app_v10.py 변동사항 요약

## 핵심 변경: Arize Phoenix 수준 Observability — Tracing + Agent Analysis

v9의 단순 점수 로깅에서 벗어나, Arize Phoenix가 제공하는 두 관점(시스템 성능 디버깅 / AI 의사결정 분석)을 코드 레벨로 완전 구현.

---

## 최종 스택
| 구성 | v9 (rag_app_v9.py) | v10 (rag_app_v10.py) |
|------|---------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 파이프라인 | 의도분해 + Pre-filter + Rerank | 동일 |
| 멀티문서 추론 | Step1→2→3 | 동일 |
| **Latency 추적** | 전체 합계만 | **단계별 span 측정 (Tracer 클래스)** |
| **Token 추적** | 없음 | **모든 LLM 호출의 prompt/completion/total 토큰** |
| **병목 감지** | 없음 | **가장 느린 span 자동 식별** |
| **의사결정 로그** | 없음 | **각 단계의 결정 근거(decision) 텍스트** |
| **환각 원인 분석** | "있음"만 기록 | **유형(fabrication/distortion/over-generalization) + 발생 원인 + 개선 제안** |
| **UI 탭 수** | 2개 (챗봇 / 평가) | **3개 (챗봇 / 트레이싱 / 에이전트 분석)** |

---

## 변경 사항

### 1. `Tracer` 클래스 — Span 기반 단계별 추적
Arize Phoenix의 span 개념을 직접 구현. 각 단계를 start/end로 감싸 latency와 토큰을 자동 집계.
```python
class Tracer:
    def start(self, name)        # 단계 시작 시각 기록
    def end(self, name,          # 단계 종료 → span 저장
            tokens,              # {"prompt": N, "completion": N, "total": N}
            input_summary,       # 이 단계에 무엇이 들어갔는가
            output_summary,      # 이 단계에서 무엇이 나왔는가
            decision)            # 왜 이 결정을 했는가 (Agent Analysis 핵심)
    def total_tokens()           # 전체 토큰 합계
    def bottleneck()             # 가장 오래 걸린 단계 이름
```
모든 LLM 호출 함수가 `tracer` 파라미터를 받아 자동으로 span을 기록.

---

### 2. 모든 LLM 함수 — `response.usage` 토큰 캡처
```python
def _usage(response) -> dict:
    return {"prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens}

# 적용 함수: rewrite_queries, rerank_chunks,
#            step1~3, evaluate_answer, analyze_hallucination_cause
```
**이유:** 어느 단계가 토큰을 가장 많이 쓰는지 파악해야 비용 최적화 포인트를 찾을 수 있음. Step1 요약 vs Step3 답변 중 어느 쪽이 더 비싼지 실제 수치로 확인 가능.

---

### 3. 환각 원인 심층 분석 (`analyze_hallucination_cause`) — 신규 함수
환각 감지 시에만 추가 LLM 호출 1회로 원인을 구조화 분석.
```python
# v9: 환각여부: "있음" 만 기록
# v10: 환각 감지 시 추가 분석
{
    "환각_주장":  "어떤 문장이 환각인가",
    "환각_유형":  "fabrication | distortion | over-generalization",
    "근거_출처":  "해당 정보가 있어야 할 출처",
    "발생_원인":  "insufficient_context | ambiguous_chunk | llm_interpolation",
    "개선_제안":  "이 환각을 막으려면 어떻게 해야 하는가"
}
```
| 유형 | 설명 |
|------|------|
| fabrication | 문서에 전혀 없는 내용을 만들어냄 |
| distortion | 문서 내용을 잘못 해석하거나 과장 |
| over-generalization | 특수 사례를 일반 규칙으로 확대 |

| 발생 원인 | 설명 |
|-----------|------|
| insufficient_context | 관련 청크가 검색되지 않아 LLM이 메꿈 |
| ambiguous_chunk | 청크 내용이 모호해 LLM이 잘못 해석 |
| llm_interpolation | 문맥상 자연스럽게 이어붙이는 과정에서 발생 |

---

### 4. 탭 3개 분리

#### 🔬 트레이싱 탭 (시스템 성능 디버깅)
```
[요약 지표]
총 트레이스 수 | 평균 응답 시간 | 평균 총 토큰 | 가장 빈번한 병목 단계

[차트]
- 응답 시간 추이 (line chart)
- 토큰 사용량 추이 (line chart)

[트레이스 상세 (드릴다운)]
- Waterfall: 각 단계 progress bar + ms 표시
- 단계별 토큰 bar chart
- API 호출 흐름 테이블 (단계/소요/토큰/입력/출력)
```

#### 🧠 에이전트 분석 탭 (AI 의사결정 추적)
```
[요약 지표]
평균 정확도 | 평균 관련성 | 환각 없음 비율 | 환각 감지 건수

[차트]
- 환각 여부 분포 bar chart
- 정확도/관련성 추이 line chart
- 환각 유형 분포 bar chart (fabrication/distortion/over-generalization)
- 발생 원인 분포 bar chart

[에이전트 의사결정 체인 (드릴다운)]
- 각 단계의 아이콘 + 결정 근거 텍스트 흐름
- 환각 원인 심층 분석 (유형/주장/출처/원인/개선제안)
- 평가 점수 + 토큰 + 병목 정보
```

---

## 로그 데이터 구조 변화
```python
# v9 로그 항목
{"timestamp", "question", "queries", "retrieved_chunks",
 "answer", "evaluation", "latency_ms", "mode"}

# v10 로그 항목 (추가된 필드)
{...,
 "trace_id": "a3f9b2c1",           # 트레이스 식별자
 "spans": [                         # 단계별 상세 기록
     {"name": "query_rewriting",
      "duration_ms": 450,
      "tokens": {"prompt": 120, "completion": 80, "total": 200},
      "input_summary": "원본 질문",
      "output_summary": "쿼리 4개 생성",
      "decision": "의도 분해형 3개 + 원본"},
     ...
 ],
 "total_tokens": {"prompt": 2450, "completion": 680, "total": 3130},
 "bottleneck": "step3_answer",      # 가장 느린 단계
 "hallucination_analysis": {        # 환각 시에만 존재
     "환각_유형": "fabrication",
     "발생_원인": "insufficient_context",
     "개선_제안": "관련 청크 top_k 확대 필요"
 }
}
```

---

## Arize Phoenix 대응 매핑
| Arize Phoenix 기능 | v10 구현 |
|--------------------|----------|
| Span tracing | `Tracer` 클래스의 start/end span |
| Latency waterfall | 트레이싱 탭 progress bar |
| Token usage | `_usage()` + 단계별 bar chart |
| Bottleneck detection | `tracer.bottleneck()` |
| LLM decision reasoning | span의 `decision` 필드 |
| Hallucination detection | `evaluate_answer()` |
| Hallucination root cause | `analyze_hallucination_cause()` |
| Evaluation scores | 정확도/관련성 metric |
| Export | CSV 다운로드 (trace_id 포함) |

## LLM 호출 횟수 비교
| 단계 | v9 | v10 |
|------|----|-----|
| 쿼리 리라이팅 | 1 | 1 |
| Rerank | 1 | 1 |
| Step1 요약 | 1 | 1 |
| Step2 분석 | 1 | 1 |
| Step3 답변 | 1 | 1 |
| 평가 | 1 | 1 |
| 환각 원인 분석 | — | +1 (환각 감지 시에만) |
| **합계** | **6회** | **6~7회** |

## 성능 개선 기대 포인트
- 병목 구간 식별로 최적화 우선순위 결정 가능 (느린 단계만 집중 개선)
- 토큰 비용 분석으로 가장 비싼 단계 파악 → 프롬프트 압축 포인트 발견
- 환각 유형·원인 데이터 누적 → 어느 청킹 방식/top_k에서 fabrication이 많은지 통계 분석 가능
- 의사결정 체인으로 "왜 이 답변이 나왔는가"를 단계별로 역추적 가능
