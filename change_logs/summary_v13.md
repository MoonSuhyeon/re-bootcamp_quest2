# rag_app_v13.py 변동사항 요약

## 핵심 변경: 쿼리 라우팅 엔진 — 질문 의도에 따라 검색 전략 자동 결정

v12의 기능을 그대로 유지하면서, 매 질문마다 LLM이 검색 파라미터를 동적으로 결정하는 라우팅 레이어 추가.

---

## 최종 스택
| 구성 | v12 (rag_app_v12.py) | v13 (rag_app_v13.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Dense + BM25 RRF | 동일 |
| 임베딩 캐시 | EmbeddingCache | 동일 |
| NDCG@k | 자동 계산 | 동일 |
| **검색 전략 결정** | 사이드바 수동 설정 | **쿼리 라우팅 엔진 자동 결정** |
| **의도 분류** | 없음 | **6개 유형 자동 분류** |
| **라우팅 로그** | 없음 | **route_decision 필드 저장** |
| **UI** | 3탭 | 3탭 (라우팅 expander 추가) |

---

## 변경 사항

### 1. `ROUTING_SYSTEM_PROMPT` — 라우팅 지시 프롬프트

질문을 6가지 의도로 분류하고 검색 전략 JSON을 반환하도록 지시:

| 의도 유형 | 설명 | 검색 전략 |
|-----------|------|-----------|
| `factual_lookup` | 정확한 사실·수치·날짜 | BM25 비중 높게, recall 강화 |
| `definition` | 짧고 명확한 정의 | 가벼운 검색 (top_k=2) |
| `multi_hop` | 여러 문서 연결 추론 | query 분해 필수, top_k 높게 |
| `reasoning` | 검색+재정렬+종합 | reranker heavy + dense 강화 |
| `exploratory` | 넓은 의미 탐색 | recall 최우선, 다양한 청크 |
| `ambiguous` | 의도 불분명 | recall 최우선 후 필터링 |

---

### 2. `route_query(question, tracer)` — 신규 함수

```python
def route_query(question: str, tracer: Tracer = None) -> dict:
    # LLM에게 ROUTING_SYSTEM_PROMPT 전달
    # response_format={"type": "json_object"} 로 JSON 강제
    # 파싱 실패 시 ambiguous fallback 반환
    # 반환 형식:
    {
        "의도": "reasoning",
        "검색_전략": {
            "dense_weight": 0.7, "bm25_weight": 0.3,
            "reranker_사용여부": true, "reranker_모드": "heavy",
            "top_k": 3, "query_rewrite_필요": true,
            "query_분해_필요": false, "recall_우선순위": false
        },
        "메타데이터_전략": {"메타데이터_필터_사용": false, "선호_출처": [], "시간_가중치": "없음"},
        "설명": "복잡한 추론이 필요하므로 reranker heavy 적용"
    }
```

---

### 3. `_apply_routing(route, defaults)` — 라우팅 결과 → 파이프라인 파라미터 변환

```python
# 라우팅 결과를 실제 파이프라인 파라미터로 변환
{
    "use_bm25":          bm25_weight > 0.3,
    "use_reranking":     reranker_사용여부,
    "top_k":             top_k (2~5 정수),
    "use_query_rewrite": query_rewrite_필요
}
```

---

### 4. 파이프라인 Step 0 추가 — 라우팅 실행

```
기존 (v12): 쿼리 리라이팅 → 검색 → prefilter → rerank → ...

v13:
[Step 0] 🧭 쿼리 라우팅 → 의도 분류 + 전략 결정
  ↓
[Step 1] 리라이팅 (라우팅이 query_rewrite_필요: true 이면 실행)
  ↓
[Step 2] Hybrid 검색 (라우팅이 bm25_weight > 0.3 이면 BM25 활성화)
  ↓
[Step 3] Pre-filter
  ↓
[Step 4] Rerank (라우팅이 reranker_사용여부: true 이면 실행)
  ↓
...
```

---

### 5. UI 변화

#### 사이드바
- **자동 쿼리 라우팅 토글** (기본값: ON) 추가
- ON 시 수동 설정 항목(쿼리 리라이팅, BM25, 리랭킹) 자동 비활성화
- OFF 시 v12와 동일한 수동 제어

#### 챗봇 탭
- **🧭 쿼리 라우팅 결정 expander** 추가:
  - 의도 분류 결과 + 설명
  - 검색 전략 JSON
  - 실제 적용된 파라미터
  - 메타데이터 전략

#### 트레이싱 탭
- **라우팅 의도 분포 bar chart** 추가
- 요약 지표에 **주요 의도** 항목 추가

#### 에이전트 분석 탭
- 각 로그 expander에 **라우팅 의도 + 설명** 표시
- 의사결정 체인에 `query_routing` span 추가 (🧭 아이콘)

---

## 로그 데이터 구조 변화

```python
# v13 추가 필드
{
    "route_decision": {
        "의도": "reasoning",
        "검색_전략": {...},
        "메타데이터_전략": {...},
        "설명": "..."
    }
}
```

---

## LLM 호출 횟수

| 단계 | v13 |
|------|-----|
| **쿼리 라우팅** | **+1 (신규)** |
| 쿼리 리라이팅 | 1 (캐시 히트 시 0) |
| Rerank | 0~1 (라우팅에 따라) |
| Step1 요약 | 1 |
| Step2 분석 | 1 |
| Step3 답변 | 1 |
| 평가 | 1 |
| 환각 원인 분석 | +1 (환각 시에만) |
| **합계** | **7~8회** |

## 성능 개선 기대 포인트
- factual_lookup 질문에서 BM25 비중 자동 증가 → 키워드 검색 정밀도 향상
- definition 질문에서 top_k=2, 리랭킹 OFF → 응답 속도 단축 + 토큰 절감
- multi_hop 질문에서 query 분해 자동 실행 → recall 향상
- 의도별 통계 누적 → 어떤 질문 유형이 많은지 데이터 수집 가능
