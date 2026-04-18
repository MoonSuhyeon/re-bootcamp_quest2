# v25 변동사항 요약

## v24 → v25 핵심 변경점

### 1. Multi-Hop Reasoning (Agentic RAG) [NEW]

복합 질문("A와 B의 차이점", "원인과 결과", "비교 분석")을 엔진이 스스로 계획하고 단계별로 검색합니다.

**흐름:**
```
질문 → needs_multihop() 판단 → decompose() 분해
  → hop1: "A 찾기" 검색 + 중간답변
  → hop2: "B 찾기" 검색 + 중간답변
  → hop3: "비교하기" 검색 + 중간답변
  → synthesize() 최종 종합 답변
```

**새 클래스:** `MultiHopPlanner`
| 메서드 | 역할 |
|--------|------|
| `needs_multihop(question)` | LLM이 복합 질문 여부 판단 (키워드 + LLM 2단계) |
| `decompose(question)` | sub-query 목록 생성 (최대 `MULTIHOP_MAX_HOPS`개) |
| `synthesize(question, hop_results)` | 각 hop 결과 종합 → 최종 답변 |

**새 함수:** `run_multihop_pipeline()` — hop별 검색+중간답변 실행기

---

### 2. Self-RAG (Self-Correction Retrieval) [NEW]

검색 직후 LLM이 "이 결과가 질문에 답하기 충분한가?"를 판단하고, 부족하면 쿼리를 재작성해 재검색합니다.

**흐름:**
```
검색 결과 →  check_sufficiency() 판단
  → 충분: 그대로 진행
  → 부족: rewrite_hint로 쿼리 재작성 → 재검색 → 결과 병합 (최대 SELF_RAG_MAX_ITER회)
```

**새 클래스:** `SelfRAGChecker`
| 메서드 | 역할 |
|--------|------|
| `check_sufficiency(question, chunks)` | `{sufficient, reason, rewrite_hint}` JSON 반환 |

---

### 3. config.py 신규 상수

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `MULTIHOP_MAX_HOPS` | 4 | hop 최대 수 |
| `SELF_RAG_MAX_ITER` | 3 | Self-RAG 재검색 최대 횟수 |

---

### 4. API 변경

`POST /chat` 요청 바디에 2개 필드 추가:

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `use_multihop` | `false` | Multi-Hop 파이프라인 활성화 |
| `use_self_rag` | `false` | Self-RAG 재검색 루프 활성화 |

---

## 파일 변경 요약

| 파일 | 변경 내용 |
|------|-----------|
| `rag_engine.py` | `MultiHopPlanner`, `SelfRAGChecker` 클래스 추가; `run_multihop_pipeline()` 추가; `run_rag_pipeline()`, `process_rag_query()` 파라미터 확장 |
| `config.py` | `MULTIHOP_MAX_HOPS`, `SELF_RAG_MAX_ITER` 추가 |
| `routers/chat.py` | `ChatRequest`에 `use_multihop`, `use_self_rag` 필드 추가 |
| `server_api.py` | 버전 25.0.0으로 업데이트 |

## 포트폴리오 관점

- **Multi-Hop**: 단순 검색을 넘어 "계획 → 실행 → 종합"하는 Agentic 패턴 구현
- **Self-RAG**: 2023년 Self-RAG 논문의 핵심 아이디어(검색 자기 평가) 적용
- 두 기능 모두 기본값 `false` → 기존 파이프라인과 완전 하위 호환
