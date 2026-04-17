# rag_app_v8.py 변동사항 요약

## 핵심 변경: 코드 레벨 4가지 개선 (성능·비용·안정성)

v7의 기능은 그대로 유지하면서 실무 적용 수준의 설계 결함 4개를 수정.

---

## 최종 스택
| 구성 | v7 (rag_app_v7.py) | v8 (rag_app_v8.py) |
|------|---------------------|---------------------|
| 임베딩 | text-embedding-3-small | text-embedding-3-small (동일) |
| LLM | gpt-4o-mini | gpt-4o-mini (동일) |
| 벡터 DB | FAISS - 코사인 유사도 | FAISS - 코사인 유사도 (동일) |
| 쿼리 생성 | 단순 paraphrase | **의도 분해형 쿼리 우선** |
| 검색 후보 | top_k_per_query=5 | **top_k_per_query=20 → Pre-filter → Rerank** |
| Pre-filter | 없음 | **임베딩 유사도 top N 컷 (신규)** |
| chunk 출처 추적 | `dict(zip())` → 덮어쓰기 위험 | **`(chunk, source)` 튜플 쌍 유지** |
| Step 2 분석 | 공통점 / 차이점 / 핵심 정보 | **+ 불확실성 / 누락 정보 항목 추가** |

---

## 변경 사항

### 1. 쿼리 리라이팅 — 의도 분해형 우선 (`rewrite_queries`)
단순 paraphrase 대신 질문의 핵심 의도를 정의·원인·결과·방법·조건으로 분해한 쿼리 우선 생성.
```python
# 변경 전 (v7: 단순 표현 변형)
"주어진 질문과 의미는 같지만 다른 표현으로 n개 생성"

# 변경 후 (v8: 의도 분해 우선)
"1) 의도 분해형 쿼리 (정의·원인·결과·방법·조건으로 분리) 우선
 2) 표현 다양화 쿼리 (동의어·구조 변경) 보조"
# 예: '계약 해지 방법은?'
# → '계약 해지의 정의와 요건' / '해지 가능한 조건과 사유' / '해지 절차 및 방법'
```
**이유:** paraphrase는 같은 의미의 다른 표현이라 임베딩 공간에서 원본과 가깝게 위치해 실질적 검색 다양성이 낮음. 의도 분해는 질문의 서로 다른 측면을 독립 쿼리로 만들어 더 넓은 영역의 청크를 커버함.

---

### 2. Embedding Pre-filter 추가 (`prefilter_by_similarity`)
retrieve(20개) → 임베딩 유사도 상위 N개 컷 → rerank(N개) → top_k 로 3단계 깔때기 구조.
```python
# 변경 전 (v7: retrieve 소수 → 바로 rerank)
retrieve(top_k_per_query=5)  # 후보 적음
→ rerank_chunks()

# 변경 후 (v8: retrieve 많이 → 빠른 embedding 컷 → LLM rerank)
retrieve(top_k_per_query=20)                     # 후보 넓게
→ prefilter_by_similarity(query, items, top_n=10) # 임베딩으로 빠르게 절반 컷
→ rerank_chunks(query, filtered, top_k)          # LLM은 적은 수만 처리
```
**이유:** retrieve를 적게 하면 LLM rerank 비용은 낮지만 좋은 청크를 놓침(Recall 저하). retrieve를 많이 하고 바로 rerank하면 토큰이 폭발적으로 증가. Pre-filter는 추가 API 호출 없이(임베딩 재사용) 후보를 줄여 Recall과 비용 둘 다 해결.

---

### 3. chunk 출처 추적 — `(chunk, source)` 튜플로 통일
파이프라인 전체에서 청크와 출처를 튜플 쌍으로 묶어 전달. dict 방식 제거.
```python
# 변경 전 (v7: dict → 동일 텍스트 청크 덮어쓰기 위험)
chunk_to_source = dict(zip(candidate_chunks, candidate_sources))
# 문제: 두 파일에 같은 문장이 있으면 마지막 파일로 덮어씀

# 변경 후 (v8: 튜플 쌍 유지 → 인덱스 기반, 덮어쓰기 없음)
result_items = [(chunk, source), ...]          # retrieve 단계부터 튜플
filtered_items = prefilter_by_similarity(...)  # 튜플 그대로 전달
ranked_items = rerank_chunks(...)              # [(chunk, source, score), ...] 반환
```
**이유:** 계약서 A와 계약서 B에 동일한 조항 문장이 있을 경우 dict 방식은 어느 파일에서 왔는지 잘못 표시됨. 튜플은 순서가 보장되므로 어떤 경우에도 출처가 정확함.

---

### 4. Step 2 — 불확실성·누락 정보 항목 추가 (`step2_analyze_relationships`)
공통점·차이점에 "불확실성 / 누락 정보" 섹션 추가. Step3에서 확신도 결정에 활용.
```python
# 변경 전 (v7)
"**공통점** / **차이점** / **핵심 정보**"

# 변경 후 (v8)
"**공통점** / **차이점** / **핵심 정보** / **불확실성 / 누락 정보**"
# → Step3 system prompt에도 반영:
# "불확실성/누락 정보가 있으면 확신도에 반드시 반영할 것"
```
**이유:** 공통점·차이점만 분석하면 "문서에 없는 정보"를 LLM이 암묵적으로 채워 hallucination 발생. 누락 정보를 명시적으로 추출하면 Step3에서 확신도를 낮게 표시해 사용자가 과신하지 않도록 유도.

---

## 검색 파이프라인 전체 흐름 (v8)
```
질문 입력
  → [쿼리 리라이팅] 의도 분해 우선 + 표현 다양화 (1+n개)
  → [멀티쿼리 검색] top 20/쿼리 → (chunk, source) 튜플 수집
  → [Pre-filter]   임베딩 유사도로 상위 10개 컷 (API 추가 없음)
  → [LLM Rerank]   GPT 채점 → 상위 top_k 선택
  → [Step 1]       청크별 요약 (배치, LLM 1회)
  → [Step 2]       공통점·차이점·불확실성 분석 (LLM 1회)
  → [Step 3]       최종 구조화 답변 (LLM 1회)
```

## LLM 호출 횟수 비교
| 단계 | v7 | v8 |
|------|----|----|
| 쿼리 리라이팅 | 1회 | 1회 |
| Pre-filter | — | 0회 (임베딩만 사용) |
| Rerank | 1회 (후보 많음) | 1회 (후보 적음, 토큰 절감) |
| Step1 요약 | 1회 | 1회 |
| Step2 분석 | 1회 | 1회 |
| Step3 답변 | 1회 | 1회 |
| **합계** | **5회** | **5회 (rerank 토큰 대폭 감소)** |

## 성능 개선 기대 포인트
- 검색 Recall 향상: 의도 분해 쿼리로 질문의 여러 측면을 커버
- Rerank 비용 절감: Pre-filter로 LLM이 처리할 청크 수 절반 이하로 축소
- 출처 정확성 보장: 튜플 구조로 동일 텍스트 청크도 올바른 파일명 표시
- Hallucination 억제 강화: 누락 정보 명시 → 확신도 자동 하향 조정
