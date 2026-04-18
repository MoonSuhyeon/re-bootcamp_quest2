# rag_app_v4.py 변동사항 요약

## 핵심 변경: 검색 召回率(Recall) + 정확도(Precision) 동시 개선

v3의 단일 쿼리 검색 한계를 쿼리 리라이팅과 LLM 리랭킹으로 해결.

---

## 최종 스택
| 구성 | v3 (rag_app_v3.py) | v4 (rag_app_v4.py) |
|------|---------------------|---------------------|
| 임베딩 | text-embedding-3-small | text-embedding-3-small (동일) |
| LLM | gpt-4o-mini | gpt-4o-mini (동일) |
| 벡터 DB | FAISS - 코사인 유사도 | FAISS - 코사인 유사도 (동일) |
| 청킹 | 문단/문장 단위 | 문단/문장 단위 (동일) |
| 쿼리 처리 | 단일 쿼리 | 쿼리 리라이팅 (원본 + 변형 n개) |
| 검색 후처리 | 없음 | LLM 배치 리랭킹 |

---

## 변경 사항

### 1. 쿼리 리라이팅 (Multi-Query Retrieval)
원본 질문을 GPT가 다른 표현으로 n개 재작성 → 각각 검색 → 중복 제거 후 합산.
```python
# 변경 전 (단일 쿼리)
context_chunks = retrieve(prompt, index, chunks, top_k=3)

# 변경 후 (멀티 쿼리)
queries = rewrite_queries(original_query, n=3)
# 예시: ["계약 해지 조건은?", "계약을 끝낼 수 있는 경우", "해지 가능한 상황"]
candidate_chunks = retrieve_multi_query(queries, index, chunks, top_k_per_query=5)
```
**문제:** 사용자가 특정 표현으로만 질문하면, 문서가 다른 단어를 써도 임베딩 유사도가 낮아 관련 청크를 놓침.
**해결:** 동의어/구조 변경으로 재작성된 쿼리를 추가 투입해 더 넓게 검색(Recall ↑).

---

### 2. LLM 배치 리랭킹
후보 청크 전체를 단 1번의 API 호출로 0~10점 채점 → 점수 순 정렬 → 상위 top_k만 사용.
```python
def rerank_chunks(query, chunks, top_k=3):
    # 모든 청크를 하나의 프롬프트에 담아 일괄 채점
    chunks_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
    # GPT 출력 예시: "1: 8\n2: 2\n3: 7\n..."
    # → 점수 파싱 → 정렬 → 상위 top_k 반환
```
**문제:** 멀티 쿼리로 후보가 많아지면 LLM에 불필요한 청크까지 전달되어 답변 품질 저하.
**해결:** LLM이 직접 관련성 점수를 매겨 진짜 관련 청크만 압축 선발(Precision ↑).

---

### 3. UI 개선
- 쿼리 리라이팅 / 리랭킹 토글 스위치 (개별 ON/OFF)
- 생성된 변형 쿼리 목록 expander로 확인 가능
- 참고 문서에 관련성 점수(x/10) 표시
- `st.status`로 각 단계 진행 상황 실시간 표시

---

## 검색 파이프라인 변화
```
[v3] 질문 → 임베딩 검색 → top_k 청크 → LLM 답변

[v4] 질문
  → [1단계] 쿼리 리라이팅 (1+n개 쿼리 생성)
  → [2단계] 멀티 쿼리 임베딩 검색 (후보 청크 다수 수집)
  → [3단계] LLM 배치 리랭킹 (0~10점 채점 → 상위 top_k 선택)
  → [4단계] LLM 답변 생성
```

## 성능 개선 기대 포인트
- Recall 향상: 다양한 표현의 쿼리로 더 많은 관련 청크 수집
- Precision 향상: LLM 리랭킹으로 노이즈 청크 제거 후 핵심만 전달
- 두 방식의 조합으로 검색 품질이 단순 top_k 대비 크게 개선 기대
