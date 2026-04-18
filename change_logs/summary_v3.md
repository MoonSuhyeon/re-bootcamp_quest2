# rag_app_v3.py 변동사항 요약

## 핵심 변경: 검색 품질 개선

v2의 성능 불량 원인 2가지를 정확히 진단하고 수정.

---

## 최종 스택
| 구성 | v2 (rag_app_v2.py) | v3 (rag_app_v3.py) |
|------|---------------------|---------------------|
| 임베딩 | text-embedding-3-small | text-embedding-3-small (동일) |
| LLM | gpt-4o-mini | gpt-4o-mini (동일) |
| 벡터 DB | FAISS - L2 거리 | FAISS - 코사인 유사도 |
| 청킹 | 글자 수 기준 (500자) | 문단/문장 단위 |
| UI | Streamlit | Streamlit (동일) |

---

## 변경 사항

### 1. 청킹 방식 개선: 글자 수 기준 → 문단/문장 단위
```python
# 변경 전 (글자 수 기준 - 문장 중간에서 잘림)
def chunk_text(text, chunk_size=500, overlap=100):
    while start < len(text):
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

# 변경 후 (문단 → 문장 순서로 분리)
def chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)  # 문단 단위 분리
    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?。])\s+', para)  # 문장 단위 분리
            ...
```
**문제:** 글자 수로 자르면 문장 중간에서 청크가 끊겨 문맥이 손상됨 → 검색해도 의미 있는 내용을 못 찾음.
**해결:** 문단 경계 우선, 문단이 너무 크면 문장 경계에서 분리. 문장이 절대 중간에 잘리지 않음.

---

### 2. 거리 계산 방식 교체: L2 거리 → 코사인 유사도
```python
# 변경 전 (L2 거리)
index = faiss.IndexFlatL2(dim)

# 변경 후 (코사인 유사도)
def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

index = faiss.IndexFlatIP(dim)  # Inner Product
# 인덱스 추가 시 정규화
index.add(normalize(embeddings))
# 검색 시에도 정규화
index.search(normalize(query_emb), top_k)
```
**문제:** OpenAI `text-embedding-3-small`은 코사인 유사도 기준으로 학습됨. L2 거리로 검색하면 실제로 의미가 가까운 문서를 못 찾을 수 있음.
**해결:** 벡터를 정규화한 후 내적(Inner Product)을 사용. 정규화된 벡터의 내적 = 코사인 유사도.

---

## 성능 개선 기대 포인트
- 청크 품질 향상 → 검색 시 의미 있는 문맥 단위가 올바르게 전달됨
- 검색 정확도 향상 → 질문과 실제로 관련 있는 청크를 더 잘 찾아냄
- 두 문제가 모두 Retrieval 단계 문제였으므로, LLM 교체 없이 검색 품질만 올려도 답변 품질이 크게 개선될 것으로 예상
