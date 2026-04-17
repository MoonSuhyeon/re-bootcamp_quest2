# rag_app_v2.py 변동사항 요약

## 핵심 변경: 로컬 → OpenAI API

v1(app.py)의 로컬 실행 방식을 OpenAI API 기반으로 전환.

---

## 최종 스택
| 구성 | v1 (app.py) | v2 (rag_app_v2.py) |
|------|-------------|---------------------|
| 임베딩 | BAAI/bge-m3 (HuggingFace 로컬) | text-embedding-3-small (OpenAI API) |
| LLM | llama3.2 (Ollama 로컬) | gpt-4o-mini (OpenAI API) |
| 벡터 DB | FAISS (L2 거리) | FAISS (L2 거리) |
| UI | Streamlit | Streamlit |
| PDF 파싱 | pdfplumber | pdfplumber |

---

## 변경 사항

### 1. 임베딩 모델 교체: HuggingFace → OpenAI
로컬 모델 로딩 없이 API 호출로 임베딩 생성.
```python
# 변경 전 (로컬 모델 로딩)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-m3")
embeddings = embedder.encode(chunks)

# 변경 후 (API 호출)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
```
**이유:** OpenAI API 키 결제 완료. 로컬 모델 대비 RAM 사용 없음, 세팅 단순화.

---

### 2. LLM 교체: Ollama llama3.2 → gpt-4o-mini
```python
# 변경 전
response = ollama.chat(model="llama3.2", messages=[...])

# 변경 후
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
)
```
**이유:** llama3.2는 성능 한계로 답변 품질 불량. gpt-4o-mini는 저비용(질문 1회 ~$0.001)으로 성능 대폭 향상 기대.

---

### 3. API 키 관리: .env + python-dotenv
```python
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

---

### 4. 시스템 프롬프트 한국어화
```python
# 변경 전 (영어 프롬프트)
"You must answer ONLY using the provided context."

# 변경 후 (한국어 프롬프트)
"주어진 컨텍스트만 사용해서 답변하세요."
```

---

## 남은 한계 (v2 기준)
- 청킹이 여전히 글자 수(500자) 기준이라 문장 중간에서 잘릴 수 있음
- FAISS가 L2 거리를 사용하는데, OpenAI 임베딩은 코사인 유사도에 최적화되어 있어 검색 품질 저하 가능
