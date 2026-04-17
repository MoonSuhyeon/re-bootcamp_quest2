# RAG 시스템 변동 사항 요약

## 최종 스택
| 구성 | 도구 |
|------|------|
| UI | Streamlit |
| 임베딩 | BAAI/bge-m3 (HuggingFace) |
| 벡터 DB | FAISS |
| LLM | Ollama llama3.2 (로컬) |
| PDF 파싱 | pdfplumber |

---

## 변동 사항 히스토리

### 1. 초기 UI 구성
Streamlit으로 껍데기 페이지 생성. RAG 연결 없이 채팅 UI만 구성.
```python
# 채팅 UI 기본 구조
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = f"(RAG 미연결) '{prompt}'에 대한 답변입니다."
```

---

### 2. HuggingFace RAG 연결 (초기)
임베딩 + flan-t5 생성 모델 + FAISS 연결.
```python
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")
```
**문제:** 새 버전 transformers에서 `text2text-generation` task 이름 오류 발생.

---

### 3. PDF 파서 교체: PyPDF2 → pdfplumber
한글 PDF 추출 시 깨짐 현상 발생.
```python
# 변경 전 (한글 깨짐)
reader = PyPDF2.PdfReader(io.BytesIO(file.read()))

# 변경 후 (한글 정상 추출)
with pdfplumber.open(io.BytesIO(file.read())) as pdf:
    for page in pdf.pages:
        extracted = page.extract_text()
```

---

### 4. 임베딩 모델 교체: paraphrase-multilingual → bge-m3
한국어 성능이 더 뛰어난 다국어 임베딩 모델로 교체.
```python
# 변경 전
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 변경 후 (한국어 포함 100개 언어 지원, 고성능)
embedder = SentenceTransformer("BAAI/bge-m3")
```

---

### 5. 청킹 파라미터 조정
```python
# 변경 전
def chunk_text(text, chunk_size=300, overlap=50):

# 변경 후 (더 많은 문맥 포함)
def chunk_text(text, chunk_size=500, overlap=100):
```

---

### 6. top_k 슬라이더 추가
사이드바에서 검색 청크 수를 3~5 사이로 동적 조절 가능하게 변경.
```python
top_k = st.slider("검색 청크 수 (top_k)", min_value=3, max_value=5, value=3)
```

---

### 7. 시스템 프롬프트 추가
hallucination 방지를 위한 RAG 필수 프롬프트 적용.
```python
prompt = (
    "You must answer ONLY using the provided context.\n"
    "If the answer is not in the context, say \"모르겠습니다\".\n"
    "Always answer in Korean. Do not use Chinese characters.\n\n"
    "[Context]\n"
    f"{context}\n\n"
    "[Question]\n"
    f"{question}"
)
```

---

### 8. LLM 교체: flan-t5 → Ollama llama3.2
transformers 버전 오류 및 영어 전용 한계로 Ollama 로컬 LLM으로 전환.
```python
# 변경 전 (영어 전용, 버전 오류)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 변경 후 (한국어 지원, 로컬 실행)
import ollama
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)
```
**참고:** llama3은 RAM 4.6GB 필요로 메모리 부족 → llama3.2(2GB)로 다운그레이드.

---

## 현재 최종 코드 구조
```
app.py
├── load_embedder()       # bge-m3 임베딩 모델 로딩 (캐시)
├── extract_text()        # PDF/TXT 텍스트 추출 (pdfplumber)
├── chunk_text()          # 텍스트 청킹 (size=500, overlap=100)
├── build_index()         # FAISS 인덱스 생성
├── retrieve()            # 유사 청크 검색 (top_k)
├── generate_answer()     # Ollama llama3.2로 답변 생성
└── Streamlit UI          # 사이드바 + 채팅 인터페이스
```

## 남은 한계
- llama3.2가 작은 모델이라 가끔 한자 혼용 현상 발생 (프롬프트로 억제 중)
- RAM 여유 생기면 llama3.1:8b 등 더 큰 모델로 교체 권장
