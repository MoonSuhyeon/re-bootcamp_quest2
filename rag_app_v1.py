import streamlit as st
from sentence_transformers import SentenceTransformer
import ollama
import faiss
import numpy as np
import pdfplumber
import io

st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# --- 모델 로딩 (최초 1회만) ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")


# --- 텍스트 추출 ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

# --- 텍스트 청크 분할 ---
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

# --- FAISS 인덱스 생성 ---
def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index

# --- 관련 청크 검색 ---
def retrieve(query, index, chunks, embedder, top_k=3):
    query_emb = embedder.encode([query]).astype(np.float32)
    _, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# --- 답변 생성 ---
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You must answer ONLY using the provided context.\n"
        "If the answer is not in the context, say \"모르겠습니다\".\n"
        "Always answer in Korean. Do not use Chinese characters.\n\n"
        "[Context]\n"
        f"{context}\n\n"
        "[Question]\n"
        f"{question}"
    )
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# --- 세션 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []


# --- 사이드바 ---
with st.sidebar:
    st.title("📚 RAG 챗봇")
    st.markdown("---")

    st.subheader("문서 업로드")
    uploaded_files = st.file_uploader(
        "파일을 업로드하세요",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF 또는 TXT 파일을 업로드하세요"
    )

    if uploaded_files:
        if st.button("문서 처리하기", use_container_width=True, type="primary"):
            embedder = load_embedder()
            all_chunks = []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    chunks = chunk_text(text)
                    all_chunks.extend(chunks)
                st.session_state.chunks = all_chunks
                st.session_state.index = build_index(all_chunks, embedder)
            st.success(f"완료! {len(all_chunks)}개 청크 생성됨")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크 인덱싱됨")

    st.markdown("---")
    top_k = st.slider("검색 청크 수 (top_k)", min_value=3, max_value=5, value=3)

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("문서 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.index = None
        st.session_state.chunks = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by HuggingFace + FAISS")


# --- 메인 화면 ---
st.title("💬 문서 기반 챗봇")

if st.session_state.index is None:
    st.info("왼쪽 사이드바에서 문서를 업로드하고 '문서 처리하기'를 눌러주세요.")

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 사용자 입력
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if st.session_state.index is None:
            response = "먼저 문서를 업로드하고 처리해주세요."
            st.write(response)
        else:
            with st.spinner("답변 생성 중..."):
                embedder = load_embedder()
                context_chunks = retrieve(prompt, st.session_state.index, st.session_state.chunks, embedder, top_k=top_k)
                response = generate_answer(prompt, context_chunks)
                st.write(response)
                with st.expander("참고한 문서 내용 보기"):
                    for i, chunk in enumerate(context_chunks):
                        st.caption(f"[{i+1}] {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": response})
