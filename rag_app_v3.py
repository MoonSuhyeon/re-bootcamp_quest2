import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pdfplumber
import io
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide"
)


# --- 임베딩 생성 ---
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data], dtype=np.float32)


# --- 코사인 유사도를 위한 정규화 ---
def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


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


# --- 문단/문장 단위 청킹 (문장 중간에서 자르지 않음) ---
def chunk_text(text, chunk_size=500):
    # 문단 단위로 먼저 분리
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # 문단이 chunk_size 초과면 문장 단위로 분리
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            sentences = re.split(r'(?<=[.!?。])\s+', para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) > chunk_size and temp:
                    chunks.append(temp.strip())
                    temp = sent
                else:
                    temp = temp + " " + sent if temp else sent
            if temp:
                chunks.append(temp.strip())
        else:
            # chunk_size 안에 들어오면 문단을 합침
            if len(current) + len(para) > chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# --- FAISS 인덱스 생성 (코사인 유사도) ---
def build_index(chunks):
    embeddings = normalize(get_embeddings(chunks))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = 정규화된 벡터의 코사인 유사도
    index.add(embeddings)
    return index


# --- 관련 청크 검색 ---
def retrieve(query, index, chunks, top_k=3):
    query_emb = normalize(get_embeddings([query]))
    _, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# --- 답변 생성 ---
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다. "
                    "반드시 제공된 컨텍스트만 사용해서 답변하세요. "
                    "컨텍스트에 없는 내용은 '해당 문서에서 관련 내용을 찾을 수 없습니다.'라고 답하세요. "
                    "항상 한국어로 답변하세요."
                )
            },
            {
                "role": "user",
                "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"
            }
        ]
    )
    return response.choices[0].message.content


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
            all_chunks = []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    chunks = chunk_text(text)
                    all_chunks.extend(chunks)
                st.session_state.chunks = all_chunks
                st.session_state.index = build_index(all_chunks)
            st.success(f"완료! {len(all_chunks)}개 청크 생성됨")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크 인덱싱됨")

    st.markdown("---")
    top_k = st.slider("검색 청크 수 (top_k)", min_value=1, max_value=5, value=3)

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
    st.caption("Powered by OpenAI + FAISS")


# --- 메인 화면 ---
st.title("💬 문서 기반 챗봇")

if st.session_state.index is None:
    st.info("왼쪽 사이드바에서 문서를 업로드하고 '문서 처리하기'를 눌러주세요.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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
                context_chunks = retrieve(prompt, st.session_state.index, st.session_state.chunks, top_k=top_k)
                response = generate_answer(prompt, context_chunks)
                st.write(response)
                with st.expander("참고한 문서 내용 보기"):
                    for i, chunk in enumerate(context_chunks):
                        st.caption(f"[{i+1}] {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": response})
