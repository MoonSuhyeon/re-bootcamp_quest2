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
    page_title="RAG 챗봇 v4",
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


# --- 문단/문장 단위 청킹 ---
def chunk_text(text, chunk_size=500):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
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
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# --- [NEW] 쿼리 리라이팅: 원본 쿼리 + n개 변형 생성 ---
def rewrite_queries(original_query, n=3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 문서 검색 성능을 높이는 쿼리 전문가입니다. "
                    f"주어진 질문과 의미는 같지만 다른 표현으로 {n}개의 검색 쿼리를 생성하세요. "
                    "동의어 활용, 문장 구조 변경, 핵심 키워드 강조 등의 방법을 사용하세요. "
                    "각 쿼리를 새 줄에 작성하고, 번호·기호 없이 쿼리 텍스트만 출력하세요."
                )
            },
            {
                "role": "user",
                "content": f"원본 질문: {original_query}"
            }
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    # 원본 쿼리를 첫 번째로, 변형 쿼리를 최대 n개 추가
    return [original_query] + variants[:n]


# --- [NEW] 멀티 쿼리 검색: 여러 쿼리로 검색 후 중복 제거 ---
def retrieve_multi_query(queries, index, chunks, top_k_per_query=5):
    seen_indices = set()
    result_chunks = []

    for query in queries:
        query_emb = normalize(get_embeddings([query]))
        _, indices = index.search(query_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen_indices:
                seen_indices.add(i)
                result_chunks.append(chunks[i])

    return result_chunks


# --- [NEW] LLM 배치 리랭킹: 한 번의 API 호출로 전체 청크 점수화 ---
def rerank_chunks(query, chunks, top_k=3):
    if not chunks:
        return []

    chunks_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문에 대해 각 문서 청크의 관련성을 0~10점으로 평가하세요.\n"
                    "반드시 아래 형식으로만 출력하세요 (다른 텍스트 금지):\n"
                    "1: 8\n"
                    "2: 3\n"
                    "3: 7\n"
                    "...\n"
                    "10점: 질문에 직접적인 답변 포함, 0점: 전혀 무관"
                )
            },
            {
                "role": "user",
                "content": f"질문: {query}\n\n문서 청크들:\n{chunks_text}"
            }
        ]
    )

    scores = {}
    for line in response.choices[0].message.content.strip().split('\n'):
        parts = line.split(':')
        if len(parts) == 2:
            try:
                idx = int(parts[0].strip()) - 1
                score = float(parts[1].strip())
                if 0 <= idx < len(chunks):
                    scores[idx] = score
            except ValueError:
                pass

    scored_chunks = [(chunks[i], scores.get(i, 0.0)) for i in range(len(chunks))]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]


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
    st.title("📚 RAG 챗봇 v4")
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
    st.subheader("검색 설정")

    use_query_rewrite = st.toggle("쿼리 리라이팅 사용", value=True, help="원본 쿼리를 여러 방식으로 재작성해 검색 召回율(recall)을 높입니다")
    num_rewrites = st.slider("리라이팅 쿼리 수", min_value=1, max_value=5, value=3, disabled=not use_query_rewrite)

    use_reranking = st.toggle("리랭킹 사용", value=True, help="LLM으로 검색된 청크를 재평가해 정확도(precision)를 높입니다")
    top_k = st.slider("최종 사용 청크 수 (top_k)", min_value=1, max_value=5, value=3)

    if use_query_rewrite:
        st.caption(f"검색: 쿼리 {num_rewrites+1}개 × 5개 청크 → 리랭킹 → 상위 {top_k}개")
    else:
        st.caption(f"검색: 단일 쿼리 → 상위 {top_k}개")

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
    st.caption("Powered by OpenAI + FAISS | v4: Query Rewriting + Reranking")


# --- 메인 화면 ---
st.title("💬 문서 기반 챗봇 v4")
st.caption("쿼리 리라이팅 + LLM 리랭킹으로 검색 품질 향상")

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
                # 1단계: 쿼리 리라이팅
                if use_query_rewrite:
                    with st.status("쿼리 리라이팅 중...", expanded=False) as status:
                        queries = rewrite_queries(prompt, n=num_rewrites)
                        status.update(label=f"쿼리 {len(queries)}개 생성 완료", state="complete")
                else:
                    queries = [prompt]

                # 2단계: 멀티 쿼리 검색 (리라이팅 ON) 또는 단일 쿼리 검색
                if use_query_rewrite:
                    candidate_chunks = retrieve_multi_query(
                        queries, st.session_state.index, st.session_state.chunks,
                        top_k_per_query=5
                    )
                else:
                    query_emb = normalize(get_embeddings([prompt]))
                    _, indices = st.session_state.index.search(query_emb, top_k)
                    candidate_chunks = [st.session_state.chunks[i] for i in indices[0] if i < len(st.session_state.chunks)]

                # 3단계: 리랭킹
                if use_reranking and len(candidate_chunks) > top_k:
                    with st.status("리랭킹 중...", expanded=False) as status:
                        scored_chunks = rerank_chunks(prompt, candidate_chunks, top_k=top_k)
                        context_chunks = [chunk for chunk, _ in scored_chunks]
                        status.update(label=f"리랭킹 완료 → 상위 {top_k}개 선택", state="complete")
                else:
                    scored_chunks = [(chunk, None) for chunk in candidate_chunks[:top_k]]
                    context_chunks = candidate_chunks[:top_k]

                # 4단계: 답변 생성
                response = generate_answer(prompt, context_chunks)
                st.write(response)

                # --- 검색 과정 상세 보기 ---
                if use_query_rewrite:
                    with st.expander("🔍 생성된 쿼리 보기"):
                        for i, q in enumerate(queries):
                            label = "**[원본]**" if i == 0 else f"**[변형 {i}]**"
                            st.markdown(f"{label} {q}")

                with st.expander("📄 참고한 문서 내용 보기"):
                    for i, (chunk, score) in enumerate(scored_chunks):
                        score_badge = f" _(관련성 점수: {score:.1f}/10)_" if score is not None else ""
                        st.caption(f"[{i+1}]{score_badge}")
                        st.write(chunk)
                        if i < len(scored_chunks) - 1:
                            st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
