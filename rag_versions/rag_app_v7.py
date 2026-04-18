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
    page_title="RAG 챗봇 v7",
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


# --- 문단/문장 기반 청킹 + overlap ---
def chunk_text_with_overlap(text, chunk_size=500, overlap=100):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    raw_chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                raw_chunks.append(current.strip())
                current = ""
            sentences = re.split(r'(?<=[.!?。])\s+', para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) > chunk_size and temp:
                    raw_chunks.append(temp.strip())
                    temp = sent
                else:
                    temp = temp + " " + sent if temp else sent
            if temp:
                raw_chunks.append(temp.strip())
        else:
            if len(current) + len(para) > chunk_size and current:
                raw_chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

    if current:
        raw_chunks.append(current.strip())
    raw_chunks = [c for c in raw_chunks if c]

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + " " + raw_chunks[i])
    return overlapped


# --- 의미 기반(Semantic) 청킹 + overlap ---
def chunk_text_semantic(text, chunk_size=500, overlap=100):
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 3:
        return chunk_text_with_overlap(text, chunk_size, overlap)

    embeddings = normalize(get_embeddings(sentences))
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    threshold = mean_sim - 0.5 * std_sim

    breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
    breakpoints = [0] + breakpoints + [len(sentences)]

    raw_chunks = []
    for i in range(len(breakpoints) - 1):
        group = sentences[breakpoints[i]:breakpoints[i + 1]]
        current = ""
        for sent in group:
            if len(current) + len(sent) > chunk_size and current:
                raw_chunks.append(current.strip())
                current = sent
            else:
                current = current + " " + sent if current else sent
        if current:
            raw_chunks.append(current.strip())

    raw_chunks = [c for c in raw_chunks if c]

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + " " + raw_chunks[i])
    return overlapped


# --- FAISS 인덱스 생성 ---
def build_index(chunks):
    embeddings = normalize(get_embeddings(chunks))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# --- 쿼리 리라이팅 ---
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
            {"role": "user", "content": f"원본 질문: {original_query}"}
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    return [original_query] + variants[:n]


# --- 멀티 쿼리 검색 ---
def retrieve_multi_query(queries, index, chunks, top_k_per_query=5):
    seen_indices = set()
    result_indices = []
    for query in queries:
        query_emb = normalize(get_embeddings([query]))
        _, indices = index.search(query_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen_indices:
                seen_indices.add(i)
                result_indices.append(i)
    return result_indices


# --- LLM 배치 리랭킹 ---
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
                    "1: 8\n2: 3\n3: 7\n"
                    "10점: 질문에 직접적인 답변 포함, 0점: 전혀 무관"
                )
            },
            {"role": "user", "content": f"질문: {query}\n\n문서 청크들:\n{chunks_text}"}
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

    scored = [(chunks[i], scores.get(i, 0.0)) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =====================================================================
# [NEW] 멀티문서 Reasoning — 3단계 파이프라인
# =====================================================================

def step1_summarize_chunks(question, chunks):
    """Step 1: 질문 관점에서 각 청크 핵심 요약 (배치 처리)"""
    chunks_text = "\n\n".join([f"[{i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문과 관련하여 각 청크의 핵심 내용을 2~3문장으로 요약하세요.\n"
                    "질문과 무관한 내용은 제외하고, 관련 정보만 간결히 추출하세요.\n"
                    "반드시 아래 형식으로만 출력하세요 (다른 텍스트 금지):\n"
                    "[1]: 요약 내용\n"
                    "[2]: 요약 내용\n"
                    "..."
                )
            },
            {"role": "user", "content": f"질문: {question}\n\n청크들:\n{chunks_text}"}
        ]
    )

    summaries = {}
    for line in response.choices[0].message.content.strip().split('\n'):
        m = re.match(r'\[(\d+)\]:\s*(.+)', line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(chunks):
                summaries[idx] = m.group(2).strip()

    # 파싱 실패한 청크는 원문 앞부분으로 대체
    return [summaries.get(i, chunks[i][:120] + "...") for i in range(len(chunks))]


def step2_analyze_relationships(question, summaries, sources):
    """Step 2: 청크 요약들 간 공통점/차이점 추출"""
    summaries_text = "\n".join(
        [f"[출처 {i+1} | {sources[i]}] {s}" for i, s in enumerate(summaries)]
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "여러 문서 청크의 요약을 분석하여 질문에 답하기 위한 공통점과 차이점을 추출하세요.\n"
                    "반드시 아래 형식으로 출력하세요:\n\n"
                    "**공통점**\n"
                    "- (여러 출처에서 일치하는 내용)\n\n"
                    "**차이점**\n"
                    "- (출처 간 상충하거나 다른 내용, 없으면 '없음')\n\n"
                    "**핵심 정보**\n"
                    "- (질문 답변에 가장 중요한 정보 요약)"
                )
            },
            {"role": "user", "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
        ]
    )
    return response.choices[0].message.content


def step3_generate_final_answer(question, chunks, summaries, analysis):
    """Step 3: 요약·분석 결과를 활용해 최종 구조화 답변 생성"""
    numbered_context = "\n\n".join(
        [f"[출처 {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )
    summaries_text = "\n".join([f"[출처 {i+1}] {s}" for i, s in enumerate(summaries)])

    system_prompt = """당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다.
[분석 결과]를 참고해 [참고 문서]에서 근거를 찾아 답변하세요.

[필수 답변 규칙]
1. 반드시 아래 3단계 구조로 답변하세요:

   **📌 요약**
   질문에 대한 핵심 답변을 1~2문장으로 간결하게 작성.

   **📖 근거**
   각 주장마다 반드시 [출처 N] 형태로 청크 번호를 인용.
   여러 출처 동시 인용 가능: [출처 1][출처 3]

   **✅ 결론**
   최종 판단을 작성하고 마지막 줄에 확신도 명시.
   형식: 확신도: 높음 / 보통 / 낮음 — (이유 한 줄)

2. 확신도 기준:
   - 높음: 직접적인 근거가 명확히 있음
   - 보통: 관련 내용은 있으나 추론 필요
   - 낮음: 근거가 매우 부족하거나 간접적

3. 컨텍스트에 없는 내용은 절대 지어내지 마세요.
4. 항상 한국어로 답변하세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"[청크 요약]\n{summaries_text}\n\n"
                    f"[공통점·차이점 분석]\n{analysis}\n\n"
                    f"[참고 문서 원문]\n{numbered_context}\n\n"
                    f"[질문]\n{question}"
                )
            }
        ]
    )
    return response.choices[0].message.content


# v6 단일 호출 답변 (멀티문서 추론 OFF 시 사용)
def generate_answer_simple(question, context_chunks):
    numbered_context = "\n\n".join(
        [f"[출처 {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    system_prompt = """당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다.

[필수 답변 규칙]
1. 반드시 아래 3단계 구조로 답변하세요:
   **📌 요약** / **📖 근거** ([출처 N] 인용 필수) / **✅ 결론** (확신도: 높음/보통/낮음)
2. 컨텍스트에 없는 내용은 절대 지어내지 마세요.
3. 항상 한국어로 답변하세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[참고 문서]\n{numbered_context}\n\n[질문]\n{question}"}
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
if "chunk_sources" not in st.session_state:
    st.session_state.chunk_sources = []  # 청크별 출처 파일명 (parallel list)


# --- 사이드바 ---
with st.sidebar:
    st.title("📚 RAG 챗봇 v7")
    st.markdown("---")

    st.subheader("문서 업로드")
    uploaded_files = st.file_uploader(
        "파일을 업로드하세요",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    st.subheader("청킹 설정")
    chunking_mode = st.radio(
        "청킹 방식",
        ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"],
        help="의미 기반은 문장 임베딩 유사도로 자연스러운 주제 경계를 찾습니다"
    )
    chunk_size = st.slider("청크 크기", min_value=200, max_value=1000, value=500, step=100)
    overlap = st.slider("Overlap 크기", min_value=0, max_value=200, value=100, step=20)

    if uploaded_files:
        if st.button("문서 처리하기", use_container_width=True, type="primary"):
            all_chunks = []
            all_sources = []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    if chunking_mode == "의미 기반(Semantic) + Overlap":
                        file_chunks = chunk_text_semantic(text, chunk_size=chunk_size, overlap=overlap)
                    else:
                        file_chunks = chunk_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(file_chunks)
                    all_sources.extend([f.name] * len(file_chunks))  # 파일명 추적

                st.session_state.chunks = all_chunks
                st.session_state.chunk_sources = all_sources
                st.session_state.index = build_index(all_chunks)
            st.success(f"완료! {len(all_chunks)}개 청크 생성됨")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크 인덱싱됨")
        # 파일별 청크 수 표시
        if st.session_state.chunk_sources:
            from collections import Counter
            source_counts = Counter(st.session_state.chunk_sources)
            for fname, cnt in source_counts.items():
                st.caption(f"  └ {fname}: {cnt}개")

    st.markdown("---")
    st.subheader("검색 설정")
    use_query_rewrite = st.toggle("쿼리 리라이팅 사용", value=True)
    num_rewrites = st.slider("리라이팅 쿼리 수", min_value=1, max_value=5, value=3, disabled=not use_query_rewrite)
    use_reranking = st.toggle("리랭킹 사용", value=True)
    top_k = st.slider("최종 사용 청크 수 (top_k)", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.subheader("추론 설정")
    use_multidoc_reasoning = st.toggle(
        "멀티문서 추론 사용",
        value=True,
        help="Step1: 청크 요약 → Step2: 공통점/차이점 분석 → Step3: 최종 답변"
    )

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("문서 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.chunk_sources = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by OpenAI + FAISS | v7: Multi-Doc Reasoning")


# --- 메인 화면 ---
st.title("💬 문서 기반 챗봇 v7")
st.caption("멀티문서 Reasoning: 청크 요약 → 공통점/차이점 분석 → 최종 답변")

if st.session_state.index is None:
    st.info("왼쪽 사이드바에서 문서를 업로드하고 '문서 처리하기'를 눌러주세요.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.index is None:
            response = "먼저 문서를 업로드하고 처리해주세요."
            st.markdown(response)
        else:
            with st.spinner("답변 생성 중..."):
                # ── 1단계: 쿼리 리라이팅 ──────────────────────────────
                if use_query_rewrite:
                    with st.status("🔄 쿼리 리라이팅 중...", expanded=False) as s:
                        queries = rewrite_queries(prompt, n=num_rewrites)
                        s.update(label=f"✅ 쿼리 {len(queries)}개 생성 완료", state="complete")
                else:
                    queries = [prompt]

                # ── 2단계: 검색 ───────────────────────────────────────
                if use_query_rewrite:
                    candidate_indices = retrieve_multi_query(
                        queries, st.session_state.index, st.session_state.chunks, top_k_per_query=5
                    )
                    candidate_chunks = [st.session_state.chunks[i] for i in candidate_indices]
                    candidate_sources = [st.session_state.chunk_sources[i] if st.session_state.chunk_sources else "알 수 없음" for i in candidate_indices]
                else:
                    query_emb = normalize(get_embeddings([prompt]))
                    _, raw_indices = st.session_state.index.search(query_emb, top_k)
                    candidate_chunks = [st.session_state.chunks[i] for i in raw_indices[0] if i < len(st.session_state.chunks)]
                    candidate_sources = [st.session_state.chunk_sources[i] if st.session_state.chunk_sources else "알 수 없음" for i in raw_indices[0] if i < len(st.session_state.chunks)]

                # ── 3단계: 리랭킹 ─────────────────────────────────────
                if use_reranking and len(candidate_chunks) > top_k:
                    with st.status("🔄 리랭킹 중...", expanded=False) as s:
                        scored = rerank_chunks(prompt, candidate_chunks, top_k=top_k)
                        # 점수 기준 재정렬 후 source도 맞춰서 재정렬
                        reranked_chunks = [c for c, _ in scored]
                        reranked_scores = [sc for _, sc in scored]
                        # source는 텍스트 일치로 재매핑
                        chunk_to_source = dict(zip(candidate_chunks, candidate_sources))
                        reranked_sources = [chunk_to_source.get(c, "알 수 없음") for c in reranked_chunks]
                        s.update(label=f"✅ 리랭킹 완료 → 상위 {top_k}개 선택", state="complete")
                else:
                    reranked_chunks = candidate_chunks[:top_k]
                    reranked_scores = [None] * len(reranked_chunks)
                    reranked_sources = candidate_sources[:top_k]

                # ── 4단계: 멀티문서 Reasoning (3-Step) ───────────────
                if use_multidoc_reasoning and len(reranked_chunks) > 0:

                    with st.status("📝 Step 1: 각 청크 요약 중...", expanded=False) as s:
                        summaries = step1_summarize_chunks(prompt, reranked_chunks)
                        s.update(label=f"✅ Step 1 완료: {len(summaries)}개 청크 요약", state="complete")

                    with st.status("🔍 Step 2: 공통점·차이점 분석 중...", expanded=False) as s:
                        analysis = step2_analyze_relationships(prompt, summaries, reranked_sources)
                        s.update(label="✅ Step 2 완료: 관계 분석 완료", state="complete")

                    with st.status("✍️ Step 3: 최종 답변 생성 중...", expanded=False) as s:
                        response = step3_generate_final_answer(prompt, reranked_chunks, summaries, analysis)
                        s.update(label="✅ Step 3 완료: 답변 생성", state="complete")

                else:
                    response = generate_answer_simple(prompt, reranked_chunks)

                st.markdown(response)

                # ── 중간 추론 과정 표시 ────────────────────────────────
                if use_query_rewrite:
                    with st.expander("🔍 생성된 쿼리 보기"):
                        for i, q in enumerate(queries):
                            label = "**[원본]**" if i == 0 else f"**[변형 {i}]**"
                            st.markdown(f"{label} {q}")

                if use_multidoc_reasoning and len(reranked_chunks) > 0:
                    with st.expander("📝 Step 1: 청크별 요약 보기"):
                        for i, (chunk_summary, src) in enumerate(zip(summaries, reranked_sources)):
                            score_txt = f" | 관련성: {reranked_scores[i]:.1f}/10" if reranked_scores[i] is not None else ""
                            st.caption(f"[출처 {i+1}] {src}{score_txt}")
                            st.info(chunk_summary)
                            if i < len(summaries) - 1:
                                st.divider()

                    with st.expander("🔍 Step 2: 공통점·차이점 분석 보기"):
                        st.markdown(analysis)

                with st.expander("📄 출처 원문 보기"):
                    for i, (chunk, src, score) in enumerate(zip(reranked_chunks, reranked_sources, reranked_scores)):
                        score_badge = f" _(관련성: {score:.1f}/10)_" if score is not None else ""
                        st.caption(f"[출처 {i+1}] **{src}**{score_badge}")
                        st.write(chunk)
                        if i < len(reranked_chunks) - 1:
                            st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
