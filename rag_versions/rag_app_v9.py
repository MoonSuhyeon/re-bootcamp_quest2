import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pdfplumber
import io
import os
import re
import json
import time
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 로그 저장 경로 (스크립트와 같은 폴더)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_eval_logs.json")

st.set_page_config(
    page_title="RAG 챗봇 v9",
    page_icon="📚",
    layout="wide"
)


# =====================================================================
# 기본 유틸
# =====================================================================

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data], dtype=np.float32)


def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


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


# =====================================================================
# 청킹
# =====================================================================

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


def build_index(chunks):
    embeddings = normalize(get_embeddings(chunks))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# =====================================================================
# 검색 파이프라인
# =====================================================================

def rewrite_queries(original_query, n=3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 문서 검색 성능을 높이는 쿼리 전문가입니다.\n"
                    "주어진 질문을 아래 두 가지 방식으로 재작성하세요:\n\n"
                    "1) 의도 분해형 쿼리 (정의·원인·결과·방법·조건으로 분리) 우선\n"
                    "2) 표현 다양화 쿼리 (동의어·문장 구조 변경)\n\n"
                    f"의도 분해형을 우선으로 총 {n}개를 생성하세요.\n"
                    "각 쿼리를 새 줄에, 번호·기호 없이 텍스트만 출력하세요."
                )
            },
            {"role": "user", "content": f"원본 질문: {original_query}"}
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    return [original_query] + variants[:n]


def retrieve_multi_query(queries, index, chunks, sources, top_k_per_query=20):
    seen_indices = set()
    result_items = []
    for query in queries:
        query_emb = normalize(get_embeddings([query]))
        _, indices = index.search(query_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen_indices:
                seen_indices.add(i)
                result_items.append((chunks[i], sources[i] if sources else "알 수 없음"))
    return result_items


def prefilter_by_similarity(query, items, top_n=10):
    if len(items) <= top_n:
        return items
    chunks = [item[0] for item in items]
    query_emb = normalize(get_embeddings([query]))[0]
    chunk_embs = normalize(get_embeddings(chunks))
    similarities = chunk_embs @ query_emb
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [items[i] for i in top_indices]


def rerank_chunks(query, items, top_k=3):
    if not items:
        return []
    chunks_text = "\n\n".join([f"[{i+1}] {item[0]}" for i, item in enumerate(items)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문에 대해 각 문서 청크의 관련성을 0~10점으로 평가하세요.\n"
                    "반드시 아래 형식으로만 출력하세요:\n1: 8\n2: 3\n..."
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
                if 0 <= idx < len(items):
                    scores[idx] = score
            except ValueError:
                pass
    scored = [(items[i][0], items[i][1], scores.get(i, 0.0)) for i in range(len(items))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


# =====================================================================
# 멀티문서 Reasoning
# =====================================================================

def step1_summarize_chunks(question, chunks):
    chunks_text = "\n\n".join([f"[{i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문과 관련하여 각 청크의 핵심 내용을 2~3문장으로 요약하세요.\n"
                    "반드시 아래 형식으로만 출력하세요:\n[1]: 요약 내용\n[2]: 요약 내용\n..."
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
    return [summaries.get(i, chunks[i][:120] + "...") for i in range(len(chunks))]


def step2_analyze_relationships(question, summaries, sources):
    summaries_text = "\n".join(
        [f"[출처 {i+1} | {sources[i]}] {s}" for i, s in enumerate(summaries)]
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "여러 문서 청크의 요약을 분석하여 질문에 답하기 위한 정보를 구조화하세요.\n"
                    "반드시 아래 형식으로 출력하세요:\n\n"
                    "**공통점**\n- ...\n\n"
                    "**차이점**\n- ...\n\n"
                    "**핵심 정보**\n- ...\n\n"
                    "**불확실성 / 누락 정보**\n- ..."
                )
            },
            {"role": "user", "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
        ]
    )
    return response.choices[0].message.content


def step3_generate_final_answer(question, chunks, summaries, analysis):
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    summaries_text = "\n".join([f"[출처 {i+1}] {s}" for i, s in enumerate(summaries)])

    system_prompt = """당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다.

[필수 답변 규칙]
1. 반드시 아래 3단계 구조로 답변하세요:
   **📌 요약** — 핵심 답변 1~2문장
   **📖 근거** — 각 주장마다 [출처 N] 인용 필수
   **✅ 결론** — 최종 판단 + 확신도: 높음/보통/낮음 — (이유)
2. 불확실성/누락 정보가 있으면 확신도에 반드시 반영하세요.
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
                    f"[분석]\n{analysis}\n\n"
                    f"[참고 문서 원문]\n{numbered_context}\n\n"
                    f"[질문]\n{question}"
                )
            }
        ]
    )
    return response.choices[0].message.content


def generate_answer_simple(question, items):
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{item[0]}" for i, item in enumerate(items)])
    system_prompt = """당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다.
반드시 **📌 요약** / **📖 근거** ([출처 N] 인용) / **✅ 결론** (확신도) 구조로 답변하세요.
컨텍스트에 없는 내용은 절대 지어내지 마세요. 항상 한국어로 답변하세요."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[참고 문서]\n{numbered_context}\n\n[질문]\n{question}"}
        ]
    )
    return response.choices[0].message.content


# =====================================================================
# [NEW] 평가 — LLM 기반 정확도·관련성·환각 판정
# =====================================================================

def evaluate_answer(question, context_chunks, answer):
    """
    답변을 3가지 기준으로 자동 평가.
    - 정확도 (1~5): 문서 내용에 비춰 사실적으로 맞는가
    - 관련성 (1~5): 질문에 제대로 답하고 있는가
    - 환각 여부: 문서에 없는 내용이 답변에 포함됐는가
    """
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문·문서·답변을 분석해 품질을 평가하세요.\n"
                    "반드시 아래 형식으로만 출력하세요 (다른 텍스트 금지):\n"
                    "정확도: N\n"
                    "관련성: N\n"
                    "환각여부: 없음|부분적|있음\n"
                    "환각근거: (환각 내용 요약, 없으면 '없음')\n\n"
                    "정확도 기준: 5=문서 내용과 완전 일치, 1=사실과 다름\n"
                    "관련성 기준: 5=질문에 완벽히 답함, 1=전혀 답하지 않음\n"
                    "환각 기준: 문서에 없는 내용을 사실처럼 서술하면 '있음'"
                )
            },
            {
                "role": "user",
                "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"
            }
        ]
    )

    result = {"정확도": 0, "관련성": 0, "환각여부": "알 수 없음", "환각근거": ""}
    for line in response.choices[0].message.content.strip().split('\n'):
        if line.startswith("정확도:"):
            try:
                result["정확도"] = int(line.split(':')[1].strip())
            except ValueError:
                pass
        elif line.startswith("관련성:"):
            try:
                result["관련성"] = int(line.split(':')[1].strip())
            except ValueError:
                pass
        elif line.startswith("환각여부:"):
            result["환각여부"] = line.split(':')[1].strip()
        elif line.startswith("환각근거:"):
            result["환각근거"] = line.split(':', 1)[1].strip()
    return result


# =====================================================================
# [NEW] 로그 저장·불러오기
# =====================================================================

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_log(entry: dict):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def build_log_entry(question, queries, ranked_items, answer, evaluation, latency_ms, mode):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "queries": queries,
        "retrieved_chunks": [
            {"text": chunk[:200] + ("..." if len(chunk) > 200 else ""),
             "source": src,
             "rerank_score": round(score, 2) if score is not None else None}
            for chunk, src, score in ranked_items
        ],
        "answer": answer,
        "evaluation": evaluation,
        "latency_ms": latency_ms,
        "mode": mode  # "multidoc" or "simple"
    }


# =====================================================================
# 세션 초기화
# =====================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_sources" not in st.session_state:
    st.session_state.chunk_sources = []


# =====================================================================
# 탭 레이아웃 — 챗봇 / 평가 로그 대시보드
# =====================================================================

tab_chat, tab_eval = st.tabs(["💬 챗봇", "📊 평가 로그"])


# =====================================================================
# 사이드바 (공통)
# =====================================================================

with st.sidebar:
    st.title("📚 RAG 챗봇 v9")
    st.markdown("---")

    st.subheader("문서 업로드")
    uploaded_files = st.file_uploader(
        "파일을 업로드하세요", type=["pdf", "txt"], accept_multiple_files=True
    )

    st.subheader("청킹 설정")
    chunking_mode = st.radio(
        "청킹 방식",
        ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"],
    )
    chunk_size = st.slider("청크 크기", 200, 1000, 500, 100)
    overlap = st.slider("Overlap 크기", 0, 200, 100, 20)

    if uploaded_files:
        if st.button("문서 처리하기", use_container_width=True, type="primary"):
            all_chunks, all_sources = [], []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    fn = chunk_text_semantic if chunking_mode == "의미 기반(Semantic) + Overlap" else chunk_text_with_overlap
                    fc = fn(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(fc)
                    all_sources.extend([f.name] * len(fc))
                st.session_state.chunks = all_chunks
                st.session_state.chunk_sources = all_sources
                st.session_state.index = build_index(all_chunks)
            st.success(f"완료! {len(all_chunks)}개 청크 생성됨")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크 인덱싱됨")
        for fname, cnt in Counter(st.session_state.chunk_sources).items():
            st.caption(f"  └ {fname}: {cnt}개")

    st.markdown("---")
    st.subheader("검색 설정")
    use_query_rewrite = st.toggle("쿼리 리라이팅 사용", value=True)
    num_rewrites = st.slider("리라이팅 쿼리 수", 1, 5, 3, disabled=not use_query_rewrite)
    prefilter_n = st.slider("Pre-filter 후보 수", 5, 20, 10)
    use_reranking = st.toggle("리랭킹 사용", value=True)
    top_k = st.slider("최종 사용 청크 수 (top_k)", 1, 5, 3)

    st.markdown("---")
    st.subheader("추론 / 평가 설정")
    use_multidoc_reasoning = st.toggle("멀티문서 추론 사용", value=True)
    auto_evaluate = st.toggle(
        "자동 평가 사용",
        value=True,
        help="답변 생성 후 정확도·관련성·환각 여부를 자동으로 평가하고 로그에 저장합니다"
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
    if st.button("로그 초기화", use_container_width=True):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        st.success("로그가 삭제됐습니다.")
        st.rerun()

    st.markdown("---")
    st.caption("Powered by OpenAI + FAISS | v9: Logging + Eval Dashboard")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================

with tab_chat:
    st.title("💬 문서 기반 챗봇 v9")
    st.caption("의도 분해 쿼리 · Pre-filter → Rerank · 멀티문서 추론 · 자동 평가 로깅")

    if st.session_state.index is None:
        st.info("왼쪽 사이드바에서 문서를 업로드하고 '문서 처리하기'를 눌러주세요.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        t_start = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.index is None:
                response = "먼저 문서를 업로드하고 처리해주세요."
                st.markdown(response)
            else:
                with st.spinner("답변 생성 중..."):
                    # 1. 쿼리 리라이팅
                    if use_query_rewrite:
                        with st.status("🔄 쿼리 리라이팅 중...", expanded=False) as s:
                            queries = rewrite_queries(prompt, n=num_rewrites)
                            s.update(label=f"✅ 쿼리 {len(queries)}개 생성", state="complete")
                    else:
                        queries = [prompt]

                    # 2. 멀티쿼리 검색
                    candidate_items = retrieve_multi_query(
                        queries, st.session_state.index,
                        st.session_state.chunks, st.session_state.chunk_sources,
                        top_k_per_query=20
                    )

                    # 3. Embedding Pre-filter
                    with st.status(f"⚡ Pre-filter: {len(candidate_items)}개 → {prefilter_n}개", expanded=False) as s:
                        filtered_items = prefilter_by_similarity(prompt, candidate_items, top_n=prefilter_n)
                        s.update(label=f"✅ Pre-filter 완료: {len(filtered_items)}개 선별", state="complete")

                    # 4. LLM Rerank
                    if use_reranking and len(filtered_items) > top_k:
                        with st.status("🔄 리랭킹 중...", expanded=False) as s:
                            ranked_items = rerank_chunks(prompt, filtered_items, top_k=top_k)
                            s.update(label=f"✅ 리랭킹 완료 → 상위 {top_k}개", state="complete")
                    else:
                        ranked_items = [(item[0], item[1], None) for item in filtered_items[:top_k]]

                    final_chunks  = [item[0] for item in ranked_items]
                    final_sources = [item[1] for item in ranked_items]
                    final_scores  = [item[2] for item in ranked_items]

                    # 5. 답변 생성
                    if use_multidoc_reasoning:
                        with st.status("📝 Step 1: 청크 요약...", expanded=False) as s:
                            summaries = step1_summarize_chunks(prompt, final_chunks)
                            s.update(label=f"✅ Step 1 완료", state="complete")
                        with st.status("🔍 Step 2: 관계 분석...", expanded=False) as s:
                            analysis = step2_analyze_relationships(prompt, summaries, final_sources)
                            s.update(label="✅ Step 2 완료", state="complete")
                        with st.status("✍️ Step 3: 답변 생성...", expanded=False) as s:
                            response = step3_generate_final_answer(prompt, final_chunks, summaries, analysis)
                            s.update(label="✅ Step 3 완료", state="complete")
                        mode = "multidoc"
                    else:
                        response = generate_answer_simple(prompt, ranked_items)
                        summaries, analysis = [], ""
                        mode = "simple"

                    latency_ms = int((time.time() - t_start) * 1000)

                    # 6. 자동 평가 + 로그 저장
                    evaluation = {}
                    if auto_evaluate:
                        with st.status("🧪 품질 평가 중...", expanded=False) as s:
                            evaluation = evaluate_answer(prompt, final_chunks, response)
                            log_entry = build_log_entry(
                                prompt, queries, ranked_items,
                                response, evaluation, latency_ms, mode
                            )
                            save_log(log_entry)
                            s.update(
                                label=f"✅ 평가 완료 | 정확도 {evaluation.get('정확도', '-')}/5 · 관련성 {evaluation.get('관련성', '-')}/5 · 환각 {evaluation.get('환각여부', '-')}",
                                state="complete"
                            )

                st.markdown(response)

                # 평가 뱃지
                if evaluation:
                    hall = evaluation.get('환각여부', '')
                    hall_color = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else "🔴")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("⏱ 응답 시간", f"{latency_ms/1000:.1f}s")
                    col2.metric("📐 정확도", f"{evaluation.get('정확도', '-')}/5")
                    col3.metric("🎯 관련성", f"{evaluation.get('관련성', '-')}/5")
                    col4.metric(f"{hall_color} 환각", hall)
                    if evaluation.get('환각근거') and evaluation.get('환각근거') != "없음":
                        st.warning(f"⚠️ 환각 감지: {evaluation['환각근거']}")

                # expander
                if use_query_rewrite:
                    with st.expander("🔍 생성된 쿼리 보기"):
                        for i, q in enumerate(queries):
                            st.markdown(f"{'**[원본]**' if i == 0 else f'**[변형 {i}]**'} {q}")

                if use_multidoc_reasoning and summaries:
                    with st.expander("📝 Step 1: 청크별 요약"):
                        for i, (s, src, sc) in enumerate(zip(summaries, final_sources, final_scores)):
                            sc_txt = f" | 관련성: {sc:.1f}/10" if sc is not None else ""
                            st.caption(f"[출처 {i+1}] {src}{sc_txt}")
                            st.info(s)
                            if i < len(summaries) - 1:
                                st.divider()

                    with st.expander("🔍 Step 2: 공통점·차이점·불확실성 분석"):
                        st.markdown(analysis)

                with st.expander("📄 출처 원문 보기"):
                    for i, (chunk, src, sc) in enumerate(zip(final_chunks, final_sources, final_scores)):
                        sc_badge = f" _(관련성: {sc:.1f}/10)_" if sc is not None else ""
                        st.caption(f"[출처 {i+1}] **{src}**{sc_badge}")
                        st.write(chunk)
                        if i < len(final_chunks) - 1:
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})


# =====================================================================
# TAB 2 — 평가 로그 대시보드 (Arize Phoenix 스타일)
# =====================================================================

with tab_eval:
    st.title("📊 평가 로그 대시보드")
    st.caption("질문 · 검색 청크 · 답변 · 정확도 · 관련성 · 환각 여부 기록")

    logs = load_logs()

    if not logs:
        st.info("아직 로그가 없습니다. 챗봇 탭에서 질문하면 자동으로 기록됩니다.")
    else:
        # ── 상단 요약 지표 ──────────────────────────────────────────
        total = len(logs)
        eval_logs = [l for l in logs if l.get("evaluation")]
        avg_acc  = round(sum(l["evaluation"].get("정확도", 0) for l in eval_logs) / len(eval_logs), 2) if eval_logs else 0
        avg_rel  = round(sum(l["evaluation"].get("관련성", 0) for l in eval_logs) / len(eval_logs), 2) if eval_logs else 0
        hall_counts = Counter(l["evaluation"].get("환각여부", "알 수 없음") for l in eval_logs)
        hall_none_pct = round(hall_counts.get("없음", 0) / len(eval_logs) * 100) if eval_logs else 0
        avg_lat = round(sum(l.get("latency_ms", 0) for l in logs) / total / 1000, 2)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("총 질문 수",    f"{total}건")
        c2.metric("평균 정확도",   f"{avg_acc}/5")
        c3.metric("평균 관련성",   f"{avg_rel}/5")
        c4.metric("환각 없음 비율", f"{hall_none_pct}%")
        c5.metric("평균 응답 시간", f"{avg_lat}s")

        st.markdown("---")

        # ── 환각 분포 바 차트 ───────────────────────────────────────
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("환각 여부 분포")
            hall_data = {"없음": hall_counts.get("없음", 0),
                         "부분적": hall_counts.get("부분적", 0),
                         "있음": hall_counts.get("있음", 0)}
            st.bar_chart(hall_data)

        with col_chart2:
            st.subheader("정확도 / 관련성 추이")
            if len(eval_logs) >= 2:
                chart_data = {
                    "정확도": [l["evaluation"].get("정확도", 0) for l in eval_logs],
                    "관련성": [l["evaluation"].get("관련성", 0) for l in eval_logs],
                }
                st.line_chart(chart_data)
            else:
                st.caption("2건 이상 쌓이면 추이 차트가 표시됩니다.")

        st.markdown("---")

        # ── 로그 테이블 ─────────────────────────────────────────────
        st.subheader("전체 로그")

        # 최신 순 정렬
        for i, log in enumerate(reversed(logs)):
            ev = log.get("evaluation", {})
            hall = ev.get("환각여부", "-")
            hall_icon = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else ("🔴" if hall == "있음" else "⚪"))
            acc = ev.get("정확도", "-")
            rel = ev.get("관련성", "-")
            lat = f"{log.get('latency_ms', 0)/1000:.1f}s"
            ts  = log.get("timestamp", "")
            mode_badge = "🧠 멀티추론" if log.get("mode") == "multidoc" else "⚡ 단순"

            label = f"[{ts}] {mode_badge} | {log['question'][:60]}{'...' if len(log['question']) > 60 else ''} | 정확도 {acc}/5 · 관련성 {rel}/5 · 환각 {hall_icon} · {lat}"
            with st.expander(label):
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.markdown("**❓ 질문**")
                    st.write(log["question"])

                    if log.get("queries"):
                        st.markdown("**🔍 사용된 쿼리**")
                        for q in log["queries"]:
                            st.caption(f"• {q}")

                    st.markdown("**📄 검색된 청크**")
                    for j, chunk_info in enumerate(log.get("retrieved_chunks", [])):
                        sc = chunk_info.get("rerank_score")
                        sc_txt = f" _(관련성: {sc}/10)_" if sc is not None else ""
                        st.caption(f"[출처 {j+1}] **{chunk_info.get('source', '')}**{sc_txt}")
                        st.text(chunk_info.get("text", ""))

                    st.markdown("**💬 최종 답변**")
                    st.markdown(log.get("answer", ""))

                with col_b:
                    st.markdown("**🧪 평가 결과**")
                    if ev:
                        st.metric("정확도", f"{ev.get('정확도', '-')}/5")
                        st.metric("관련성", f"{ev.get('관련성', '-')}/5")
                        hall_val = ev.get("환각여부", "-")
                        color = "🟢" if hall_val == "없음" else ("🟡" if hall_val == "부분적" else "🔴")
                        st.metric("환각 여부", f"{color} {hall_val}")
                        if ev.get("환각근거") and ev.get("환각근거") != "없음":
                            st.warning(f"환각 내용: {ev['환각근거']}")
                    st.metric("응답 시간", lat)
                    st.caption(f"추론 모드: {mode_badge}")

        # ── CSV 다운로드 ─────────────────────────────────────────────
        st.markdown("---")
        import csv
        csv_rows = []
        headers = ["timestamp", "question", "정확도", "관련성", "환각여부", "환각근거", "latency_ms", "mode"]
        csv_rows.append(headers)
        for log in logs:
            ev = log.get("evaluation", {})
            csv_rows.append([
                log.get("timestamp", ""),
                log.get("question", ""),
                ev.get("정확도", ""),
                ev.get("관련성", ""),
                ev.get("환각여부", ""),
                ev.get("환각근거", ""),
                log.get("latency_ms", ""),
                log.get("mode", "")
            ])

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerows(csv_rows)

        st.download_button(
            label="⬇️ 로그 CSV 다운로드",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name=f"rag_eval_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
