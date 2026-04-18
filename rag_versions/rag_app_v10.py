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
import uuid
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_eval_logs_v10.json")

st.set_page_config(page_title="RAG 챗봇 v10", page_icon="📚", layout="wide")


# =====================================================================
# [NEW] Tracer — 단계별 latency + token 사용량 추적
# =====================================================================

class Tracer:
    """
    각 단계(span)의 시작/종료 시간, 토큰 사용량, 결정 근거를 기록.
    Arize Phoenix의 span 개념과 동일한 구조.
    """
    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:8]
        self.spans: list[dict] = []
        self._active: dict = {}

    def start(self, name: str):
        self._active[name] = time.time()

    def end(self, name: str, tokens: dict = None,
            input_summary: str = "", output_summary: str = "",
            decision: str = "", error: str = ""):
        start = self._active.pop(name, time.time())
        span = {
            "name": name,
            "duration_ms": int((time.time() - start) * 1000),
            "tokens": tokens or {"prompt": 0, "completion": 0, "total": 0},
            "input_summary": input_summary,
            "output_summary": output_summary,
            "decision": decision,
            "error": error,
        }
        self.spans.append(span)
        return span

    def total_tokens(self) -> dict:
        p = sum(s["tokens"]["prompt"] for s in self.spans)
        c = sum(s["tokens"]["completion"] for s in self.spans)
        return {"prompt": p, "completion": c, "total": p + c}

    def total_latency_ms(self) -> int:
        return sum(s["duration_ms"] for s in self.spans)

    def bottleneck(self) -> str:
        if not self.spans:
            return "-"
        slowest = max(self.spans, key=lambda s: s["duration_ms"])
        return slowest["name"]


def _usage(response) -> dict:
    u = response.usage
    return {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
            "total": u.total_tokens}


# =====================================================================
# 기본 유틸
# =====================================================================

def get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
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
    raw_chunks, current = [], ""
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
        overlapped.append(raw_chunks[i - 1][-overlap:] + " " + raw_chunks[i])
    return overlapped


def chunk_text_semantic(text, chunk_size=500, overlap=100):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', text) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return chunk_text_with_overlap(text, chunk_size, overlap)
    embeddings = normalize(get_embeddings(sentences))
    similarities = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    threshold = np.mean(similarities) - 0.5 * np.std(similarities)
    breakpoints = [0] + [i + 1 for i, sim in enumerate(similarities) if sim < threshold] + [len(sentences)]
    raw_chunks = []
    for i in range(len(breakpoints) - 1):
        current = ""
        for sent in sentences[breakpoints[i]:breakpoints[i + 1]]:
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
        overlapped.append(raw_chunks[i - 1][-overlap:] + " " + raw_chunks[i])
    return overlapped


def build_index(chunks):
    embeddings = normalize(get_embeddings(chunks))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# =====================================================================
# 검색 파이프라인 — 모든 LLM 함수가 (결과, usage) 반환
# =====================================================================

def rewrite_queries(original_query, n=3, tracer: Tracer = None):
    tracer and tracer.start("query_rewriting")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 문서 검색 성능을 높이는 쿼리 전문가입니다.\n"
                    "주어진 질문을 아래 두 가지 방식으로 재작성하세요:\n"
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
    queries = [original_query] + variants[:n]
    if tracer:
        tracer.end("query_rewriting",
                   tokens=_usage(response),
                   input_summary=f"원본: {original_query[:60]}",
                   output_summary=f"쿼리 {len(queries)}개 생성",
                   decision=f"의도 분해형 우선 {n}개 요청 → 변형 {len(variants)}개 생성됨")
    return queries


def retrieve_multi_query(queries, index, chunks, sources, top_k_per_query=20, tracer: Tracer = None):
    tracer and tracer.start("embedding_search")
    seen, items = set(), []
    for query in queries:
        query_emb = normalize(get_embeddings([query]))
        _, indices = index.search(query_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen:
                seen.add(i)
                items.append((chunks[i], sources[i] if sources else "알 수 없음"))
    if tracer:
        tracer.end("embedding_search",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개",
                   output_summary=f"후보 청크 {len(items)}개 수집",
                   decision=f"쿼리당 top {top_k_per_query}, 중복 제거 후 {len(items)}개")
    return items


def prefilter_by_similarity(query, items, top_n=10, tracer: Tracer = None):
    tracer and tracer.start("prefilter")
    if len(items) <= top_n:
        tracer and tracer.end("prefilter",
                              input_summary=f"{len(items)}개 (컷 불필요)",
                              output_summary=f"{len(items)}개 유지",
                              decision="후보 수가 top_n 이하라 그대로 통과")
        return items
    chunks = [item[0] for item in items]
    query_emb = normalize(get_embeddings([query]))[0]
    sims = normalize(get_embeddings(chunks)) @ query_emb
    top_idx = np.argsort(sims)[::-1][:top_n]
    result = [items[i] for i in top_idx]
    cutoff_score = round(float(sims[top_idx[-1]]), 4)
    if tracer:
        tracer.end("prefilter",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {len(result)}개 선별 (컷오프 유사도: {cutoff_score})",
                   decision=f"임베딩 코사인 유사도 ≥ {cutoff_score} 인 {top_n}개 통과, {len(items)-top_n}개 제거")
    return result


def rerank_chunks(query, items, top_k=3, tracer: Tracer = None):
    tracer and tracer.start("rerank")
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
                sc = float(parts[1].strip())
                if 0 <= idx < len(items):
                    scores[idx] = sc
            except ValueError:
                pass
    scored = [(items[i][0], items[i][1], scores.get(i, 0.0)) for i in range(len(items))]
    scored.sort(key=lambda x: x[2], reverse=True)
    result = scored[:top_k]
    score_str = ", ".join([f"{s[2]:.1f}" for s in result])
    if tracer:
        tracer.end("rerank",
                   tokens=_usage(response),
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {top_k}개 선택 (점수: {score_str})",
                   decision=f"LLM 0~10 채점 → 점수 순 정렬 → 상위 {top_k}개 반환")
    return result


# =====================================================================
# 멀티문서 Reasoning — (결과, usage) 반환
# =====================================================================

def step1_summarize_chunks(question, chunks, tracer: Tracer = None):
    tracer and tracer.start("step1_summarize")
    chunks_text = "\n\n".join([f"[{i+1}]\n{c}" for i, c in enumerate(chunks)])
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
    result = [summaries.get(i, chunks[i][:120] + "...") for i in range(len(chunks))]
    if tracer:
        tracer.end("step1_summarize",
                   tokens=_usage(response),
                   input_summary=f"{len(chunks)}개 청크",
                   output_summary=f"{len(result)}개 요약 완료",
                   decision="질문 관점 핵심 추출, 무관 내용 제거")
    return result


def step2_analyze_relationships(question, summaries, sources, tracer: Tracer = None):
    tracer and tracer.start("step2_analyze")
    summaries_text = "\n".join([f"[출처 {i+1} | {sources[i]}] {s}" for i, s in enumerate(summaries)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "여러 문서 청크의 요약을 분석하여 질문에 답하기 위한 정보를 구조화하세요.\n"
                    "반드시 아래 형식으로 출력하세요:\n\n"
                    "**공통점**\n- ...\n\n**차이점**\n- ...\n\n"
                    "**핵심 정보**\n- ...\n\n**불확실성 / 누락 정보**\n- ..."
                )
            },
            {"role": "user", "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        # 불확실성 항목이 실제로 있는지 감지
        has_uncertainty = "없음" not in result.split("**불확실성")[-1][:60] if "**불확실성" in result else False
        tracer.end("step2_analyze",
                   tokens=_usage(response),
                   input_summary=f"{len(summaries)}개 요약",
                   output_summary="공통점/차이점/핵심/불확실성 구조화 완료",
                   decision=f"불확실성·누락 감지: {'있음' if has_uncertainty else '없음'}")
    return result


def step3_generate_final_answer(question, chunks, summaries, analysis, tracer: Tracer = None):
    tracer and tracer.start("step3_answer")
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(chunks)])
    summaries_text = "\n".join([f"[출처 {i+1}] {s}" for i, s in enumerate(summaries)])
    system_prompt = """당신은 주어진 문서를 기반으로 질문에 답하는 어시스턴트입니다.
[필수 답변 규칙]
1. 반드시 **📌 요약** / **📖 근거** ([출처 N] 인용 필수) / **✅ 결론** (확신도: 높음/보통/낮음) 구조로 답변.
2. 불확실성/누락 정보가 있으면 확신도에 반드시 반영.
3. 컨텍스트에 없는 내용은 절대 지어내지 마세요.
4. 항상 한국어로 답변하세요."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"[청크 요약]\n{summaries_text}\n\n"
                f"[분석]\n{analysis}\n\n"
                f"[참고 문서 원문]\n{numbered_context}\n\n"
                f"[질문]\n{question}"
            )}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("step3_answer",
                   tokens=_usage(response),
                   input_summary="요약 + 분석 + 원문",
                   output_summary=f"답변 {len(result)}자 생성",
                   decision="Step1 요약 + Step2 분석을 힌트로 최종 구조화 답변")
    return result


def generate_answer_simple(question, items, tracer: Tracer = None):
    tracer and tracer.start("step3_answer")
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{item[0]}" for i, item in enumerate(items)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "문서 기반 어시스턴트. **📌 요약** / **📖 근거** ([출처 N]) / **✅ 결론** (확신도) 구조. 한국어."},
            {"role": "user", "content": f"[참고 문서]\n{numbered_context}\n\n[질문]\n{question}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("step3_answer", tokens=_usage(response),
                   input_summary="원문 직접 전달", output_summary=f"답변 {len(result)}자",
                   decision="단순 모드: 요약·분석 없이 원문 → 답변")
    return result


# =====================================================================
# [NEW] 평가 + 환각 원인 심층 분석
# =====================================================================

def evaluate_answer(question, context_chunks, answer, tracer: Tracer = None):
    tracer and tracer.start("evaluation")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 질문·문서·답변을 분석해 품질을 평가하세요.\n"
                    "반드시 아래 형식으로만 출력하세요:\n"
                    "정확도: N\n관련성: N\n환각여부: 없음|부분적|있음\n환각근거: ..."
                )
            },
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
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
    if tracer:
        tracer.end("evaluation",
                   tokens=_usage(response),
                   input_summary="질문 + 문서 + 답변",
                   output_summary=f"정확도 {result['정확도']}/5 · 관련성 {result['관련성']}/5 · 환각 {result['환각여부']}",
                   decision="문서↔답변 직접 비교 채점")
    return result


def analyze_hallucination_cause(question, context_chunks, answer, hall_type, tracer: Tracer = None):
    """
    환각이 감지됐을 때만 호출.
    - 어느 주장이 환각인지
    - 어떤 유형인지 (fabrication/distortion/over-generalization)
    - 왜 발생했는지 (context 부족 / 청크 모호 / LLM 보간)
    """
    if hall_type == "없음":
        return None
    tracer and tracer.start("hallucination_analysis")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "답변에서 환각(hallucination)이 감지됐습니다. 원인을 분석하세요.\n"
                    "반드시 아래 형식으로 출력하세요:\n"
                    "환각_주장: (어떤 문장/내용이 환각인가)\n"
                    "환각_유형: fabrication|distortion|over-generalization\n"
                    "근거_출처: (해당 정보가 어느 출처에 있어야 하는가, 없으면 '없음')\n"
                    "발생_원인: insufficient_context|ambiguous_chunk|llm_interpolation\n"
                    "개선_제안: (이 환각을 막으려면 어떻게 해야 하는가 한 줄)\n\n"
                    "유형 설명:\n"
                    "- fabrication: 문서에 전혀 없는 내용을 만들어냄\n"
                    "- distortion: 문서 내용을 잘못 해석하거나 과장\n"
                    "- over-generalization: 특수 사례를 일반 규칙으로 확대 적용"
                )
            },
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ]
    )
    raw = response.choices[0].message.content.strip()
    result = {"환각_주장": "", "환각_유형": "", "근거_출처": "", "발생_원인": "", "개선_제안": "", "raw": raw}
    for line in raw.split('\n'):
        for key in ["환각_주장", "환각_유형", "근거_출처", "발생_원인", "개선_제안"]:
            if line.startswith(f"{key}:"):
                result[key] = line.split(':', 1)[1].strip()
    if tracer:
        tracer.end("hallucination_analysis",
                   tokens=_usage(response),
                   input_summary=f"환각 유형: {hall_type}",
                   output_summary=f"유형: {result['환각_유형']} / 원인: {result['발생_원인']}",
                   decision="환각 감지 시에만 실행 (추가 LLM 호출)")
    return result


# =====================================================================
# 로그 저장·불러오기
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


def build_log_entry(question, queries, ranked_items, answer,
                    evaluation, hall_cause, tracer: Tracer, mode):
    return {
        "trace_id": tracer.trace_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "queries": queries,
        "retrieved_chunks": [
            {"text": c[:200] + ("..." if len(c) > 200 else ""), "source": s,
             "rerank_score": round(sc, 2) if sc is not None else None}
            for c, s, sc in ranked_items
        ],
        "answer": answer,
        "evaluation": evaluation,
        "hallucination_analysis": hall_cause,
        "spans": tracer.spans,
        "total_tokens": tracer.total_tokens(),
        "total_latency_ms": tracer.total_latency_ms(),
        "bottleneck": tracer.bottleneck(),
        "mode": mode,
    }


# =====================================================================
# 세션 초기화
# =====================================================================
for key, default in [("messages", []), ("index", None), ("chunks", []), ("chunk_sources", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# 탭 레이아웃
# =====================================================================
tab_chat, tab_trace, tab_agent = st.tabs(["💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석"])


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("📚 RAG 챗봇 v10")
    st.markdown("---")

    uploaded_files = st.file_uploader("파일을 업로드하세요", type=["pdf", "txt"], accept_multiple_files=True)
    chunking_mode = st.radio("청킹 방식", ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"])
    chunk_size = st.slider("청크 크기", 200, 1000, 500, 100)
    overlap = st.slider("Overlap 크기", 0, 200, 100, 20)

    if uploaded_files:
        if st.button("문서 처리하기", use_container_width=True, type="primary"):
            all_chunks, all_sources = [], []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    fn = chunk_text_semantic if "Semantic" in chunking_mode else chunk_text_with_overlap
                    fc = fn(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(fc)
                    all_sources.extend([f.name] * len(fc))
                st.session_state.chunks = all_chunks
                st.session_state.chunk_sources = all_sources
                st.session_state.index = build_index(all_chunks)
            st.success(f"완료! {len(all_chunks)}개 청크")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크")
        for fname, cnt in Counter(st.session_state.chunk_sources).items():
            st.caption(f"  └ {fname}: {cnt}개")

    st.markdown("---")
    use_query_rewrite = st.toggle("쿼리 리라이팅", value=True)
    num_rewrites = st.slider("리라이팅 수", 1, 5, 3, disabled=not use_query_rewrite)
    prefilter_n = st.slider("Pre-filter 수", 5, 20, 10)
    use_reranking = st.toggle("리랭킹", value=True)
    top_k = st.slider("최종 청크 수 (top_k)", 1, 5, 3)
    use_multidoc = st.toggle("멀티문서 추론", value=True)
    auto_evaluate = st.toggle("자동 평가 + 로깅", value=True)

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("문서 초기화", use_container_width=True):
        st.session_state.update({"messages": [], "index": None, "chunks": [], "chunk_sources": []})
        st.rerun()
    if st.button("로그 초기화", use_container_width=True):
        os.path.exists(LOG_FILE) and os.remove(LOG_FILE)
        st.success("로그 삭제 완료")
        st.rerun()
    st.caption("v10: Full Tracing + Agent Analysis")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================
with tab_chat:
    st.title("💬 문서 기반 챗봇 v10")
    if st.session_state.index is None:
        st.info("사이드바에서 문서를 업로드하고 처리해주세요.")
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
                tracer = Tracer()
                with st.spinner("답변 생성 중..."):
                    # 1. 쿼리 리라이팅
                    if use_query_rewrite:
                        with st.status("🔄 쿼리 리라이팅...", expanded=False) as s:
                            queries = rewrite_queries(prompt, n=num_rewrites, tracer=tracer)
                            s.update(label=f"✅ 쿼리 {len(queries)}개", state="complete")
                    else:
                        queries = [prompt]

                    # 2. 검색
                    candidate_items = retrieve_multi_query(
                        queries, st.session_state.index,
                        st.session_state.chunks, st.session_state.chunk_sources,
                        top_k_per_query=20, tracer=tracer
                    )

                    # 3. Pre-filter
                    with st.status(f"⚡ Pre-filter...", expanded=False) as s:
                        filtered = prefilter_by_similarity(prompt, candidate_items, prefilter_n, tracer)
                        s.update(label=f"✅ {len(filtered)}개 선별", state="complete")

                    # 4. Rerank
                    if use_reranking and len(filtered) > top_k:
                        with st.status("🔄 리랭킹...", expanded=False) as s:
                            ranked = rerank_chunks(prompt, filtered, top_k, tracer)
                            s.update(label=f"✅ 상위 {top_k}개", state="complete")
                    else:
                        ranked = [(item[0], item[1], None) for item in filtered[:top_k]]
                        tracer.start("rerank")
                        tracer.end("rerank", input_summary=f"{len(filtered)}개",
                                   output_summary=f"{len(ranked)}개 (리랭킹 OFF)", decision="리랭킹 비활성화")

                    final_chunks  = [r[0] for r in ranked]
                    final_sources = [r[1] for r in ranked]
                    final_scores  = [r[2] for r in ranked]

                    # 5. 답변 생성
                    if use_multidoc:
                        with st.status("📝 Step1 요약...", expanded=False) as s:
                            summaries = step1_summarize_chunks(prompt, final_chunks, tracer)
                            s.update(label="✅ Step1 완료", state="complete")
                        with st.status("🔍 Step2 분석...", expanded=False) as s:
                            analysis = step2_analyze_relationships(prompt, summaries, final_sources, tracer)
                            s.update(label="✅ Step2 완료", state="complete")
                        with st.status("✍️ Step3 답변...", expanded=False) as s:
                            response = step3_generate_final_answer(prompt, final_chunks, summaries, analysis, tracer)
                            s.update(label="✅ Step3 완료", state="complete")
                        mode = "multidoc"
                    else:
                        response = generate_answer_simple(prompt, ranked, tracer)
                        summaries, analysis = [], ""
                        mode = "simple"

                    # 6. 평가 + 환각 원인 분석
                    evaluation, hall_cause = {}, None
                    if auto_evaluate:
                        with st.status("🧪 평가 중...", expanded=False) as s:
                            evaluation = evaluate_answer(prompt, final_chunks, response, tracer)
                            if evaluation.get("환각여부", "없음") != "없음":
                                hall_cause = analyze_hallucination_cause(
                                    prompt, final_chunks, response,
                                    evaluation["환각여부"], tracer
                                )
                            save_log(build_log_entry(
                                prompt, queries, ranked, response,
                                evaluation, hall_cause, tracer, mode
                            ))
                            hall = evaluation.get("환각여부", "-")
                            s.update(label=f"✅ 정확도 {evaluation.get('정확도','-')}/5 · 관련성 {evaluation.get('관련성','-')}/5 · 환각 {hall}", state="complete")

                st.markdown(response)

                # 평가 뱃지
                if evaluation:
                    hall = evaluation.get("환각여부", "")
                    hc = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else "🔴")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("⏱ 응답", f"{tracer.total_latency_ms()/1000:.1f}s")
                    c2.metric("🔤 토큰", f"{tracer.total_tokens()['total']:,}")
                    c3.metric("📐 정확도", f"{evaluation.get('정확도','-')}/5")
                    c4.metric("🎯 관련성", f"{evaluation.get('관련성','-')}/5")
                    c5.metric(f"{hc} 환각", hall)
                    if hall_cause:
                        st.error(f"⚠️ 환각 유형: **{hall_cause.get('환각_유형','')}** | 원인: {hall_cause.get('발생_원인','')} | 개선: {hall_cause.get('개선_제안','')}")

                # expander
                if use_query_rewrite:
                    with st.expander("🔍 생성된 쿼리"):
                        for i, q in enumerate(queries):
                            st.markdown(f"{'**[원본]**' if i == 0 else f'**[변형 {i}]**'} {q}")
                if use_multidoc and summaries:
                    with st.expander("📝 Step1 청크 요약"):
                        for i, (s, src, sc) in enumerate(zip(summaries, final_sources, final_scores)):
                            st.caption(f"[출처 {i+1}] {src}" + (f" | 관련성 {sc:.1f}/10" if sc else ""))
                            st.info(s)
                            if i < len(summaries) - 1:
                                st.divider()
                    with st.expander("🔍 Step2 공통점·차이점·불확실성"):
                        st.markdown(analysis)
                with st.expander("📄 출처 원문"):
                    for i, (c, src, sc) in enumerate(zip(final_chunks, final_sources, final_scores)):
                        st.caption(f"[출처 {i+1}] **{src}**" + (f" _(관련성 {sc:.1f}/10)_" if sc else ""))
                        st.write(c)
                        if i < len(final_chunks) - 1:
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})


# =====================================================================
# TAB 2 — 트레이싱 대시보드
# =====================================================================
with tab_trace:
    st.title("🔬 트레이싱 대시보드")
    st.caption("단계별 latency · 토큰 사용량 · 병목 구간 · API 흐름")

    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다. 챗봇 탭에서 질문하면 자동 기록됩니다.")
    else:
        # 요약 지표
        total_tokens_all = [l.get("total_tokens", {}).get("total", 0) for l in logs]
        total_lat_all = [l.get("total_latency_ms", 0) for l in logs]
        bottlenecks = Counter(l.get("bottleneck", "") for l in logs)
        most_bottleneck = bottlenecks.most_common(1)[0][0] if bottlenecks else "-"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 트레이스 수", len(logs))
        c2.metric("평균 응답 시간", f"{sum(total_lat_all)/len(total_lat_all)/1000:.1f}s")
        c3.metric("평균 총 토큰", f"{int(sum(total_tokens_all)/len(total_tokens_all)):,}")
        c4.metric("가장 빈번한 병목", most_bottleneck)

        st.markdown("---")

        # 응답 시간 추이
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("응답 시간 추이 (초)")
            st.line_chart([l / 1000 for l in total_lat_all])
        with col2:
            st.subheader("토큰 사용량 추이")
            st.line_chart(total_tokens_all)

        st.markdown("---")
        st.subheader("트레이스 상세 (최신 순)")

        for log in reversed(logs):
            spans = log.get("spans", [])
            total_ms = log.get("total_latency_ms", 1)
            tok = log.get("total_tokens", {})
            ts = log.get("timestamp", "")
            q = log.get("question", "")[:60]
            bn = log.get("bottleneck", "-")

            with st.expander(f"[{ts}] {q}... | ⏱ {total_ms/1000:.1f}s | 🔤 {tok.get('total',0):,} tokens | 병목: {bn}"):
                # Waterfall 차트
                st.markdown("**⏱ 단계별 Latency (Waterfall)**")
                for span in spans:
                    dur = span["duration_ms"]
                    pct = dur / total_ms if total_ms > 0 else 0
                    err_icon = " ❌" if span.get("error") else ""
                    c1, c2, c3 = st.columns([2, 6, 1])
                    c1.caption(span["name"] + err_icon)
                    c2.progress(min(pct, 1.0))
                    c3.caption(f"{dur}ms")

                st.markdown("---")

                # 토큰 breakdown
                st.markdown("**🔤 단계별 토큰 사용량**")
                tok_data = {s["name"]: s["tokens"]["total"] for s in spans if s["tokens"]["total"] > 0}
                if tok_data:
                    st.bar_chart(tok_data)
                    cols = st.columns(3)
                    cols[0].metric("Prompt 토큰", tok.get("prompt", 0))
                    cols[1].metric("Completion 토큰", tok.get("completion", 0))
                    cols[2].metric("총 토큰", tok.get("total", 0))

                st.markdown("---")

                # API 흐름 테이블
                st.markdown("**🔗 API 호출 흐름**")
                flow_rows = []
                for span in spans:
                    flow_rows.append({
                        "단계": span["name"],
                        "소요(ms)": span["duration_ms"],
                        "토큰": span["tokens"]["total"],
                        "입력": span.get("input_summary", "")[:60],
                        "출력": span.get("output_summary", "")[:60],
                    })
                if flow_rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)


# =====================================================================
# TAB 3 — 에이전트 분석 대시보드
# =====================================================================
with tab_agent:
    st.title("🧠 에이전트 분석 대시보드")
    st.caption("AI 의사결정 추적 · 환각 원인 심층 분석 · 개선 제안")

    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        eval_logs = [l for l in logs if l.get("evaluation")]

        # 요약 지표
        avg_acc = round(sum(l["evaluation"].get("정확도", 0) for l in eval_logs) / len(eval_logs), 2) if eval_logs else 0
        avg_rel = round(sum(l["evaluation"].get("관련성", 0) for l in eval_logs) / len(eval_logs), 2) if eval_logs else 0
        hall_counts = Counter(l["evaluation"].get("환각여부", "") for l in eval_logs)
        hall_none_pct = round(hall_counts.get("없음", 0) / len(eval_logs) * 100) if eval_logs else 0
        hall_logs = [l for l in eval_logs if l["evaluation"].get("환각여부", "없음") != "없음"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("평균 정확도", f"{avg_acc}/5")
        c2.metric("평균 관련성", f"{avg_rel}/5")
        c3.metric("환각 없음 비율", f"{hall_none_pct}%")
        c4.metric("환각 감지 건수", len(hall_logs))

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("환각 여부 분포")
            st.bar_chart({"없음": hall_counts.get("없음", 0),
                          "부분적": hall_counts.get("부분적", 0),
                          "있음": hall_counts.get("있음", 0)})
        with col2:
            st.subheader("정확도 / 관련성 추이")
            if len(eval_logs) >= 2:
                st.line_chart({"정확도": [l["evaluation"].get("정확도", 0) for l in eval_logs],
                               "관련성": [l["evaluation"].get("관련성", 0) for l in eval_logs]})
            else:
                st.caption("2건 이상 필요")

        # 환각 원인 분석 섹션
        if hall_logs:
            st.markdown("---")
            st.subheader("⚠️ 환각 감지 케이스 심층 분석")
            cause_type_counts = Counter()
            root_cause_counts = Counter()
            for l in hall_logs:
                hc = l.get("hallucination_analysis") or {}
                cause_type_counts[hc.get("환각_유형", "미분석")] += 1
                root_cause_counts[hc.get("발생_원인", "미분석")] += 1

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**환각 유형 분포**")
                if cause_type_counts:
                    st.bar_chart(dict(cause_type_counts))
            with cc2:
                st.markdown("**발생 원인 분포**")
                if root_cause_counts:
                    st.bar_chart(dict(root_cause_counts))

        st.markdown("---")
        st.subheader("전체 에이전트 의사결정 로그 (최신 순)")

        for log in reversed(logs):
            spans = log.get("spans", [])
            ev = log.get("evaluation", {})
            hc = log.get("hallucination_analysis")
            hall = ev.get("환각여부", "-")
            hi = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else ("🔴" if hall == "있음" else "⚪"))
            ts = log.get("timestamp", "")
            q  = log.get("question", "")[:55]

            label = f"[{ts}] {q}... | 정확도 {ev.get('정확도','-')}/5 · 관련성 {ev.get('관련성','-')}/5 · {hi} 환각 {hall}"
            with st.expander(label):
                col_a, col_b = st.columns([3, 2])

                with col_a:
                    st.markdown("**❓ 질문**")
                    st.write(log.get("question", ""))

                    st.markdown("**🤖 AI 의사결정 체인**")
                    for span in spans:
                        icon_map = {
                            "query_rewriting": "🔄", "embedding_search": "🔍",
                            "prefilter": "⚡", "rerank": "📊",
                            "step1_summarize": "📝", "step2_analyze": "🔬",
                            "step3_answer": "✍️", "evaluation": "🧪",
                            "hallucination_analysis": "⚠️"
                        }
                        icon = icon_map.get(span["name"], "▶")
                        decision = span.get("decision", "")
                        st.markdown(
                            f"{icon} **{span['name']}** `{span['duration_ms']}ms`  \n"
                            f"↳ {span.get('output_summary', '')}  \n"
                            + (f"💡 *{decision}*" if decision else "")
                        )
                        if span != spans[-1]:
                            st.markdown("↓")

                    st.markdown("**💬 최종 답변**")
                    st.markdown(log.get("answer", ""))

                with col_b:
                    st.markdown("**🧪 평가 결과**")
                    if ev:
                        st.metric("정확도", f"{ev.get('정확도','-')}/5")
                        st.metric("관련성", f"{ev.get('관련성','-')}/5")
                        st.metric("환각 여부", f"{hi} {hall}")
                        if ev.get("환각근거") and ev.get("환각근거") != "없음":
                            st.warning(f"감지 내용: {ev['환각근거']}")

                    if hc:
                        st.markdown("**🔬 환각 원인 분석**")
                        st.error(f"**유형**: {hc.get('환각_유형','')}")
                        st.write(f"**주장**: {hc.get('환각_주장','')}")
                        st.write(f"**근거 출처**: {hc.get('근거_출처','')}")
                        st.write(f"**발생 원인**: {hc.get('발생_원인','')}")
                        st.info(f"💡 개선 제안: {hc.get('개선_제안','')}")

                    st.markdown("**⏱ 성능**")
                    st.metric("총 응답 시간", f"{log.get('total_latency_ms',0)/1000:.1f}s")
                    tok = log.get("total_tokens", {})
                    st.metric("총 토큰", f"{tok.get('total',0):,}")
                    st.caption(f"병목: {log.get('bottleneck', '-')}")
                    st.caption(f"추론 모드: {log.get('mode','')}")

        # CSV 다운로드
        st.markdown("---")
        import csv
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["timestamp", "trace_id", "question", "정확도", "관련성",
                         "환각여부", "환각유형", "발생원인", "개선제안",
                         "total_latency_ms", "total_tokens", "bottleneck", "mode"])
        for log in logs:
            ev = log.get("evaluation", {})
            hc = log.get("hallucination_analysis") or {}
            writer.writerow([
                log.get("timestamp",""), log.get("trace_id",""),
                log.get("question",""),
                ev.get("정확도",""), ev.get("관련성",""), ev.get("환각여부",""),
                hc.get("환각_유형",""), hc.get("발생_원인",""), hc.get("개선_제안",""),
                log.get("total_latency_ms",""), log.get("total_tokens",{}).get("total",""),
                log.get("bottleneck",""), log.get("mode","")
            ])
        st.download_button(
            "⬇️ 분석 로그 CSV 다운로드",
            csv_buf.getvalue().encode("utf-8-sig"),
            file_name=f"rag_agent_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True
        )
