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
import pickle
import hashlib
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE         = os.path.join(_BASE, "rag_eval_logs_v16.json")
EMBED_CACHE_FILE = os.path.join(_BASE, "embed_cache_v16.pkl")

st.set_page_config(page_title="RAG 챗봇 v16", page_icon="📚", layout="wide")


# =====================================================================
# 쿼리 라우팅 시스템 프롬프트
# =====================================================================

ROUTING_SYSTEM_PROMPT = """\
너는 프로덕션 RAG 시스템의 쿼리 라우팅 및 검색 제어 엔진이다.
질문을 분석해 최적의 검색 전략을 결정하라.

의도 분류 기준:
- factual_lookup : 정확한 사실·수치·날짜 검색 → BM25 비중 높게, recall 강화
- definition     : 짧고 명확한 정의 질문 → 가벼운 검색으로 충분
- multi_hop      : 여러 문서를 연결해 추론 → query 분해 필수
- reasoning      : 검색 + 재정렬 + 종합 → reranker + dense 강화
- exploratory    : 넓은 의미 탐색 → 높은 recall, 다양한 청크
- ambiguous      : 의도 불분명 → recall 최우선 후 필터링

반드시 아래 JSON 형식으로만 출력하라:
{
  "의도": "<위 분류 중 하나>",
  "검색_전략": {
    "dense_weight": <0.0~1.0>,
    "bm25_weight": <0.0~1.0>,
    "reranker_사용여부": <true|false>,
    "reranker_모드": "<light|heavy>",
    "top_k": <2~5 정수>,
    "query_rewrite_필요": <true|false>,
    "query_분해_필요": <true|false>,
    "recall_우선순위": <true|false>
  },
  "메타데이터_전략": {
    "메타데이터_필터_사용": <true|false>,
    "선호_출처": [],
    "시간_가중치": "<최근|과거|없음>"
  },
  "설명": "<한 줄 이유>"
}"""


# =====================================================================
# Ablation Study 설정
# =====================================================================

ABLATION_CONFIGS = [
    {"id": "full",        "name": "✅ Full Pipeline",    "query_rewrite": True,  "bm25": True,  "rerank": True},
    {"id": "no_rewrite",  "name": "❌ No Query Rewrite", "query_rewrite": False, "bm25": True,  "rerank": True},
    {"id": "no_bm25",     "name": "❌ No BM25 (Dense만)", "query_rewrite": True,  "bm25": False, "rerank": True},
    {"id": "no_rerank",   "name": "❌ No Rerank",        "query_rewrite": True,  "bm25": True,  "rerank": False},
    {"id": "dense_rerank","name": "⚡ Dense + Rerank",   "query_rewrite": False, "bm25": False, "rerank": True},
    {"id": "minimal",     "name": "🔥 Minimal (기본만)", "query_rewrite": False, "bm25": False, "rerank": False},
]


# =====================================================================
# Tracer
# =====================================================================

class Tracer:
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
        return max(self.spans, key=lambda s: s["duration_ms"])["name"]


def _usage(response) -> dict:
    u = response.usage
    return {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
            "total": u.total_tokens}


# =====================================================================
# Embedding Cache
# =====================================================================

class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        self._cache: dict = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self._cache, f)

    def size(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path):
            os.remove(self.path)
        self.hits = self.misses = 0


embed_cache = EmbeddingCache(EMBED_CACHE_FILE)


def get_embeddings(texts: list) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in response.data], dtype=np.float32)


def get_embeddings_cached(texts: list) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    result_map: dict = {}
    need_api: list = []
    for text in texts:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in embed_cache._cache:
            result_map[text] = embed_cache._cache[key]
            embed_cache.hits += 1
        else:
            if text not in result_map:
                need_api.append(text)
    if need_api:
        embed_cache.misses += len(need_api)
        new_embs = get_embeddings(need_api)
        for text, emb in zip(need_api, new_embs):
            key = hashlib.md5(text.encode("utf-8")).hexdigest()
            embed_cache._cache[key] = emb
            result_map[text] = emb
        embed_cache._save()
    return np.array([result_map[t] for t in texts], dtype=np.float32)


def compute_ndcg(ordered_items: list, score_dict: dict, k: int) -> float:
    if not score_dict:
        return 0.0
    n = min(k, len(ordered_items))
    if n == 0:
        return 0.0
    dcg  = sum(score_dict.get(i, 0.0) / np.log2(i + 2) for i in range(n))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(score_dict.values(), reverse=True)[:n]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# =====================================================================
# [NEW v16] 검색 품질 리포트
# =====================================================================

# NDCG 품질 등급 경계
_NDCG_EXCELLENT = 0.9
_NDCG_GOOD      = 0.7
_NDCG_FAIR      = 0.5


def compute_search_quality_report(ndcg_prefilter: float, use_bm25: bool,
                                   n_candidates: int) -> dict:
    """
    [v16] 임베딩 검색 품질을 정량화해 리포트 반환.

    ndcg_prefilter : prefilter 출력 순서의 NDCG (리랭커 점수 기준) → Dense 검색 품질
    reranker_gain  : 리랭커가 개선할 수 있는 여지 = 1.0 - ndcg_prefilter
    quality_label  : excellent / good / fair / poor
    diagnosis      : 한 줄 진단 메시지
    """
    reranker_gain = round(1.0 - ndcg_prefilter, 4)

    if ndcg_prefilter >= _NDCG_EXCELLENT:
        quality_label = "excellent"
        diagnosis = "임베딩 검색이 이미 최적 순서 → 리랭커는 확인 역할만 수행"
    elif ndcg_prefilter >= _NDCG_GOOD:
        quality_label = "good"
        diagnosis = "임베딩 검색 품질 양호 → 리랭커가 소폭 개선"
    elif ndcg_prefilter >= _NDCG_FAIR:
        quality_label = "fair"
        diagnosis = "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미"
    else:
        quality_label = "poor"
        diagnosis = "임베딩 검색 품질 낮음 → 청킹 전략·임베딩 모델 개선 검토 필요"

    return {
        "ndcg_prefilter":  ndcg_prefilter,
        "reranker_gain":   reranker_gain,
        "quality_label":   quality_label,
        "diagnosis":       diagnosis,
        "use_bm25":        use_bm25,
        "n_candidates":    n_candidates,
    }


def linkify_citations(text: str) -> str:
    return re.sub(
        r'\[출처 (\d+)\]',
        lambda m: (
            f'<a href="#source-{m.group(1)}" '
            f'style="color:#1976D2;font-weight:bold;text-decoration:none;">'
            f'[출처 {m.group(1)}]</a>'
        ),
        text
    )


# =====================================================================
# 기본 유틸
# =====================================================================

def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def extract_text(file) -> str:
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


def build_index(chunks: list):
    embeddings = normalize(get_embeddings_cached(chunks))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# =====================================================================
# 쿼리 라우팅
# =====================================================================

def route_query(question: str, tracer: Tracer = None) -> dict:
    tracer and tracer.start("query_routing")
    resp_obj = None
    try:
        resp_obj = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user",   "content": f"사용자 질문: {question}"}
            ]
        )
        result = json.loads(resp_obj.choices[0].message.content)
    except Exception as e:
        result = {
            "의도": "ambiguous",
            "검색_전략": {"dense_weight": 0.5, "bm25_weight": 0.5,
                         "reranker_사용여부": True, "reranker_모드": "heavy",
                         "top_k": 3, "query_rewrite_필요": True,
                         "query_분해_필요": False, "recall_우선순위": True},
            "메타데이터_전략": {"메타데이터_필터_사용": False, "선호_출처": [], "시간_가중치": "없음"},
            "설명": f"라우팅 실패 → fallback: {str(e)}"
        }
    if tracer:
        strategy = result.get("검색_전략", {})
        tok = _usage(resp_obj) if resp_obj else {"prompt": 0, "completion": 0, "total": 0}
        tracer.end("query_routing", tokens=tok,
                   input_summary=question[:60],
                   output_summary=f"의도: {result.get('의도','-')} | top_k: {strategy.get('top_k','-')}",
                   decision=result.get("설명", ""))
    return result


def _apply_routing(route: dict, defaults: dict) -> dict:
    s = route.get("검색_전략", {})
    return {
        "use_bm25":          s.get("bm25_weight", 0.5) > 0.3,
        "use_reranking":     s.get("reranker_사용여부", defaults["use_reranking"]),
        "top_k":             int(s.get("top_k", defaults["top_k"])),
        "use_query_rewrite": s.get("query_rewrite_필요", defaults["use_query_rewrite"]),
    }


# =====================================================================
# 검색 파이프라인
# =====================================================================

def rewrite_queries(original_query: str, n: int = 3, tracer: Tracer = None,
                    use_session_cache: bool = True):
    if use_session_cache:
        cache_key = f"{original_query}||{n}"
        session_cache = st.session_state.get("rewrite_cache", {})
        if cache_key in session_cache:
            cached = session_cache[cache_key]
            if tracer:
                tracer.start("query_rewriting")
                tracer.end("query_rewriting",
                           input_summary=f"원본: {original_query[:60]}",
                           output_summary=f"캐시 히트 → {len(cached)}개 재사용",
                           decision="세션 캐시 히트")
            return cached

    tracer and tracer.start("query_rewriting")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 검색 전문가. 질문을 두 방식으로 재작성:\n"
                "1) 의도 분해형 (정의·원인·결과·방법·조건) 우선\n"
                "2) 표현 다양화 (동의어·구조 변경)\n"
                f"의도 분해형 우선 총 {n}개. 번호·기호 없이 줄당 하나."
            )},
            {"role": "user", "content": f"원본 질문: {original_query}"}
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    queries = [original_query] + variants[:n]

    if use_session_cache:
        cache_key = f"{original_query}||{n}"
        session_cache = st.session_state.get("rewrite_cache", {})
        session_cache[cache_key] = queries
        st.session_state.rewrite_cache = session_cache

    if tracer:
        tracer.end("query_rewriting", tokens=_usage(response),
                   input_summary=f"원본: {original_query[:60]}",
                   output_summary=f"쿼리 {len(queries)}개 생성",
                   decision=f"의도 분해형 우선 → 변형 {len(variants)}개")
    return queries


def retrieve_hybrid(queries: list, index, chunks: list, sources: list,
                    top_k_per_query: int = 20, use_bm25: bool = True,
                    tracer: Tracer = None) -> list:
    tracer and tracer.start("embedding_search")
    seen_dense, dense_items = set(), []
    for query in queries:
        query_emb = normalize(get_embeddings_cached([query]))
        _, indices = index.search(query_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen_dense:
                seen_dense.add(i)
                dense_items.append((chunks[i], sources[i] if sources else "알 수 없음"))

    if not use_bm25 or not BM25_AVAILABLE:
        if tracer:
            tracer.end("embedding_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                       input_summary=f"쿼리 {len(queries)}개",
                       output_summary=f"Dense only: {len(dense_items)}개",
                       decision="BM25 비활성화")
        return dense_items

    tokenized_corpus = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    seen_bm25, bm25_items = set(), []
    for query in queries:
        scores_arr = bm25.get_scores(query.split())
        top_idx = np.argsort(scores_arr)[::-1][:top_k_per_query]
        for i in top_idx:
            if i < len(chunks) and i not in seen_bm25:
                seen_bm25.add(i)
                bm25_items.append((chunks[i], sources[i] if sources else "알 수 없음"))

    RRF_K = 60
    rrf_scores: dict = {}
    item_by_key: dict = {}

    def _key(item):
        return hashlib.md5((item[0][:120] + item[1]).encode("utf-8")).hexdigest()

    for rank, item in enumerate(dense_items):
        k = _key(item)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
        item_by_key[k] = item
    for rank, item in enumerate(bm25_items):
        k = _key(item)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
        item_by_key[k] = item

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    result = [item_by_key[k] for k in sorted_keys]

    if tracer:
        tracer.end("embedding_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개",
                   output_summary=f"Dense {len(dense_items)} + BM25 {len(bm25_items)} → RRF {len(result)}개",
                   decision=f"RRF(k={RRF_K})")
    return result


def prefilter_by_similarity(query: str, items: list, top_n: int = 10,
                             tracer: Tracer = None) -> list:
    tracer and tracer.start("prefilter")
    if len(items) <= top_n:
        tracer and tracer.end("prefilter", input_summary=f"{len(items)}개",
                              output_summary=f"{len(items)}개 유지", decision="top_n 이하")
        return items
    chunk_texts = [item[0] for item in items]
    query_emb   = normalize(get_embeddings_cached([query]))[0]
    chunk_embs  = normalize(get_embeddings_cached(chunk_texts))
    sims        = chunk_embs @ query_emb
    top_idx     = np.argsort(sims)[::-1][:top_n]
    result      = [items[i] for i in top_idx]
    cutoff      = round(float(sims[top_idx[-1]]), 4)
    if tracer:
        tracer.end("prefilter", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {len(result)}개 (컷오프: {cutoff})",
                   decision=f"캐시 임베딩 코사인 ≥ {cutoff}")
    return result


def rerank_chunks(query: str, items: list, top_k: int = 3,
                  tracer: Tracer = None) -> tuple:
    tracer and tracer.start("rerank")
    if not items:
        return [], {}
    chunks_text = "\n\n".join([f"[{i+1}] {item[0]}" for i, item in enumerate(items)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문 대비 각 청크 관련성을 0~10점으로. 형식: 1: 8\n2: 3\n..."},
            {"role": "user",   "content": f"질문: {query}\n\n청크들:\n{chunks_text}"}
        ]
    )
    all_scores: dict = {}
    for line in response.choices[0].message.content.strip().split('\n'):
        parts = line.split(':')
        if len(parts) == 2:
            try:
                idx = int(parts[0].strip()) - 1
                sc  = float(parts[1].strip())
                if 0 <= idx < len(items):
                    all_scores[idx] = sc
            except ValueError:
                pass
    scored = [(items[i][0], items[i][1], all_scores.get(i, 0.0)) for i in range(len(items))]
    scored.sort(key=lambda x: x[2], reverse=True)
    result = scored[:top_k]
    if tracer:
        tracer.end("rerank", tokens=_usage(response),
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {top_k}개 (점수: {', '.join([f'{s[2]:.1f}' for s in result])})",
                   decision=f"LLM 0~10 채점 → 상위 {top_k}개")
    return result, all_scores


def step1_summarize_chunks(question, chunks, tracer: Tracer = None):
    tracer and tracer.start("step1_summarize")
    chunks_text = "\n\n".join([f"[{i+1}]\n{c}" for i, c in enumerate(chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문 관련해 각 청크 2~3문장 요약. 형식: [1]: 요약\n[2]: 요약\n..."},
            {"role": "user",   "content": f"질문: {question}\n\n청크들:\n{chunks_text}"}
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
        tracer.end("step1_summarize", tokens=_usage(response),
                   input_summary=f"{len(chunks)}개 청크", output_summary=f"{len(result)}개 요약",
                   decision="질문 관점 핵심 추출")
    return result


def step2_analyze_relationships(question, summaries, sources, tracer: Tracer = None):
    tracer and tracer.start("step2_analyze")
    summaries_text = "\n".join([f"[출처 {i+1} | {sources[i]}] {s}" for i, s in enumerate(summaries)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "청크 요약 구조화. 형식:\n**공통점**\n- ...\n\n**차이점**\n- ...\n\n**핵심 정보**\n- ...\n\n**불확실성 / 누락 정보**\n- ..."},
            {"role": "user",   "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        has_unc = "없음" not in result.split("**불확실성")[-1][:60] if "**불확실성" in result else False
        tracer.end("step2_analyze", tokens=_usage(response),
                   input_summary=f"{len(summaries)}개 요약",
                   output_summary="공통점/차이점/핵심/불확실성 완료",
                   decision=f"불확실성: {'있음' if has_unc else '없음'}")
    return result


def step3_generate_final_answer(question, chunks, summaries, analysis, tracer: Tracer = None):
    tracer and tracer.start("step3_answer")
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(chunks)])
    summaries_text   = "\n".join([f"[출처 {i+1}] {s}" for i, s in enumerate(summaries)])
    system_prompt = ("문서 기반 어시스턴트. 규칙: "
                     "1. **📌 요약** / **📖 근거** ([출처 N] 인용) / **✅ 결론** (확신도: 높음/보통/낮음). "
                     "2. 불확실성 있으면 확신도 반영. 3. 문서 밖 내용 금지. 4. 한국어.")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"[청크 요약]\n{summaries_text}\n\n[분석]\n{analysis}\n\n"
                f"[참고 문서 원문]\n{numbered_context}\n\n[질문]\n{question}"
            )}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("step3_answer", tokens=_usage(response),
                   input_summary="요약+분석+원문", output_summary=f"답변 {len(result)}자",
                   decision="Step1+Step2 힌트로 최종 답변")
    return result


def generate_answer_simple(question, items, tracer: Tracer = None):
    tracer and tracer.start("step3_answer")
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{item[0]}" for i, item in enumerate(items)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "문서 기반 어시스턴트. **📌 요약** / **📖 근거** / **✅ 결론** 구조. 한국어."},
            {"role": "user",   "content": f"[참고 문서]\n{numbered_context}\n\n[질문]\n{question}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("step3_answer", tokens=_usage(response),
                   input_summary="원문 직접", output_summary=f"답변 {len(result)}자",
                   decision="단순 모드")
    return result


# =====================================================================
# 평가 시스템 (v15 동일)
# =====================================================================

def evaluate_answer(question, context_chunks, answer, tracer: Tracer = None):
    tracer and tracer.start("evaluation")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "답변 품질 진단 전문가. 아래 형식으로만 출력하라.\n\n"
                "정확도: <1~5 정수, 5 초과 금지>\n"
                "관련성: <1~5 정수, 5 초과 금지>\n"
                "환각여부: <없음|부분적|있음>\n"
                "환각근거: <환각이 없으면 '없음', 있으면 구체적 설명>\n"
                "신뢰도: <높음|보통|낮음>\n"
                "불일치_항목: <문서와 답변이 어긋나는 항목, 없으면 '없음'>\n"
                "누락_정보: <문서에 있지만 답변에 빠진 핵심 정보, 없으면 '없음'>\n"
                "개선_제안: <답변 품질을 높이기 위한 구체적 제안 1문장>"
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ]
    )
    result = {
        "정확도": 0, "관련성": 0, "환각여부": "알 수 없음", "환각근거": "",
        "신뢰도": "보통", "불일치_항목": "없음", "누락_정보": "없음", "개선_제안": "",
    }
    for line in response.choices[0].message.content.strip().split('\n'):
        if line.startswith("정확도:"):
            try:
                result["정확도"] = max(1, min(5, int(re.sub(r'[^0-9]', '', line.split(':', 1)[1].strip()))))
            except ValueError:
                pass
        elif line.startswith("관련성:"):
            try:
                result["관련성"] = max(1, min(5, int(re.sub(r'[^0-9]', '', line.split(':', 1)[1].strip()))))
            except ValueError:
                pass
        elif line.startswith("환각여부:"):
            result["환각여부"] = line.split(':', 1)[1].strip()
        elif line.startswith("환각근거:"):
            result["환각근거"] = line.split(':', 1)[1].strip()
        elif line.startswith("신뢰도:"):
            result["신뢰도"] = line.split(':', 1)[1].strip()
        elif line.startswith("불일치_항목:"):
            result["불일치_항목"] = line.split(':', 1)[1].strip()
        elif line.startswith("누락_정보:"):
            result["누락_정보"] = line.split(':', 1)[1].strip()
        elif line.startswith("개선_제안:"):
            result["개선_제안"] = line.split(':', 1)[1].strip()
    if tracer:
        tracer.end("evaluation", tokens=_usage(response),
                   input_summary="질문+문서+답변",
                   output_summary=f"정확도 {result['정확도']}/5 · 관련성 {result['관련성']}/5 · 신뢰도 {result['신뢰도']}",
                   decision="구조화 품질 진단")
    return result


def analyze_hallucination_cause(question, context_chunks, answer, hall_type, tracer: Tracer = None):
    if hall_type == "없음":
        return None
    tracer and tracer.start("hallucination_analysis")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "환각 원인 정밀 분석 전문가. 아래 형식으로만 출력하라.\n\n"
                "환각_주장: <답변에서 사실과 다른 구체적 주장>\n"
                "환각_유형: <fabrication|distortion|over-generalization>\n"
                "심각도: <1~10 정수, 10이 가장 심각>\n"
                "근거_출처: <어느 문서 어느 부분과 충돌하는지>\n"
                "원문_인용: <문서의 실제 내용 직접 인용 (최대 2문장)>\n"
                "발생_원인: <insufficient_context|ambiguous_chunk|llm_interpolation>\n"
                "개선_제안: <재발 방지를 위한 구체적 파이프라인 개선 제안>\n"
                "수정_제안: <이 답변에서 해당 환각 부분을 어떻게 수정해야 하는지>"
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ]
    )
    raw = response.choices[0].message.content.strip()
    result = {
        "환각_주장": "", "환각_유형": "", "심각도": 0,
        "근거_출처": "", "원문_인용": "", "발생_원인": "",
        "개선_제안": "", "수정_제안": "", "raw": raw,
    }
    for line in raw.split('\n'):
        for key in ["환각_주장", "환각_유형", "근거_출처", "원문_인용", "발생_원인", "개선_제안", "수정_제안"]:
            if line.startswith(f"{key}:"):
                result[key] = line.split(':', 1)[1].strip()
        if line.startswith("심각도:"):
            try:
                result["심각도"] = max(1, min(10, int(re.sub(r'[^0-9]', '', line.split(':', 1)[1].strip()))))
            except ValueError:
                pass
    if tracer:
        tracer.end("hallucination_analysis", tokens=_usage(response),
                   input_summary=f"환각: {hall_type} | 유형: {result['환각_유형']}",
                   output_summary=f"심각도: {result['심각도']}/10 | 원인: {result['발생_원인']}",
                   decision="환각 감지 시에만 실행")
    return result


def build_quality_report(evaluation: dict, hall_cause: dict = None) -> dict:
    acc  = evaluation.get("정확도", 0)
    rel  = evaluation.get("관련성", 0)
    hall = evaluation.get("환각여부", "없음")
    hall_penalty = {"없음": 0.0, "부분적": 0.5, "있음": 1.5}.get(hall, 0.0)
    overall = round(max(0.0, min(5.0, (acc + rel) / 2 - hall_penalty)), 2)
    grade = "A" if overall >= 4.5 else "B" if overall >= 3.5 else "C" if overall >= 2.5 else "D" if overall >= 1.5 else "F"

    issues = []
    if acc < 3:
        issues.append(f"낮은 정확도 ({acc}/5)")
    if rel < 3:
        issues.append(f"낮은 관련성 ({rel}/5)")
    if hall != "없음":
        severity = (hall_cause or {}).get("심각도", "-")
        issues.append(f"환각 감지: {hall} (심각도 {severity}/10)")
    mismatch = evaluation.get("불일치_항목", "없음")
    if mismatch and mismatch != "없음":
        issues.append(f"문서-답변 불일치: {mismatch[:60]}")
    missing = evaluation.get("누락_정보", "없음")
    if missing and missing != "없음":
        issues.append(f"누락 정보: {missing[:60]}")
    if evaluation.get("신뢰도") == "낮음":
        issues.append("전반적 신뢰도 낮음")

    recommendations = []
    if evaluation.get("개선_제안"):
        recommendations.append(evaluation["개선_제안"])
    if hall_cause and hall_cause.get("수정_제안"):
        recommendations.append(f"[환각 수정] {hall_cause['수정_제안']}")
    if hall_cause and hall_cause.get("개선_제안"):
        recommendations.append(f"[파이프라인] {hall_cause['개선_제안']}")

    return {
        "overall_score":   overall,
        "grade":           grade,
        "issues":          issues,
        "recommendations": recommendations,
    }


# =====================================================================
# Ablation Study 파이프라인 (v16: search_quality_report 포함)
# =====================================================================

def run_single_config(question: str, config: dict,
                      index, chunks: list, sources: list,
                      top_k: int = 3, prefilter_n: int = 10) -> dict:
    tracer = Tracer()
    t_start = time.time()
    try:
        if config["query_rewrite"]:
            queries = rewrite_queries(question, n=2, tracer=tracer, use_session_cache=False)
        else:
            queries = [question]

        candidates = retrieve_hybrid(queries, index, chunks, sources,
                                      top_k_per_query=15,
                                      use_bm25=config["bm25"] and BM25_AVAILABLE,
                                      tracer=tracer)
        filtered = prefilter_by_similarity(question, candidates, prefilter_n, tracer)

        ndcg_prefilter = None
        sqr = None
        if config["rerank"] and len(filtered) > top_k:
            ranked, all_scores = rerank_chunks(question, filtered, top_k, tracer)
            ndcg_prefilter = compute_ndcg(filtered, all_scores, k=top_k)
            sqr = compute_search_quality_report(ndcg_prefilter, config["bm25"], len(filtered))
        else:
            ranked = [(item[0], item[1], None) for item in filtered[:top_k]]

        final_chunks  = [r[0] for r in ranked]
        final_sources = [r[1] for r in ranked]

        summaries  = step1_summarize_chunks(question, final_chunks, tracer)
        analysis   = step2_analyze_relationships(question, summaries, final_sources, tracer)
        answer     = step3_generate_final_answer(question, final_chunks, summaries, analysis, tracer)
        evaluation = evaluate_answer(question, final_chunks, answer, tracer)
        qr         = build_quality_report(evaluation)

        return {
            "config_id":           config["id"],
            "config_name":         config["name"],
            "query_rewrite":       config["query_rewrite"],
            "bm25":                config["bm25"],
            "rerank":              config["rerank"],
            "accuracy":            evaluation.get("정확도", 0),
            "relevance":           evaluation.get("관련성", 0),
            "hallucination":       evaluation.get("환각여부", "-"),
            "confidence":          evaluation.get("신뢰도", "-"),
            "overall_score":       qr["overall_score"],
            "grade":               qr["grade"],
            "ndcg_prefilter":      ndcg_prefilter,                    # [NEW v16]
            "reranker_gain":       sqr["reranker_gain"] if sqr else None,  # [NEW v16]
            "search_quality":      sqr["quality_label"] if sqr else None,  # [NEW v16]
            "latency_ms":          tracer.total_latency_ms(),
            "total_tokens":        tracer.total_tokens()["total"],
            "answer":              answer,
            "error":               None,
        }
    except Exception as e:
        return {
            "config_id": config["id"], "config_name": config["name"],
            "query_rewrite": config["query_rewrite"], "bm25": config["bm25"], "rerank": config["rerank"],
            "accuracy": 0, "relevance": 0, "hallucination": "-", "confidence": "-",
            "overall_score": 0, "grade": "F",
            "ndcg_prefilter": None, "reranker_gain": None, "search_quality": None,
            "latency_ms": int((time.time() - t_start) * 1000),
            "total_tokens": 0, "answer": "", "error": str(e),
        }


# =====================================================================
# 로그
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
                    evaluation, hall_cause, tracer: Tracer, mode,
                    ndcg=None, route_decision=None,
                    quality_report=None, search_quality_report=None):  # [NEW v16]
    sqr = search_quality_report or {}
    return {
        "trace_id":              tracer.trace_id,
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":              question,
        "queries":               queries,
        "retrieved_chunks": [
            {"text": c[:200] + ("..." if len(c) > 200 else ""), "source": s,
             "rerank_score": round(sc, 2) if sc is not None else None}
            for c, s, sc in ranked_items
        ],
        "answer":                answer,
        "evaluation":            evaluation,
        "hallucination_analysis": hall_cause,
        "quality_report":        quality_report,
        "search_quality_report": sqr,                          # [NEW v16]
        "ndcg_at_k":             ndcg,
        "reranker_gain":         sqr.get("reranker_gain"),     # [NEW v16]
        "spans":                 tracer.spans,
        "total_tokens":          tracer.total_tokens(),
        "total_latency_ms":      tracer.total_latency_ms(),
        "bottleneck":            tracer.bottleneck(),
        "mode":                  mode,
        "route_decision":        route_decision,
        "embed_cache_size":      embed_cache.size(),
        "embed_cache_hits":      embed_cache.hits,
        "embed_cache_misses":    embed_cache.misses,
    }


# =====================================================================
# 세션 초기화
# =====================================================================
for key, default in [
    ("messages", []), ("index", None), ("chunks", []),
    ("chunk_sources", []), ("rewrite_cache", {}),
    ("ablation_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# 탭 레이아웃 (5탭)
# =====================================================================
tab_chat, tab_trace, tab_agent, tab_ablation, tab_search = st.tabs(
    ["💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation Study", "🔍 검색 품질 분석"]
)


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("📚 RAG 챗봇 v16")
    st.markdown("---")

    uploaded_files = st.file_uploader("파일을 업로드하세요", type=["pdf", "txt"], accept_multiple_files=True)
    chunking_mode  = st.radio("청킹 방식", ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"])
    chunk_size     = st.slider("청크 크기", 200, 1000, 500, 100)
    overlap        = st.slider("Overlap 크기", 0, 200, 100, 20)

    if uploaded_files:
        if st.button("문서 처리하기", use_container_width=True, type="primary"):
            all_chunks, all_sources = [], []
            with st.spinner("문서 처리 중..."):
                for f in uploaded_files:
                    text = extract_text(f)
                    fn   = chunk_text_semantic if "Semantic" in chunking_mode else chunk_text_with_overlap
                    fc   = fn(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(fc)
                    all_sources.extend([f.name] * len(fc))
                st.session_state.chunks        = all_chunks
                st.session_state.chunk_sources = all_sources
                st.session_state.index         = build_index(all_chunks)
            st.success(f"완료! {len(all_chunks)}개 청크")

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크")
        for fname, cnt in Counter(st.session_state.chunk_sources).items():
            st.caption(f"  └ {fname}: {cnt}개")

    st.markdown("---")
    auto_routing = st.toggle("🧭 자동 쿼리 라우팅", value=True)
    st.caption("수동 설정 (라우팅 OFF 시)")
    use_query_rewrite = st.toggle("쿼리 리라이팅", value=True,  disabled=auto_routing)
    num_rewrites      = st.slider("리라이팅 수", 1, 5, 3,       disabled=auto_routing or not use_query_rewrite)
    if BM25_AVAILABLE:
        use_bm25 = st.toggle("Hybrid 검색 (BM25)", value=True, disabled=auto_routing)
    else:
        st.warning("rank_bm25 미설치 → `pip install rank-bm25`")
        use_bm25 = False
    prefilter_n   = st.slider("Pre-filter 수", 5, 20, 10)
    use_reranking = st.toggle("리랭킹", value=True,             disabled=auto_routing)
    top_k         = st.slider("최종 청크 수 (top_k)", 1, 5, 3)
    use_multidoc  = st.toggle("멀티문서 추론", value=True)
    auto_evaluate = st.toggle("자동 평가 + 로깅", value=True)

    st.markdown("---")
    st.caption(f"💾 임베딩 캐시: {embed_cache.size()}개")
    hit_total = embed_cache.hits + embed_cache.misses
    if hit_total > 0:
        st.caption(f"   히트율: {embed_cache.hits*100//hit_total}%")

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("문서 초기화", use_container_width=True):
        st.session_state.update({"messages": [], "index": None, "chunks": [], "chunk_sources": []})
        st.rerun()
    if st.button("로그 초기화", use_container_width=True):
        os.path.exists(LOG_FILE) and os.remove(LOG_FILE)
        st.rerun()
    if st.button("임베딩 캐시 초기화", use_container_width=True):
        embed_cache.clear()
        st.rerun()
    st.caption("v16: 검색 품질 정량 측정 시스템")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================
with tab_chat:
    st.title("💬 문서 기반 챗봇 v16")
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
                route_decision = None
                defaults = {"use_bm25": use_bm25, "use_reranking": use_reranking,
                            "top_k": top_k, "use_query_rewrite": use_query_rewrite}
                with st.spinner("답변 생성 중..."):
                    eff = defaults.copy()
                    if auto_routing:
                        with st.status("🧭 쿼리 라우팅...", expanded=False) as s:
                            route_decision = route_query(prompt, tracer)
                            intent = route_decision.get("의도", "-")
                            eff = _apply_routing(route_decision, defaults)
                            s.update(label=f"✅ 의도: {intent} | top_k: {eff['top_k']}", state="complete")

                    if eff["use_query_rewrite"]:
                        with st.status("🔄 쿼리 리라이팅...", expanded=False) as s:
                            queries = rewrite_queries(prompt, n=num_rewrites, tracer=tracer)
                            s.update(label=f"✅ 쿼리 {len(queries)}개", state="complete")
                    else:
                        queries = [prompt]

                    candidate_items = retrieve_hybrid(
                        queries, st.session_state.index,
                        st.session_state.chunks, st.session_state.chunk_sources,
                        top_k_per_query=20, use_bm25=eff["use_bm25"], tracer=tracer
                    )

                    with st.status("⚡ Pre-filter...", expanded=False) as s:
                        filtered = prefilter_by_similarity(prompt, candidate_items, prefilter_n, tracer)
                        s.update(label=f"✅ {len(filtered)}개 선별", state="complete")

                    ndcg_k = None
                    sqr    = None
                    eff_top_k = eff["top_k"]
                    if eff["use_reranking"] and len(filtered) > eff_top_k:
                        with st.status("🔄 리랭킹...", expanded=False) as s:
                            ranked, all_scores = rerank_chunks(prompt, filtered, eff_top_k, tracer)
                            ndcg_k = compute_ndcg(filtered, all_scores, k=eff_top_k)
                            sqr    = compute_search_quality_report(ndcg_k, eff["use_bm25"], len(filtered))
                            s.update(
                                label=(
                                    f"✅ 상위 {eff_top_k}개 | "
                                    f"Dense NDCG@{eff_top_k}: {ndcg_k:.3f} | "
                                    f"Reranker Gain: {sqr['reranker_gain']:.3f} | "
                                    f"{sqr['quality_label']}"
                                ),
                                state="complete"
                            )
                    else:
                        ranked = [(item[0], item[1], None) for item in filtered[:eff_top_k]]
                        tracer.start("rerank")
                        tracer.end("rerank", input_summary=f"{len(filtered)}개",
                                   output_summary=f"{len(ranked)}개 (OFF)", decision="리랭킹 비활성화")

                    final_chunks  = [r[0] for r in ranked]
                    final_sources = [r[1] for r in ranked]
                    final_scores  = [r[2] for r in ranked]

                    if use_multidoc:
                        with st.status("📝 Step1...", expanded=False) as s:
                            summaries = step1_summarize_chunks(prompt, final_chunks, tracer)
                            s.update(label="✅ Step1 완료", state="complete")
                        with st.status("🔍 Step2...", expanded=False) as s:
                            analysis = step2_analyze_relationships(prompt, summaries, final_sources, tracer)
                            s.update(label="✅ Step2 완료", state="complete")
                        with st.status("✍️ Step3...", expanded=False) as s:
                            response = step3_generate_final_answer(prompt, final_chunks, summaries, analysis, tracer)
                            s.update(label="✅ Step3 완료", state="complete")
                        mode = "multidoc"
                    else:
                        response = generate_answer_simple(prompt, ranked, tracer)
                        summaries, analysis = [], ""
                        mode = "simple"

                    evaluation, hall_cause, quality_report = {}, None, None
                    if auto_evaluate:
                        with st.status("🧪 평가 중...", expanded=False) as s:
                            evaluation = evaluate_answer(prompt, final_chunks, response, tracer)
                            hall = evaluation.get("환각여부", "없음")
                            if hall != "없음":
                                hall_cause = analyze_hallucination_cause(
                                    prompt, final_chunks, response, hall, tracer)
                            quality_report = build_quality_report(evaluation, hall_cause)
                            save_log(build_log_entry(
                                prompt, queries, ranked, response,
                                evaluation, hall_cause, tracer, mode,
                                ndcg=ndcg_k, route_decision=route_decision,
                                quality_report=quality_report,
                                search_quality_report=sqr))
                            grade = quality_report["grade"]
                            score = quality_report["overall_score"]
                            s.update(label=f"✅ 정확도 {evaluation.get('정확도','-')}/5 · 관련성 {evaluation.get('관련성','-')}/5 · 등급 {grade}({score})", state="complete")

                st.markdown(linkify_citations(response), unsafe_allow_html=True)

                if evaluation:
                    hall = evaluation.get("환각여부", "")
                    hc   = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else "🔴")
                    conf = evaluation.get("신뢰도", "-")
                    cols = st.columns(7)
                    cols[0].metric("⏱ 응답",    f"{tracer.total_latency_ms()/1000:.1f}s")
                    cols[1].metric("🔤 토큰",    f"{tracer.total_tokens()['total']:,}")
                    cols[2].metric("📐 정확도",  f"{evaluation.get('정확도','-')}/5")
                    cols[3].metric("🎯 관련성",  f"{evaluation.get('관련성','-')}/5")
                    cols[4].metric(f"{hc} 환각", hall)
                    cols[5].metric("🧭 신뢰도",  conf)
                    # [NEW v16] Dense NDCG + Reranker Gain 병렬 표시
                    if sqr:
                        cols[6].metric(
                            f"🔍 Dense NDCG@{eff_top_k}",
                            f"{ndcg_k:.3f}",
                            delta=f"Gain {sqr['reranker_gain']:.3f}"
                        )
                    else:
                        cols[6].metric(f"📊 NDCG@{eff_top_k}", "-")

                    # [NEW v16] 검색 품질 리포트 expander
                    if sqr:
                        label_color = {"excellent": "🟢", "good": "🔵", "fair": "🟡", "poor": "🔴"}.get(sqr["quality_label"], "⚪")
                        with st.expander(f"🔍 검색 품질 리포트 — {label_color} {sqr['quality_label'].upper()} (Dense NDCG {ndcg_k:.3f} / Reranker Gain {sqr['reranker_gain']:.3f})"):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Dense NDCG@k",   f"{sqr['ndcg_prefilter']:.4f}")
                            c2.metric("Reranker Gain",  f"{sqr['reranker_gain']:.4f}",
                                      help="= 1.0 − Dense NDCG. 리랭커가 순서를 얼마나 개선했는가")
                            c3.metric("후보 수",         f"{sqr['n_candidates']}개")
                            st.info(f"📋 {sqr['diagnosis']}")

                    if quality_report:
                        grade = quality_report["grade"]
                        score = quality_report["overall_score"]
                        grade_color = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(grade, "⚪")
                        with st.expander(f"🩺 품질 리포트 — 등급 {grade_color} {grade} (종합점수 {score}/5)"):
                            if quality_report["issues"]:
                                st.markdown("**⚠️ 감지된 문제**")
                                for issue in quality_report["issues"]:
                                    st.markdown(f"- {issue}")
                            if quality_report["recommendations"]:
                                st.markdown("**💡 개선 제안**")
                                for rec in quality_report["recommendations"]:
                                    st.markdown(f"- {rec}")
                            if evaluation.get("불일치_항목") and evaluation["불일치_항목"] != "없음":
                                st.markdown(f"**🔍 문서-답변 불일치**: {evaluation['불일치_항목']}")
                            if evaluation.get("누락_정보") and evaluation["누락_정보"] != "없음":
                                st.markdown(f"**📭 누락 정보**: {evaluation['누락_정보']}")

                    if hall_cause:
                        severity = hall_cause.get("심각도", "-")
                        sev_color = "🔴" if isinstance(severity, int) and severity >= 7 else ("🟡" if isinstance(severity, int) and severity >= 4 else "🟢")
                        st.error(
                            f"{sev_color} **심각도 {severity}/10** | "
                            f"유형: {hall_cause.get('환각_유형','')} | "
                            f"원인: {hall_cause.get('발생_원인','')} | "
                            f"원문: _{hall_cause.get('원문_인용','')[:80]}_"
                        )
                        if hall_cause.get("수정_제안"):
                            st.warning(f"🔧 수정 제안: {hall_cause['수정_제안']}")

                if route_decision:
                    with st.expander("🧭 쿼리 라우팅 결정"):
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.markdown(f"**의도**: `{route_decision.get('의도','-')}`  \n**설명**: {route_decision.get('설명','-')}")
                            st.json(route_decision.get("검색_전략", {}))
                        with col_r2:
                            st.markdown("**적용된 파라미터**")
                            st.json(eff)

                if eff["use_query_rewrite"]:
                    with st.expander("🔍 생성된 쿼리"):
                        for i, q in enumerate(queries):
                            st.markdown(f"{'**[원본]**' if i == 0 else f'**[변형 {i}]**'} {q}")
                if use_multidoc and summaries:
                    with st.expander("📝 Step1 청크 요약"):
                        for i, (s, src, sc) in enumerate(zip(summaries, final_sources, final_scores)):
                            st.caption(f"[출처 {i+1}] {src}" + (f" | {sc:.1f}/10" if sc else ""))
                            st.info(s)
                            if i < len(summaries) - 1:
                                st.divider()
                    with st.expander("🔍 Step2 분석"):
                        st.markdown(analysis)

                st.markdown("**📄 출처 원문**")
                for i, (c, src, sc) in enumerate(zip(final_chunks, final_sources, final_scores)):
                    st.markdown(f'<div id="source-{i+1}"></div>', unsafe_allow_html=True)
                    st.caption(f"[출처 {i+1}] **{src}**" + (f" _(관련성 {sc:.1f}/10)_" if sc else ""))
                    st.write(c)
                    if i < len(final_chunks) - 1:
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})


# =====================================================================
# TAB 2 — 트레이싱 (v15 동일)
# =====================================================================
with tab_trace:
    st.title("🔬 트레이싱 대시보드")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        total_tokens_all = [l.get("total_tokens", {}).get("total", 0) for l in logs]
        total_lat_all    = [l.get("total_latency_ms", 0) for l in logs]
        ndcg_vals        = [l["ndcg_at_k"] for l in logs if l.get("ndcg_at_k") is not None]
        avg_ndcg         = round(sum(ndcg_vals)/len(ndcg_vals), 3) if ndcg_vals else None
        intent_counts    = Counter((l.get("route_decision") or {}).get("의도", "미분류") for l in logs)
        last = logs[-1]
        hits, miss = last.get("embed_cache_hits", 0), last.get("embed_cache_misses", 0)
        hit_rate = f"{hits*100//(hits+miss)}%" if (hits+miss) > 0 else "-"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("총 트레이스", len(logs))
        c2.metric("평균 응답",   f"{sum(total_lat_all)/len(total_lat_all)/1000:.1f}s")
        c3.metric("평균 토큰",   f"{int(sum(total_tokens_all)/len(total_tokens_all)):,}")
        c4.metric("캐시 히트율", hit_rate)
        c5.metric("평균 NDCG@k", str(avg_ndcg) if avg_ndcg else "-")
        c6.metric("주요 의도",   intent_counts.most_common(1)[0][0] if intent_counts else "-")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("응답 시간 추이 (초)")
            st.line_chart([l/1000 for l in total_lat_all])
        with col2:
            st.subheader("토큰 사용량 추이")
            st.line_chart(total_tokens_all)

        if ndcg_vals:
            st.subheader("NDCG@k 추이 (Dense Retrieval 품질)")
            st.line_chart([l.get("ndcg_at_k") or 0 for l in logs])

        if intent_counts:
            st.subheader("🧭 라우팅 의도 분포")
            st.bar_chart(dict(intent_counts))

        st.markdown("---")
        st.subheader("트레이스 상세 (최신 순)")
        for log in reversed(logs):
            spans    = log.get("spans", [])
            total_ms = log.get("total_latency_ms", 1)
            tok      = log.get("total_tokens", {})
            ts       = log.get("timestamp", "")
            q        = log.get("question", "")[:50]
            nd       = log.get("ndcg_at_k")
            rg       = log.get("reranker_gain")
            intent   = (log.get("route_decision") or {}).get("의도", "-")

            with st.expander(
                f"[{ts}] {q}... | ⏱ {total_ms/1000:.1f}s | 🔤 {tok.get('total',0):,} | {intent}"
                + (f" | NDCG {nd:.3f}" if nd else "")
                + (f" | Gain {rg:.3f}" if rg else "")
            ):
                for span in spans:
                    dur = span["duration_ms"]
                    pct = dur / total_ms if total_ms > 0 else 0
                    c1, c2, c3 = st.columns([2, 6, 1])
                    c1.caption(span["name"] + (" ❌" if span.get("error") else ""))
                    c2.progress(min(pct, 1.0))
                    c3.caption(f"{dur}ms")
                st.markdown("---")
                tok_data = {s["name"]: s["tokens"]["total"] for s in spans if s["tokens"]["total"] > 0}
                if tok_data:
                    st.bar_chart(tok_data)
                flow_rows = [{"단계": s["name"], "소요(ms)": s["duration_ms"], "토큰": s["tokens"]["total"],
                              "입력": s.get("input_summary","")[:60], "출력": s.get("output_summary","")[:60]}
                             for s in spans]
                if flow_rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)


# =====================================================================
# TAB 3 — 에이전트 분석 (v15 동일)
# =====================================================================
with tab_agent:
    st.title("🧠 에이전트 분석 대시보드")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        import pandas as pd

        eval_logs   = [l for l in logs if l.get("evaluation")]
        avg_acc     = round(sum(l["evaluation"].get("정확도",0) for l in eval_logs)/len(eval_logs), 2) if eval_logs else 0
        avg_rel     = round(sum(l["evaluation"].get("관련성",0) for l in eval_logs)/len(eval_logs), 2) if eval_logs else 0
        hall_counts = Counter(l["evaluation"].get("환각여부","") for l in eval_logs)
        hall_logs   = [l for l in eval_logs if l["evaluation"].get("환각여부","없음") != "없음"]
        ndcg_vals   = [l["ndcg_at_k"] for l in logs if l.get("ndcg_at_k") is not None]
        avg_ndcg    = round(sum(ndcg_vals)/len(ndcg_vals), 3) if ndcg_vals else None
        qr_logs     = [l for l in logs if l.get("quality_report")]
        avg_overall = round(sum(l["quality_report"]["overall_score"] for l in qr_logs)/len(qr_logs), 2) if qr_logs else None
        grade_counts= Counter(l["quality_report"]["grade"] for l in qr_logs)
        conf_counts = Counter(l["evaluation"].get("신뢰도", "보통") for l in eval_logs)
        hall_none_pct = round(hall_counts.get("없음",0)/len(eval_logs)*100) if eval_logs else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("평균 정확도",    f"{avg_acc}/5")
        c2.metric("평균 관련성",    f"{avg_rel}/5")
        c3.metric("환각 없음 비율", f"{hall_none_pct}%")
        c4.metric("환각 감지",      len(hall_logs))
        c5.metric("평균 NDCG@k",   str(avg_ndcg) if avg_ndcg else "-")
        c6.metric("평균 종합점수",  f"{avg_overall}/5" if avg_overall else "-")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("환각 여부 분포")
            st.bar_chart({"없음": hall_counts.get("없음",0), "부분적": hall_counts.get("부분적",0), "있음": hall_counts.get("있음",0)})
        with col2:
            st.subheader("정확도 / 관련성 추이")
            if len(eval_logs) >= 2:
                st.line_chart({"정확도": [l["evaluation"].get("정확도",0) for l in eval_logs],
                               "관련성": [l["evaluation"].get("관련성",0) for l in eval_logs]})
            else:
                st.caption("2건 이상 필요")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🧭 신뢰도 분포")
            if conf_counts:
                st.bar_chart({"높음": conf_counts.get("높음",0),
                              "보통": conf_counts.get("보통",0),
                              "낮음": conf_counts.get("낮음",0)})
        with col4:
            st.subheader("🏅 답변 등급 분포")
            if grade_counts:
                st.bar_chart(dict(sorted(grade_counts.items())))

        if qr_logs and len(qr_logs) >= 2:
            st.subheader("📈 종합 품질 점수 추이 (overall_score)")
            st.line_chart([l["quality_report"]["overall_score"] for l in qr_logs])

        if hall_logs:
            st.markdown("---")
            cause_counts  = Counter((l.get("hallucination_analysis") or {}).get("환각_유형","미분석") for l in hall_logs)
            root_counts   = Counter((l.get("hallucination_analysis") or {}).get("발생_원인","미분석") for l in hall_logs)
            severity_vals = [int((l.get("hallucination_analysis") or {}).get("심각도", 0) or 0) for l in hall_logs]
            avg_severity  = round(sum(severity_vals)/len(severity_vals), 1) if severity_vals else 0

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown("**환각 유형 분포**")
                if cause_counts:
                    st.bar_chart(dict(cause_counts))
            with cc2:
                st.markdown("**발생 원인 분포**")
                if root_counts:
                    st.bar_chart(dict(root_counts))
            with cc3:
                st.markdown("**평균 심각도**")
                st.metric("환각 평균 심각도", f"{avg_severity}/10",
                          delta="높음 주의" if avg_severity >= 7 else ("보통" if avg_severity >= 4 else "낮음"))

        st.markdown("---")
        for log in reversed(logs[:10]):
            ev     = log.get("evaluation", {})
            qr     = log.get("quality_report", {})
            hall   = ev.get("환각여부", "-")
            hi     = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else ("🔴" if hall == "있음" else "⚪"))
            ts     = log.get("timestamp", "")
            q      = log.get("question","")[:45]
            nd     = log.get("ndcg_at_k")
            rg     = log.get("reranker_gain")
            intent = (log.get("route_decision") or {}).get("의도","-")
            grade  = (qr or {}).get("grade", "-")
            score  = (qr or {}).get("overall_score", "-")

            with st.expander(
                f"[{ts}] {q}... | {ev.get('정확도','-')}/5 · {ev.get('관련성','-')}/5 · "
                f"{hi} {hall} | 등급 {grade}({score}) | {intent}"
                + (f" | NDCG {nd:.3f}" if nd else "")
                + (f" | Gain {rg:.3f}" if rg else "")
            ):
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    if log.get("route_decision"):
                        rd = log["route_decision"]
                        st.markdown(f"**🧭 라우팅**: `{rd.get('의도','-')}` — {rd.get('설명','-')}")
                    st.markdown("**💬 최종 답변**")
                    st.markdown(log.get("answer",""))
                    if qr and (qr.get("issues") or qr.get("recommendations")):
                        st.markdown("---")
                        if qr.get("issues"):
                            st.markdown("**⚠️ 감지된 문제**")
                            for issue in qr["issues"]:
                                st.markdown(f"  - {issue}")
                        if qr.get("recommendations"):
                            st.markdown("**💡 개선 제안**")
                            for rec in qr["recommendations"]:
                                st.markdown(f"  - {rec}")
                    ha = log.get("hallucination_analysis")
                    if ha:
                        st.markdown("---")
                        st.markdown("**🔬 환각 상세 분석**")
                        cols_ha = st.columns(2)
                        cols_ha[0].markdown(f"유형: `{ha.get('환각_유형','-')}`")
                        cols_ha[0].markdown(f"심각도: `{ha.get('심각도','-')}/10`")
                        cols_ha[1].markdown(f"발생 원인: `{ha.get('발생_원인','-')}`")
                        if ha.get("원문_인용"):
                            st.caption(f"📄 원문 인용: _{ha['원문_인용'][:120]}_")
                        if ha.get("수정_제안"):
                            st.warning(f"🔧 {ha['수정_제안']}")
                with col_b:
                    if ev:
                        st.metric("정확도", f"{ev.get('정확도','-')}/5")
                        st.metric("관련성", f"{ev.get('관련성','-')}/5")
                        st.metric("신뢰도", ev.get("신뢰도", "-"))
                        st.metric("환각",   f"{hi} {hall}")
                    if qr:
                        st.metric("종합점수", f"{qr.get('overall_score','-')}/5")
                        st.metric("등급",    qr.get("grade", "-"))
                    st.metric("응답 시간", f"{log.get('total_latency_ms',0)/1000:.1f}s")
                    if nd:
                        st.metric("Dense NDCG", f"{nd:.3f}")
                    if rg is not None:
                        st.metric("Reranker Gain", f"{rg:.3f}")


# =====================================================================
# TAB 4 — Ablation Study (v16: 검색 품질 지표 추가)
# =====================================================================
with tab_ablation:
    st.title("🧬 Ablation Study")
    st.caption("동일 질문에 여러 파이프라인 구성을 자동 실행해 성능 기여도를 비교합니다.")

    if st.session_state.index is None:
        st.info("먼저 사이드바에서 문서를 업로드하고 처리해주세요.")
    else:
        st.markdown("---")
        st.subheader("🔬 실험 설정")

        abl_question = st.text_input(
            "실험 질문 (동일 질문으로 모든 config 실행)",
            placeholder="예: 계약 해지 절차는 어떻게 되나요?",
        )

        config_names   = [c["name"] for c in ABLATION_CONFIGS]
        selected_names = st.multiselect("실험할 Config 선택", options=config_names, default=config_names)
        selected_configs = [c for c in ABLATION_CONFIGS if c["name"] in selected_names]

        abl_top_k       = st.slider("Ablation top_k", 1, 5, 3, key="abl_top_k")
        abl_prefilter_n = st.slider("Ablation pre-filter 수", 5, 20, 10, key="abl_pf")

        n_configs = len(selected_configs)
        st.info(f"선택된 config {n_configs}개 × 약 6 LLM 호출 = **약 {n_configs*6}회 API 호출** 예상.")

        run_btn = st.button("🧪 Ablation 실행", type="primary",
                            disabled=not abl_question or not selected_configs,
                            use_container_width=True)

        if run_btn and abl_question:
            results = []
            progress_bar = st.progress(0)
            status_text  = st.empty()
            for i, config in enumerate(selected_configs):
                status_text.markdown(f"**실행 중** ({i+1}/{n_configs}): `{config['name']}`")
                results.append(run_single_config(
                    abl_question, config,
                    st.session_state.index, st.session_state.chunks, st.session_state.chunk_sources,
                    top_k=abl_top_k, prefilter_n=abl_prefilter_n
                ))
                progress_bar.progress((i + 1) / n_configs)
            status_text.markdown("✅ **모든 실험 완료**")
            st.session_state.ablation_results = results

        if st.session_state.ablation_results:
            results = st.session_state.ablation_results
            import pandas as pd

            st.markdown("---")
            st.subheader("📊 실험 결과 비교")

            df = pd.DataFrame([{
                "Config":           r["config_name"],
                "정확도 (/5)":      r["accuracy"],
                "관련성 (/5)":      r["relevance"],
                "신뢰도":           r.get("confidence", "-"),
                "종합점수 (/5)":    r.get("overall_score", "-"),
                "등급":             r.get("grade", "-"),
                "환각여부":         r["hallucination"],
                "Dense NDCG":       round(r["ndcg_prefilter"], 3) if r.get("ndcg_prefilter") else "-",  # [NEW v16]
                "Reranker Gain":    round(r["reranker_gain"], 3) if r.get("reranker_gain") else "-",     # [NEW v16]
                "검색 품질":        r.get("search_quality", "-"),                                         # [NEW v16]
                "응답(ms)":         r["latency_ms"],
                "토큰":             r["total_tokens"],
                "오류":             "❌" if r.get("error") else "✅",
            } for r in results])

            st.dataframe(df, use_container_width=True, hide_index=True)

            valid = [r for r in results if not r.get("error")]
            if valid:
                st.markdown("---")
                st.subheader("🔍 자동 분석 결과")

                best_acc    = max(valid, key=lambda r: r["accuracy"])
                best_rel    = max(valid, key=lambda r: r["relevance"])
                best_score  = max(valid, key=lambda r: r.get("overall_score", 0))
                fastest     = min(valid, key=lambda r: r["latency_ms"])
                cheapest    = min(valid, key=lambda r: r["total_tokens"])

                # [NEW v16] 검색 품질 기준 최고 config
                rerank_valid = [r for r in valid if r.get("ndcg_prefilter") is not None]
                best_ndcg    = max(rerank_valid, key=lambda r: r["ndcg_prefilter"]) if rerank_valid else None

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("🏆 최고 정확도",   best_acc["config_name"].split()[-1],
                            delta=f"정확도 {best_acc['accuracy']}/5")
                col2.metric("🎯 최고 관련성",   best_rel["config_name"].split()[-1],
                            delta=f"관련성 {best_rel['relevance']}/5")
                col3.metric("🥇 최고 종합점수", best_score["config_name"].split()[-1],
                            delta=f"등급 {best_score.get('grade','-')} ({best_score.get('overall_score','-')})")
                col4.metric("⚡ 최저 지연",     f"{fastest['latency_ms']/1000:.1f}s",
                            delta=fastest["config_name"].replace("✅","").replace("❌","").replace("⚡","").strip())
                if best_ndcg:  # [NEW v16]
                    col5.metric("🔍 최고 Dense NDCG",
                                f"{best_ndcg['ndcg_prefilter']:.3f}",
                                delta=best_ndcg["config_name"].replace("✅","").replace("❌","").replace("⚡","").strip())
                else:
                    col5.metric("💰 최소 토큰", f"{cheapest['total_tokens']:,}")

                st.markdown("**📋 분석 요약**")
                lines = []
                if best_acc["config_id"] == "full":
                    lines.append("- Full Pipeline이 정확도 1위 → 모든 구성 요소가 품질에 기여")
                else:
                    lines.append(f"- **{best_acc['config_name']}** 이 Full Pipeline보다 정확도 높음 → 해당 구성 요소 조합 권장")

                full_result = next((r for r in valid if r["config_id"] == "full"), None)
                no_rewrite  = next((r for r in valid if r["config_id"] == "no_rewrite"), None)
                no_bm25     = next((r for r in valid if r["config_id"] == "no_bm25"), None)
                no_rerank   = next((r for r in valid if r["config_id"] == "no_rerank"), None)

                if full_result and no_rewrite:
                    diff = full_result["accuracy"] - no_rewrite["accuracy"]
                    lines.append(f"- 쿼리 리라이팅 기여도: 정확도 +{diff:+.1f} ({'유효' if diff > 0 else '미미'})")
                if full_result and no_bm25:
                    diff = full_result["accuracy"] - no_bm25["accuracy"]
                    # [NEW v16] BM25의 Dense NDCG 영향도 비교
                    ndcg_diff = (full_result.get("ndcg_prefilter") or 0) - (no_bm25.get("ndcg_prefilter") or 0)
                    lines.append(f"- BM25 하이브리드 기여도: 정확도 +{diff:+.1f} / Dense NDCG +{ndcg_diff:+.3f} ({'유효' if diff > 0 or ndcg_diff > 0.05 else '미미'})")
                if full_result and no_rerank:
                    diff = full_result["accuracy"] - no_rerank["accuracy"]
                    lines.append(f"- 리랭킹 기여도: 정확도 +{diff:+.1f} ({'유효' if diff > 0 else '미미'})")
                if rerank_valid:
                    avg_ndcg_ab = round(sum(r["ndcg_prefilter"] for r in rerank_valid) / len(rerank_valid), 3)
                    avg_gain_ab = round(sum(r["reranker_gain"] for r in rerank_valid) / len(rerank_valid), 3)
                    lines.append(f"- 실험 평균 Dense NDCG: {avg_ndcg_ab} / 평균 Reranker Gain: {avg_gain_ab}")
                no_hall = [r for r in valid if r["hallucination"] == "없음"]
                if no_hall:
                    lines.append(f"- 환각 없음 config: {', '.join([r['config_name'] for r in no_hall])}")

                for line in lines:
                    st.markdown(line)

                st.markdown("---")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("**정확도 / 관련성 / 종합점수 비교**")
                    chart_data = pd.DataFrame({
                        "Config":   [r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip() for r in valid],
                        "정확도":   [r["accuracy"] for r in valid],
                        "관련성":   [r["relevance"] for r in valid],
                        "종합점수": [r.get("overall_score", 0) for r in valid],
                    }).set_index("Config")
                    st.bar_chart(chart_data)

                with col_c2:
                    if rerank_valid:  # [NEW v16] Dense NDCG / Reranker Gain 비교
                        st.markdown("**Dense NDCG / Reranker Gain 비교**")
                        ndcg_chart = pd.DataFrame({
                            "Config":        [r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip() for r in rerank_valid],
                            "Dense NDCG":    [r["ndcg_prefilter"] for r in rerank_valid],
                            "Reranker Gain": [r["reranker_gain"] for r in rerank_valid],
                        }).set_index("Config")
                        st.bar_chart(ndcg_chart)
                    else:
                        st.markdown("**응답 시간 비교 (ms)**")
                        st.bar_chart({r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip(): r["latency_ms"] for r in valid})

                st.markdown("---")
                st.subheader("💬 Config별 답변 비교")
                for r in results:
                    err_tag   = f" ❌ {r['error']}" if r.get("error") else ""
                    ndcg_tag  = f" | Dense NDCG {r['ndcg_prefilter']:.3f}" if r.get("ndcg_prefilter") else ""
                    grade_tag = f" | 등급 {r.get('grade','-')}({r.get('overall_score','-')})" if r.get("grade") else ""
                    with st.expander(
                        f"{r['config_name']} | 정확도 {r['accuracy']}/5 · 관련성 {r['relevance']}/5 · {r['latency_ms']}ms{ndcg_tag}{grade_tag}{err_tag}"
                    ):
                        col_ans, col_meta = st.columns([3, 1])
                        with col_ans:
                            st.markdown(r.get("answer", "오류로 답변 없음"))
                        with col_meta:
                            st.metric("정확도",      f"{r['accuracy']}/5")
                            st.metric("관련성",      f"{r['relevance']}/5")
                            st.metric("신뢰도",      r.get("confidence", "-"))
                            st.metric("종합점수",    f"{r.get('overall_score','-')}/5")
                            st.metric("등급",        r.get("grade", "-"))
                            st.metric("환각",        r["hallucination"])
                            if r.get("ndcg_prefilter"):
                                st.metric("Dense NDCG",    f"{r['ndcg_prefilter']:.3f}")
                                st.metric("Reranker Gain", f"{r['reranker_gain']:.3f}")
                                st.metric("검색 품질",     r.get("search_quality", "-"))
                            st.metric("응답",        f"{r['latency_ms']/1000:.1f}s")
                            st.metric("토큰",        f"{r['total_tokens']:,}")

                st.markdown("---")
                import csv as csv_module
                abl_buf = io.StringIO()
                abl_writer = csv_module.writer(abl_buf)
                abl_writer.writerow(["config_name","query_rewrite","bm25","rerank",
                                     "accuracy","relevance","confidence","overall_score","grade",
                                     "hallucination","ndcg_prefilter","reranker_gain","search_quality",
                                     "latency_ms","total_tokens","error"])
                for r in results:
                    abl_writer.writerow([
                        r["config_name"], r["query_rewrite"], r["bm25"], r["rerank"],
                        r["accuracy"], r["relevance"], r.get("confidence",""), r.get("overall_score",""), r.get("grade",""),
                        r["hallucination"], r.get("ndcg_prefilter",""), r.get("reranker_gain",""), r.get("search_quality",""),
                        r["latency_ms"], r["total_tokens"], r.get("error","")
                    ])
                st.download_button(
                    "⬇️ Ablation 결과 CSV 다운로드",
                    abl_buf.getvalue().encode("utf-8-sig"),
                    file_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True
                )


# =====================================================================
# [NEW v16] TAB 5 — 검색 품질 분석
# =====================================================================
with tab_search:
    st.title("🔍 검색 품질 분석")
    st.caption(
        "Dense Retrieval(임베딩 검색)의 순서 품질을 NDCG@k로 측정하고, "
        "리랭커가 그 위에서 얼마나 개선했는지 정량적으로 분석합니다."
    )

    logs = load_logs()
    sq_logs = [l for l in logs if l.get("ndcg_at_k") is not None]

    if not sq_logs:
        st.info("리랭킹이 활성화된 질문이 아직 없습니다. 챗봇에서 리랭킹을 켠 후 질문해보세요.")
    else:
        import pandas as pd

        ndcg_vals    = [l["ndcg_at_k"] for l in sq_logs]
        gain_vals    = [l["reranker_gain"] for l in sq_logs if l.get("reranker_gain") is not None]
        sqr_vals     = [l.get("search_quality_report", {}) for l in sq_logs]

        avg_ndcg     = round(sum(ndcg_vals)/len(ndcg_vals), 4)
        avg_gain     = round(sum(gain_vals)/len(gain_vals), 4) if gain_vals else None
        excellent_n  = sum(1 for v in ndcg_vals if v >= _NDCG_EXCELLENT)
        poor_n       = sum(1 for v in ndcg_vals if v < _NDCG_FAIR)
        bm25_on_ndcg = [l["ndcg_at_k"] for l in sq_logs if (l.get("search_quality_report") or {}).get("use_bm25")]
        bm25_off_ndcg= [l["ndcg_at_k"] for l in sq_logs if not (l.get("search_quality_report") or {}).get("use_bm25")]

        # 요약 지표
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("평균 Dense NDCG",   f"{avg_ndcg:.4f}")
        c2.metric("평균 Reranker Gain", f"{avg_gain:.4f}" if avg_gain else "-",
                  help="= 1.0 − Dense NDCG. 리랭커가 순서를 개선한 여지")
        c3.metric("Excellent (≥0.9)",   f"{excellent_n}건")
        c4.metric("Poor (<0.5)",         f"{poor_n}건",
                  delta="개선 필요" if poor_n > 0 else None,
                  delta_color="inverse")
        c5.metric("총 측정 건수",        f"{len(sq_logs)}건")

        st.markdown("---")

        # Dense NDCG 추이 + Reranker Gain 추이
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Dense NDCG@k 추이")
            st.line_chart(ndcg_vals)
            st.caption("1.0에 가까울수록 임베딩 검색이 이미 최적 순서를 만들고 있음")
        with col2:
            if gain_vals:
                st.subheader("📉 Reranker Gain 추이")
                st.line_chart(gain_vals)
                st.caption("값이 클수록 리랭커가 순서를 크게 개선했음 (임베딩 품질 보완 효과)")

        st.markdown("---")

        # NDCG 분포 + BM25 비교
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📊 NDCG 품질 등급 분포")
            buckets = {
                "Excellent (≥0.9)": sum(1 for v in ndcg_vals if v >= _NDCG_EXCELLENT),
                "Good (0.7~0.9)":   sum(1 for v in ndcg_vals if _NDCG_GOOD <= v < _NDCG_EXCELLENT),
                "Fair (0.5~0.7)":   sum(1 for v in ndcg_vals if _NDCG_FAIR <= v < _NDCG_GOOD),
                "Poor (<0.5)":      sum(1 for v in ndcg_vals if v < _NDCG_FAIR),
            }
            st.bar_chart(buckets)

        with col4:
            st.subheader("🆚 BM25 ON vs OFF — Dense NDCG 비교")
            if bm25_on_ndcg and bm25_off_ndcg:
                avg_on  = round(sum(bm25_on_ndcg)/len(bm25_on_ndcg), 4)
                avg_off = round(sum(bm25_off_ndcg)/len(bm25_off_ndcg), 4)
                st.bar_chart({"BM25 ON": avg_on, "BM25 OFF": avg_off})
                st.caption(f"BM25 ON 평균: {avg_on} / OFF 평균: {avg_off}")
            elif bm25_on_ndcg:
                st.info(f"BM25 ON 데이터만 있음 (평균 NDCG: {round(sum(bm25_on_ndcg)/len(bm25_on_ndcg),4)})")
            elif bm25_off_ndcg:
                st.info(f"BM25 OFF 데이터만 있음 (평균 NDCG: {round(sum(bm25_off_ndcg)/len(bm25_off_ndcg),4)})")
            else:
                st.caption("BM25 ON/OFF 비교 데이터 없음")

        st.markdown("---")

        # 의도별 평균 NDCG
        intent_ndcg: dict = {}
        for l in sq_logs:
            intent = (l.get("route_decision") or {}).get("의도", "미분류")
            intent_ndcg.setdefault(intent, []).append(l["ndcg_at_k"])
        if intent_ndcg:
            st.subheader("🧭 의도(Intent)별 평균 Dense NDCG")
            avg_by_intent = {k: round(sum(v)/len(v), 4) for k, v in intent_ndcg.items()}
            st.bar_chart(avg_by_intent)
            st.caption("특정 의도에서 NDCG가 낮으면 해당 질문 유형에 맞는 청킹 전략 검토 필요")

        st.markdown("---")

        # 저품질 질문 목록
        poor_logs = [l for l in sq_logs if l["ndcg_at_k"] < _NDCG_GOOD]
        if poor_logs:
            st.subheader(f"⚠️ 검색 품질 낮은 질문 (NDCG < {_NDCG_GOOD}) — {len(poor_logs)}건")
            st.caption("이 질문들은 임베딩 검색 순서가 부정확 → 청킹 크기·임베딩 모델·문서 전처리 개선 검토")
            for l in sorted(poor_logs, key=lambda x: x["ndcg_at_k"]):
                nd  = l["ndcg_at_k"]
                rg  = l.get("reranker_gain", "-")
                ts  = l.get("timestamp", "")
                q   = l.get("question", "")
                sqr = l.get("search_quality_report", {})
                label_color = "🔴" if nd < _NDCG_FAIR else "🟡"
                with st.expander(f"{label_color} NDCG {nd:.4f} | Gain {rg:.3f if isinstance(rg, float) else rg} | [{ts}] {q[:60]}..."):
                    c_a, c_b = st.columns([3, 1])
                    with c_a:
                        st.markdown(f"**질문**: {q}")
                        st.info(f"🩺 {sqr.get('diagnosis', '진단 없음')}")
                    with c_b:
                        st.metric("Dense NDCG",    f"{nd:.4f}")
                        st.metric("Reranker Gain", f"{rg:.4f}" if isinstance(rg, float) else str(rg))
                        st.metric("후보 수",        sqr.get("n_candidates", "-"))
                        st.metric("BM25 사용",      "✅" if sqr.get("use_bm25") else "❌")
        else:
            st.success(f"모든 질문의 Dense NDCG ≥ {_NDCG_GOOD} — 검색 품질 양호 ✅")

        # 전체 데이터 테이블
        st.markdown("---")
        st.subheader("📋 전체 검색 품질 데이터")
        rows = []
        for l in sq_logs:
            sqr = l.get("search_quality_report", {})
            rows.append({
                "시간":          l.get("timestamp",""),
                "질문":          l.get("question","")[:50],
                "Dense NDCG":    l["ndcg_at_k"],
                "Reranker Gain": l.get("reranker_gain", "-"),
                "품질 등급":      sqr.get("quality_label", "-"),
                "BM25":          "✅" if sqr.get("use_bm25") else "❌",
                "후보 수":        sqr.get("n_candidates", "-"),
                "의도":          (l.get("route_decision") or {}).get("의도", "-"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # CSV 다운로드
        import csv as csv_module
        sq_buf = io.StringIO()
        sq_writer = csv_module.writer(sq_buf)
        sq_writer.writerow(["timestamp","question","ndcg_prefilter","reranker_gain",
                             "quality_label","use_bm25","n_candidates","intent"])
        for row in rows:
            sq_writer.writerow([
                row["시간"], row["질문"], row["Dense NDCG"], row["Reranker Gain"],
                row["품질 등급"], row["BM25"], row["후보 수"], row["의도"]
            ])
        st.download_button(
            "⬇️ 검색 품질 데이터 CSV 다운로드",
            sq_buf.getvalue().encode("utf-8-sig"),
            file_name=f"search_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True
        )
