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
LOG_FILE          = os.path.join(_BASE, "rag_eval_logs_v19.json")
EMBED_CACHE_FILE  = os.path.join(_BASE, "embed_cache_v19.pkl")
ANSWER_CACHE_FILE = os.path.join(_BASE, "answer_cache_v19.json")

ANSWER_CACHE_TTL_SEC = 1800   # 30분
QUERY_CACHE_TTL_SEC  = 3600   # 1시간

st.set_page_config(page_title="RAG 챗봇 v19", page_icon="📚", layout="wide")


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
# Fallback 설정
# =====================================================================

MAX_RETRIES          = 2
FALLBACK_MIN_ACC     = 3
FALLBACK_HALL_TYPES  = ("부분적", "있음")


# =====================================================================
# Dynamic Retrieval 프로필 (v18)
# =====================================================================

DYNAMIC_RETRIEVAL_PROFILES = {
    "definition":     {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": False, "label": "정의형 → BM25 ↑, 빠른 처리"},
    "factual_lookup": {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": None,  "label": "Fact형 → Rerank 강화"},
    "reasoning":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True,  "label": "분석형 → MultiDoc ↑, Rerank 강화"},
    "multi_hop":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 8, "top_k_delta": 1, "multidoc_override": True,  "label": "멀티홉 → MultiDoc 강제, 후보 확장"},
    "exploratory":    {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True,  "label": "탐색형 → BM25 ↑, 넓은 Recall"},
    "ambiguous":      {"bm25_boost": True,  "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 0, "multidoc_override": None,  "label": "의도불명 → BM25 + 강한 Rerank"},
}


def should_fallback(evaluation: dict) -> tuple:
    acc  = evaluation.get("정확도", 5)
    hall = evaluation.get("환각여부", "없음")
    reasons = []
    if acc < FALLBACK_MIN_ACC:
        reasons.append(f"정확도 {acc}/5 < {FALLBACK_MIN_ACC}")
    if hall in FALLBACK_HALL_TYPES:
        reasons.append(f"환각 감지: {hall}")
    return (bool(reasons), " + ".join(reasons))


def escalate_params(base_eff: dict, base_prefilter_n: int, attempt: int) -> tuple:
    new_eff = base_eff.copy()
    new_eff["use_bm25"]          = True
    new_eff["use_reranking"]     = True
    new_eff["use_query_rewrite"] = True
    new_eff["top_k"]             = min(5, base_eff["top_k"] + attempt)
    new_prefilter_n              = min(20, base_prefilter_n + attempt * 5)
    return new_eff, new_prefilter_n


def apply_dynamic_retrieval(intent: str, eff: dict, prefilter_n: int, use_multidoc: bool) -> tuple:
    profile = DYNAMIC_RETRIEVAL_PROFILES.get(intent)
    if not profile:
        return eff, prefilter_n, use_multidoc, None
    new_eff = eff.copy()
    if profile["bm25_boost"] and BM25_AVAILABLE:
        new_eff["use_bm25"] = True
    if profile["rerank_force"]:
        new_eff["use_reranking"] = True
    if profile["top_k_delta"]:
        new_eff["top_k"] = min(5, eff["top_k"] + profile["top_k_delta"])
    new_pf_n     = min(20, prefilter_n + profile["prefilter_delta"])
    new_multidoc = use_multidoc if profile["multidoc_override"] is None else profile["multidoc_override"]
    return new_eff, new_pf_n, new_multidoc, profile["label"]


# =====================================================================
# [NEW v19] 캐시 전략 — QueryResultCache / AnswerCache
# =====================================================================

class QueryResultCache:
    """
    [v19] 검색 결과 캐시 (세션 기반, TTL 있음).
    동일 질문 + 동일 설정 → 임베딩 검색·BM25·prefilter 스킵.
    """
    def __init__(self):
        self.hits   = 0
        self.misses = 0

    def _store(self) -> dict:
        if "query_result_cache_store" not in st.session_state:
            st.session_state.query_result_cache_store = {}
        return st.session_state.query_result_cache_store

    def _key(self, question: str, use_bm25: bool, prefilter_n: int) -> str:
        return hashlib.md5(f"{question}|{use_bm25}|{prefilter_n}".encode()).hexdigest()

    def get(self, question: str, use_bm25: bool, prefilter_n: int,
            ttl: int = QUERY_CACHE_TTL_SEC) -> list | None:
        store = self._store()
        key   = self._key(question, use_bm25, prefilter_n)
        if key in store:
            entry = store[key]
            if time.time() - entry["ts"] < ttl:
                self.hits += 1
                return entry["items"]
            del store[key]
        self.misses += 1
        return None

    def set(self, question: str, use_bm25: bool, prefilter_n: int, items: list):
        store = self._store()
        key   = self._key(question, use_bm25, prefilter_n)
        store[key] = {"items": items, "ts": time.time()}

    def size(self) -> int:
        return len(self._store())

    def clear(self):
        st.session_state.query_result_cache_store = {}
        self.hits = self.misses = 0


class AnswerCache:
    """
    [v19] 답변 캐시 (파일 기반, 짧은 TTL).
    동일 질문 → 모든 LLM 생성·평가 단계 스킵. 비용 절감 핵심.
    """
    def __init__(self, path: str):
        self.path = path
        self._cache: dict = self._load()
        self.hits   = 0
        self.misses = 0

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def get(self, key: str, ttl: int = ANSWER_CACHE_TTL_SEC) -> dict | None:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < ttl:
                self.hits += 1
                return entry["data"]
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, data: dict):
        self._cache[key] = {"data": data, "ts": time.time()}
        self._save()

    def size(self) -> int:
        return len(self._cache)

    def valid_size(self, ttl: int = ANSWER_CACHE_TTL_SEC) -> int:
        now = time.time()
        return sum(1 for v in self._cache.values() if now - v["ts"] < ttl)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path):
            os.remove(self.path)
        self.hits = self.misses = 0


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


embed_cache        = EmbeddingCache(EMBED_CACHE_FILE)
query_result_cache = QueryResultCache()
answer_cache       = AnswerCache(ANSWER_CACHE_FILE)


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


_NDCG_EXCELLENT = 0.9
_NDCG_GOOD      = 0.7
_NDCG_FAIR      = 0.5


def compute_search_quality_report(ndcg_prefilter: float, use_bm25: bool, n_candidates: int) -> dict:
    reranker_gain = round(1.0 - ndcg_prefilter, 4)
    if ndcg_prefilter >= _NDCG_EXCELLENT:
        quality_label, diagnosis = "excellent", "임베딩 검색이 이미 최적 순서 → 리랭커는 확인 역할만 수행"
    elif ndcg_prefilter >= _NDCG_GOOD:
        quality_label, diagnosis = "good",      "임베딩 검색 품질 양호 → 리랭커가 소폭 개선"
    elif ndcg_prefilter >= _NDCG_FAIR:
        quality_label, diagnosis = "fair",      "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미"
    else:
        quality_label, diagnosis = "poor",      "임베딩 검색 품질 낮음 → 청킹 전략·임베딩 모델 개선 검토 필요"
    return {"ndcg_prefilter": ndcg_prefilter, "reranker_gain": reranker_gain,
            "quality_label": quality_label, "diagnosis": diagnosis,
            "use_bm25": use_bm25, "n_candidates": n_candidates}


def linkify_citations(text: str) -> str:
    return re.sub(
        r'\[출처 (\d+)\]',
        lambda m: f'<a href="#source-{m.group(1)}" style="color:#1976D2;font-weight:bold;text-decoration:none;">[출처 {m.group(1)}]</a>',
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
# [NEW v19] 키워드 추출 (LLM 미사용, TF 기반)
# =====================================================================

_STOPWORDS = {
    "이", "가", "을", "를", "의", "에", "는", "은", "과", "와", "도", "로", "으로",
    "에서", "에게", "부터", "까지", "그", "이것", "저", "것", "및", "등", "위해",
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "with", "by", "from", "this", "that", "have", "has", "be",
}


def extract_keywords_simple(text: str, top_k: int = 10) -> str:
    words    = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text)
    filtered = [w for w in words if w.lower() not in _STOPWORDS]
    counts   = Counter(filtered)
    return " ".join([w for w, _ in counts.most_common(top_k)])


# =====================================================================
# [NEW v19] Multi-Vector Index 구축
# =====================================================================

def build_multi_vector_index(chunks: list) -> dict:
    """
    [v19] 청크당 3종 벡터 생성:
    ① 청크 전체 텍스트 (dense)
    ② 개별 문장 벡터 (sentence-level)
    ③ 키워드 문자열 벡터 (keyword)
    """
    # ① Chunk-level
    chunk_embs = normalize(get_embeddings_cached(chunks))
    chunk_idx  = faiss.IndexFlatIP(chunk_embs.shape[1])
    chunk_idx.add(chunk_embs)

    # ② Sentence-level
    all_sents:    list = []
    sent_to_chunk: list = []
    for ci, chunk in enumerate(chunks):
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        if not sents:
            sents = [chunk[:200]]
        for s in sents:
            all_sents.append(s)
            sent_to_chunk.append(ci)

    sent_embs = normalize(get_embeddings_cached(all_sents))
    sent_idx  = faiss.IndexFlatIP(sent_embs.shape[1])
    sent_idx.add(sent_embs)

    # ③ Keyword-level
    kw_strings = [extract_keywords_simple(c, top_k=10) for c in chunks]
    kw_embs    = normalize(get_embeddings_cached(kw_strings))
    kw_idx     = faiss.IndexFlatIP(kw_embs.shape[1])
    kw_idx.add(kw_embs)

    return {
        "chunk_index":   chunk_idx,
        "sent_index":    sent_idx,
        "kw_index":      kw_idx,
        "sent_to_chunk": sent_to_chunk,
        "n_chunks":      len(chunks),
        "n_sentences":   len(all_sents),
    }


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
# 검색 파이프라인 단계들
# =====================================================================

def rewrite_queries(original_query: str, n: int = 3, tracer: Tracer = None,
                    use_session_cache: bool = True):
    if use_session_cache:
        cache_key     = f"{original_query}||{n}"
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
    queries  = [original_query] + variants[:n]

    if use_session_cache:
        cache_key     = f"{original_query}||{n}"
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


# =====================================================================
# [NEW v19] Multi-Vector Retrieval
# =====================================================================

def retrieve_multi_vector(queries: list, mv_index: dict,
                           chunks: list, sources: list,
                           top_k_per_query: int = 20,
                           use_bm25: bool = True,
                           tracer: Tracer = None) -> list:
    """
    [v19] 청크·문장·키워드 3개 인덱스 검색 후 RRF 통합.
    BM25도 추가 레이어로 포함.
    """
    tracer and tracer.start("embedding_search")

    chunk_idx      = mv_index["chunk_index"]
    sent_idx       = mv_index["sent_index"]
    kw_idx         = mv_index["kw_index"]
    sent_to_chunk  = mv_index["sent_to_chunk"]
    n_chunks       = mv_index["n_chunks"]
    n_sents        = mv_index["n_sentences"]

    RRF_K = 60
    rrf_scores: dict = {}

    for query in queries:
        q_emb = normalize(get_embeddings_cached([query]))

        # ① Chunk-level search
        _, cr = chunk_idx.search(q_emb, min(top_k_per_query, n_chunks))
        for rank, ci in enumerate(cr[0]):
            if 0 <= ci < n_chunks:
                rrf_scores[ci] = rrf_scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)

        # ② Sentence-level → 부모 청크로 매핑
        _, sr = sent_idx.search(q_emb, min(top_k_per_query * 2, n_sents))
        sent_rank_map: dict = {}
        for rank, si in enumerate(sr[0]):
            if 0 <= si < len(sent_to_chunk):
                ci = sent_to_chunk[si]
                if ci not in sent_rank_map:
                    sent_rank_map[ci] = rank
        for ci, rank in sent_rank_map.items():
            rrf_scores[ci] = rrf_scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)

        # ③ Keyword-level search
        kw_query = extract_keywords_simple(query, top_k=8)
        if kw_query.strip():
            kq_emb = normalize(get_embeddings_cached([kw_query]))
            _, kr  = kw_idx.search(kq_emb, min(top_k_per_query, n_chunks))
            for rank, ci in enumerate(kr[0]):
                if 0 <= ci < n_chunks:
                    rrf_scores[ci] = rrf_scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)

    # ④ BM25 (선택)
    if use_bm25 and BM25_AVAILABLE:
        tokenized_corpus = [c.split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        for query in queries:
            scores_arr = bm25.get_scores(query.split())
            top_idx = np.argsort(scores_arr)[::-1][:top_k_per_query]
            for rank, ci in enumerate(top_idx):
                if ci < n_chunks:
                    rrf_scores[ci] = rrf_scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)

    sorted_ci = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    result = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]

    if tracer:
        tracer.end("embedding_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개",
                   output_summary=f"Multi-Vector (청크+문장+키워드{'+BM25' if use_bm25 else ''}) → {len(result)}개",
                   decision="3-벡터 RRF 통합")
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


# =====================================================================
# [NEW v19] Context Compression — 핵심 문장 추출
# =====================================================================

def compress_chunks(question: str, chunks: list,
                    max_sentences: int = 5,
                    min_sim: float = 0.25,
                    tracer: Tracer = None) -> tuple:
    """
    [v19] 각 청크를 문장 단위로 분해 후 질문 유사도 기반 핵심 문장만 추출.
    LLM 호출 없음 — 임베딩 코사인 유사도로 필터링.
    Returns: (compressed_chunks, compression_stats)
    """
    tracer and tracer.start("context_compression")

    q_emb = normalize(get_embeddings_cached([question]))[0]
    compressed = []
    stats      = []

    for chunk in chunks:
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        if len(sents) <= 2:
            compressed.append(chunk)
            stats.append({"original": len(chunk), "compressed": len(chunk), "ratio": 1.0})
            continue

        sent_embs = normalize(get_embeddings_cached(sents))
        sims      = sent_embs @ q_emb

        qualified = [(i, float(sims[i])) for i in range(len(sents)) if sims[i] >= min_sim]
        if not qualified:
            qualified = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)[:2]

        qualified  = sorted(qualified, key=lambda x: x[1], reverse=True)[:max_sentences]
        keep_idx   = sorted([i for i, _ in qualified])
        comp_text  = " ".join(sents[i] for i in keep_idx)

        compressed.append(comp_text)
        ratio = round(len(comp_text) / max(len(chunk), 1), 2)
        stats.append({"original": len(chunk), "compressed": len(comp_text), "ratio": ratio})

    if tracer:
        orig_total = sum(s["original"] for s in stats)
        comp_total = sum(s["compressed"] for s in stats)
        avg_ratio  = round(comp_total / max(orig_total, 1), 2)
        tracer.end("context_compression",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(chunks)}개 청크 ({orig_total}자)",
                   output_summary=f"압축 후 {comp_total}자 (평균 {avg_ratio:.0%})",
                   decision=f"문장 유사도 ≥ {min_sim} + 상위 {max_sentences}문장")
    return compressed, stats


# =====================================================================
# 답변 생성 단계들
# =====================================================================

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
        elif line.startswith("환각여부:"):   result["환각여부"]   = line.split(':', 1)[1].strip()
        elif line.startswith("환각근거:"):   result["환각근거"]   = line.split(':', 1)[1].strip()
        elif line.startswith("신뢰도:"):     result["신뢰도"]     = line.split(':', 1)[1].strip()
        elif line.startswith("불일치_항목:"): result["불일치_항목"] = line.split(':', 1)[1].strip()
        elif line.startswith("누락_정보:"):  result["누락_정보"]  = line.split(':', 1)[1].strip()
        elif line.startswith("개선_제안:"):  result["개선_제안"]  = line.split(':', 1)[1].strip()
    if tracer:
        tracer.end("evaluation", tokens=_usage(response),
                   input_summary="질문+문서+답변",
                   output_summary=f"정확도 {result['정확도']}/5 · 환각 {result['환각여부']} · 신뢰도 {result['신뢰도']}",
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
                   input_summary=f"환각: {hall_type}",
                   output_summary=f"심각도: {result['심각도']}/10 | 원인: {result['발생_원인']}",
                   decision="환각 감지 시에만 실행")
    return result


def build_quality_report(evaluation: dict, hall_cause: dict = None) -> dict:
    acc  = evaluation.get("정확도", 0)
    rel  = evaluation.get("관련성", 0)
    hall = evaluation.get("환각여부", "없음")
    hall_penalty = {"없음": 0.0, "부분적": 0.5, "있음": 1.5}.get(hall, 0.0)
    overall = round(max(0.0, min(5.0, (acc + rel) / 2 - hall_penalty)), 2)
    grade   = "A" if overall >= 4.5 else "B" if overall >= 3.5 else "C" if overall >= 2.5 else "D" if overall >= 1.5 else "F"
    issues  = []
    if acc < 3:
        issues.append(f"낮은 정확도 ({acc}/5)")
    if rel < 3:
        issues.append(f"낮은 관련성 ({rel}/5)")
    if hall != "없음":
        issues.append(f"환각 감지: {hall} (심각도 {(hall_cause or {}).get('심각도', '-')}/10)")
    mismatch = evaluation.get("불일치_항목", "없음")
    if mismatch and mismatch != "없음":
        issues.append(f"문서-답변 불일치: {mismatch[:60]}")
    missing = evaluation.get("누락_정보", "없음")
    if missing and missing != "없음":
        issues.append(f"누락 정보: {missing[:60]}")
    if evaluation.get("신뢰도") == "낮음":
        issues.append("전반적 신뢰도 낮음")
    recs = []
    if evaluation.get("개선_제안"):
        recs.append(evaluation["개선_제안"])
    if hall_cause and hall_cause.get("수정_제안"):
        recs.append(f"[환각 수정] {hall_cause['수정_제안']}")
    if hall_cause and hall_cause.get("개선_제안"):
        recs.append(f"[파이프라인] {hall_cause['개선_제안']}")
    return {"overall_score": overall, "grade": grade, "issues": issues, "recommendations": recs}


def critique_answer(question: str, context_chunks: list, draft: str, tracer: Tracer = None) -> str:
    tracer and tracer.start("critique")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "당신은 RAG 답변 품질 심사 전문가입니다.\n"
                "아래 Draft 답변을 문서와 대조해 비판적으로 검토하세요.\n\n"
                "반드시 아래 형식으로만 출력하세요:\n"
                "**문제점**\n- (문서와 불일치하거나 부정확한 내용)\n\n"
                "**누락**\n- (문서에는 있지만 답변에서 빠진 중요 정보)\n\n"
                "**개선 방향**\n- (구체적으로 어떻게 고쳐야 하는지)\n\n"
                "문제가 없으면 각 항목에 '없음'을 쓰세요."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft 답변]\n{draft}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("critique", tokens=_usage(response),
                   input_summary=f"Draft {len(draft)}자",
                   output_summary=f"비판 {len(result)}자",
                   decision="Draft 문제점·누락·개선 방향 도출")
    return result


def refine_answer(question: str, context_chunks: list, draft: str, critique: str,
                  tracer: Tracer = None) -> str:
    tracer and tracer.start("refine")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 기반 어시스턴트. 아래 Draft 답변과 Critique를 참고해 개선된 최종 답변을 작성하세요.\n"
                "규칙: 1. **📌 요약** / **📖 근거** ([출처 N] 인용) / **✅ 결론** (확신도: 높음/보통/낮음).\n"
                "2. Critique의 문제점·누락 항목을 반드시 반영하세요.\n"
                "3. 문서에 없는 내용은 절대 추가하지 마세요.\n"
                "4. 한국어로 작성하세요."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft 답변]\n{draft}\n\n[Critique]\n{critique}"}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        tracer.end("refine", tokens=_usage(response),
                   input_summary=f"Draft {len(draft)}자 + Critique",
                   output_summary=f"Refined 답변 {len(result)}자",
                   decision="Critique 반영 최종 답변")
    return result


# =====================================================================
# 단일 파이프라인 실행 (v19: 캐시 전략 + Context Compression + Multi-Vector)
# =====================================================================

def run_rag_pipeline(question: str, eff: dict,
                     index, chunks: list, sources: list,
                     prefilter_n: int, use_multidoc: bool,
                     num_rewrites: int = 3,
                     use_session_cache: bool = True,
                     use_self_refine: bool = False,
                     use_compression: bool = False,
                     mv_index: dict = None) -> dict:
    """
    [v19] RAG 파이프라인.
    신규: ① 답변 캐시 체크 ② 쿼리 결과 캐시 ③ Multi-Vector 검색 ④ Context Compression
    """
    tracer = Tracer()

    # ── [NEW v19] 답변 캐시 체크 ──────────────────────────────────
    if use_session_cache:
        ans_key    = hashlib.md5(question.encode("utf-8")).hexdigest()
        cached_ans = answer_cache.get(ans_key)
        if cached_ans:
            tracer.start("answer_cache_hit")
            tracer.end("answer_cache_hit",
                       input_summary=question[:60],
                       output_summary="답변 캐시 히트 → 모든 LLM 호출 스킵",
                       decision=f"TTL {ANSWER_CACHE_TTL_SEC//60}분 이내 캐시")
            return {
                "tracer": tracer, "queries": [question],
                "ranked": [], "final_chunks": [], "final_sources": [], "final_scores": [],
                "gen_chunks": [], "summaries": [], "analysis": "",
                "answer":       cached_ans["answer"],
                "draft_answer": cached_ans["answer"],
                "critique":     None,
                "mode":         "answer_cache_hit",
                "evaluation":   cached_ans["evaluation"],
                "hall_cause":   None,
                "quality_report": cached_ans["quality_report"],
                "ndcg_k": None, "sqr": None,
                "eff": eff.copy(), "prefilter_n": prefilter_n,
                "cache_hit": "answer", "compression_stats": None,
            }

    # 1. Query rewriting
    if eff["use_query_rewrite"]:
        queries = rewrite_queries(question, n=num_rewrites, tracer=tracer,
                                  use_session_cache=use_session_cache)
    else:
        queries = [question]

    # ── [NEW v19] 쿼리 결과 캐시 체크 ────────────────────────────
    cache_hit  = None
    qr_cached  = query_result_cache.get(question, eff["use_bm25"], prefilter_n) if use_session_cache else None

    if qr_cached is not None:
        filtered  = qr_cached
        cache_hit = "query"
        tracer.start("embedding_search")
        tracer.end("embedding_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary="쿼리 결과 캐시 히트",
                   output_summary=f"{len(filtered)}개 (캐시 재사용)",
                   decision=f"TTL {QUERY_CACHE_TTL_SEC//60}분 이내 캐시")
        tracer.start("prefilter")
        tracer.end("prefilter", input_summary="캐시", output_summary="캐시", decision="캐시 히트 스킵")
    else:
        # 2. 검색
        if mv_index:
            candidates = retrieve_multi_vector(queries, mv_index, chunks, sources,
                                               top_k_per_query=20, use_bm25=eff["use_bm25"], tracer=tracer)
        else:
            candidates = retrieve_hybrid(queries, index, chunks, sources,
                                         top_k_per_query=20, use_bm25=eff["use_bm25"], tracer=tracer)
        # 3. Pre-filter
        filtered = prefilter_by_similarity(question, candidates, prefilter_n, tracer)
        if use_session_cache:
            query_result_cache.set(question, eff["use_bm25"], prefilter_n, filtered)

    # 4. Reranking
    ndcg_k    = None
    sqr       = None
    eff_top_k = eff["top_k"]
    if eff["use_reranking"] and len(filtered) > eff_top_k:
        ranked, all_scores = rerank_chunks(question, filtered, eff_top_k, tracer)
        ndcg_k = compute_ndcg(filtered, all_scores, k=eff_top_k)
        sqr    = compute_search_quality_report(ndcg_k, eff["use_bm25"], len(filtered))
    else:
        ranked = [(item[0], item[1], None) for item in filtered[:eff_top_k]]
        tracer.start("rerank")
        tracer.end("rerank", input_summary=f"{len(filtered)}개",
                   output_summary=f"{len(ranked)}개 (OFF)", decision="리랭킹 비활성화")

    final_chunks  = [r[0] for r in ranked]
    final_sources = [r[1] for r in ranked]
    final_scores  = [r[2] for r in ranked]

    # ── [NEW v19] 5. Context Compression ─────────────────────────
    compression_stats = None
    gen_chunks        = final_chunks
    if use_compression and final_chunks:
        gen_chunks, compression_stats = compress_chunks(question, final_chunks, tracer=tracer)

    # 6. 답변 생성
    if use_multidoc and gen_chunks:
        summaries = step1_summarize_chunks(question, gen_chunks, tracer)
        analysis  = step2_analyze_relationships(question, summaries, final_sources, tracer)
        answer    = step3_generate_final_answer(question, gen_chunks, summaries, analysis, tracer)
        mode = "multidoc"
    else:
        ranked_simple = [(gen_chunks[i], final_sources[i], final_scores[i]) for i in range(len(gen_chunks))]
        answer    = generate_answer_simple(question, ranked_simple, tracer)
        summaries, analysis = [], ""
        mode = "simple"

    # 7. Self-Refinement
    draft_answer = answer
    critique     = None
    if use_self_refine and gen_chunks:
        critique = critique_answer(question, gen_chunks, draft_answer, tracer)
        answer   = refine_answer(question, gen_chunks, draft_answer, critique, tracer)

    # 8. 평가
    evaluation  = evaluate_answer(question, gen_chunks, answer, tracer)
    hall        = evaluation.get("환각여부", "없음")
    hall_cause  = None
    if hall != "없음":
        hall_cause = analyze_hallucination_cause(question, gen_chunks, answer, hall, tracer)
    quality_report = build_quality_report(evaluation, hall_cause)

    # ── [NEW v19] 답변 캐시 저장 ─────────────────────────────────
    if use_session_cache:
        ans_key = hashlib.md5(question.encode("utf-8")).hexdigest()
        answer_cache.set(ans_key, {
            "answer": answer, "evaluation": evaluation, "quality_report": quality_report
        })

    return {
        "tracer": tracer, "queries": queries,
        "ranked": ranked,
        "final_chunks": final_chunks, "final_sources": final_sources, "final_scores": final_scores,
        "gen_chunks": gen_chunks,
        "summaries": summaries, "analysis": analysis,
        "answer": answer, "draft_answer": draft_answer, "critique": critique,
        "mode": mode,
        "evaluation": evaluation, "hall_cause": hall_cause, "quality_report": quality_report,
        "ndcg_k": ndcg_k, "sqr": sqr,
        "eff": eff.copy(), "prefilter_n": prefilter_n,
        "cache_hit": cache_hit, "compression_stats": compression_stats,
    }


# =====================================================================
# Ablation Study
# =====================================================================

def run_single_config(question: str, config: dict,
                      index, chunks: list, sources: list,
                      top_k: int = 3, prefilter_n: int = 10) -> dict:
    eff = {
        "use_bm25":          config["bm25"] and BM25_AVAILABLE,
        "use_reranking":     config["rerank"],
        "top_k":             top_k,
        "use_query_rewrite": config["query_rewrite"],
    }
    t_start = time.time()
    try:
        r = run_rag_pipeline(question, eff, index, chunks, sources,
                             prefilter_n=prefilter_n, use_multidoc=True,
                             num_rewrites=2, use_session_cache=False,
                             use_self_refine=False, use_compression=False, mv_index=None)
        qr  = r["quality_report"]
        sqr = r["sqr"] or {}
        return {
            "config_id": config["id"], "config_name": config["name"],
            "query_rewrite": config["query_rewrite"], "bm25": config["bm25"], "rerank": config["rerank"],
            "accuracy": r["evaluation"].get("정확도", 0), "relevance": r["evaluation"].get("관련성", 0),
            "hallucination": r["evaluation"].get("환각여부", "-"), "confidence": r["evaluation"].get("신뢰도", "-"),
            "overall_score": qr["overall_score"], "grade": qr["grade"],
            "ndcg_prefilter": r["ndcg_k"], "reranker_gain": sqr.get("reranker_gain"),
            "search_quality": sqr.get("quality_label"),
            "latency_ms": r["tracer"].total_latency_ms(), "total_tokens": r["tracer"].total_tokens()["total"],
            "answer": r["answer"], "error": None,
        }
    except Exception as e:
        return {
            "config_id": config["id"], "config_name": config["name"],
            "query_rewrite": config["query_rewrite"], "bm25": config["bm25"], "rerank": config["rerank"],
            "accuracy": 0, "relevance": 0, "hallucination": "-", "confidence": "-",
            "overall_score": 0, "grade": "F", "ndcg_prefilter": None, "reranker_gain": None,
            "search_quality": None, "latency_ms": int((time.time() - t_start) * 1000),
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
                    quality_report=None, search_quality_report=None,
                    fallback_triggered=False, fallback_attempts=0, fallback_history=None,
                    self_refinement=None, dynamic_retrieval_profile=None,
                    cache_hit=None, compression_stats=None, mv_retrieval=False):
    sqr = search_quality_report or {}
    return {
        "trace_id":                  tracer.trace_id,
        "timestamp":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":                  question,
        "queries":                   queries,
        "retrieved_chunks": [
            {"text": c[:200] + ("..." if len(c) > 200 else ""), "source": s,
             "rerank_score": round(sc, 2) if sc is not None else None}
            for c, s, sc in ranked_items
        ],
        "answer":                    answer,
        "evaluation":                evaluation,
        "hallucination_analysis":    hall_cause,
        "quality_report":            quality_report,
        "search_quality_report":     sqr,
        "ndcg_at_k":                 ndcg,
        "reranker_gain":             sqr.get("reranker_gain"),
        "fallback_triggered":        fallback_triggered,
        "fallback_attempts":         fallback_attempts,
        "fallback_history":          fallback_history or [],
        "self_refinement":           self_refinement,
        "dynamic_retrieval_profile": dynamic_retrieval_profile,
        # [NEW v19]
        "cache_hit":                 cache_hit,
        "compression_stats":         compression_stats,
        "mv_retrieval":              mv_retrieval,
        "spans":                     tracer.spans,
        "total_tokens":              tracer.total_tokens(),
        "total_latency_ms":          tracer.total_latency_ms(),
        "bottleneck":                tracer.bottleneck(),
        "mode":                      mode,
        "route_decision":            route_decision,
        "embed_cache_size":          embed_cache.size(),
        "embed_cache_hits":          embed_cache.hits,
        "embed_cache_misses":        embed_cache.misses,
    }


# =====================================================================
# 세션 초기화
# =====================================================================
for key, default in [
    ("messages", []), ("index", None), ("chunks", []),
    ("chunk_sources", []), ("rewrite_cache", {}),
    ("ablation_results", []), ("mv_index", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# 탭 레이아웃
# =====================================================================
tab_chat, tab_trace, tab_agent, tab_ablation, tab_search = st.tabs(
    ["💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation Study", "🔍 검색 품질 분석"]
)


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("📚 RAG 챗봇 v19")
    st.markdown("---")

    uploaded_files = st.file_uploader("파일을 업로드하세요", type=["pdf", "txt"], accept_multiple_files=True)
    chunking_mode  = st.radio("청킹 방식", ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"])
    chunk_size     = st.slider("청크 크기", 200, 1000, 500, 100)
    overlap        = st.slider("Overlap 크기", 0, 200, 100, 20)

    # [NEW v19] Multi-Vector 토글 (문서 처리 전에 선택)
    use_multi_vector = st.toggle("🔢 Multi-Vector 인덱싱", value=True,
                                  help="청크·문장·키워드 3종 벡터 동시 구축 → 검색 정확도 향상 (인덱싱 시간 증가)")

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

                if use_multi_vector:
                    with st.spinner("Multi-Vector 인덱스 구축 중 (청크+문장+키워드)..."):
                        mv_idx = build_multi_vector_index(all_chunks)
                        st.session_state.index    = mv_idx["chunk_index"]
                        st.session_state.mv_index = mv_idx
                    st.success(f"완료! {len(all_chunks)}개 청크 | {mv_idx['n_sentences']}개 문장 벡터")
                else:
                    st.session_state.index    = build_index(all_chunks)
                    st.session_state.mv_index = None
                    st.success(f"완료! {len(all_chunks)}개 청크")
            # 문서 변경 시 쿼리 캐시 무효화
            query_result_cache.clear()

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크")
        for fname, cnt in Counter(st.session_state.chunk_sources).items():
            st.caption(f"  └ {fname}: {cnt}개")
        if st.session_state.mv_index:
            n_s = st.session_state.mv_index.get("n_sentences", 0)
            st.caption(f"  🔢 문장 벡터: {n_s}개")

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
    st.caption("🔄 Fallback 설정")
    enable_fallback = st.toggle("Fallback 자동 재시도", value=True,
                                help=f"정확도 < {FALLBACK_MIN_ACC} 또는 환각 감지 시 자동 재시도 (최대 {MAX_RETRIES}회)")
    st.caption(f"  트리거: 정확도 < {FALLBACK_MIN_ACC} / 환각 감지")

    st.markdown("---")
    st.caption("✏️ Self-Refinement 설정")
    enable_self_refine = st.toggle("Self-Refinement (Draft→Critique→Refine)", value=True,
                                   help="초안 생성 후 자기 비판 → 개선 답변 (+2 LLM 호출)")

    st.markdown("---")
    st.caption("🎯 Dynamic Retrieval 설정")
    enable_dynamic_retrieval = st.toggle("의도별 검색 전략 자동 조정", value=True,
                                          disabled=not auto_routing)

    st.markdown("---")
    # [NEW v19] 캐시 전략 설정
    st.caption("⚡ 캐시 전략 설정")
    enable_answer_cache = st.toggle("답변 캐시 (Answer Cache)", value=True,
                                     help=f"동일 질문 재요청 시 LLM 전체 스킵 (TTL {ANSWER_CACHE_TTL_SEC//60}분)")
    enable_query_cache  = st.toggle("쿼리 결과 캐시 (Query Cache)", value=True,
                                     help=f"동일 질문 + 설정 재요청 시 검색 단계 스킵 (TTL {QUERY_CACHE_TTL_SEC//60}분)")
    st.caption(f"  답변 캐시: {answer_cache.valid_size()}개 유효 / {answer_cache.size()}개 전체")
    st.caption(f"  쿼리 캐시: {query_result_cache.size()}개")
    st.caption(f"  임베딩 캐시: {embed_cache.size()}개")
    hit_total = embed_cache.hits + embed_cache.misses
    if hit_total > 0:
        st.caption(f"  임베딩 히트율: {embed_cache.hits*100//hit_total}%")

    st.markdown("---")
    # [NEW v19] Context Compression 설정
    st.caption("🗜️ Context Compression 설정")
    enable_compression = st.toggle("Context Compression", value=True,
                                    help="청크에서 핵심 문장만 추출 → 토큰↓, 노이즈↓ (LLM 호출 없음)")
    comp_max_sentences = st.slider("최대 추출 문장 수", 2, 8, 5, disabled=not enable_compression)
    comp_min_sim       = st.slider("최소 유사도", 0.1, 0.5, 0.25, 0.05, disabled=not enable_compression)
    if enable_compression:
        st.caption(f"  유사도 ≥ {comp_min_sim:.2f} 문장 추출 (상위 {comp_max_sentences}개)")

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("문서 초기화", use_container_width=True):
        st.session_state.update({"messages": [], "index": None, "chunks": [], "chunk_sources": [], "mv_index": None})
        query_result_cache.clear()
        st.rerun()
    if st.button("로그 초기화", use_container_width=True):
        os.path.exists(LOG_FILE) and os.remove(LOG_FILE)
        st.rerun()
    if st.button("캐시 전체 초기화", use_container_width=True):
        embed_cache.clear()
        query_result_cache.clear()
        answer_cache.clear()
        st.rerun()
    if st.button("답변 캐시만 초기화", use_container_width=True):
        answer_cache.clear()
        st.rerun()
    st.caption("v19: Cache 전략 분리 + Context Compression + Multi-Vector")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================
with tab_chat:
    st.title("💬 문서 기반 챗봇 v19")
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
                route_decision        = None
                dynamic_profile_label = None
                defaults = {"use_bm25": use_bm25, "use_reranking": use_reranking,
                            "top_k": top_k, "use_query_rewrite": use_query_rewrite}

                eff           = defaults.copy()
                cur_multidoc  = use_multidoc
                cur_prefilter = prefilter_n

                # ── Step 0: 라우팅 ─────────────────────────────
                if auto_routing:
                    with st.status("🧭 쿼리 라우팅...", expanded=False) as s:
                        _rt = Tracer()
                        route_decision = route_query(prompt, _rt)
                        intent = route_decision.get("의도", "-")
                        eff = _apply_routing(route_decision, defaults)
                        if enable_dynamic_retrieval:
                            eff, cur_prefilter, cur_multidoc, dynamic_profile_label = \
                                apply_dynamic_retrieval(intent, eff, prefilter_n, use_multidoc)
                        mv_tag = " | 🔢 Multi-Vector" if st.session_state.mv_index else ""
                        pf_tag = f" | 프로필: {dynamic_profile_label}" if dynamic_profile_label else ""
                        s.update(label=f"✅ 의도: {intent} | top_k: {eff['top_k']}{pf_tag}{mv_tag}", state="complete")

                # ── Fallback 루프 ──────────────────────────────
                attempt_num             = 0
                fallback_triggered_flag = False
                fallback_history        = []
                best_result             = None

                with st.status("🤔 답변 생성 중 (시도 1)...", expanded=False) as s:
                    cur_result = run_rag_pipeline(
                        prompt, eff,
                        st.session_state.index, st.session_state.chunks, st.session_state.chunk_sources,
                        prefilter_n=cur_prefilter, use_multidoc=cur_multidoc,
                        num_rewrites=num_rewrites,
                        use_session_cache=enable_query_cache and enable_answer_cache,
                        use_self_refine=enable_self_refine,
                        use_compression=enable_compression,
                        mv_index=st.session_state.mv_index,
                    )
                    ev0 = cur_result["evaluation"]
                    qr0 = cur_result["quality_report"]
                    fb_needed, fb_reason = should_fallback(ev0) if enable_fallback else (False, "")

                    cache_icon  = "⚡" if cur_result.get("cache_hit") else ""
                    comp_icon   = " 🗜️" if cur_result.get("compression_stats") else ""
                    refine_icon = " ✏️" if cur_result.get("critique") else ""
                    status_icon = "✅" if not fb_needed else "⚠️"
                    s.update(
                        label=(
                            f"{status_icon} 시도 1 완료{cache_icon}{comp_icon}{refine_icon} | "
                            f"정확도 {ev0.get('정확도','-')}/5 · 환각 {ev0.get('환각여부','-')} · 등급 {qr0['grade']}"
                            + (f" → Fallback 예정: {fb_reason}" if fb_needed else "")
                        ),
                        state="complete"
                    )

                fallback_history.append({
                    "attempt": 0, "trigger": None, "eff": cur_result["eff"],
                    "prefilter_n": cur_result["prefilter_n"],
                    "accuracy": ev0.get("정확도", 0), "hallucination": ev0.get("환각여부", "-"),
                    "overall_score": qr0["overall_score"], "grade": qr0["grade"],
                    "tokens": cur_result["tracer"].total_tokens()["total"],
                    "latency_ms": cur_result["tracer"].total_latency_ms(),
                })
                best_result = cur_result

                while fb_needed and attempt_num < MAX_RETRIES:
                    attempt_num += 1
                    fallback_triggered_flag = True
                    new_eff, new_pf_n = escalate_params(eff, cur_prefilter, attempt_num)

                    with st.status(f"🔄 Fallback 재시도 {attempt_num}/{MAX_RETRIES} — {fb_reason}", expanded=False) as s:
                        cur_result = run_rag_pipeline(
                            prompt, new_eff,
                            st.session_state.index, st.session_state.chunks, st.session_state.chunk_sources,
                            prefilter_n=new_pf_n, use_multidoc=cur_multidoc,
                            num_rewrites=num_rewrites, use_session_cache=False,
                            use_self_refine=enable_self_refine,
                            use_compression=enable_compression,
                            mv_index=st.session_state.mv_index,
                        )
                        ev_n     = cur_result["evaluation"]
                        qr_n     = cur_result["quality_report"]
                        improved = qr_n["overall_score"] > best_result["quality_report"]["overall_score"]
                        fb_needed, fb_reason = should_fallback(ev_n)
                        s.update(
                            label=(
                                f"{'✅ 개선됨' if improved else '➡️ 유지'} 재시도 {attempt_num} | "
                                f"정확도 {ev_n.get('정확도','-')}/5 · 환각 {ev_n.get('환각여부','-')} · 등급 {qr_n['grade']}"
                                + (f" → 추가 재시도: {fb_reason}" if fb_needed else "")
                            ),
                            state="complete"
                        )

                    fallback_history.append({
                        "attempt": attempt_num, "trigger": fallback_history[-1].get("trigger") or fb_reason,
                        "eff": cur_result["eff"], "prefilter_n": cur_result["prefilter_n"],
                        "accuracy": ev_n.get("정확도", 0), "hallucination": ev_n.get("환각여부", "-"),
                        "overall_score": qr_n["overall_score"], "grade": qr_n["grade"],
                        "tokens": cur_result["tracer"].total_tokens()["total"],
                        "latency_ms": cur_result["tracer"].total_latency_ms(),
                    })
                    if improved:
                        best_result = cur_result

                # ── 최종 결과 ──────────────────────────────────
                final          = best_result
                response       = final["answer"]
                evaluation     = final["evaluation"]
                hall_cause     = final["hall_cause"]
                quality_report = final["quality_report"]
                ndcg_k         = final["ndcg_k"]
                sqr            = final["sqr"]
                queries        = final["queries"]
                ranked         = final["ranked"]
                final_chunks   = final["final_chunks"]
                final_sources  = final["final_sources"]
                final_scores   = final["final_scores"]
                gen_chunks     = final["gen_chunks"]
                summaries      = final["summaries"]
                analysis       = final["analysis"]
                mode           = final["mode"]
                final_eff      = final["eff"]
                tracer         = final["tracer"]
                draft_answer   = final["draft_answer"]
                critique       = final["critique"]
                cache_hit      = final.get("cache_hit")
                comp_stats     = final.get("compression_stats")

                self_ref_log = None
                if enable_self_refine and critique:
                    self_ref_log = {"enabled": True, "draft": draft_answer[:500], "critique": critique}

                if auto_evaluate:
                    save_log(build_log_entry(
                        prompt, queries, ranked, response,
                        evaluation, hall_cause, tracer, mode,
                        ndcg=ndcg_k, route_decision=route_decision,
                        quality_report=quality_report, search_quality_report=sqr,
                        fallback_triggered=fallback_triggered_flag,
                        fallback_attempts=attempt_num, fallback_history=fallback_history,
                        self_refinement=self_ref_log,
                        dynamic_retrieval_profile=dynamic_profile_label,
                        cache_hit=cache_hit, compression_stats=comp_stats,
                        mv_retrieval=bool(st.session_state.mv_index),
                    ))

                # ── 답변 표시 ───────────────────────────────────
                # 답변 캐시 히트 배너
                if cache_hit == "answer":
                    st.success(f"⚡ 답변 캐시 히트 — TTL {ANSWER_CACHE_TTL_SEC//60}분 이내 캐시 재사용 (모든 LLM 호출 스킵)")
                elif cache_hit == "query":
                    st.info("🔁 쿼리 결과 캐시 히트 — 검색 단계 스킵")

                st.markdown(linkify_citations(response), unsafe_allow_html=True)

                if evaluation:
                    hall = evaluation.get("환각여부", "")
                    hc   = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else "🔴")
                    cols = st.columns(8)
                    cols[0].metric("⏱ 응답",   f"{tracer.total_latency_ms()/1000:.1f}s")
                    cols[1].metric("🔤 토큰",   f"{tracer.total_tokens()['total']:,}")
                    cols[2].metric("📐 정확도", f"{evaluation.get('정확도','-')}/5")
                    cols[3].metric("🎯 관련성", f"{evaluation.get('관련성','-')}/5")
                    cols[4].metric(f"{hc} 환각", hall)
                    cols[5].metric("🧭 신뢰도", evaluation.get("신뢰도", "-"))
                    if fallback_triggered_flag:
                        cols[6].metric("🔄 재시도", f"{attempt_num}회",
                                       delta=f"등급 {fallback_history[0]['grade']}→{quality_report['grade']}",
                                       delta_color="normal")
                    else:
                        cols[6].metric("🔄 재시도", "없음")
                    cols[7].metric("🔍 Dense NDCG",
                                   f"{ndcg_k:.3f}" if ndcg_k else "-",
                                   delta=f"Gain {sqr['reranker_gain']:.3f}" if sqr else None)

                # [NEW v19] Context Compression expander
                if comp_stats:
                    orig_total = sum(s["original"] for s in comp_stats)
                    comp_total = sum(s["compressed"] for s in comp_stats)
                    avg_ratio  = round(comp_total / max(orig_total, 1) * 100)
                    with st.expander(f"🗜️ Context Compression — 토큰 {100-avg_ratio}% 절감 ({orig_total}자 → {comp_total}자)"):
                        for i, s in enumerate(comp_stats):
                            st.caption(f"청크 {i+1}: {s['original']}자 → {s['compressed']}자 ({s['ratio']*100:.0f}%)")

                # Self-Refinement expander
                if enable_self_refine and critique:
                    with st.expander("✏️ Self-Refinement 내역 (Draft → Critique → Refined)"):
                        st.markdown("**📝 Draft 답변**")
                        st.info(draft_answer)
                        st.markdown("**🔎 Critique (자기 비판)**")
                        st.warning(critique)
                        st.markdown("**✅ Refined 최종 답변** ← 위에 표시된 답변")

                # Fallback 히스토리 expander
                if fallback_triggered_flag:
                    with st.expander(f"🔄 Fallback 실행 내역 ({attempt_num}회 재시도)"):
                        for h in fallback_history:
                            is_best = h["overall_score"] == quality_report["overall_score"]
                            st.markdown(f"**{'🏆 최종 채택' if is_best else '📋'} 시도 {h['attempt']+1}** — 트리거: `{h.get('trigger') or '최초 시도'}`")
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("정확도",   f"{h['accuracy']}/5")
                            c2.metric("환각",     h["hallucination"])
                            c3.metric("종합점수", f"{h['overall_score']}/5")
                            c4.metric("등급",     h["grade"])
                            c5.metric("토큰",     f"{h['tokens']:,}")
                            eff_h = h["eff"]
                            st.caption(f"top_k={eff_h['top_k']} | prefilter={h['prefilter_n']} | BM25={'ON' if eff_h['use_bm25'] else 'OFF'} | Rerank={'ON' if eff_h['use_reranking'] else 'OFF'}")
                            if h["attempt"] < len(fallback_history) - 1:
                                st.divider()

                if sqr:
                    lc = {"excellent": "🟢", "good": "🔵", "fair": "🟡", "poor": "🔴"}.get(sqr["quality_label"], "⚪")
                    with st.expander(f"🔍 검색 품질 리포트 — {lc} {sqr['quality_label'].upper()}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Dense NDCG@k", f"{sqr['ndcg_prefilter']:.4f}")
                        c2.metric("Reranker Gain", f"{sqr['reranker_gain']:.4f}")
                        c3.metric("후보 수",        f"{sqr['n_candidates']}개")
                        st.info(f"📋 {sqr['diagnosis']}")

                if quality_report:
                    grade = quality_report["grade"]
                    score = quality_report["overall_score"]
                    gc = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(grade, "⚪")
                    with st.expander(f"🩺 품질 리포트 — 등급 {gc} {grade} ({score}/5)"):
                        if quality_report["issues"]:
                            st.markdown("**⚠️ 감지된 문제**")
                            for issue in quality_report["issues"]:
                                st.markdown(f"- {issue}")
                        if quality_report["recommendations"]:
                            st.markdown("**💡 개선 제안**")
                            for rec in quality_report["recommendations"]:
                                st.markdown(f"- {rec}")

                if hall_cause:
                    severity  = hall_cause.get("심각도", "-")
                    sev_color = "🔴" if isinstance(severity, int) and severity >= 7 else ("🟡" if isinstance(severity, int) and severity >= 4 else "🟢")
                    st.error(f"{sev_color} **심각도 {severity}/10** | 유형: {hall_cause.get('환각_유형','')} | 원인: {hall_cause.get('발생_원인','')} | 원문: _{hall_cause.get('원문_인용','')[:80]}_")
                    if hall_cause.get("수정_제안"):
                        st.warning(f"🔧 수정 제안: {hall_cause['수정_제안']}")

                if route_decision:
                    with st.expander("🧭 쿼리 라우팅 결정"):
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.markdown(f"**의도**: `{route_decision.get('의도','-')}`  \n**설명**: {route_decision.get('설명','-')}")
                            st.json(route_decision.get("검색_전략", {}))
                            if dynamic_profile_label:
                                st.success(f"🎯 Dynamic Retrieval 프로필: {dynamic_profile_label}")
                        with col_r2:
                            st.markdown("**적용된 파라미터 (최종)**")
                            st.json(final_eff)

                if final_eff["use_query_rewrite"]:
                    with st.expander("🔍 생성된 쿼리"):
                        for i, q in enumerate(queries):
                            st.markdown(f"{'**[원본]**' if i == 0 else f'**[변형 {i}]**'} {q}")

                if cur_multidoc and summaries:
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
# TAB 2 — 트레이싱
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
        last = logs[-1]
        hits, miss = last.get("embed_cache_hits", 0), last.get("embed_cache_misses", 0)
        hit_rate    = f"{hits*100//(hits+miss)}%" if (hits+miss) > 0 else "-"
        fb_count    = sum(1 for l in logs if l.get("fallback_triggered"))
        cache_hits  = sum(1 for l in logs if l.get("cache_hit"))       # [NEW v19]
        ans_hits    = sum(1 for l in logs if l.get("cache_hit") == "answer")  # [NEW v19]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("총 트레이스",     len(logs))
        c2.metric("평균 응답",       f"{sum(total_lat_all)/len(total_lat_all)/1000:.1f}s")
        c3.metric("평균 토큰",       f"{int(sum(total_tokens_all)/len(total_tokens_all)):,}")
        c4.metric("🔄 Fallback 발생", f"{fb_count}회")
        c5.metric("⚡ 캐시 히트",     f"{cache_hits}회")      # [NEW v19]
        c6.metric("⚡ 답변 캐시 히트", f"{ans_hits}회")        # [NEW v19]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("응답 시간 추이 (초)")
            st.line_chart([l/1000 for l in total_lat_all])
        with col2:
            st.subheader("토큰 사용량 추이")
            st.line_chart(total_tokens_all)

        intent_counts = Counter((l.get("route_decision") or {}).get("의도", "미분류") for l in logs)
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
            fb_tag   = f" | 🔄 {log.get('fallback_attempts',0)}회" if log.get("fallback_triggered") else ""
            sr_tag   = " | ✏️" if log.get("self_refinement") else ""
            ch_tag   = f" | ⚡{log['cache_hit']}" if log.get("cache_hit") else ""

            with st.expander(
                f"[{ts}] {q}... | ⏱ {total_ms/1000:.1f}s | 🔤 {tok.get('total',0):,} | {intent}"
                + (f" | NDCG {nd:.3f}" if nd else "") + fb_tag + sr_tag + ch_tag
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
# TAB 3 — 에이전트 분석 (v19: 캐시 통계 + Compression 통계 추가)
# =====================================================================
with tab_agent:
    st.title("🧠 에이전트 분석 대시보드")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        import pandas as pd

        eval_logs    = [l for l in logs if l.get("evaluation")]
        avg_acc      = round(sum(l["evaluation"].get("정확도",0) for l in eval_logs)/len(eval_logs), 2) if eval_logs else 0
        avg_rel      = round(sum(l["evaluation"].get("관련성",0) for l in eval_logs)/len(eval_logs), 2) if eval_logs else 0
        hall_counts  = Counter(l["evaluation"].get("환각여부","") for l in eval_logs)
        hall_logs    = [l for l in eval_logs if l["evaluation"].get("환각여부","없음") != "없음"]
        ndcg_vals    = [l["ndcg_at_k"] for l in logs if l.get("ndcg_at_k") is not None]
        avg_ndcg     = round(sum(ndcg_vals)/len(ndcg_vals), 3) if ndcg_vals else None
        qr_logs      = [l for l in logs if l.get("quality_report")]
        avg_overall  = round(sum(l["quality_report"]["overall_score"] for l in qr_logs)/len(qr_logs), 2) if qr_logs else None
        grade_counts = Counter(l["quality_report"]["grade"] for l in qr_logs)
        conf_counts  = Counter(l["evaluation"].get("신뢰도","보통") for l in eval_logs)
        hall_none_pct = round(hall_counts.get("없음",0)/len(eval_logs)*100) if eval_logs else 0
        fb_logs      = [l for l in logs if l.get("fallback_triggered")]
        fb_rate      = round(len(fb_logs)/len(logs)*100) if logs else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("평균 정확도",    f"{avg_acc}/5")
        c2.metric("평균 관련성",    f"{avg_rel}/5")
        c3.metric("환각 없음 비율", f"{hall_none_pct}%")
        c4.metric("평균 NDCG@k",   str(avg_ndcg) if avg_ndcg else "-")
        c5.metric("평균 종합점수",  f"{avg_overall}/5" if avg_overall else "-")
        c6.metric("🔄 Fallback 율", f"{fb_rate}%")

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
            st.bar_chart({"높음": conf_counts.get("높음",0), "보통": conf_counts.get("보통",0), "낮음": conf_counts.get("낮음",0)})
        with col4:
            st.subheader("🏅 답변 등급 분포")
            if grade_counts:
                st.bar_chart(dict(sorted(grade_counts.items())))

        # [NEW v19] 캐시 전략 통계
        st.markdown("---")
        st.subheader("⚡ 캐시 전략 통계")
        ans_hit_logs   = [l for l in logs if l.get("cache_hit") == "answer"]
        qry_hit_logs   = [l for l in logs if l.get("cache_hit") == "query"]
        no_cache_logs  = [l for l in logs if not l.get("cache_hit")]
        ans_hit_rate   = round(len(ans_hit_logs)/len(logs)*100) if logs else 0
        qry_hit_rate   = round(len(qry_hit_logs)/len(logs)*100) if logs else 0

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("답변 캐시 히트",  f"{len(ans_hit_logs)}건 ({ans_hit_rate}%)")
        cc2.metric("쿼리 캐시 히트",  f"{len(qry_hit_logs)}건 ({qry_hit_rate}%)")
        cc3.metric("캐시 미사용",     f"{len(no_cache_logs)}건")
        # 절감 토큰 추정 (답변 캐시 히트는 평균 토큰 절감)
        avg_tokens = round(sum(l.get("total_tokens",{}).get("total",0) for l in no_cache_logs) / max(len(no_cache_logs),1))
        estimated_saved = len(ans_hit_logs) * avg_tokens
        cc4.metric("예상 절감 토큰",  f"{estimated_saved:,}")

        if ans_hit_logs or qry_hit_logs or no_cache_logs:
            st.bar_chart({"답변 캐시 히트": len(ans_hit_logs),
                          "쿼리 캐시 히트": len(qry_hit_logs),
                          "캐시 미사용":    len(no_cache_logs)})

        # [NEW v19] Context Compression 통계
        comp_logs = [l for l in logs if l.get("compression_stats")]
        if comp_logs:
            st.markdown("---")
            st.subheader(f"🗜️ Context Compression 통계 ({len(comp_logs)}건)")
            all_ratios = []
            for l in comp_logs:
                for s in l["compression_stats"]:
                    if s.get("original", 0) > 0:
                        all_ratios.append(s["compressed"] / s["original"])
            if all_ratios:
                avg_comp = round(sum(all_ratios) / len(all_ratios) * 100)
                cmp1, cmp2, cmp3 = st.columns(3)
                cmp1.metric("평균 압축률",   f"{avg_comp}% (원본 대비)")
                cmp2.metric("평균 토큰 절감", f"{100-avg_comp}%")
                cmp3.metric("압축 적용 건수", f"{len(comp_logs)}건")

        # Dynamic Retrieval 프로필 분포
        profile_counts = Counter(l.get("dynamic_retrieval_profile") or "프로필 없음" for l in logs)
        if profile_counts:
            st.markdown("---")
            st.subheader("🎯 Dynamic Retrieval 프로필 분포")
            st.bar_chart(dict(profile_counts))

        # Multi-Vector 사용 통계
        mv_logs = [l for l in logs if l.get("mv_retrieval")]
        if mv_logs or logs:
            st.markdown("---")
            st.subheader("🔢 Multi-Vector 검색 사용 현황")
            mv_rate = round(len(mv_logs)/len(logs)*100) if logs else 0
            mv1, mv2 = st.columns(2)
            mv1.metric("Multi-Vector 사용", f"{len(mv_logs)}건 ({mv_rate}%)")
            mv2.metric("단일 벡터 사용",    f"{len(logs)-len(mv_logs)}건")

        # Fallback 분석
        if fb_logs:
            st.markdown("---")
            avg_att = round(sum(l.get("fallback_attempts",0) for l in fb_logs)/len(fb_logs), 1)
            first_acc = [h["accuracy"] for l in fb_logs for h in l.get("fallback_history",[]) if h["attempt"] == 0]
            final_acc = [l["evaluation"].get("정확도",0) for l in fb_logs]
            avg_first = round(sum(first_acc)/len(first_acc), 2) if first_acc else 0
            avg_final = round(sum(final_acc)/len(final_acc), 2) if final_acc else 0

            st.subheader(f"🔄 Fallback 분석 ({len(fb_logs)}건 / {fb_rate}%)")
            fb1, fb2, fb3 = st.columns(3)
            fb1.metric("Fallback 발생",     f"{len(fb_logs)}건")
            fb2.metric("평균 재시도",        f"{avg_att}회")
            fb3.metric("정확도 개선",
                       f"{avg_first:.2f} → {avg_final:.2f}",
                       delta=f"+{avg_final-avg_first:.2f}" if avg_final > avg_first else f"{avg_final-avg_first:.2f}",
                       delta_color="normal")

        if hall_logs:
            st.markdown("---")
            cause_counts  = Counter((l.get("hallucination_analysis") or {}).get("환각_유형","미분석") for l in hall_logs)
            root_counts   = Counter((l.get("hallucination_analysis") or {}).get("발생_원인","미분석") for l in hall_logs)
            severity_vals = [int((l.get("hallucination_analysis") or {}).get("심각도", 0) or 0) for l in hall_logs]
            avg_severity  = round(sum(severity_vals)/len(severity_vals), 1) if severity_vals else 0
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown("**환각 유형 분포**")
                if cause_counts: st.bar_chart(dict(cause_counts))
            with cc2:
                st.markdown("**발생 원인 분포**")
                if root_counts: st.bar_chart(dict(root_counts))
            with cc3:
                st.markdown("**평균 심각도**")
                st.metric("환각 평균 심각도", f"{avg_severity}/10")

        st.markdown("---")
        for log in reversed(logs[:10]):
            ev     = log.get("evaluation", {})
            qr     = log.get("quality_report", {})
            hall   = ev.get("환각여부", "-")
            hi     = "🟢" if hall == "없음" else ("🟡" if hall == "부분적" else ("🔴" if hall == "있음" else "⚪"))
            ts     = log.get("timestamp", "")
            q      = log.get("question","")[:45]
            nd     = log.get("ndcg_at_k")
            grade  = (qr or {}).get("grade", "-")
            score  = (qr or {}).get("overall_score", "-")
            intent = (log.get("route_decision") or {}).get("의도","-")
            fb_tag = f" | 🔄 {log.get('fallback_attempts',0)}회" if log.get("fallback_triggered") else ""
            ch_tag = f" | ⚡{log['cache_hit']}" if log.get("cache_hit") else ""

            with st.expander(
                f"[{ts}] {q}... | {ev.get('정확도','-')}/5 · {hi} {hall} | 등급 {grade}({score}) | {intent}{fb_tag}{ch_tag}"
                + (f" | NDCG {nd:.3f}" if nd else "")
            ):
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    if log.get("dynamic_retrieval_profile"):
                        st.success(f"🎯 {log['dynamic_retrieval_profile']}")
                    if log.get("cache_hit"):
                        st.info(f"⚡ 캐시 히트: {log['cache_hit']}")
                    sr = log.get("self_refinement")
                    if sr:
                        with st.expander("✏️ Self-Refinement 내역"):
                            st.info(sr.get("draft","")[:300] + "...")
                            st.warning(sr.get("critique","")[:300] + "...")
                    st.markdown("**💬 최종 답변**")
                    st.markdown(log.get("answer",""))
                with col_b:
                    if ev:
                        st.metric("정확도",  f"{ev.get('정확도','-')}/5")
                        st.metric("관련성",  f"{ev.get('관련성','-')}/5")
                        st.metric("신뢰도",  ev.get("신뢰도", "-"))
                    if qr:
                        st.metric("종합점수", f"{qr.get('overall_score','-')}/5")
                        st.metric("등급",     qr.get("grade", "-"))
                    if log.get("cache_hit"):
                        st.metric("캐시", f"⚡ {log['cache_hit']}")
                    st.metric("응답", f"{log.get('total_latency_ms',0)/1000:.1f}s")


# =====================================================================
# TAB 4 — Ablation Study
# =====================================================================
with tab_ablation:
    st.title("🧬 Ablation Study")
    if st.session_state.index is None:
        st.info("먼저 문서를 업로드하고 처리해주세요.")
    else:
        import pandas as pd
        st.markdown("---")
        abl_question    = st.text_input("실험 질문", placeholder="예: 계약 해지 절차는 어떻게 되나요?")
        config_names    = [c["name"] for c in ABLATION_CONFIGS]
        selected_names  = st.multiselect("실험할 Config", options=config_names, default=config_names)
        selected_configs = [c for c in ABLATION_CONFIGS if c["name"] in selected_names]
        abl_top_k        = st.slider("Ablation top_k", 1, 5, 3, key="abl_top_k")
        abl_prefilter_n  = st.slider("Ablation pre-filter 수", 5, 20, 10, key="abl_pf")
        n_configs        = len(selected_configs)
        st.info(f"선택된 config {n_configs}개 × 약 6 LLM 호출 = **약 {n_configs*6}회** 예상.")

        if st.button("🧪 Ablation 실행", type="primary", disabled=not abl_question or not selected_configs, use_container_width=True):
            results, progress_bar, status_text = [], st.progress(0), st.empty()
            for i, config in enumerate(selected_configs):
                status_text.markdown(f"**실행 중** ({i+1}/{n_configs}): `{config['name']}`")
                results.append(run_single_config(abl_question, config,
                                                 st.session_state.index, st.session_state.chunks,
                                                 st.session_state.chunk_sources,
                                                 top_k=abl_top_k, prefilter_n=abl_prefilter_n))
                progress_bar.progress((i + 1) / n_configs)
            status_text.markdown("✅ **모든 실험 완료**")
            st.session_state.ablation_results = results

        if st.session_state.ablation_results:
            results = st.session_state.ablation_results
            st.markdown("---")
            df = pd.DataFrame([{
                "Config":        r["config_name"],
                "정확도 (/5)":   r["accuracy"],   "관련성 (/5)":  r["relevance"],
                "신뢰도":        r.get("confidence","-"), "종합점수": r.get("overall_score","-"),
                "등급":          r.get("grade","-"), "환각여부":   r["hallucination"],
                "Dense NDCG":    round(r["ndcg_prefilter"],3) if r.get("ndcg_prefilter") else "-",
                "Reranker Gain": round(r["reranker_gain"],3)  if r.get("reranker_gain")  else "-",
                "검색 품질":     r.get("search_quality","-"),
                "응답(ms)":      r["latency_ms"],  "토큰":        r["total_tokens"],
                "오류":          "❌" if r.get("error") else "✅",
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)

            valid = [r for r in results if not r.get("error")]
            if valid:
                best_acc   = max(valid, key=lambda r: r["accuracy"])
                best_score = max(valid, key=lambda r: r.get("overall_score",0))
                fastest    = min(valid, key=lambda r: r["latency_ms"])
                rr_valid   = [r for r in valid if r.get("ndcg_prefilter") is not None]
                best_ndcg  = max(rr_valid, key=lambda r: r["ndcg_prefilter"]) if rr_valid else None

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🏆 최고 정확도",   best_acc["config_name"].split()[-1],   delta=f"정확도 {best_acc['accuracy']}/5")
                col2.metric("🥇 최고 종합점수", best_score["config_name"].split()[-1], delta=f"등급 {best_score.get('grade','-')}")
                col3.metric("⚡ 최저 지연",     f"{fastest['latency_ms']/1000:.1f}s")
                if best_ndcg:
                    col4.metric("🔍 최고 Dense NDCG", f"{best_ndcg['ndcg_prefilter']:.3f}")

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    chart_data = pd.DataFrame({
                        "Config":   [r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip() for r in valid],
                        "정확도":   [r["accuracy"] for r in valid],
                        "관련성":   [r["relevance"] for r in valid],
                        "종합점수": [r.get("overall_score",0) for r in valid],
                    }).set_index("Config")
                    st.bar_chart(chart_data)
                with col_c2:
                    if rr_valid:
                        ndcg_chart = pd.DataFrame({
                            "Config":        [r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip() for r in rr_valid],
                            "Dense NDCG":    [r["ndcg_prefilter"] for r in rr_valid],
                            "Reranker Gain": [r["reranker_gain"] for r in rr_valid],
                        }).set_index("Config")
                        st.bar_chart(ndcg_chart)

                for r in results:
                    with st.expander(f"{r['config_name']} | 정확도 {r['accuracy']}/5 · 등급 {r.get('grade','-')} | {r['latency_ms']}ms"):
                        col_ans, col_meta = st.columns([3, 1])
                        with col_ans:
                            st.markdown(r.get("answer","오류로 답변 없음"))
                        with col_meta:
                            st.metric("정확도",   f"{r['accuracy']}/5")
                            st.metric("관련성",   f"{r['relevance']}/5")
                            st.metric("종합점수", f"{r.get('overall_score','-')}/5")
                            st.metric("환각",     r["hallucination"])
                            if r.get("ndcg_prefilter"):
                                st.metric("Dense NDCG",    f"{r['ndcg_prefilter']:.3f}")
                                st.metric("Reranker Gain", f"{r['reranker_gain']:.3f}")
                            st.metric("응답",  f"{r['latency_ms']/1000:.1f}s")
                            st.metric("토큰",  f"{r['total_tokens']:,}")

                import csv as csv_module
                abl_buf = io.StringIO()
                abl_writer = csv_module.writer(abl_buf)
                abl_writer.writerow(["config_name","accuracy","relevance","confidence","overall_score","grade",
                                     "hallucination","ndcg_prefilter","reranker_gain","search_quality","latency_ms","total_tokens","error"])
                for r in results:
                    abl_writer.writerow([r["config_name"],r["accuracy"],r["relevance"],r.get("confidence",""),
                                         r.get("overall_score",""),r.get("grade",""),r["hallucination"],
                                         r.get("ndcg_prefilter",""),r.get("reranker_gain",""),r.get("search_quality",""),
                                         r["latency_ms"],r["total_tokens"],r.get("error","")])
                st.download_button("⬇️ Ablation 결과 CSV", abl_buf.getvalue().encode("utf-8-sig"),
                                   file_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv", use_container_width=True)


# =====================================================================
# TAB 5 — 검색 품질 분석
# =====================================================================
with tab_search:
    st.title("🔍 검색 품질 분석")
    logs    = load_logs()
    sq_logs = [l for l in logs if l.get("ndcg_at_k") is not None]

    if not sq_logs:
        st.info("리랭킹이 활성화된 질문이 아직 없습니다.")
    else:
        import pandas as pd
        ndcg_vals  = [l["ndcg_at_k"] for l in sq_logs]
        gain_vals  = [l["reranker_gain"] for l in sq_logs if l.get("reranker_gain") is not None]
        avg_ndcg   = round(sum(ndcg_vals)/len(ndcg_vals), 4)
        avg_gain   = round(sum(gain_vals)/len(gain_vals), 4) if gain_vals else None
        excellent_n = sum(1 for v in ndcg_vals if v >= _NDCG_EXCELLENT)
        poor_n      = sum(1 for v in ndcg_vals if v < _NDCG_FAIR)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("평균 Dense NDCG",   f"{avg_ndcg:.4f}")
        c2.metric("평균 Reranker Gain", f"{avg_gain:.4f}" if avg_gain else "-")
        c3.metric("Excellent (≥0.9)",   f"{excellent_n}건")
        c4.metric("Poor (<0.5)",         f"{poor_n}건", delta_color="inverse")
        c5.metric("총 측정 건수",        f"{len(sq_logs)}건")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Dense NDCG@k 추이")
            st.line_chart(ndcg_vals)
        with col2:
            if gain_vals:
                st.subheader("📉 Reranker Gain 추이")
                st.line_chart(gain_vals)

        st.markdown("---")
        bm25_on  = [l["ndcg_at_k"] for l in sq_logs if (l.get("search_quality_report") or {}).get("use_bm25")]
        bm25_off = [l["ndcg_at_k"] for l in sq_logs if not (l.get("search_quality_report") or {}).get("use_bm25")]
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📊 NDCG 품질 등급 분포")
            st.bar_chart({
                "Excellent (≥0.9)": sum(1 for v in ndcg_vals if v >= _NDCG_EXCELLENT),
                "Good (0.7~0.9)":   sum(1 for v in ndcg_vals if _NDCG_GOOD <= v < _NDCG_EXCELLENT),
                "Fair (0.5~0.7)":   sum(1 for v in ndcg_vals if _NDCG_FAIR <= v < _NDCG_GOOD),
                "Poor (<0.5)":      sum(1 for v in ndcg_vals if v < _NDCG_FAIR),
            })
        with col4:
            st.subheader("🆚 BM25 ON vs OFF NDCG")
            if bm25_on and bm25_off:
                st.bar_chart({"BM25 ON": round(sum(bm25_on)/len(bm25_on),4),
                              "BM25 OFF": round(sum(bm25_off)/len(bm25_off),4)})
            elif bm25_on:
                st.info(f"BM25 ON만 있음 (평균 {round(sum(bm25_on)/len(bm25_on),4)})")
            else:
                st.caption("BM25 비교 데이터 없음")

        intent_ndcg: dict = {}
        for l in sq_logs:
            intent = (l.get("route_decision") or {}).get("의도", "미분류")
            intent_ndcg.setdefault(intent, []).append(l["ndcg_at_k"])
        if intent_ndcg:
            st.markdown("---")
            st.subheader("🧭 의도별 평균 Dense NDCG")
            st.bar_chart({k: round(sum(v)/len(v), 4) for k, v in intent_ndcg.items()})

        poor_logs = [l for l in sq_logs if l["ndcg_at_k"] < _NDCG_GOOD]
        if poor_logs:
            st.markdown("---")
            st.subheader(f"⚠️ 검색 품질 낮은 질문 (NDCG < {_NDCG_GOOD}) — {len(poor_logs)}건")
            for l in sorted(poor_logs, key=lambda x: x["ndcg_at_k"]):
                nd  = l["ndcg_at_k"]
                sqr = l.get("search_quality_report", {})
                lc  = "🔴" if nd < _NDCG_FAIR else "🟡"
                with st.expander(f"{lc} NDCG {nd:.4f} | {l.get('question','')[:60]}..."):
                    st.info(f"🩺 {sqr.get('diagnosis','진단 없음')}")

        st.markdown("---")
        rows = [{"시간":         l.get("timestamp",""),
                 "질문":         l.get("question","")[:50],
                 "Dense NDCG":   l["ndcg_at_k"],
                 "Reranker Gain": l.get("reranker_gain","-"),
                 "품질":         (l.get("search_quality_report") or {}).get("quality_label","-"),
                 "BM25":         "✅" if (l.get("search_quality_report") or {}).get("use_bm25") else "❌",
                 "의도":         (l.get("route_decision") or {}).get("의도","-"),
                 "Multi-Vector": "🔢" if l.get("mv_retrieval") else "-",   # [NEW v19]
                 "캐시 히트":    f"⚡{l['cache_hit']}" if l.get("cache_hit") else "-",  # [NEW v19]
                 "Fallback":     "🔄" if l.get("fallback_triggered") else "-",
                 "Self-Refine":  "✏️" if l.get("self_refinement") else "-"}
                for l in sq_logs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
