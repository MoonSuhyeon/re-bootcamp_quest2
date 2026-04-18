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
import math
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed  # [NEW v21]
from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE             = os.path.join(_BASE, "rag_eval_logs_v21.json")
EMBED_CACHE_FILE     = os.path.join(_BASE, "embed_cache_v21.pkl")
ANSWER_CACHE_FILE    = os.path.join(_BASE, "answer_cache_v21.json")
FAILURE_DATASET_FILE = os.path.join(_BASE, "failure_dataset_v21.json")

ANSWER_CACHE_TTL_SEC      = 1800
QUERY_CACHE_TTL_SEC       = 3600
FAILURE_THRESHOLD_ACCURACY = 3
PARALLEL_MAX_WORKERS      = 4    # [NEW v21] 병렬 검색 스레드 수
DEDUP_THRESHOLD_DEFAULT   = 0.85 # [NEW v21] Selective Context 중복 임계값

# 수치 계산 감지 패턴 [NEW v21]
CALC_PATTERNS = [
    r'\d+[,.]\d+', r'얼마', r'몇\s', r'합계', r'총\s', r'평균',
    r'비율', r'퍼센트', r'%', r'증가', r'감소', r'차이', r'계산',
    r'합산', r'곱하', r'나누', r'더하', r'빼'
]

st.set_page_config(page_title="RAG 챗봇 v21", page_icon="📚", layout="wide")


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
# Ablation / Fallback / Dynamic Retrieval 설정
# =====================================================================

ABLATION_CONFIGS = [
    {"id": "full",        "name": "✅ Full Pipeline",    "query_rewrite": True,  "bm25": True,  "rerank": True},
    {"id": "no_rewrite",  "name": "❌ No Query Rewrite", "query_rewrite": False, "bm25": True,  "rerank": True},
    {"id": "no_bm25",     "name": "❌ No BM25 (Dense만)", "query_rewrite": True,  "bm25": False, "rerank": True},
    {"id": "no_rerank",   "name": "❌ No Rerank",        "query_rewrite": True,  "bm25": True,  "rerank": False},
    {"id": "dense_rerank","name": "⚡ Dense + Rerank",   "query_rewrite": False, "bm25": False, "rerank": True},
    {"id": "minimal",     "name": "🔥 Minimal (기본만)", "query_rewrite": False, "bm25": False, "rerank": False},
]

MAX_RETRIES         = 2
FALLBACK_MIN_ACC    = 3
FALLBACK_HALL_TYPES = ("부분적", "있음")

DYNAMIC_RETRIEVAL_PROFILES = {
    "definition":     {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": False, "label": "정의형 → BM25 ↑, 빠른 처리"},
    "factual_lookup": {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 3, "top_k_delta": 0, "multidoc_override": None,  "label": "Fact형 → Rerank 강화"},
    "reasoning":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True,  "label": "분석형 → MultiDoc ↑, Rerank 강화"},
    "multi_hop":      {"bm25_boost": False, "rerank_force": True,  "prefilter_delta": 8, "top_k_delta": 1, "multidoc_override": True,  "label": "멀티홉 → MultiDoc 강제, 후보 확장"},
    "exploratory":    {"bm25_boost": True,  "rerank_force": False, "prefilter_delta": 5, "top_k_delta": 1, "multidoc_override": True,  "label": "탐색형 → BM25 ↑, 넓은 Recall"},
    "ambiguous":      {"bm25_boost": True,  "rerank_force": True,  "prefilter_delta": 5, "top_k_delta": 0, "multidoc_override": None,  "label": "의도불명 → BM25 + 강한 Rerank"},
}


def should_fallback(evaluation: dict) -> tuple:
    acc, hall = evaluation.get("정확도", 5), evaluation.get("환각여부", "없음")
    reasons = []
    if acc < FALLBACK_MIN_ACC:
        reasons.append(f"정확도 {acc}/5 < {FALLBACK_MIN_ACC}")
    if hall in FALLBACK_HALL_TYPES:
        reasons.append(f"환각 감지: {hall}")
    return (bool(reasons), " + ".join(reasons))


def escalate_params(base_eff: dict, base_prefilter_n: int, attempt: int) -> tuple:
    new_eff = base_eff.copy()
    new_eff["use_bm25"] = new_eff["use_reranking"] = new_eff["use_query_rewrite"] = True
    new_eff["top_k"] = min(5, base_eff["top_k"] + attempt)
    return new_eff, min(20, base_prefilter_n + attempt * 5)


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
    new_pf_n = min(20, prefilter_n + profile["prefilter_delta"])
    new_multidoc = use_multidoc if profile["multidoc_override"] is None else profile["multidoc_override"]
    return new_eff, new_pf_n, new_multidoc, profile["label"]


# =====================================================================
# 캐시 전략 (v19~)
# =====================================================================

class QueryResultCache:
    def __init__(self):
        self.hits = self.misses = 0

    def _store(self) -> dict:
        if "query_result_cache_store" not in st.session_state:
            st.session_state.query_result_cache_store = {}
        return st.session_state.query_result_cache_store

    def _key(self, question, use_bm25, prefilter_n):
        return hashlib.md5(f"{question}|{use_bm25}|{prefilter_n}".encode()).hexdigest()

    def get(self, question, use_bm25, prefilter_n, ttl=QUERY_CACHE_TTL_SEC):
        store, key = self._store(), self._key(question, use_bm25, prefilter_n)
        if key in store:
            e = store[key]
            if time.time() - e["ts"] < ttl:
                self.hits += 1
                return e["items"]
            del store[key]
        self.misses += 1
        return None

    def set(self, question, use_bm25, prefilter_n, items):
        store = self._store()
        store[self._key(question, use_bm25, prefilter_n)] = {"items": items, "ts": time.time()}

    def size(self):
        return len(self._store())

    def clear(self):
        st.session_state.query_result_cache_store = {}
        self.hits = self.misses = 0


class AnswerCache:
    def __init__(self, path):
        self.path = path
        self._cache = self._load()
        self.hits = self.misses = 0

    def _load(self):
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

    def get(self, key, ttl=ANSWER_CACHE_TTL_SEC):
        if key in self._cache:
            e = self._cache[key]
            if time.time() - e["ts"] < ttl:
                self.hits += 1
                return e["data"]
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, key, data):
        self._cache[key] = {"data": data, "ts": time.time()}
        self._save()

    def size(self):
        return len(self._cache)

    def valid_size(self, ttl=ANSWER_CACHE_TTL_SEC):
        now = time.time()
        return sum(1 for v in self._cache.values() if now - v["ts"] < ttl)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path):
            os.remove(self.path)
        self.hits = self.misses = 0


# =====================================================================
# 실패 데이터셋 (v20~)
# =====================================================================

class FailureDataset:
    def __init__(self, path):
        self.path = path
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, entry):
        self._data.append(entry)
        self._save()

    def get_all(self):
        return self._data

    def size(self):
        return len(self._data)

    def clear(self):
        self._data = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def export_finetune_jsonl(self) -> bytes:
        lines = []
        for entry in self._data:
            chunks_text = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(entry.get("chunks", []))])
            hint = entry.get("improvement_hint") or entry.get("answer", "")
            msg = {
                "messages": [
                    {"role": "system",    "content": "문서 기반 어시스턴트. **📌 요약** / **📖 근거** / **✅ 결론** 구조. 한국어."},
                    {"role": "user",      "content": f"[참고 문서]\n{chunks_text}\n\n[질문]\n{entry.get('question','')}"},
                    {"role": "assistant", "content": hint},
                ],
                "_meta": {"failure_types": entry.get("failure_types", []), "evaluation": entry.get("evaluation", {})}
            }
            lines.append(json.dumps(msg, ensure_ascii=False))
        return "\n".join(lines).encode("utf-8")

    def export_problems_json(self) -> bytes:
        export = [{
            "id": e.get("id",""), "timestamp": e.get("timestamp",""),
            "question": e.get("question",""), "failure_types": e.get("failure_types",[]),
            "accuracy": (e.get("evaluation") or {}).get("정확도","-"),
            "hallucination": (e.get("evaluation") or {}).get("환각여부","-"),
            "improvement_hint": e.get("improvement_hint",""),
            "issues": (e.get("quality_report") or {}).get("issues",[]),
        } for e in self._data]
        return json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")


def classify_failure_types(evaluation, quality_report, sqr=None):
    types = []
    acc, rel, hall = evaluation.get("정확도",5), evaluation.get("관련성",5), evaluation.get("환각여부","없음")
    miss = evaluation.get("누락_정보","없음")
    if acc <= FAILURE_THRESHOLD_ACCURACY: types.append("low_accuracy")
    if rel <= FAILURE_THRESHOLD_ACCURACY: types.append("low_relevance")
    if hall in ("부분적","있음"):         types.append("hallucination")
    if miss and miss != "없음":           types.append("incomplete_answer")
    if sqr and sqr.get("quality_label") in ("poor","fair"): types.append("retrieval_failure")
    return types


def generate_improvement_hint(question, chunks, answer, evaluation, failure_types, tracer=None):
    tracer and tracer.start("improvement_hint")
    context  = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(chunks)])
    fail_str = ", ".join(failure_types) if failure_types else "미분류"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "당신은 RAG 파이프라인 개선 전문가입니다.\n"
                "반드시 아래 형식으로 출력하세요:\n"
                "**실패 원인 분석**\n- ...\n\n**청크 개선 제안**\n- ...\n\n"
                "**쿼리 개선 제안**\n- ...\n\n**개선된 답변 방향**\n- ..."
            )},
            {"role": "user", "content": (
                f"[실패 유형] {fail_str}\n\n[질문]\n{question}\n\n[참고 문서]\n{context}\n\n"
                f"[실패한 답변]\n{answer}\n\n[평가]\n"
                f"정확도: {evaluation.get('정확도','-')}/5\n환각여부: {evaluation.get('환각여부','-')}\n"
                f"누락_정보: {evaluation.get('누락_정보','-')}\n개선_제안: {evaluation.get('개선_제안','-')}"
            )}
        ]
    )
    result = response.choices[0].message.content
    if tracer:
        u = response.usage
        tracer.end("improvement_hint",
                   tokens={"prompt": u.prompt_tokens, "completion": u.completion_tokens, "total": u.total_tokens},
                   input_summary=f"실패 유형: {fail_str}", output_summary=f"힌트 {len(result)}자",
                   decision="실패 케이스 자동 분석")
    return result


def build_failure_entry(question, answer, chunks, sources, evaluation, quality_report,
                         failure_types, improvement_hint=None, ndcg=None, sqr=None,
                         mode="", mv_retrieval=False):
    return {
        "id": str(uuid.uuid4())[:12], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question, "answer": answer, "chunks": chunks, "sources": sources,
        "evaluation": evaluation, "quality_report": quality_report,
        "failure_types": failure_types, "improvement_hint": improvement_hint,
        "retrieval_info": {"ndcg": ndcg, "quality_label": (sqr or {}).get("quality_label"),
                           "mode": mode, "mv_retrieval": mv_retrieval},
    }


# =====================================================================
# Tracer
# =====================================================================

class Tracer:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:8]
        self.spans: list = []
        self._active: dict = {}

    def start(self, name):
        self._active[name] = time.time()

    def end(self, name, tokens=None, input_summary="", output_summary="", decision="", error=""):
        start = self._active.pop(name, time.time())
        span = {"name": name, "duration_ms": int((time.time()-start)*1000),
                "tokens": tokens or {"prompt":0,"completion":0,"total":0},
                "input_summary": input_summary, "output_summary": output_summary,
                "decision": decision, "error": error}
        self.spans.append(span)
        return span

    def total_tokens(self):
        p = sum(s["tokens"]["prompt"] for s in self.spans)
        c = sum(s["tokens"]["completion"] for s in self.spans)
        return {"prompt": p, "completion": c, "total": p+c}

    def total_latency_ms(self):
        return sum(s["duration_ms"] for s in self.spans)

    def bottleneck(self):
        return max(self.spans, key=lambda s: s["duration_ms"])["name"] if self.spans else "-"


def _usage(response):
    u = response.usage
    return {"prompt": u.prompt_tokens, "completion": u.completion_tokens, "total": u.total_tokens}


# =====================================================================
# Embedding Cache
# =====================================================================

class EmbeddingCache:
    def __init__(self, path):
        self.path = path
        self._cache = self._load()
        self.hits = self.misses = 0

    def _load(self):
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

    def size(self): return len(self._cache)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path): os.remove(self.path)
        self.hits = self.misses = 0


embed_cache        = EmbeddingCache(EMBED_CACHE_FILE)
query_result_cache = QueryResultCache()
answer_cache       = AnswerCache(ANSWER_CACHE_FILE)
failure_dataset    = FailureDataset(FAILURE_DATASET_FILE)


def get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in response.data], dtype=np.float32)


def get_embeddings_cached(texts):
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    result_map, need_api = {}, []
    for text in texts:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in embed_cache._cache:
            result_map[text] = embed_cache._cache[key]
            embed_cache.hits += 1
        elif text not in result_map:
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


def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compute_ndcg(ordered_items, score_dict, k):
    if not score_dict: return 0.0
    n = min(k, len(ordered_items))
    if n == 0: return 0.0
    dcg  = sum(score_dict.get(i, 0.0) / np.log2(i+2) for i in range(n))
    idcg = sum(r / np.log2(i+2) for i, r in enumerate(sorted(score_dict.values(), reverse=True)[:n]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


_NDCG_EXCELLENT, _NDCG_GOOD, _NDCG_FAIR = 0.9, 0.7, 0.5


def compute_search_quality_report(ndcg_prefilter, use_bm25, n_candidates):
    reranker_gain = round(1.0 - ndcg_prefilter, 4)
    if ndcg_prefilter >= _NDCG_EXCELLENT:
        quality_label, diagnosis = "excellent", "임베딩 검색이 이미 최적 순서 → 리랭커는 확인 역할만 수행"
    elif ndcg_prefilter >= _NDCG_GOOD:
        quality_label, diagnosis = "good", "임베딩 검색 품질 양호 → 리랭커가 소폭 개선"
    elif ndcg_prefilter >= _NDCG_FAIR:
        quality_label, diagnosis = "fair", "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미"
    else:
        quality_label, diagnosis = "poor", "임베딩 검색 품질 낮음 → 청킹 전략·임베딩 모델 개선 검토 필요"
    return {"ndcg_prefilter": ndcg_prefilter, "reranker_gain": reranker_gain,
            "quality_label": quality_label, "diagnosis": diagnosis,
            "use_bm25": use_bm25, "n_candidates": n_candidates}


def linkify_citations(text):
    return re.sub(
        r'\[출처 (\d+)\]',
        lambda m: f'<a href="#source-{m.group(1)}" style="color:#1976D2;font-weight:bold;text-decoration:none;">[출처 {m.group(1)}]</a>',
        text
    )


# =====================================================================
# 기본 유틸
# =====================================================================

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
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
            if temp: raw_chunks.append(temp.strip())
        else:
            if len(current) + len(para) > chunk_size and current:
                raw_chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para
    if current: raw_chunks.append(current.strip())
    raw_chunks = [c for c in raw_chunks if c]
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks
    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        overlapped.append(raw_chunks[i-1][-overlap:] + " " + raw_chunks[i])
    return overlapped


def chunk_text_semantic(text, chunk_size=500, overlap=100):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', text) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return chunk_text_with_overlap(text, chunk_size, overlap)
    embeddings = normalize(get_embeddings(sentences))
    similarities = [float(np.dot(embeddings[i], embeddings[i+1])) for i in range(len(embeddings)-1)]
    threshold = np.mean(similarities) - 0.5 * np.std(similarities)
    breakpoints = [0] + [i+1 for i, sim in enumerate(similarities) if sim < threshold] + [len(sentences)]
    raw_chunks = []
    for i in range(len(breakpoints)-1):
        current = ""
        for sent in sentences[breakpoints[i]:breakpoints[i+1]]:
            if len(current) + len(sent) > chunk_size and current:
                raw_chunks.append(current.strip())
                current = sent
            else:
                current = current + " " + sent if current else sent
        if current: raw_chunks.append(current.strip())
    raw_chunks = [c for c in raw_chunks if c]
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks
    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        overlapped.append(raw_chunks[i-1][-overlap:] + " " + raw_chunks[i])
    return overlapped


def build_index(chunks):
    embeddings = normalize(get_embeddings_cached(chunks))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# =====================================================================
# 키워드 추출 (v19~)
# =====================================================================

_STOPWORDS = {
    "이","가","을","를","의","에","는","은","과","와","도","로","으로",
    "에서","에게","부터","까지","그","이것","저","것","및","등","위해",
    "the","a","an","is","are","was","were","in","on","at","to","for",
    "of","and","or","with","by","from","this","that","have","has","be",
}


def extract_keywords_simple(text, top_k=10):
    words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text)
    filtered = [w for w in words if w.lower() not in _STOPWORDS]
    counts = Counter(filtered)
    return " ".join([w for w, _ in counts.most_common(top_k)])


# =====================================================================
# Multi-Vector Index (v19~)
# =====================================================================

def build_multi_vector_index(chunks):
    chunk_embs = normalize(get_embeddings_cached(chunks))
    chunk_idx  = faiss.IndexFlatIP(chunk_embs.shape[1])
    chunk_idx.add(chunk_embs)

    all_sents, sent_to_chunk = [], []
    for ci, chunk in enumerate(chunks):
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        if not sents: sents = [chunk[:200]]
        for s in sents:
            all_sents.append(s)
            sent_to_chunk.append(ci)

    sent_embs = normalize(get_embeddings_cached(all_sents))
    sent_idx  = faiss.IndexFlatIP(sent_embs.shape[1])
    sent_idx.add(sent_embs)

    kw_strings = [extract_keywords_simple(c, top_k=10) for c in chunks]
    kw_embs    = normalize(get_embeddings_cached(kw_strings))
    kw_idx     = faiss.IndexFlatIP(kw_embs.shape[1])
    kw_idx.add(kw_embs)

    return {"chunk_index": chunk_idx, "sent_index": sent_idx, "kw_index": kw_idx,
            "sent_to_chunk": sent_to_chunk, "n_chunks": len(chunks), "n_sentences": len(all_sents)}


# =====================================================================
# [NEW v21] 병렬 검색 — ThreadPoolExecutor 4-way
# =====================================================================

def retrieve_parallel(queries: list, index, chunks: list, sources: list,
                      mv_index: dict = None, use_bm25: bool = True,
                      top_k_per_query: int = 20, tracer=None) -> tuple:
    """
    [v21] Dense / BM25 / Sentence-level / Keyword 검색을 ThreadPoolExecutor로 동시 실행.
    Fallback 재시도 시 직렬 검색 대비 대기 시간 획기적 단축.
    Returns: (result_list, parallel_ms)
    """
    tracer and tracer.start("parallel_search")
    t_start = time.time()
    n_chunks = len(chunks)
    RRF_K    = 60

    def _dense():
        scores = {}
        idx = mv_index["chunk_index"] if mv_index else index
        for query in queries:
            q_emb = normalize(get_embeddings_cached([query]))
            _, indices = idx.search(q_emb, min(top_k_per_query, n_chunks))
            for rank, ci in enumerate(indices[0]):
                if 0 <= ci < n_chunks:
                    scores[ci] = scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        return "dense", scores

    def _bm25():
        if not use_bm25 or not BM25_AVAILABLE:
            return "bm25", {}
        scores = {}
        bm25   = BM25Okapi([c.split() for c in chunks])
        for query in queries:
            arr = bm25.get_scores(query.split())
            for rank, ci in enumerate(np.argsort(arr)[::-1][:top_k_per_query]):
                if ci < n_chunks:
                    scores[int(ci)] = scores.get(int(ci), 0.0) + 1.0 / (RRF_K + rank + 1)
        return "bm25", scores

    def _sentence():
        if not mv_index: return "sentence", {}
        scores = {}
        sent_idx, stc = mv_index["sent_index"], mv_index["sent_to_chunk"]
        n_s = mv_index["n_sentences"]
        for query in queries:
            q_emb = normalize(get_embeddings_cached([query]))
            _, sr  = sent_idx.search(q_emb, min(top_k_per_query * 2, n_s))
            seen   = {}
            for rank, si in enumerate(sr[0]):
                if 0 <= si < len(stc):
                    ci = stc[si]
                    if ci not in seen: seen[ci] = rank
            for ci, rank in seen.items():
                scores[ci] = scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        return "sentence", scores

    def _keyword():
        if not mv_index: return "keyword", {}
        scores  = {}
        kw_idx  = mv_index["kw_index"]
        for query in queries:
            kw = extract_keywords_simple(query, top_k=8)
            if not kw.strip(): continue
            kq_emb = normalize(get_embeddings_cached([kw]))
            _, kr  = kw_idx.search(kq_emb, min(top_k_per_query, n_chunks))
            for rank, ci in enumerate(kr[0]):
                if 0 <= ci < n_chunks:
                    scores[int(ci)] = scores.get(int(ci), 0.0) + 1.0 / (RRF_K + rank + 1)
        return "keyword", scores

    rrf_total: dict = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as ex:
        futures = [ex.submit(_dense), ex.submit(_bm25), ex.submit(_sentence), ex.submit(_keyword)]
        for fut in as_completed(futures):
            _, partial = fut.result()
            for ci, sc in partial.items():
                rrf_total[ci] = rrf_total.get(ci, 0.0) + sc

    sorted_ci    = sorted(rrf_total, key=lambda i: rrf_total[i], reverse=True)
    result       = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
    parallel_ms  = int((time.time() - t_start) * 1000)

    if tracer:
        tracer.end("parallel_search",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개 × 4 채널 병렬",
                   output_summary=f"{len(result)}개 후보 ({parallel_ms}ms)",
                   decision=f"ThreadPoolExecutor(workers={PARALLEL_MAX_WORKERS}) RRF 통합")
    return result, parallel_ms


# =====================================================================
# [NEW v21] LongContextReorder — "Lost in the Middle" 방지
# =====================================================================

def reorder_lost_in_middle(chunks: list, scores: list) -> list:
    """
    [v21] "Lost in the Middle" 논문 기반 청크 재정렬.
    관련성 높은 청크 → 시작·끝 배치 / 낮은 청크 → 가운데 배치.
    LLM은 프롬프트 시작·끝에 주목하고 중간은 놓치는 경향 보정.
    """
    if len(chunks) <= 2:
        return chunks
    paired  = sorted(zip(scores or [0]*len(chunks), chunks),
                     key=lambda x: x[0] or 0, reverse=True)
    result  = [None] * len(paired)
    left, right = 0, len(paired) - 1
    for i, (_, chunk) in enumerate(paired):
        if i % 2 == 0:
            result[left]  = chunk
            left  += 1
        else:
            result[right] = chunk
            right -= 1
    return [c for c in result if c is not None]


# =====================================================================
# [NEW v21] Selective Context Phase 2 — Cross-chunk 중복 제거
# =====================================================================

def selective_context_phase2(question: str, chunks: list,
                              dedup_threshold: float = DEDUP_THRESHOLD_DEFAULT,
                              max_sentences_per_chunk: int = 6,
                              tracer=None) -> tuple:
    """
    [v21] Phase 2 Context Compression:
    1. 전체 청크의 문장을 모아 Cross-chunk 코사인 유사도 기반 중복 제거
    2. 질문 관련성 기반 최종 선별
    → 청크 내 중복(v19 Phase1)이 아닌 청크 간 중복 제거가 핵심
    Returns: (compressed_chunks, stats_dict)
    """
    tracer and tracer.start("selective_context")

    # 모든 문장 수집
    all_sents = []  # (chunk_idx, sent_text)
    for ci, chunk in enumerate(chunks):
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        for s in sents:
            all_sents.append((ci, s))

    if not all_sents:
        return chunks, {"original_sents": 0, "after_dedup": 0, "ratio": 1.0}

    sent_texts = [s[1] for s in all_sents]
    sent_embs  = normalize(get_embeddings_cached(sent_texts))
    q_emb      = normalize(get_embeddings_cached([question]))[0]

    # Cross-chunk 중복 제거
    kept_indices, kept_embs = [], []
    for i, emb in enumerate(sent_embs):
        is_dup = any(float(np.dot(emb, ke)) >= dedup_threshold for ke in kept_embs)
        if not is_dup:
            kept_indices.append(i)
            kept_embs.append(emb)

    # 질문 관련성 점수 계산
    kept_sents = [(all_sents[i][0], all_sents[i][1],
                   float(np.dot(sent_embs[i], q_emb)))
                  for i in kept_indices]

    # 청크별 재조합 (관련성 기준 상위 max_sentences_per_chunk 개)
    chunk_sents: dict = {ci: [] for ci in range(len(chunks))}
    for ci, text, score in kept_sents:
        chunk_sents[ci].append((text, score))

    compressed = []
    for ci in range(len(chunks)):
        sents = sorted(chunk_sents[ci], key=lambda x: x[1], reverse=True)[:max_sentences_per_chunk]
        # 원문 순서 유지를 위해 원본에서 순서 추적
        ordered = [s for s, _ in sents]
        compressed.append(" ".join(ordered) if ordered else chunks[ci][:200])

    orig_chars = sum(len(c) for c in chunks)
    comp_chars = sum(len(c) for c in compressed)
    stats = {
        "original_sents": len(all_sents),
        "after_dedup":    len(kept_indices),
        "dedup_removed":  len(all_sents) - len(kept_indices),
        "orig_chars":     orig_chars,
        "comp_chars":     comp_chars,
        "ratio":          round(comp_chars / max(orig_chars, 1), 2),
    }

    if tracer:
        tracer.end("selective_context",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(chunks)}청크 / {len(all_sents)}문장",
                   output_summary=f"중복 {stats['dedup_removed']}개 제거 → {stats['ratio']:.0%}",
                   decision=f"Cross-chunk 코사인 중복 제거 (임계값 {dedup_threshold})")
    return compressed, stats


# =====================================================================
# [NEW v21] Tool-Augmented RAG — Python 수치 계산
# =====================================================================

def detect_calc_intent(question: str, chunks: list) -> bool:
    """[v21] 수치 계산이 필요한 질문인지 감지"""
    q_calc = any(re.search(p, question) for p in CALC_PATTERNS)
    ctx_nums = any(re.search(r'\d{2,}', c) for c in chunks)
    return q_calc and ctx_nums


def _safe_eval(code: str, data: dict):
    """[v21] 제한된 환경에서 Python 수식 안전 실행"""
    safe_globals = {
        "__builtins__": {},
        "math": math, "round": round, "abs": abs, "sum": sum,
        "max": max, "min": min, "len": len, "int": int, "float": float,
        "list": list, "range": range, "sorted": sorted,
    }
    safe_globals.update({k: v for k, v in data.items() if isinstance(v, (int, float, list))})
    try:
        exec(compile(code, "<string>", "exec"), safe_globals)
        return safe_globals.get("result", None)
    except Exception:
        try:
            import ast
            tree = ast.parse(code, mode='eval')
            return eval(compile(tree, '', 'eval'), safe_globals)
        except Exception:
            return None


def tool_augmented_answer(question: str, chunks: list, tracer=None) -> tuple:
    """
    [v21] Tool-Augmented 답변 생성:
    1. LLM이 문서에서 수치 데이터 추출 + Python 코드 생성
    2. 안전한 Python 실행으로 정확한 계산 결과 도출
    3. 계산 결과를 LLM에 주입 → '수치 환각' 방지
    Returns: (answer, python_code, calc_result) or (None, None, None)
    """
    if not detect_calc_intent(question, chunks):
        return None, None, None

    tracer and tracer.start("tool_extract")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(chunks)])

    # Step 1: LLM이 수치 추출 + Python 코드 생성
    resp1 = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "문서에서 수치 계산에 필요한 데이터를 추출하고 Python 코드를 생성하세요.\n"
                "반드시 JSON 형식으로만 출력하세요:\n"
                "{\n"
                "  \"has_calculation\": true 또는 false,\n"
                "  \"data\": {\"변수명\": 숫자값},\n"
                "  \"python_code\": \"result = 변수명 * 2\",\n"
                "  \"explanation\": \"계산 내용 한 줄 설명\"\n"
                "}\n"
                "python_code는 반드시 마지막 줄에 result = ... 형태로 끝나야 합니다."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[문서]\n{context}"}
        ]
    )

    try:
        extracted = json.loads(resp1.choices[0].message.content)
    except Exception:
        extracted = {"has_calculation": False}

    if tracer:
        tracer.end("tool_extract", tokens=_usage(resp1),
                   input_summary=question[:60],
                   output_summary=f"계산 필요: {extracted.get('has_calculation', False)}",
                   decision="수치 추출 + Python 코드 생성")

    if not extracted.get("has_calculation", False):
        return None, None, None

    python_code = extracted.get("python_code", "")
    calc_result = _safe_eval(python_code, extracted.get("data", {}))

    if calc_result is None:
        return None, python_code, None

    # Step 2: 계산 결과 기반 최종 답변 생성
    tracer and tracer.start("tool_answer")
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 기반 어시스턴트. 아래 구조로 답변하세요:\n"
                "**📌 요약** / **📖 근거** ([출처 N] 인용) / "
                "**🔢 계산 결과** (Python으로 정확히 계산된 값 사용) / "
                "**✅ 결론** (확신도: 높음/보통/낮음)\n"
                "중요: 계산 결과는 제공된 Python 값을 그대로 쓰세요. 절대 재계산하지 마세요."
            )},
            {"role": "user", "content": (
                f"[참고 문서]\n{context}\n\n[질문]\n{question}\n\n"
                f"[Python 계산 결과 — 이 값이 정확합니다]\n"
                f"코드: {python_code}\n결과: {calc_result}\n"
                f"설명: {extracted.get('explanation', '')}"
            )}
        ]
    )
    answer = resp2.choices[0].message.content
    if tracer:
        tracer.end("tool_answer", tokens=_usage(resp2),
                   input_summary=f"계산 결과: {calc_result}",
                   output_summary=f"Tool-Augmented 답변 {len(answer)}자",
                   decision="Python 계산 결과 주입 → 수치 환각 방지")
    return answer, python_code, str(calc_result)


# =====================================================================
# 쿼리 라우팅
# =====================================================================

def route_query(question, tracer=None):
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
            "검색_전략": {"dense_weight":0.5,"bm25_weight":0.5,"reranker_사용여부":True,
                          "reranker_모드":"heavy","top_k":3,"query_rewrite_필요":True,
                          "query_분해_필요":False,"recall_우선순위":True},
            "메타데이터_전략": {"메타데이터_필터_사용":False,"선호_출처":[],"시간_가중치":"없음"},
            "설명": f"라우팅 실패 → fallback: {str(e)}"
        }
    if tracer:
        s   = result.get("검색_전략", {})
        tok = _usage(resp_obj) if resp_obj else {"prompt":0,"completion":0,"total":0}
        tracer.end("query_routing", tokens=tok,
                   input_summary=question[:60],
                   output_summary=f"의도: {result.get('의도','-')} | top_k: {s.get('top_k','-')}",
                   decision=result.get("설명",""))
    return result


def _apply_routing(route, defaults):
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

def rewrite_queries(original_query, n=3, tracer=None, use_session_cache=True):
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
                f"문서 검색 전문가. 의도 분해형 우선 총 {n}개 재작성. 번호·기호 없이 줄당 하나."
            )},
            {"role": "user", "content": f"원본 질문: {original_query}"}
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    queries  = [original_query] + variants[:n]

    if use_session_cache:
        session_cache = st.session_state.get("rewrite_cache", {})
        session_cache[f"{original_query}||{n}"] = queries
        st.session_state.rewrite_cache = session_cache

    if tracer:
        tracer.end("query_rewriting", tokens=_usage(response),
                   input_summary=f"원본: {original_query[:60]}",
                   output_summary=f"쿼리 {len(queries)}개 생성",
                   decision=f"의도 분해형 우선")
    return queries


def retrieve_hybrid(queries, index, chunks, sources, top_k_per_query=20, use_bm25=True, tracer=None):
    tracer and tracer.start("embedding_search")
    seen_dense, dense_items = set(), []
    for query in queries:
        q_emb = normalize(get_embeddings_cached([query]))
        _, indices = index.search(q_emb, top_k_per_query)
        for i in indices[0]:
            if i < len(chunks) and i not in seen_dense:
                seen_dense.add(i)
                dense_items.append((chunks[i], sources[i] if sources else "알 수 없음"))

    if not use_bm25 or not BM25_AVAILABLE:
        tracer and tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                              input_summary=f"쿼리 {len(queries)}개",
                              output_summary=f"Dense only: {len(dense_items)}개",
                              decision="BM25 비활성화")
        return dense_items

    bm25 = BM25Okapi([c.split() for c in chunks])
    seen_bm25, bm25_items = set(), []
    for query in queries:
        scores_arr = bm25.get_scores(query.split())
        for i in np.argsort(scores_arr)[::-1][:top_k_per_query]:
            if i < len(chunks) and i not in seen_bm25:
                seen_bm25.add(i)
                bm25_items.append((chunks[i], sources[i] if sources else "알 수 없음"))

    RRF_K = 60
    rrf_scores, item_by_key = {}, {}
    def _key(item): return hashlib.md5((item[0][:120]+item[1]).encode()).hexdigest()
    for rank, item in enumerate(dense_items):
        k = _key(item)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0/(RRF_K+rank+1)
        item_by_key[k] = item
    for rank, item in enumerate(bm25_items):
        k = _key(item)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0/(RRF_K+rank+1)
        item_by_key[k] = item

    result = [item_by_key[k] for k in sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)]
    if tracer:
        tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                   input_summary=f"쿼리 {len(queries)}개",
                   output_summary=f"Dense {len(dense_items)} + BM25 {len(bm25_items)} → RRF {len(result)}개",
                   decision="RRF(k=60)")
    return result


def prefilter_by_similarity(query, items, top_n=10, tracer=None):
    tracer and tracer.start("prefilter")
    if len(items) <= top_n:
        tracer and tracer.end("prefilter", input_summary=f"{len(items)}개",
                              output_summary=f"{len(items)}개 유지", decision="top_n 이하")
        return items
    chunk_texts = [item[0] for item in items]
    q_emb  = normalize(get_embeddings_cached([query]))[0]
    sims   = normalize(get_embeddings_cached(chunk_texts)) @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_n]
    result  = [items[i] for i in top_idx]
    cutoff  = round(float(sims[top_idx[-1]]), 4)
    if tracer:
        tracer.end("prefilter", tokens={"prompt":0,"completion":0,"total":0},
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {len(result)}개 (컷오프: {cutoff})",
                   decision=f"코사인 ≥ {cutoff}")
    return result


def rerank_chunks(query, items, top_k=3, tracer=None):
    tracer and tracer.start("rerank")
    if not items: return [], {}
    chunks_text = "\n\n".join([f"[{i+1}] {item[0]}" for i, item in enumerate(items)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "질문 대비 각 청크 관련성을 0~10점으로. 형식: 1: 8\n2: 3\n..."},
            {"role": "user",   "content": f"질문: {query}\n\n청크들:\n{chunks_text}"}
        ]
    )
    all_scores = {}
    for line in response.choices[0].message.content.strip().split('\n'):
        parts = line.split(':')
        if len(parts) == 2:
            try:
                idx = int(parts[0].strip()) - 1
                sc  = float(parts[1].strip())
                if 0 <= idx < len(items): all_scores[idx] = sc
            except ValueError:
                pass
    scored = [(items[i][0], items[i][1], all_scores.get(i, 0.0)) for i in range(len(items))]
    scored.sort(key=lambda x: x[2], reverse=True)
    result = scored[:top_k]
    if tracer:
        tracer.end("rerank", tokens=_usage(response),
                   input_summary=f"{len(items)}개 후보",
                   output_summary=f"상위 {top_k}개 (점수: {', '.join([f'{s[2]:.1f}' for s in result])})",
                   decision=f"LLM 0~10 채점")
    return result, all_scores


def compress_chunks(question, chunks, max_sentences=5, min_sim=0.25, tracer=None):
    tracer and tracer.start("context_compression")
    q_emb = normalize(get_embeddings_cached([question]))[0]
    compressed, stats = [], []
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
        stats.append({"original": len(chunk), "compressed": len(comp_text),
                      "ratio": round(len(comp_text)/max(len(chunk),1), 2)})
    if tracer:
        orig_t = sum(s["original"] for s in stats)
        comp_t = sum(s["compressed"] for s in stats)
        avg_r  = round(comp_t/max(orig_t,1), 2)
        tracer.end("context_compression", tokens={"prompt":0,"completion":0,"total":0},
                   input_summary=f"{len(chunks)}개 청크 ({orig_t}자)",
                   output_summary=f"압축 후 {comp_t}자 (평균 {avg_r:.0%})",
                   decision=f"문장 유사도 ≥ {min_sim}")
    return compressed, stats


# =====================================================================
# 답변 생성 단계들
# =====================================================================

def step1_summarize_chunks(question, chunks, tracer=None):
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
            if 0 <= idx < len(chunks): summaries[idx] = m.group(2).strip()
    result = [summaries.get(i, chunks[i][:120]+"...") for i in range(len(chunks))]
    tracer and tracer.end("step1_summarize", tokens=_usage(response),
                           input_summary=f"{len(chunks)}개 청크",
                           output_summary=f"{len(result)}개 요약",
                           decision="질문 관점 핵심 추출")
    return result


def step2_analyze_relationships(question, summaries, sources, tracer=None):
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
    tracer and tracer.end("step2_analyze", tokens=_usage(response),
                           input_summary=f"{len(summaries)}개 요약",
                           output_summary="공통점/차이점/핵심/불확실성 완료",
                           decision="불확실성 분석")
    return result


def step3_generate_final_answer(question, chunks, summaries, analysis, tracer=None):
    tracer and tracer.start("step3_answer")
    numbered_context = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(chunks)])
    summaries_text   = "\n".join([f"[출처 {i+1}] {s}" for i, s in enumerate(summaries)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 기반 어시스턴트. 규칙: "
                "1. **📌 요약** / **📖 근거** ([출처 N] 인용) / **✅ 결론** (확신도: 높음/보통/낮음). "
                "2. 불확실성 있으면 확신도 반영. 3. 문서 밖 내용 금지. 4. 한국어."
            )},
            {"role": "user", "content": (
                f"[청크 요약]\n{summaries_text}\n\n[분석]\n{analysis}\n\n"
                f"[참고 문서 원문]\n{numbered_context}\n\n[질문]\n{question}"
            )}
        ]
    )
    result = response.choices[0].message.content
    tracer and tracer.end("step3_answer", tokens=_usage(response),
                           input_summary="요약+분석+원문",
                           output_summary=f"답변 {len(result)}자",
                           decision="Step1+Step2 힌트로 최종 답변")
    return result


def generate_answer_simple(question, items, tracer=None):
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
    tracer and tracer.end("step3_answer", tokens=_usage(response),
                           input_summary="원문 직접",
                           output_summary=f"답변 {len(result)}자",
                           decision="단순 모드")
    return result


def evaluate_answer(question, context_chunks, answer, tracer=None):
    tracer and tracer.start("evaluation")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "답변 품질 진단 전문가. 아래 형식으로만 출력하라.\n\n"
                "정확도: <1~5 정수>\n관련성: <1~5 정수>\n환각여부: <없음|부분적|있음>\n"
                "환각근거: <없으면 '없음'>\n신뢰도: <높음|보통|낮음>\n"
                "불일치_항목: <없으면 '없음'>\n누락_정보: <없으면 '없음'>\n개선_제안: <1문장>"
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ]
    )
    result = {"정확도":0,"관련성":0,"환각여부":"알 수 없음","환각근거":"",
              "신뢰도":"보통","불일치_항목":"없음","누락_정보":"없음","개선_제안":""}
    for line in response.choices[0].message.content.strip().split('\n'):
        if line.startswith("정확도:"):
            try: result["정확도"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
            except ValueError: pass
        elif line.startswith("관련성:"):
            try: result["관련성"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
            except ValueError: pass
        elif line.startswith("환각여부:"):   result["환각여부"]   = line.split(':',1)[1].strip()
        elif line.startswith("환각근거:"):   result["환각근거"]   = line.split(':',1)[1].strip()
        elif line.startswith("신뢰도:"):     result["신뢰도"]     = line.split(':',1)[1].strip()
        elif line.startswith("불일치_항목:"): result["불일치_항목"] = line.split(':',1)[1].strip()
        elif line.startswith("누락_정보:"):  result["누락_정보"]  = line.split(':',1)[1].strip()
        elif line.startswith("개선_제안:"):  result["개선_제안"]  = line.split(':',1)[1].strip()
    tracer and tracer.end("evaluation", tokens=_usage(response),
                           input_summary="질문+문서+답변",
                           output_summary=f"정확도 {result['정확도']}/5 · 환각 {result['환각여부']}",
                           decision="구조화 품질 진단")
    return result


def analyze_hallucination_cause(question, context_chunks, answer, hall_type, tracer=None):
    if hall_type == "없음": return None
    tracer and tracer.start("hallucination_analysis")
    context  = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "환각 원인 정밀 분석 전문가. 아래 형식으로만 출력하라.\n\n"
                "환각_주장: ...\n환각_유형: <fabrication|distortion|over-generalization>\n"
                "심각도: <1~10>\n근거_출처: ...\n원문_인용: ...\n"
                "발생_원인: <insufficient_context|ambiguous_chunk|llm_interpolation>\n"
                "개선_제안: ...\n수정_제안: ..."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ]
    )
    raw    = response.choices[0].message.content.strip()
    result = {"환각_주장":"","환각_유형":"","심각도":0,"근거_출처":"","원문_인용":"",
              "발생_원인":"","개선_제안":"","수정_제안":"","raw":raw}
    for line in raw.split('\n'):
        for key in ["환각_주장","환각_유형","근거_출처","원문_인용","발생_원인","개선_제안","수정_제안"]:
            if line.startswith(f"{key}:"): result[key] = line.split(':',1)[1].strip()
        if line.startswith("심각도:"):
            try: result["심각도"] = max(1, min(10, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
            except ValueError: pass
    tracer and tracer.end("hallucination_analysis", tokens=_usage(response),
                           input_summary=f"환각: {hall_type}",
                           output_summary=f"심각도: {result['심각도']}/10",
                           decision="환각 감지 시에만 실행")
    return result


def build_quality_report(evaluation, hall_cause=None):
    acc, rel, hall = evaluation.get("정확도",0), evaluation.get("관련성",0), evaluation.get("환각여부","없음")
    hall_penalty   = {"없음":0.0,"부분적":0.5,"있음":1.5}.get(hall, 0.0)
    overall = round(max(0.0, min(5.0, (acc+rel)/2 - hall_penalty)), 2)
    grade   = "A" if overall>=4.5 else "B" if overall>=3.5 else "C" if overall>=2.5 else "D" if overall>=1.5 else "F"
    issues  = []
    if acc < 3: issues.append(f"낮은 정확도 ({acc}/5)")
    if rel < 3: issues.append(f"낮은 관련성 ({rel}/5)")
    if hall != "없음": issues.append(f"환각 감지: {hall} (심각도 {(hall_cause or {}).get('심각도','-')}/10)")
    mismatch = evaluation.get("불일치_항목","없음")
    if mismatch and mismatch != "없음": issues.append(f"문서-답변 불일치: {mismatch[:60]}")
    missing = evaluation.get("누락_정보","없음")
    if missing and missing != "없음": issues.append(f"누락 정보: {missing[:60]}")
    if evaluation.get("신뢰도") == "낮음": issues.append("전반적 신뢰도 낮음")
    recs = []
    if evaluation.get("개선_제안"): recs.append(evaluation["개선_제안"])
    if hall_cause and hall_cause.get("수정_제안"): recs.append(f"[환각 수정] {hall_cause['수정_제안']}")
    if hall_cause and hall_cause.get("개선_제안"): recs.append(f"[파이프라인] {hall_cause['개선_제안']}")
    return {"overall_score": overall, "grade": grade, "issues": issues, "recommendations": recs}


def critique_answer(question, context_chunks, draft, tracer=None):
    tracer and tracer.start("critique")
    context  = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "RAG 답변 품질 심사 전문가.\n"
                "반드시 아래 형식으로만 출력하세요:\n"
                "**문제점**\n- ...\n\n**누락**\n- ...\n\n**개선 방향**\n- ...\n"
                "문제가 없으면 각 항목에 '없음'."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft 답변]\n{draft}"}
        ]
    )
    result = response.choices[0].message.content
    tracer and tracer.end("critique", tokens=_usage(response),
                           input_summary=f"Draft {len(draft)}자",
                           output_summary=f"비판 {len(result)}자",
                           decision="Draft 문제점·누락·개선 방향 도출")
    return result


def refine_answer(question, context_chunks, draft, critique, tracer=None):
    tracer and tracer.start("refine")
    context  = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(context_chunks)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 기반 어시스턴트. Draft + Critique를 참고해 개선된 최종 답변 작성.\n"
                "규칙: 1. **📌 요약** / **📖 근거** / **✅ 결론** 구조.\n"
                "2. Critique의 문제점·누락 항목을 반드시 반영. 3. 문서 외 내용 금지. 4. 한국어."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft]\n{draft}\n\n[Critique]\n{critique}"}
        ]
    )
    result = response.choices[0].message.content
    tracer and tracer.end("refine", tokens=_usage(response),
                           input_summary=f"Draft {len(draft)}자 + Critique",
                           output_summary=f"Refined 답변 {len(result)}자",
                           decision="Critique 반영 최종 답변")
    return result


# =====================================================================
# 단일 파이프라인 실행 (v21)
# =====================================================================

def run_rag_pipeline(question: str, eff: dict,
                     index, chunks: list, sources: list,
                     prefilter_n: int, use_multidoc: bool,
                     num_rewrites: int = 3,
                     use_session_cache: bool = True,
                     use_self_refine: bool = False,
                     use_compression: bool = False,
                     mv_index: dict = None,
                     auto_save_failure: bool = True,
                     gen_improvement_hint: bool = False,
                     # [NEW v21]
                     use_parallel_search: bool = True,
                     use_lim_reorder: bool = True,
                     use_selective_context: bool = False,
                     selective_dedup_thresh: float = DEDUP_THRESHOLD_DEFAULT,
                     use_tool_augment: bool = False) -> dict:
    """
    [v21] RAG 파이프라인.
    신규: ① 병렬 검색(ThreadPoolExecutor) ② LongContextReorder ③ Selective Context Phase2 ④ Tool-Augmented
    """
    tracer = Tracer()

    # ── 답변 캐시 체크 ────────────────────────────────────────────
    if use_session_cache:
        ans_key    = hashlib.md5(question.encode("utf-8")).hexdigest()
        cached_ans = answer_cache.get(ans_key)
        if cached_ans:
            tracer.start("answer_cache_hit")
            tracer.end("answer_cache_hit", input_summary=question[:60],
                       output_summary="답변 캐시 히트 → 모든 LLM 호출 스킵",
                       decision=f"TTL {ANSWER_CACHE_TTL_SEC//60}분 이내 캐시")
            return {
                "tracer": tracer, "queries": [question],
                "ranked": [], "final_chunks": [], "final_sources": [], "final_scores": [],
                "gen_chunks": [], "summaries": [], "analysis": "",
                "answer": cached_ans["answer"], "draft_answer": cached_ans["answer"], "critique": None,
                "mode": "answer_cache_hit",
                "evaluation": cached_ans["evaluation"], "hall_cause": None,
                "quality_report": cached_ans["quality_report"],
                "ndcg_k": None, "sqr": None, "eff": eff.copy(), "prefilter_n": prefilter_n,
                "cache_hit": "answer", "compression_stats": None, "selective_stats": None,
                "failure_types": [], "failure_saved": False,
                "parallel_ms": None, "tool_code": None, "calc_result": None, "tool_used": False,
            }

    # 1. Query rewriting
    queries = rewrite_queries(question, n=num_rewrites, tracer=tracer,
                               use_session_cache=use_session_cache) if eff["use_query_rewrite"] else [question]

    # 2. 쿼리 결과 캐시 체크
    cache_hit = None
    qr_cached = query_result_cache.get(question, eff["use_bm25"], prefilter_n) if use_session_cache else None
    parallel_ms = None

    if qr_cached is not None:
        filtered  = qr_cached
        cache_hit = "query"
        tracer.start("embedding_search")
        tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                   input_summary="쿼리 결과 캐시 히트",
                   output_summary=f"{len(filtered)}개 (캐시 재사용)",
                   decision=f"TTL {QUERY_CACHE_TTL_SEC//60}분 이내 캐시")
        tracer.start("prefilter")
        tracer.end("prefilter", input_summary="캐시", output_summary="캐시", decision="캐시 히트 스킵")
    else:
        # 3. 검색 — [NEW v21] 병렬 OR 기존 순차
        if use_parallel_search:
            candidates, parallel_ms = retrieve_parallel(
                queries, index, chunks, sources,
                mv_index=mv_index, use_bm25=eff["use_bm25"],
                top_k_per_query=20, tracer=tracer
            )
        elif mv_index:
            candidates = _retrieve_mv_sequential(queries, mv_index, chunks, sources, eff["use_bm25"], tracer)
        else:
            candidates = retrieve_hybrid(queries, index, chunks, sources,
                                         use_bm25=eff["use_bm25"], tracer=tracer)
        filtered = prefilter_by_similarity(question, candidates, prefilter_n, tracer)
        if use_session_cache:
            query_result_cache.set(question, eff["use_bm25"], prefilter_n, filtered)

    # 4. Reranking
    ndcg_k, sqr = None, None
    eff_top_k   = eff["top_k"]
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

    # [NEW v21] 4b. LongContextReorder
    if use_lim_reorder and len(final_chunks) > 2:
        final_chunks = reorder_lost_in_middle(final_chunks, final_scores)

    # 5. Context Compression
    compression_stats = None
    selective_stats   = None
    gen_chunks        = final_chunks

    if use_selective_context and final_chunks:
        # [NEW v21] Phase 2: Cross-chunk 중복 제거
        gen_chunks, selective_stats = selective_context_phase2(
            question, final_chunks,
            dedup_threshold=selective_dedup_thresh, tracer=tracer
        )
    elif use_compression and final_chunks:
        # Phase 1 (v19)
        gen_chunks, compression_stats = compress_chunks(question, final_chunks, tracer=tracer)

    # 6. 답변 생성
    tool_code   = None
    calc_result = None
    tool_used   = False

    # [NEW v21] 6a. Tool-Augmented 먼저 시도
    if use_tool_augment and gen_chunks:
        ta_answer, tool_code, calc_result = tool_augmented_answer(question, gen_chunks, tracer)
        if ta_answer:
            answer    = ta_answer
            tool_used = True
            summaries, analysis = [], ""
            mode = "tool_augmented"
        else:
            tool_used = False

    if not tool_used:
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

    # 답변 캐시 저장
    if use_session_cache:
        ans_key = hashlib.md5(question.encode("utf-8")).hexdigest()
        answer_cache.set(ans_key, {"answer": answer, "evaluation": evaluation, "quality_report": quality_report})

    # 실패 케이스 저장
    failure_types, failure_saved = [], False
    if auto_save_failure:
        failure_types = classify_failure_types(evaluation, quality_report, sqr)
        if failure_types:
            hint = generate_improvement_hint(question, gen_chunks, answer, evaluation, failure_types, tracer) \
                   if gen_improvement_hint and gen_chunks else None
            failure_dataset.add(build_failure_entry(
                question, answer, gen_chunks, final_sources,
                evaluation, quality_report, failure_types,
                improvement_hint=hint, ndcg=ndcg_k, sqr=sqr,
                mode=mode, mv_retrieval=(mv_index is not None)
            ))
            failure_saved = True

    return {
        "tracer": tracer, "queries": queries,
        "ranked": ranked, "final_chunks": final_chunks, "final_sources": final_sources,
        "final_scores": final_scores, "gen_chunks": gen_chunks,
        "summaries": summaries, "analysis": analysis,
        "answer": answer, "draft_answer": draft_answer, "critique": critique,
        "mode": mode,
        "evaluation": evaluation, "hall_cause": hall_cause, "quality_report": quality_report,
        "ndcg_k": ndcg_k, "sqr": sqr, "eff": eff.copy(), "prefilter_n": prefilter_n,
        "cache_hit": cache_hit, "compression_stats": compression_stats, "selective_stats": selective_stats,
        "failure_types": failure_types, "failure_saved": failure_saved,
        "parallel_ms": parallel_ms, "tool_code": tool_code,
        "calc_result": calc_result, "tool_used": tool_used,
    }


def _retrieve_mv_sequential(queries, mv_index, chunks, sources, use_bm25, tracer):
    """Multi-Vector 순차 검색 (병렬 비활성 시 폴백)"""
    tracer and tracer.start("embedding_search")
    RRF_K, n_chunks = 60, mv_index["n_chunks"]
    rrf = {}
    for query in queries:
        q_emb = normalize(get_embeddings_cached([query]))
        _, cr = mv_index["chunk_index"].search(q_emb, min(20, n_chunks))
        for rank, ci in enumerate(cr[0]):
            if 0 <= ci < n_chunks:
                rrf[ci] = rrf.get(ci, 0.0) + 1.0/(RRF_K+rank+1)
        _, sr = mv_index["sent_index"].search(q_emb, min(40, mv_index["n_sentences"]))
        seen = {}
        for rank, si in enumerate(sr[0]):
            if 0 <= si < len(mv_index["sent_to_chunk"]):
                ci = mv_index["sent_to_chunk"][si]
                if ci not in seen: seen[ci] = rank
        for ci, rank in seen.items():
            rrf[ci] = rrf.get(ci, 0.0) + 1.0/(RRF_K+rank+1)
    if use_bm25 and BM25_AVAILABLE:
        bm25 = BM25Okapi([c.split() for c in chunks])
        for query in queries:
            arr = bm25.get_scores(query.split())
            for rank, ci in enumerate(np.argsort(arr)[::-1][:20]):
                rrf[int(ci)] = rrf.get(int(ci), 0.0) + 1.0/(RRF_K+rank+1)
    sorted_ci = sorted(rrf, key=lambda i: rrf[i], reverse=True)
    result = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
    tracer and tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                           input_summary=f"쿼리 {len(queries)}개",
                           output_summary=f"Multi-Vector 순차 {len(result)}개",
                           decision="순차 Multi-Vector")
    return result


# =====================================================================
# Ablation Study
# =====================================================================

def run_single_config(question, config, index, chunks, sources, top_k=3, prefilter_n=10):
    eff = {"use_bm25": config["bm25"] and BM25_AVAILABLE, "use_reranking": config["rerank"],
           "top_k": top_k, "use_query_rewrite": config["query_rewrite"]}
    t_start = time.time()
    try:
        r = run_rag_pipeline(question, eff, index, chunks, sources,
                             prefilter_n=prefilter_n, use_multidoc=True,
                             num_rewrites=2, use_session_cache=False,
                             use_self_refine=False, use_compression=False, mv_index=None,
                             auto_save_failure=False, use_parallel_search=False)
        qr, sqr = r["quality_report"], r["sqr"] or {}
        return {"config_id": config["id"], "config_name": config["name"],
                "query_rewrite": config["query_rewrite"], "bm25": config["bm25"], "rerank": config["rerank"],
                "accuracy": r["evaluation"].get("정확도",0), "relevance": r["evaluation"].get("관련성",0),
                "hallucination": r["evaluation"].get("환각여부","-"), "confidence": r["evaluation"].get("신뢰도","-"),
                "overall_score": qr["overall_score"], "grade": qr["grade"],
                "ndcg_prefilter": r["ndcg_k"], "reranker_gain": sqr.get("reranker_gain"),
                "search_quality": sqr.get("quality_label"),
                "latency_ms": r["tracer"].total_latency_ms(), "total_tokens": r["tracer"].total_tokens()["total"],
                "answer": r["answer"], "error": None}
    except Exception as e:
        return {"config_id": config["id"], "config_name": config["name"],
                "query_rewrite": config["query_rewrite"], "bm25": config["bm25"], "rerank": config["rerank"],
                "accuracy": 0, "relevance": 0, "hallucination": "-", "confidence": "-",
                "overall_score": 0, "grade": "F", "ndcg_prefilter": None, "reranker_gain": None,
                "search_quality": None, "latency_ms": int((time.time()-t_start)*1000),
                "total_tokens": 0, "answer": "", "error": str(e)}


# =====================================================================
# 로그
# =====================================================================

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except json.JSONDecodeError: return []
    return []


def save_log(entry):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def build_log_entry(question, queries, ranked_items, answer,
                    evaluation, hall_cause, tracer, mode,
                    ndcg=None, route_decision=None,
                    quality_report=None, search_quality_report=None,
                    fallback_triggered=False, fallback_attempts=0, fallback_history=None,
                    self_refinement=None, dynamic_retrieval_profile=None,
                    cache_hit=None, compression_stats=None, mv_retrieval=False,
                    failure_types=None, failure_saved=False,
                    parallel_ms=None, tool_used=False, selective_stats=None):
    sqr = search_quality_report or {}
    return {
        "trace_id":      tracer.trace_id,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":      question, "queries": queries,
        "retrieved_chunks": [
            {"text": c[:200]+("..." if len(c)>200 else ""), "source": s,
             "rerank_score": round(sc,2) if sc is not None else None}
            for c, s, sc in ranked_items
        ],
        "answer": answer, "evaluation": evaluation,
        "hallucination_analysis": hall_cause, "quality_report": quality_report,
        "search_quality_report": sqr, "ndcg_at_k": ndcg, "reranker_gain": sqr.get("reranker_gain"),
        "fallback_triggered": fallback_triggered, "fallback_attempts": fallback_attempts,
        "fallback_history": fallback_history or [],
        "self_refinement": self_refinement, "dynamic_retrieval_profile": dynamic_retrieval_profile,
        "cache_hit": cache_hit, "compression_stats": compression_stats,
        "mv_retrieval": mv_retrieval, "failure_types": failure_types or [], "failure_saved": failure_saved,
        # [NEW v21]
        "parallel_ms":     parallel_ms,
        "tool_used":       tool_used,
        "selective_stats": selective_stats,
        "spans":           tracer.spans, "total_tokens": tracer.total_tokens(),
        "total_latency_ms": tracer.total_latency_ms(), "bottleneck": tracer.bottleneck(),
        "mode": mode, "route_decision": route_decision,
        "embed_cache_size": embed_cache.size(),
        "embed_cache_hits": embed_cache.hits, "embed_cache_misses": embed_cache.misses,
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
tab_chat, tab_trace, tab_agent, tab_ablation, tab_search, tab_failure, tab_v21 = st.tabs([
    "💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation",
    "🔍 검색 품질", "🚨 실패 데이터셋", "⚡ v21 분석"
])


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("📚 RAG 챗봇 v21")
    st.markdown("---")

    uploaded_files = st.file_uploader("파일을 업로드하세요", type=["pdf","txt"], accept_multiple_files=True)
    chunking_mode  = st.radio("청킹 방식", ["문단/문장 + Overlap", "의미 기반(Semantic) + Overlap"])
    chunk_size     = st.slider("청크 크기", 200, 1000, 500, 100)
    overlap        = st.slider("Overlap 크기", 0, 200, 100, 20)
    use_multi_vector = st.toggle("🔢 Multi-Vector 인덱싱", value=True)

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
                    with st.spinner("Multi-Vector 인덱스 구축 중..."):
                        mv_idx = build_multi_vector_index(all_chunks)
                        st.session_state.index    = mv_idx["chunk_index"]
                        st.session_state.mv_index = mv_idx
                    st.success(f"완료! {len(all_chunks)}개 청크 | {mv_idx['n_sentences']}개 문장 벡터")
                else:
                    st.session_state.index    = build_index(all_chunks)
                    st.session_state.mv_index = None
                    st.success(f"완료! {len(all_chunks)}개 청크")
            query_result_cache.clear()

    if st.session_state.index is not None:
        st.info(f"📄 {len(st.session_state.chunks)}개 청크")
        for fname, cnt in Counter(st.session_state.chunk_sources).items():
            st.caption(f"  └ {fname}: {cnt}개")

    st.markdown("---")
    auto_routing = st.toggle("🧭 자동 쿼리 라우팅", value=True)
    use_query_rewrite = st.toggle("쿼리 리라이팅", value=True, disabled=auto_routing)
    num_rewrites      = st.slider("리라이팅 수", 1, 5, 3, disabled=auto_routing or not use_query_rewrite)
    use_bm25     = st.toggle("Hybrid 검색 (BM25)", value=True, disabled=auto_routing) if BM25_AVAILABLE \
                   else (st.warning("rank_bm25 미설치"), False)[1]
    prefilter_n   = st.slider("Pre-filter 수", 5, 20, 10)
    use_reranking = st.toggle("리랭킹", value=True, disabled=auto_routing)
    top_k         = st.slider("최종 청크 수 (top_k)", 1, 5, 3)
    use_multidoc  = st.toggle("멀티문서 추론", value=True)
    auto_evaluate = st.toggle("자동 평가 + 로깅", value=True)

    st.markdown("---")
    st.caption("🔄 Fallback / ✏️ Self-Refinement / 🎯 Dynamic Retrieval")
    enable_fallback          = st.toggle("Fallback 자동 재시도", value=True)
    enable_self_refine       = st.toggle("Self-Refinement", value=True)
    enable_dynamic_retrieval = st.toggle("의도별 검색 전략 자동 조정", value=True, disabled=not auto_routing)

    st.markdown("---")
    st.caption("⚡ 캐시 전략")
    enable_answer_cache = st.toggle("답변 캐시", value=True)
    enable_query_cache  = st.toggle("쿼리 결과 캐시", value=True)
    st.caption(f"  답변 캐시: {answer_cache.valid_size()}개 유효 | 임베딩: {embed_cache.size()}개")

    st.markdown("---")
    st.caption("🗜️ Context Compression")
    enable_compression  = st.toggle("Phase 1 (문장 추출)", value=False)
    comp_max_sentences  = st.slider("최대 추출 문장 수", 2, 8, 5, disabled=not enable_compression)
    comp_min_sim        = st.slider("최소 유사도", 0.1, 0.5, 0.25, 0.05, disabled=not enable_compression)

    st.markdown("---")
    # ── [NEW v21] 핵심 설정 3종 ──────────────────────────────────
    st.caption("⚡ [v21] 병렬 검색")
    enable_parallel = st.toggle("병렬 검색 (ThreadPoolExecutor)", value=True,
                                 help=f"Dense / BM25 / 문장 / 키워드 4채널 동시 검색 (workers={PARALLEL_MAX_WORKERS})\nFallback 재시도 시 특히 효과적")
    enable_lim      = st.toggle("LongContextReorder", value=True,
                                 help="관련성 높은 청크를 프롬프트 시작·끝에 배치 → Lost in the Middle 방지")

    st.markdown("---")
    st.caption("🔬 [v21] Selective Context Phase 2")
    enable_selective  = st.toggle("Cross-chunk 중복 제거", value=False,
                                   help="청크 간 유사 문장을 제거 → 프롬프트 노이즈 감소 (LLM 없음)")
    selective_thresh  = st.slider("중복 임계값 (코사인)", 0.7, 0.95, DEDUP_THRESHOLD_DEFAULT, 0.05,
                                   disabled=not enable_selective)

    st.markdown("---")
    st.caption("🔧 [v21] Tool-Augmented RAG")
    enable_tool_augment = st.toggle("Python 수치 계산 도구", value=False,
                                     help="수치 계산 질문 감지 → Python으로 정확 계산 → 수치 환각 0%\n(+2 LLM 호출)")

    st.markdown("---")
    st.caption("🚨 실패 데이터셋")
    enable_failure_save = st.toggle("실패 케이스 자동 저장", value=True)
    enable_hint_gen     = st.toggle("개선 힌트 자동 생성 (+1 LLM)", value=False,
                                     disabled=not enable_failure_save)
    st.caption(f"  저장된 실패: {failure_dataset.size()}건")

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("문서 초기화", use_container_width=True):
        st.session_state.update({"messages":[],"index":None,"chunks":[],"chunk_sources":[],"mv_index":None})
        query_result_cache.clear(); st.rerun()
    if st.button("캐시 전체 초기화", use_container_width=True):
        embed_cache.clear(); query_result_cache.clear(); answer_cache.clear(); st.rerun()
    if st.button("실패 데이터셋 초기화", use_container_width=True):
        failure_dataset.clear(); st.rerun()
    if st.button("로그 초기화", use_container_width=True):
        os.path.exists(LOG_FILE) and os.remove(LOG_FILE); st.rerun()
    st.caption("v21: 병렬 검색 + Selective Context + Tool-Augmented")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================
with tab_chat:
    st.title("💬 문서 기반 챗봇 v21")
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
                route_decision, dynamic_profile_label = None, None
                defaults = {"use_bm25": use_bm25, "use_reranking": use_reranking,
                            "top_k": top_k, "use_query_rewrite": use_query_rewrite}
                eff           = defaults.copy()
                cur_multidoc  = use_multidoc
                cur_prefilter = prefilter_n

                if auto_routing:
                    with st.status("🧭 쿼리 라우팅...", expanded=False) as s:
                        _rt = Tracer()
                        route_decision = route_query(prompt, _rt)
                        intent = route_decision.get("의도", "-")
                        eff    = _apply_routing(route_decision, defaults)
                        if enable_dynamic_retrieval:
                            eff, cur_prefilter, cur_multidoc, dynamic_profile_label = \
                                apply_dynamic_retrieval(intent, eff, prefilter_n, use_multidoc)
                        par_tag = " | ⚡병렬" if enable_parallel else ""
                        pf_tag  = f" | 프로필: {dynamic_profile_label}" if dynamic_profile_label else ""
                        s.update(label=f"✅ 의도: {intent} | top_k: {eff['top_k']}{pf_tag}{par_tag}", state="complete")

                attempt_num, fallback_triggered_flag, fallback_history, best_result = 0, False, [], None

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
                        auto_save_failure=enable_failure_save,
                        gen_improvement_hint=enable_hint_gen,
                        use_parallel_search=enable_parallel,
                        use_lim_reorder=enable_lim,
                        use_selective_context=enable_selective,
                        selective_dedup_thresh=selective_thresh,
                        use_tool_augment=enable_tool_augment,
                    )
                    ev0, qr0 = cur_result["evaluation"], cur_result["quality_report"]
                    fb_needed, fb_reason = should_fallback(ev0) if enable_fallback else (False, "")

                    icons = ""
                    if cur_result.get("cache_hit"):    icons += "⚡"
                    if cur_result.get("compression_stats") or cur_result.get("selective_stats"): icons += " 🗜️"
                    if cur_result.get("critique"):     icons += " ✏️"
                    if cur_result.get("failure_saved"):icons += " 🚨"
                    if cur_result.get("tool_used"):    icons += " 🔧"
                    if cur_result.get("parallel_ms"):  icons += f" ⚡{cur_result['parallel_ms']}ms"
                    status_icon = "✅" if not fb_needed else "⚠️"
                    s.update(label=(
                        f"{status_icon} 시도 1 완료{icons} | "
                        f"정확도 {ev0.get('정확도','-')}/5 · 환각 {ev0.get('환각여부','-')} · 등급 {qr0['grade']}"
                        + (f" → Fallback: {fb_reason}" if fb_needed else "")
                    ), state="complete")

                fallback_history.append({
                    "attempt": 0, "trigger": None, "eff": cur_result["eff"],
                    "prefilter_n": cur_result["prefilter_n"],
                    "accuracy": ev0.get("정확도",0), "hallucination": ev0.get("환각여부","-"),
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
                            auto_save_failure=False,
                            use_parallel_search=enable_parallel,
                            use_lim_reorder=enable_lim,
                            use_selective_context=enable_selective,
                            selective_dedup_thresh=selective_thresh,
                            use_tool_augment=enable_tool_augment,
                        )
                        ev_n  = cur_result["evaluation"]
                        qr_n  = cur_result["quality_report"]
                        improved  = qr_n["overall_score"] > best_result["quality_report"]["overall_score"]
                        fb_needed, fb_reason = should_fallback(ev_n)
                        par_info = f" ⚡{cur_result['parallel_ms']}ms" if cur_result.get("parallel_ms") else ""
                        s.update(label=(
                            f"{'✅ 개선됨' if improved else '➡️ 유지'} 재시도 {attempt_num}{par_info} | "
                            f"정확도 {ev_n.get('정확도','-')}/5 · 등급 {qr_n['grade']}"
                        ), state="complete")

                    fallback_history.append({
                        "attempt": attempt_num, "trigger": fallback_history[-1].get("trigger") or fb_reason,
                        "eff": cur_result["eff"], "prefilter_n": cur_result["prefilter_n"],
                        "accuracy": ev_n.get("정확도",0), "hallucination": ev_n.get("환각여부","-"),
                        "overall_score": qr_n["overall_score"], "grade": qr_n["grade"],
                        "tokens": cur_result["tracer"].total_tokens()["total"],
                        "latency_ms": cur_result["tracer"].total_latency_ms(),
                    })
                    if improved: best_result = cur_result

                final         = best_result
                response      = final["answer"]
                evaluation    = final["evaluation"]
                hall_cause    = final["hall_cause"]
                quality_report= final["quality_report"]
                ndcg_k        = final["ndcg_k"]
                sqr           = final["sqr"]
                queries       = final["queries"]
                ranked        = final["ranked"]
                final_chunks  = final["final_chunks"]
                final_sources = final["final_sources"]
                final_scores  = final["final_scores"]
                gen_chunks    = final["gen_chunks"]
                summaries     = final["summaries"]
                analysis      = final["analysis"]
                mode          = final["mode"]
                final_eff     = final["eff"]
                tracer        = final["tracer"]
                draft_answer  = final["draft_answer"]
                critique      = final["critique"]
                cache_hit     = final.get("cache_hit")
                comp_stats    = final.get("compression_stats")
                sel_stats     = final.get("selective_stats")
                failure_types = final.get("failure_types", [])
                failure_saved = final.get("failure_saved", False)
                parallel_ms   = final.get("parallel_ms")
                tool_code     = final.get("tool_code")
                calc_result   = final.get("calc_result")
                tool_used     = final.get("tool_used", False)

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
                        failure_types=failure_types, failure_saved=failure_saved,
                        parallel_ms=parallel_ms, tool_used=tool_used, selective_stats=sel_stats,
                    ))

                # ── 배너 표시 ─────────────────────────────────────
                if cache_hit == "answer":
                    st.success(f"⚡ 답변 캐시 히트 (TTL {ANSWER_CACHE_TTL_SEC//60}분)")
                elif cache_hit == "query":
                    st.info("🔁 쿼리 결과 캐시 히트 — 검색 스킵")

                if parallel_ms:
                    st.info(f"⚡ 병렬 검색 완료 — {parallel_ms}ms (Dense + BM25 + 문장 + 키워드 4채널 동시)")

                if tool_used and calc_result:
                    st.success(f"🔧 Tool-Augmented — Python 계산 결과: `{calc_result}`")

                if failure_saved and failure_types:
                    type_labels = {"low_accuracy":"낮은 정확도","low_relevance":"낮은 관련성",
                                   "hallucination":"환각","incomplete_answer":"누락 정보",
                                   "retrieval_failure":"검색 실패"}
                    labels = " · ".join(type_labels.get(t,t) for t in failure_types)
                    st.warning(f"🚨 실패 케이스 저장됨 — {labels}")

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
                    cols[5].metric("🧭 신뢰도", evaluation.get("신뢰도","-"))
                    if fallback_triggered_flag:
                        cols[6].metric("🔄 재시도", f"{attempt_num}회",
                                       delta=f"등급 {fallback_history[0]['grade']}→{quality_report['grade']}",
                                       delta_color="normal")
                    else:
                        cols[6].metric("🔄 재시도", "없음")
                    cols[7].metric("🔍 Dense NDCG",
                                   f"{ndcg_k:.3f}" if ndcg_k else "-",
                                   delta=f"Gain {sqr['reranker_gain']:.3f}" if sqr else None)

                # [NEW v21] Tool-Augmented 상세
                if tool_used and tool_code:
                    with st.expander(f"🔧 Tool-Augmented 계산 내역 (결과: {calc_result})"):
                        st.markdown("**Python 코드 (LLM 생성)**")
                        st.code(tool_code, language="python")
                        st.markdown(f"**실행 결과**: `{calc_result}`")
                        st.caption("이 값이 최종 답변에 직접 사용되었습니다 — 수치 환각 0%")

                # [NEW v21] Selective Context 상세
                if sel_stats:
                    dedup_pct = round((sel_stats.get("dedup_removed",0) / max(sel_stats.get("original_sents",1),1)) * 100)
                    comp_pct  = round((1 - sel_stats.get("ratio",1)) * 100)
                    with st.expander(f"🔬 Selective Context Phase 2 — 중복 {dedup_pct}% 제거 · 압축 {comp_pct}%"):
                        st.caption(f"원문 문장: {sel_stats['original_sents']}개 → 중복 제거 후: {sel_stats['after_dedup']}개")
                        st.caption(f"원문 {sel_stats['orig_chars']}자 → 압축 {sel_stats['comp_chars']}자 (비율 {sel_stats['ratio']:.0%})")

                if comp_stats:
                    orig_total = sum(s["original"] for s in comp_stats)
                    comp_total = sum(s["compressed"] for s in comp_stats)
                    avg_ratio  = round(comp_total/max(orig_total,1)*100)
                    with st.expander(f"🗜️ Compression Phase 1 — 토큰 {100-avg_ratio}% 절감"):
                        for i, s in enumerate(comp_stats):
                            st.caption(f"청크 {i+1}: {s['original']}자 → {s['compressed']}자 ({s['ratio']*100:.0f}%)")

                if enable_self_refine and critique:
                    with st.expander("✏️ Self-Refinement (Draft → Critique → Refined)"):
                        st.markdown("**📝 Draft**"); st.info(draft_answer)
                        st.markdown("**🔎 Critique**"); st.warning(critique)
                        st.markdown("**✅ Refined** ← 위 답변")

                if fallback_triggered_flag:
                    with st.expander(f"🔄 Fallback 내역 ({attempt_num}회)"):
                        for h in fallback_history:
                            is_best = h["overall_score"] == quality_report["overall_score"]
                            st.markdown(f"**{'🏆 채택' if is_best else '📋'} 시도 {h['attempt']+1}** — `{h.get('trigger') or '최초'}`")
                            c1,c2,c3,c4,c5 = st.columns(5)
                            c1.metric("정확도",h['accuracy']); c2.metric("환각",h['hallucination'])
                            c3.metric("종합",h['overall_score']); c4.metric("등급",h['grade']); c5.metric("토큰",h['tokens'])
                            if h["attempt"] < len(fallback_history)-1: st.divider()

                if sqr:
                    lc = {"excellent":"🟢","good":"🔵","fair":"🟡","poor":"🔴"}.get(sqr["quality_label"],"⚪")
                    with st.expander(f"🔍 검색 품질 — {lc} {sqr['quality_label'].upper()}"):
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Dense NDCG@k",f"{sqr['ndcg_prefilter']:.4f}")
                        c2.metric("Reranker Gain",f"{sqr['reranker_gain']:.4f}")
                        c3.metric("후보 수",f"{sqr['n_candidates']}개")
                        st.info(f"📋 {sqr['diagnosis']}")

                if quality_report:
                    grade = quality_report["grade"]
                    gc    = {"A":"🟢","B":"🔵","C":"🟡","D":"🟠","F":"🔴"}.get(grade,"⚪")
                    with st.expander(f"🩺 품질 리포트 — {gc} {grade} ({quality_report['overall_score']}/5)"):
                        for issue in quality_report.get("issues",[]): st.markdown(f"- {issue}")
                        for rec   in quality_report.get("recommendations",[]): st.markdown(f"- {rec}")

                if route_decision:
                    with st.expander("🧭 쿼리 라우팅 결정"):
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.markdown(f"**의도**: `{route_decision.get('의도','-')}`")
                            st.json(route_decision.get("검색_전략",{}))
                            if dynamic_profile_label: st.success(f"🎯 {dynamic_profile_label}")
                        with col_r2:
                            st.markdown("**적용 파라미터**"); st.json(final_eff)

                if final_eff.get("use_query_rewrite"):
                    with st.expander("🔍 생성된 쿼리"):
                        for i, q in enumerate(queries):
                            st.markdown(f"{'**[원본]**' if i==0 else f'**[변형 {i}]**'} {q}")

                st.markdown("**📄 출처 원문**")
                for i, (c, src, sc) in enumerate(zip(final_chunks, final_sources, final_scores)):
                    st.markdown(f'<div id="source-{i+1}"></div>', unsafe_allow_html=True)
                    st.caption(f"[출처 {i+1}] **{src}**" + (f" _(관련성 {sc:.1f}/10)_" if sc else ""))
                    st.write(c)
                    if i < len(final_chunks)-1: st.divider()

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
        total_tokens_all = [l.get("total_tokens",{}).get("total",0) for l in logs]
        total_lat_all    = [l.get("total_latency_ms",0) for l in logs]
        fb_count    = sum(1 for l in logs if l.get("fallback_triggered"))
        cache_hits  = sum(1 for l in logs if l.get("cache_hit"))
        fail_count  = sum(1 for l in logs if l.get("failure_saved"))
        par_logs    = [l for l in logs if l.get("parallel_ms")]
        tool_count  = sum(1 for l in logs if l.get("tool_used"))  # [NEW v21]

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("총 트레이스",     len(logs))
        c2.metric("평균 응답",       f"{sum(total_lat_all)/len(total_lat_all)/1000:.1f}s")
        c3.metric("평균 토큰",       f"{int(sum(total_tokens_all)/len(total_tokens_all)):,}")
        c4.metric("🔄 Fallback",    f"{fb_count}회")
        c5.metric("⚡ 병렬 검색",   f"{len(par_logs)}건")  # [NEW v21]
        c6.metric("🔧 Tool 사용",   f"{tool_count}건")      # [NEW v21]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("응답 시간 추이 (초)")
            st.line_chart([l/1000 for l in total_lat_all])
        with col2:
            if par_logs:
                st.subheader("⚡ 병렬 검색 시간 추이 (ms)")
                st.line_chart([l.get("parallel_ms",0) for l in logs])

        st.markdown("---")
        st.subheader("트레이스 상세 (최신 순)")
        for log in reversed(logs):
            spans    = log.get("spans", [])
            total_ms = log.get("total_latency_ms", 1)
            tok      = log.get("total_tokens", {})
            ts       = log.get("timestamp", "")
            q        = log.get("question","")[:50]
            nd       = log.get("ndcg_at_k")
            intent   = (log.get("route_decision") or {}).get("의도","-")
            fb_tag   = f" | 🔄 {log.get('fallback_attempts',0)}회" if log.get("fallback_triggered") else ""
            sr_tag   = " | ✏️" if log.get("self_refinement") else ""
            ch_tag   = f" | ⚡{log['cache_hit']}" if log.get("cache_hit") else ""
            fl_tag   = " | 🚨" if log.get("failure_saved") else ""
            par_tag  = f" | ⚡{log['parallel_ms']}ms" if log.get("parallel_ms") else ""  # [NEW v21]
            tl_tag   = " | 🔧Tool" if log.get("tool_used") else ""                       # [NEW v21]

            with st.expander(
                f"[{ts}] {q}... | ⏱ {total_ms/1000:.1f}s | 🔤 {tok.get('total',0):,} | {intent}"
                + (f" | NDCG {nd:.3f}" if nd else "") + fb_tag + sr_tag + ch_tag + fl_tag + par_tag + tl_tag
            ):
                for span in spans:
                    dur = span["duration_ms"]
                    pct = dur/total_ms if total_ms > 0 else 0
                    c1,c2,c3 = st.columns([2,6,1])
                    c1.caption(span["name"] + (" ❌" if span.get("error") else ""))
                    c2.progress(min(pct,1.0))
                    c3.caption(f"{dur}ms")
                st.markdown("---")
                tok_data = {s["name"]:s["tokens"]["total"] for s in spans if s["tokens"]["total"]>0}
                if tok_data: st.bar_chart(tok_data)
                import pandas as pd
                flow_rows = [{"단계":s["name"],"소요(ms)":s["duration_ms"],"토큰":s["tokens"]["total"],
                              "입력":s.get("input_summary","")[:60],"출력":s.get("output_summary","")[:60]}
                             for s in spans]
                if flow_rows: st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)


# =====================================================================
# TAB 3 — 에이전트 분석
# =====================================================================
with tab_agent:
    st.title("🧠 에이전트 분석 대시보드")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        import pandas as pd
        eval_logs   = [l for l in logs if l.get("evaluation")]
        avg_acc     = round(sum(l["evaluation"].get("정확도",0) for l in eval_logs)/len(eval_logs),2) if eval_logs else 0
        avg_rel     = round(sum(l["evaluation"].get("관련성",0) for l in eval_logs)/len(eval_logs),2) if eval_logs else 0
        hall_counts = Counter(l["evaluation"].get("환각여부","") for l in eval_logs)
        ndcg_vals   = [l["ndcg_at_k"] for l in logs if l.get("ndcg_at_k") is not None]
        avg_ndcg    = round(sum(ndcg_vals)/len(ndcg_vals),3) if ndcg_vals else None
        qr_logs     = [l for l in logs if l.get("quality_report")]
        avg_overall = round(sum(l["quality_report"]["overall_score"] for l in qr_logs)/len(qr_logs),2) if qr_logs else None
        hall_none_pct = round(hall_counts.get("없음",0)/len(eval_logs)*100) if eval_logs else 0
        fb_logs     = [l for l in logs if l.get("fallback_triggered")]
        fb_rate     = round(len(fb_logs)/len(logs)*100) if logs else 0

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("평균 정확도",    f"{avg_acc}/5")
        c2.metric("평균 관련성",    f"{avg_rel}/5")
        c3.metric("환각 없음 비율", f"{hall_none_pct}%")
        c4.metric("평균 NDCG@k",   str(avg_ndcg) if avg_ndcg else "-")
        c5.metric("평균 종합점수",  f"{avg_overall}/5" if avg_overall else "-")
        c6.metric("🔄 Fallback 율", f"{fb_rate}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("환각 여부 분포")
            st.bar_chart({"없음":hall_counts.get("없음",0),"부분적":hall_counts.get("부분적",0),"있음":hall_counts.get("있음",0)})
        with col2:
            st.subheader("정확도 / 관련성 추이")
            if len(eval_logs) >= 2:
                st.line_chart({"정확도":[l["evaluation"].get("정확도",0) for l in eval_logs],
                               "관련성":[l["evaluation"].get("관련성",0) for l in eval_logs]})

        # 실패 케이스 통계
        fail_logs = [l for l in logs if l.get("failure_saved")]
        if fail_logs:
            st.markdown("---")
            all_types = []
            for l in fail_logs: all_types.extend(l.get("failure_types",[]))
            type_counts = Counter(all_types)
            st.subheader(f"🚨 실패 케이스 ({len(fail_logs)}건 / {round(len(fail_logs)/len(logs)*100)}%)")
            labeled = {"낮은 정확도":type_counts.get("low_accuracy",0),"환각":type_counts.get("hallucination",0),
                       "누락 정보":type_counts.get("incomplete_answer",0),"검색 실패":type_counts.get("retrieval_failure",0)}
            st.bar_chart(labeled)

        # [NEW v21] Tool-Augmented 통계
        tool_logs = [l for l in logs if l.get("tool_used")]
        if tool_logs or logs:
            st.markdown("---")
            st.subheader(f"🔧 Tool-Augmented RAG 사용 현황")
            tool_rate = round(len(tool_logs)/len(logs)*100) if logs else 0
            t1,t2 = st.columns(2)
            t1.metric("Tool 사용", f"{len(tool_logs)}건 ({tool_rate}%)")
            t2.metric("일반 생성", f"{len(logs)-len(tool_logs)}건")

        # per-log 상세
        st.markdown("---")
        for log in reversed(logs[:10]):
            ev    = log.get("evaluation",{})
            qr    = log.get("quality_report",{})
            hall  = ev.get("환각여부","-")
            hi    = "🟢" if hall=="없음" else ("🟡" if hall=="부분적" else ("🔴" if hall=="있음" else "⚪"))
            ts    = log.get("timestamp","")
            q     = log.get("question","")[:45]
            nd    = log.get("ndcg_at_k")
            grade = (qr or {}).get("grade","-")
            score = (qr or {}).get("overall_score","-")
            intent= (log.get("route_decision") or {}).get("의도","-")
            fb_tag = f" | 🔄{log.get('fallback_attempts',0)}회" if log.get("fallback_triggered") else ""
            ch_tag = f" | ⚡{log['cache_hit']}" if log.get("cache_hit") else ""
            tl_tag = " | 🔧" if log.get("tool_used") else ""
            fl_tag = " | 🚨" if log.get("failure_saved") else ""

            with st.expander(
                f"[{ts}] {q}... | {ev.get('정확도','-')}/5 · {hi} {hall} | 등급 {grade}({score}) | {intent}{fb_tag}{ch_tag}{tl_tag}{fl_tag}"
                + (f" | NDCG {nd:.3f}" if nd else "")
            ):
                col_a, col_b = st.columns([3,2])
                with col_a:
                    if log.get("failure_saved"):
                        st.warning(f"🚨 실패 저장: {', '.join(log.get('failure_types',[]))}")
                    if log.get("tool_used"):
                        st.success("🔧 Tool-Augmented 계산 사용")
                    st.markdown("**💬 최종 답변**")
                    st.markdown(log.get("answer",""))
                with col_b:
                    if ev:
                        st.metric("정확도", f"{ev.get('정확도','-')}/5")
                        st.metric("관련성", f"{ev.get('관련성','-')}/5")
                    if qr:
                        st.metric("종합점수", f"{qr.get('overall_score','-')}/5")
                        st.metric("등급", qr.get("grade","-"))
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
        abl_question    = st.text_input("실험 질문", placeholder="예: 계약 해지 절차는 어떻게 되나요?")
        config_names    = [c["name"] for c in ABLATION_CONFIGS]
        selected_names  = st.multiselect("실험할 Config", options=config_names, default=config_names)
        selected_configs= [c for c in ABLATION_CONFIGS if c["name"] in selected_names]
        abl_top_k       = st.slider("Ablation top_k", 1, 5, 3, key="abl_top_k")
        abl_prefilter_n = st.slider("Ablation pre-filter 수", 5, 20, 10, key="abl_pf")
        n_configs = len(selected_configs)
        st.info(f"선택된 config {n_configs}개 × 약 6 LLM 호출 = **약 {n_configs*6}회** 예상.")

        if st.button("🧪 Ablation 실행", type="primary",
                     disabled=not abl_question or not selected_configs, use_container_width=True):
            results, progress_bar, status_text = [], st.progress(0), st.empty()
            for i, config in enumerate(selected_configs):
                status_text.markdown(f"**실행 중** ({i+1}/{n_configs}): `{config['name']}`")
                results.append(run_single_config(abl_question, config,
                                                  st.session_state.index, st.session_state.chunks,
                                                  st.session_state.chunk_sources,
                                                  top_k=abl_top_k, prefilter_n=abl_prefilter_n))
                progress_bar.progress((i+1)/n_configs)
            status_text.markdown("✅ **모든 실험 완료**")
            st.session_state.ablation_results = results

        if st.session_state.ablation_results:
            results = st.session_state.ablation_results
            df = pd.DataFrame([{
                "Config": r["config_name"],
                "정확도": r["accuracy"], "관련성": r["relevance"],
                "종합점수": r.get("overall_score","-"), "등급": r.get("grade","-"),
                "환각여부": r["hallucination"],
                "Dense NDCG": round(r["ndcg_prefilter"],3) if r.get("ndcg_prefilter") else "-",
                "Reranker Gain": round(r["reranker_gain"],3) if r.get("reranker_gain") else "-",
                "응답(ms)": r["latency_ms"], "토큰": r["total_tokens"],
                "오류": "❌" if r.get("error") else "✅",
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)

            valid = [r for r in results if not r.get("error")]
            if valid:
                best_acc   = max(valid, key=lambda r: r["accuracy"])
                best_score = max(valid, key=lambda r: r.get("overall_score",0))
                fastest    = min(valid, key=lambda r: r["latency_ms"])
                col1,col2,col3 = st.columns(3)
                col1.metric("🏆 최고 정확도",   best_acc["config_name"].split()[-1],   delta=f"{best_acc['accuracy']}/5")
                col2.metric("🥇 최고 종합점수", best_score["config_name"].split()[-1], delta=f"등급 {best_score.get('grade','-')}")
                col3.metric("⚡ 최저 지연",     f"{fastest['latency_ms']/1000:.1f}s")
                chart_data = pd.DataFrame({
                    "Config": [r["config_name"].replace("✅","").replace("❌","").replace("⚡","").replace("🔥","").strip() for r in valid],
                    "정확도": [r["accuracy"] for r in valid],
                    "관련성": [r["relevance"] for r in valid],
                }).set_index("Config")
                st.bar_chart(chart_data)


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
        ndcg_vals = [l["ndcg_at_k"] for l in sq_logs]
        gain_vals = [l["reranker_gain"] for l in sq_logs if l.get("reranker_gain") is not None]
        avg_ndcg  = round(sum(ndcg_vals)/len(ndcg_vals), 4)
        avg_gain  = round(sum(gain_vals)/len(gain_vals), 4) if gain_vals else None
        excellent_n = sum(1 for v in ndcg_vals if v >= _NDCG_EXCELLENT)
        poor_n      = sum(1 for v in ndcg_vals if v < _NDCG_FAIR)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("평균 Dense NDCG",   f"{avg_ndcg:.4f}")
        c2.metric("평균 Reranker Gain", f"{avg_gain:.4f}" if avg_gain else "-")
        c3.metric("Excellent (≥0.9)",   f"{excellent_n}건")
        c4.metric("Poor (<0.5)",         f"{poor_n}건", delta_color="inverse")
        c5.metric("총 측정 건수",        f"{len(sq_logs)}건")

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("📈 Dense NDCG@k 추이")
            st.line_chart(ndcg_vals)
        with col2:
            if gain_vals:
                st.subheader("📉 Reranker Gain 추이")
                st.line_chart(gain_vals)

        rows = [{"시간": l.get("timestamp",""), "질문": l.get("question","")[:50],
                 "Dense NDCG": l["ndcg_at_k"],
                 "Reranker Gain": l.get("reranker_gain","-"),
                 "품질": (l.get("search_quality_report") or {}).get("quality_label","-"),
                 "BM25": "✅" if (l.get("search_quality_report") or {}).get("use_bm25") else "❌",
                 "의도": (l.get("route_decision") or {}).get("의도","-"),
                 "병렬": f"⚡{l['parallel_ms']}ms" if l.get("parallel_ms") else "-",  # [NEW v21]
                 "Selective": "🔬" if l.get("selective_stats") else "-",               # [NEW v21]
                 "Tool": "🔧" if l.get("tool_used") else "-",                          # [NEW v21]
                 "Fallback": "🔄" if l.get("fallback_triggered") else "-",
                 "실패": "🚨" if l.get("failure_saved") else "-"}
                for l in sq_logs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =====================================================================
# TAB 6 — 실패 데이터셋
# =====================================================================
with tab_failure:
    st.title("🚨 실패 데이터셋")
    entries = failure_dataset.get_all()
    if not entries:
        st.info("아직 저장된 실패 케이스가 없습니다.")
    else:
        all_types = []
        for e in entries: all_types.extend(e.get("failure_types",[]))
        type_counts = Counter(all_types)
        type_labels_map = {"low_accuracy":"낮은 정확도","low_relevance":"낮은 관련성",
                           "hallucination":"환각","incomplete_answer":"누락 정보","retrieval_failure":"검색 실패"}

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("총 실패",    len(entries))
        d2.metric("낮은 정확도",type_counts.get("low_accuracy",0))
        d3.metric("환각",       type_counts.get("hallucination",0))
        d4.metric("누락 정보",  type_counts.get("incomplete_answer",0))
        d5.metric("검색 실패",  type_counts.get("retrieval_failure",0))
        if type_counts:
            st.bar_chart({type_labels_map.get(k,k):v for k,v in type_counts.items()})

        st.markdown("---")
        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button("⬇️ Fine-tune JSONL (OpenAI 형식)",
                               data=failure_dataset.export_finetune_jsonl(),
                               file_name=f"failure_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                               mime="application/jsonl", use_container_width=True)
        with exp2:
            st.download_button("⬇️ 문제 분석 JSON",
                               data=failure_dataset.export_problems_json(),
                               file_name=f"failure_problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                               mime="application/json", use_container_width=True)

        st.markdown("---")
        for entry in reversed(entries):
            ev    = entry.get("evaluation",{})
            qr    = entry.get("quality_report",{})
            ftypes= entry.get("failure_types",[])
            ts    = entry.get("timestamp","")
            q     = entry.get("question","")[:60]
            acc   = ev.get("정확도","-")
            grade = (qr or {}).get("grade","-")
            type_str = " · ".join(type_labels_map.get(t,t) for t in ftypes)
            with st.expander(f"[{ts}] {q}... | 정확도 {acc}/5 · 등급 {grade} | 🚨 {type_str}"):
                t1, t2, t3 = st.tabs(["📋 개요", "💬 답변", "💡 개선 힌트"])
                with t1:
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**질문**: {entry.get('question','')}\n\n**실패**: {type_str}")
                        if ev.get("개선_제안"): st.info(f"💡 {ev['개선_제안']}")
                    with c2:
                        ri = entry.get("retrieval_info",{})
                        st.markdown(f"**정확도**: {ev.get('정확도','-')}/5 | **환각**: {ev.get('환각여부','-')}")
                        if ri.get("ndcg"): st.markdown(f"**Dense NDCG**: {ri['ndcg']:.4f}")
                        if qr: st.markdown(f"**종합**: {qr.get('overall_score','-')}/5 (등급 {qr.get('grade','-')})")
                with t2:
                    st.markdown(entry.get("answer",""))
                with t3:
                    hint = entry.get("improvement_hint")
                    if hint:
                        st.markdown(hint)
                    else:
                        st.caption("개선 힌트 없음. 아래 버튼으로 생성:")
                        if st.button("지금 생성", key=f"hint_{entry.get('id','')}"):
                            with st.spinner("생성 중..."):
                                new_hint = generate_improvement_hint(
                                    entry.get("question",""), entry.get("chunks",[]),
                                    entry.get("answer",""), entry.get("evaluation",{}),
                                    entry.get("failure_types",[])
                                )
                                for i, e in enumerate(failure_dataset._data):
                                    if e.get("id") == entry.get("id"):
                                        failure_dataset._data[i]["improvement_hint"] = new_hint
                                        failure_dataset._save()
                                        break
                                st.markdown(new_hint)


# =====================================================================
# TAB 7 — [NEW v21] v21 분석 대시보드
# =====================================================================
with tab_v21:
    st.title("⚡ v21 신기능 분석 대시보드")
    st.caption("병렬 검색 · Selective Context · Tool-Augmented RAG 성능 분석")
    logs = load_logs()

    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        import pandas as pd

        par_logs  = [l for l in logs if l.get("parallel_ms") is not None]
        sel_logs  = [l for l in logs if l.get("selective_stats")]
        tool_logs = [l for l in logs if l.get("tool_used")]
        lim_logs  = [l for l in logs if l.get("mode") not in ("answer_cache_hit",)]

        # ── 요약 지표 ────────────────────────────────────────────
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("⚡ 병렬 검색 사용",    f"{len(par_logs)}건 / {len(logs)}건")
        c2.metric("🔬 Selective Context", f"{len(sel_logs)}건 / {len(logs)}건")
        c3.metric("🔧 Tool-Augmented",    f"{len(tool_logs)}건 / {len(logs)}건")
        avg_par_ms = round(sum(l["parallel_ms"] for l in par_logs)/len(par_logs)) if par_logs else 0
        c4.metric("⚡ 평균 병렬 검색",    f"{avg_par_ms}ms")

        # ── ① 병렬 검색 ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚡ 병렬 검색 — 속도 분석")

        if par_logs:
            par_times  = [l["parallel_ms"] for l in par_logs]
            total_times= [l.get("total_latency_ms",0) for l in par_logs]
            par_ratios = [round(p/max(t,1)*100,1) for p,t in zip(par_times, total_times)]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.caption("병렬 검색 시간 추이 (ms)")
                st.line_chart(par_times)
            with col_p2:
                st.caption("병렬 검색이 전체 응답시간 중 차지 비율 (%)")
                st.line_chart(par_ratios)

            p50 = sorted(par_times)[int(len(par_times)*0.5)]
            p95 = sorted(par_times)[int(len(par_times)*0.95)] if len(par_times) >= 20 else max(par_times)
            avg_par = round(sum(par_times)/len(par_times))

            mp1,mp2,mp3 = st.columns(3)
            mp1.metric("평균 병렬 검색", f"{avg_par}ms")
            mp2.metric("P50",            f"{p50}ms")
            mp3.metric("P95",            f"{p95}ms")

            st.info(
                "**병렬 검색 효과**: Dense + BM25 + 문장 + 키워드 4채널을 동시 실행 → "
                "순차 실행 대비 이론상 ~4× 빠름. Fallback 재시도 시 효과 극대화."
            )
        else:
            st.caption("병렬 검색 사용 데이터 없음. 사이드바에서 '병렬 검색' 토글을 ON으로 설정하세요.")

        # ── ② Selective Context Phase 2 ─────────────────────────
        st.markdown("---")
        st.subheader("🔬 Selective Context Phase 2 — Cross-chunk 중복 제거")

        if sel_logs:
            dedup_rates = [round((l["selective_stats"].get("dedup_removed",0) /
                                  max(l["selective_stats"].get("original_sents",1),1))*100,1)
                           for l in sel_logs]
            comp_ratios = [round((1-l["selective_stats"].get("ratio",1))*100,1) for l in sel_logs]
            avg_dedup   = round(sum(dedup_rates)/len(dedup_rates),1) if dedup_rates else 0
            avg_comp    = round(sum(comp_ratios)/len(comp_ratios),1) if comp_ratios else 0

            sc1,sc2,sc3 = st.columns(3)
            sc1.metric("평균 중복 제거율",    f"{avg_dedup}%")
            sc2.metric("평균 컨텍스트 압축",  f"{avg_comp}%")
            sc3.metric("적용 건수",           f"{len(sel_logs)}건")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption("중복 제거율 추이 (%)")
                st.line_chart(dedup_rates)
            with col_s2:
                st.caption("컨텍스트 압축율 추이 (%)")
                st.line_chart(comp_ratios)

            # Selective Context 적용 vs 미적용 정확도 비교
            sel_acc = [l["evaluation"].get("정확도",0) for l in sel_logs if l.get("evaluation")]
            nosel_logs = [l for l in logs if not l.get("selective_stats") and l.get("evaluation")]
            nosel_acc  = [l["evaluation"].get("정확도",0) for l in nosel_logs]
            if sel_acc and nosel_acc:
                st.subheader("Selective Context 적용 vs 미적용 — 평균 정확도 비교")
                st.bar_chart({"Selective Context ON":  round(sum(sel_acc)/len(sel_acc),2),
                              "Selective Context OFF": round(sum(nosel_acc)/len(nosel_acc),2)})

            st.info(
                "**Selective Context Phase 2 원리**: 청크 내 중복(Phase 1) + 청크 간 중복 동시 제거.\n"
                "코사인 유사도 ≥ 임계값인 문장쌍에서 후순위 문장 제거 → 프롬프트 정보 밀도 ↑"
            )
        else:
            st.caption("Selective Context 사용 데이터 없음. 사이드바에서 'Cross-chunk 중복 제거' 토글을 ON으로 설정하세요.")

        # ── ③ Tool-Augmented RAG ─────────────────────────────────
        st.markdown("---")
        st.subheader("🔧 Tool-Augmented RAG — 수치 환각 분석")

        if tool_logs:
            # Tool 사용 vs 미사용 환각 비율 비교
            tool_hall = [l["evaluation"].get("환각여부","없음") for l in tool_logs if l.get("evaluation")]
            notool_logs= [l for l in logs if not l.get("tool_used") and l.get("evaluation")]
            notool_hall= [l["evaluation"].get("환각여부","없음") for l in notool_logs]

            tool_hall_rate  = round(sum(1 for h in tool_hall  if h != "없음") / max(len(tool_hall),1) * 100, 1)
            notool_hall_rate= round(sum(1 for h in notool_hall if h != "없음") / max(len(notool_hall),1) * 100, 1)

            ta1,ta2,ta3 = st.columns(3)
            ta1.metric("Tool 사용 건수",         f"{len(tool_logs)}건")
            ta2.metric("Tool 사용 시 환각율",     f"{tool_hall_rate}%", delta_color="inverse")
            ta3.metric("일반 생성 환각율",        f"{notool_hall_rate}%", delta_color="inverse")

            if tool_logs and notool_logs:
                tool_acc  = round(sum(l["evaluation"].get("정확도",0) for l in tool_logs if l.get("evaluation"))/len(tool_logs), 2)
                notool_acc= round(sum(l["evaluation"].get("정확도",0) for l in notool_logs if l.get("evaluation"))/max(len(notool_logs),1), 2)
                st.bar_chart({"Tool-Augmented 정확도":  tool_acc,
                              "일반 생성 정확도":        notool_acc})

            st.info(
                "**Tool-Augmented 작동 원리**:\n"
                "1. 수치 계산 감지 (`detect_calc_intent`) — 숫자 패턴 + 계산 키워드\n"
                "2. LLM이 문서에서 수치 추출 + Python 코드 생성\n"
                "3. `_safe_eval()`로 Python 안전 실행 → 정확한 계산 결과\n"
                "4. 계산 결과를 LLM에 주입 → 재계산 금지 지시 → **수치 환각 0%**"
            )
        else:
            st.caption("Tool-Augmented 사용 데이터 없음. 사이드바에서 'Python 수치 계산 도구' 토글을 ON으로 설정하세요.")

        # ── ④ LongContextReorder 효과 설명 ─────────────────────
        st.markdown("---")
        st.subheader("📐 LongContextReorder — Lost in the Middle 방지")
        st.markdown("""
**원리** ("Lost in the Middle" 논문 — Liu et al., 2023):

LLM은 프롬프트의 **시작 부분**과 **끝 부분**에 집중하고, **중간 부분**을 놓치는 경향이 있습니다.

```
[기존 순서]  청크1(관련↑) | 청크2(관련↓) | 청크3(관련↑↑) | 청크4(관련↓↓)
                                               ↑ LLM이 놓치기 쉬움

[재정렬 후]  청크3(관련↑↑) | 청크2(관련↓) | 청크4(관련↓↓) | 청크1(관련↑)
             ↑ 시작 = 최고 관련성                               ↑ 끝 = 차고 관련성
```

**v21 구현**: `reorder_lost_in_middle()` — 관련성 점수 기준으로 교차 배치
- 짝수 순위(0,2,4...) → 앞쪽부터
- 홀수 순위(1,3,5...) → 뒤쪽부터
- LLM 추가 호출 없음 (기존 rerank 점수 재활용)
        """)
