# rag_engine.py — v23 RAG 엔진 (UI 없는 순수 로직)
# v22 기반 + [NEW v23] LLM Compression / ToolRegistry / AsyncRAGEngine

import os
import re
import io
import json
import time
import uuid
import math
import pickle
import hashlib
import asyncio
import statistics
from datetime import datetime, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import pdfplumber
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

load_dotenv()

client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # [NEW v23]

# =====================================================================
# 경로 / 상수
# =====================================================================

_BASE                = os.path.dirname(os.path.abspath(__file__))
LOG_FILE             = os.path.join(_BASE, "rag_eval_logs_v23.json")
EMBED_CACHE_FILE     = os.path.join(_BASE, "embed_cache_v23.pkl")
ANSWER_CACHE_FILE    = os.path.join(_BASE, "answer_cache_v23.json")
FAILURE_DATASET_FILE = os.path.join(_BASE, "failure_dataset_v23.json")
USERS_FILE           = os.path.join(_BASE, "rag_users_v23.json")
USAGE_LOG_FILE       = os.path.join(_BASE, "rag_usage_v23.json")

ANSWER_CACHE_TTL_SEC       = 1800
QUERY_CACHE_TTL_SEC        = 3600
FAILURE_THRESHOLD_ACCURACY = 3
PARALLEL_MAX_WORKERS       = 4
DEDUP_THRESHOLD_DEFAULT    = 0.85
RATE_LIMIT_PER_HOUR        = 20
ALERT_ACCURACY_MIN         = 3.0
ALERT_HALL_MAX             = 0.30
ALERT_LATENCY_P95_MS       = 15_000
LLM_COMPRESS_MAX_SENTS     = 12    # [NEW v23]
TOOL_WEBSEARCH_TIMEOUT_SEC = 5     # [NEW v23]

CALC_PATTERNS = [
    r'\d+[,.]\d+', r'얼마', r'몇\s', r'합계', r'총\s', r'평균',
    r'비율', r'퍼센트', r'%', r'증가', r'감소', r'차이', r'계산',
    r'합산', r'곱하', r'나누', r'더하', r'빼',
]

# =====================================================================
# 쿼리 라우팅 프롬프트
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
    {"id": "no_bm25",     "name": "❌ No BM25 (Dense만)", "query_rewrite": True, "bm25": False, "rerank": True},
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
    return bool(reasons), " + ".join(reasons)


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
# 캐시 전략
# =====================================================================

class QueryResultCache:
    """인메모리 쿼리 결과 캐시 (v23: st.session_state 제거 → dict 기반)"""
    def __init__(self):
        self._store: dict = {}
        self.hits = self.misses = 0

    def _key(self, question, use_bm25, prefilter_n):
        return hashlib.md5(f"{question}|{use_bm25}|{prefilter_n}".encode()).hexdigest()

    def get(self, question, use_bm25, prefilter_n, ttl=QUERY_CACHE_TTL_SEC):
        key = self._key(question, use_bm25, prefilter_n)
        if key in self._store:
            e = self._store[key]
            if time.time() - e["ts"] < ttl:
                self.hits += 1
                return e["items"]
            del self._store[key]
        self.misses += 1
        return None

    def set(self, question, use_bm25, prefilter_n, items):
        self._store[self._key(question, use_bm25, prefilter_n)] = {"items": items, "ts": time.time()}

    def size(self):
        return len(self._store)

    def clear(self):
        self._store = {}
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
# 실패 데이터셋
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

    def get_by_type(self, failure_type: str):
        return [e for e in self._data if failure_type in e.get("failure_types", [])]

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


# =====================================================================
# MetricsCollector
# =====================================================================

class MetricsCollector:
    @staticmethod
    def percentile(data: list, p: float) -> float:
        if not data:
            return 0.0
        sd = sorted(data)
        idx = (len(sd) - 1) * p / 100
        low, high = int(idx), min(int(idx) + 1, len(sd) - 1)
        return sd[low] + (sd[high] - sd[low]) * (idx - low)

    def compute_from_logs(self, logs: list) -> dict:
        if not logs:
            return {}
        latencies  = [l.get("total_latency_ms", 0) for l in logs]
        accuracies = [l["evaluation"].get("정확도", 0) for l in logs if l.get("evaluation")]
        relevances = [l["evaluation"].get("관련성", 0) for l in logs if l.get("evaluation")]
        hall_list  = [l["evaluation"].get("환각여부", "없음") for l in logs if l.get("evaluation")]
        tokens     = [l.get("total_tokens", {}).get("total", 0) for l in logs]
        fail_count = sum(1 for l in logs if l.get("failure_saved"))
        cache_hits = sum(1 for l in logs if l.get("cache_hit"))
        fallbacks  = sum(1 for l in logs if l.get("fallback_triggered"))
        tool_used  = sum(1 for l in logs if l.get("tool_used"))
        par_ms     = [l["parallel_ms"] for l in logs if l.get("parallel_ms")]
        hall_rate  = round(sum(1 for h in hall_list if h != "없음") / max(len(hall_list), 1), 4)

        now     = datetime.now()
        logs_24h = [l for l in logs if self._parse_ts(l.get("timestamp","")) >= now - timedelta(hours=24)]
        logs_7d  = [l for l in logs if self._parse_ts(l.get("timestamp","")) >= now - timedelta(days=7)]

        return {
            "total_queries":      len(logs),
            "queries_24h":        len(logs_24h),
            "queries_7d":         len(logs_7d),
            "latency_p50_ms":     round(self.percentile(latencies, 50)),
            "latency_p95_ms":     round(self.percentile(latencies, 95)),
            "latency_p99_ms":     round(self.percentile(latencies, 99)),
            "latency_avg_ms":     round(statistics.mean(latencies)) if latencies else 0,
            "latency_trend":      [l.get("total_latency_ms", 0) for l in logs[-30:]],
            "accuracy_avg":       round(statistics.mean(accuracies), 2) if accuracies else 0,
            "accuracy_trend":     accuracies[-30:],
            "relevance_avg":      round(statistics.mean(relevances), 2) if relevances else 0,
            "hallucination_rate": hall_rate,
            "hall_trend":         [1 if h != "없음" else 0 for h in hall_list[-30:]],
            "token_avg":          round(statistics.mean(tokens)) if tokens else 0,
            "token_total":        sum(tokens),
            "failure_count":      fail_count,
            "cache_hit_rate":     round(cache_hits / max(len(logs), 1), 4),
            "fallback_rate":      round(fallbacks / max(len(logs), 1), 4),
            "tool_used_count":    tool_used,
            "parallel_p95_ms":    round(self.percentile(par_ms, 95)) if par_ms else 0,
        }

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    def get_alerts(self, stats: dict) -> list:
        alerts = []
        if stats.get("accuracy_avg", 5) < ALERT_ACCURACY_MIN and stats.get("total_queries", 0) >= 5:
            alerts.append({"level": "error", "icon": "🔴",
                           "message": f"평균 정확도 {stats['accuracy_avg']:.1f}/5 — 임계값 {ALERT_ACCURACY_MIN} 미달", "metric": "accuracy"})
        if stats.get("hallucination_rate", 0) > ALERT_HALL_MAX and stats.get("total_queries", 0) >= 5:
            alerts.append({"level": "warning", "icon": "🟡",
                           "message": f"환각 비율 {stats['hallucination_rate']:.0%} — 임계값 {ALERT_HALL_MAX:.0%} 초과", "metric": "hallucination"})
        if stats.get("latency_p95_ms", 0) > ALERT_LATENCY_P95_MS:
            alerts.append({"level": "warning", "icon": "🟡",
                           "message": f"P95 레이턴시 {stats['latency_p95_ms']:,}ms — 임계값 초과", "metric": "latency"})
        if not alerts:
            alerts.append({"level": "ok", "icon": "🟢", "message": "모든 지표 정상", "metric": "all"})
        return alerts

    def export_prometheus(self, stats: dict) -> str:
        ts = int(time.time() * 1000)
        lines = [
            f"# TYPE rag_queries_total counter",
            f'rag_queries_total {stats.get("total_queries", 0)} {ts}',
            f"# TYPE rag_latency_p50_ms gauge",
            f'rag_latency_p50_ms {stats.get("latency_p50_ms", 0)} {ts}',
            f"# TYPE rag_latency_p95_ms gauge",
            f'rag_latency_p95_ms {stats.get("latency_p95_ms", 0)} {ts}',
            f"# TYPE rag_accuracy_avg gauge",
            f'rag_accuracy_avg {stats.get("accuracy_avg", 0)} {ts}',
            f"# TYPE rag_hallucination_rate gauge",
            f'rag_hallucination_rate {stats.get("hallucination_rate", 0)} {ts}',
            f"# TYPE rag_cache_hit_rate gauge",
            f'rag_cache_hit_rate {stats.get("cache_hit_rate", 0)} {ts}',
            f"# TYPE rag_failure_count counter",
            f'rag_failure_count {stats.get("failure_count", 0)} {ts}',
        ]
        return "\n".join(lines)

    def export_json(self, stats: dict) -> bytes:
        export = {k: v for k, v in stats.items() if k not in ("latency_trend", "accuracy_trend", "hall_trend")}
        export["exported_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")


# =====================================================================
# UserManager
# =====================================================================

class UserManager:
    def __init__(self, users_path: str, usage_path: str):
        self.users_path = users_path
        self.usage_path = usage_path
        self._ensure_defaults()

    def _hash(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _load_users(self) -> dict:
        if os.path.exists(self.users_path):
            try:
                with open(self.users_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_users(self, data: dict):
        with open(self.users_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_usage(self) -> dict:
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_usage(self, data: dict):
        with open(self.usage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _ensure_defaults(self):
        users = self._load_users()
        changed = False
        for uname, pw, role, display in [
            ("admin", "admin123", "admin", "관리자"),
            ("demo",  "demo123",  "user",  "데모 사용자"),
        ]:
            if uname not in users:
                users[uname] = {"password_hash": self._hash(pw), "role": role,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "display_name": display}
                changed = True
        if changed:
            self._save_users(users)

    def verify_login(self, username: str, password: str) -> bool:
        u = self._load_users().get(username)
        return bool(u and u.get("password_hash") == self._hash(password))

    def get_role(self, username: str) -> str:
        return self._load_users().get(username, {}).get("role", "user")

    def get_display_name(self, username: str) -> str:
        return self._load_users().get(username, {}).get("display_name", username)

    def list_users(self) -> list:
        return [{"username": k, "role": v.get("role","user"),
                 "display_name": v.get("display_name", k),
                 "created_at": v.get("created_at","-")}
                for k, v in self._load_users().items()]

    def create_user(self, username: str, password: str, role: str = "user", display_name: str = "") -> tuple:
        if not username.strip() or not password.strip():
            return False, "사용자명과 비밀번호를 입력하세요."
        if len(password) < 6:
            return False, "비밀번호는 최소 6자 이상이어야 합니다."
        users = self._load_users()
        if username in users:
            return False, f"이미 존재하는 사용자명: {username}"
        users[username] = {"password_hash": self._hash(password), "role": role,
                           "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "display_name": display_name or username}
        self._save_users(users)
        return True, f"사용자 '{username}' 생성 완료."

    def delete_user(self, username: str) -> tuple:
        if username == "admin":
            return False, "admin 계정은 삭제할 수 없습니다."
        users = self._load_users()
        if username not in users:
            return False, f"존재하지 않는 사용자: {username}"
        del users[username]
        self._save_users(users)
        return True, f"사용자 '{username}' 삭제 완료."

    def change_password(self, username: str, new_password: str) -> tuple:
        if len(new_password) < 6:
            return False, "비밀번호는 최소 6자 이상이어야 합니다."
        users = self._load_users()
        if username not in users:
            return False, f"존재하지 않는 사용자: {username}"
        users[username]["password_hash"] = self._hash(new_password)
        self._save_users(users)
        return True, "비밀번호가 변경되었습니다."

    def record_usage(self, username: str):
        usage = self._load_usage()
        if username not in usage:
            usage[username] = []
        usage[username].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        usage[username] = usage[username][-500:]
        self._save_usage(usage)

    def check_rate_limit(self, username: str) -> tuple:
        if self._load_users().get(username, {}).get("role") == "admin":
            return True, RATE_LIMIT_PER_HOUR
        usage  = self._load_usage()
        cutoff = datetime.now() - timedelta(hours=1)
        recent = [t for t in usage.get(username, []) if MetricsCollector._parse_ts(t) >= cutoff]
        remaining = max(0, RATE_LIMIT_PER_HOUR - len(recent))
        return remaining > 0, remaining

    def get_user_stats(self, username: str) -> dict:
        usage  = self._load_usage()
        all_ts = usage.get(username, [])
        now    = datetime.now()
        h1 = [t for t in all_ts if MetricsCollector._parse_ts(t) >= now - timedelta(hours=1)]
        d1 = [t for t in all_ts if MetricsCollector._parse_ts(t) >= now - timedelta(days=1)]
        return {"total_queries": len(all_ts), "queries_1h": len(h1), "queries_24h": len(d1),
                "rate_limit": RATE_LIMIT_PER_HOUR, "remaining_1h": max(0, RATE_LIMIT_PER_HOUR - len(h1))}


# =====================================================================
# 실패 분류 / 힌트
# =====================================================================

def classify_failure_types(evaluation, quality_report, sqr=None):
    types = []
    acc  = evaluation.get("정확도", 5)
    rel  = evaluation.get("관련성", 5)
    hall = evaluation.get("환각여부", "없음")
    miss = evaluation.get("누락_정보", "없음")
    if acc <= FAILURE_THRESHOLD_ACCURACY:               types.append("low_accuracy")
    if rel <= FAILURE_THRESHOLD_ACCURACY:               types.append("low_relevance")
    if hall in ("부분적", "있음"):                       types.append("hallucination")
    if miss and miss != "없음":                          types.append("incomplete_answer")
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
        span = {"name": name, "duration_ms": int((time.time() - start) * 1000),
                "tokens": tokens or {"prompt": 0, "completion": 0, "total": 0},
                "input_summary": input_summary, "output_summary": output_summary,
                "decision": decision, "error": error}
        self.spans.append(span)
        return span

    def total_tokens(self):
        p = sum(s["tokens"]["prompt"] for s in self.spans)
        c = sum(s["tokens"]["completion"] for s in self.spans)
        return {"prompt": p, "completion": c, "total": p + c}

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

    def size(self):
        return len(self._cache)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path):
            os.remove(self.path)
        self.hits = self.misses = 0


# =====================================================================
# 전역 싱글톤 인스턴스
# =====================================================================

embed_cache        = EmbeddingCache(EMBED_CACHE_FILE)
query_result_cache = QueryResultCache()
answer_cache       = AnswerCache(ANSWER_CACHE_FILE)
failure_dataset    = FailureDataset(FAILURE_DATASET_FILE)
metrics_collector  = MetricsCollector()
user_manager       = UserManager(USERS_FILE, USAGE_LOG_FILE)

_rewrite_cache: dict = {}   # [v23] st.session_state 대체


# =====================================================================
# 임베딩
# =====================================================================

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
    return embeddings / np.where(norms == 0, 1, norms)


def compute_ndcg(ordered_items, score_dict, k):
    if not score_dict:
        return 0.0
    n = min(k, len(ordered_items))
    if n == 0:
        return 0.0
    dcg  = sum(score_dict.get(i, 0.0) / np.log2(i + 2) for i in range(n))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(score_dict.values(), reverse=True)[:n]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


_NDCG_EXCELLENT, _NDCG_GOOD, _NDCG_FAIR = 0.9, 0.7, 0.5


def compute_search_quality_report(ndcg_prefilter, use_bm25, n_candidates):
    reranker_gain = round(1.0 - ndcg_prefilter, 4)
    if ndcg_prefilter >= _NDCG_EXCELLENT:
        quality_label, diagnosis = "excellent", "임베딩 검색이 이미 최적 순서"
    elif ndcg_prefilter >= _NDCG_GOOD:
        quality_label, diagnosis = "good", "임베딩 검색 품질 양호 → 리랭커 소폭 개선"
    elif ndcg_prefilter >= _NDCG_FAIR:
        quality_label, diagnosis = "fair", "임베딩 검색 품질 보통 → 리랭커 개선 효과 유의미"
    else:
        quality_label, diagnosis = "poor", "임베딩 검색 품질 낮음 → 청킹/임베딩 개선 검토 필요"
    return {"ndcg_prefilter": ndcg_prefilter, "reranker_gain": reranker_gain,
            "quality_label": quality_label, "diagnosis": diagnosis,
            "use_bm25": use_bm25, "n_candidates": n_candidates}


# =====================================================================
# 문서 처리 유틸
# =====================================================================

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")
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
        if current:
            raw_chunks.append(current.strip())
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
# 키워드 추출 / Multi-Vector Index
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


def build_multi_vector_index(chunks):
    chunk_embs = normalize(get_embeddings_cached(chunks))
    chunk_idx  = faiss.IndexFlatIP(chunk_embs.shape[1])
    chunk_idx.add(chunk_embs)
    all_sents, sent_to_chunk = [], []
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
    kw_strings = [extract_keywords_simple(c, top_k=10) for c in chunks]
    kw_embs    = normalize(get_embeddings_cached(kw_strings))
    kw_idx     = faiss.IndexFlatIP(kw_embs.shape[1])
    kw_idx.add(kw_embs)
    return {"chunk_index": chunk_idx, "sent_index": sent_idx, "kw_index": kw_idx,
            "sent_to_chunk": sent_to_chunk, "n_chunks": len(chunks), "n_sentences": len(all_sents)}


# =====================================================================
# 병렬 검색
# =====================================================================

def retrieve_parallel(queries: list, index, chunks: list, sources: list,
                      mv_index: dict = None, use_bm25: bool = True,
                      top_k_per_query: int = 20, tracer=None) -> tuple:
    tracer and tracer.start("parallel_search")
    t_start  = time.time()
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
        if not mv_index:
            return "sentence", {}
        scores = {}
        sent_idx, stc = mv_index["sent_index"], mv_index["sent_to_chunk"]
        n_s = mv_index["n_sentences"]
        for query in queries:
            q_emb = normalize(get_embeddings_cached([query]))
            _, sr  = sent_idx.search(q_emb, min(top_k_per_query * 2, n_s))
            seen = {}
            for rank, si in enumerate(sr[0]):
                if 0 <= si < len(stc):
                    ci = stc[si]
                    if ci not in seen:
                        seen[ci] = rank
            for ci, rank in seen.items():
                scores[ci] = scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        return "sentence", scores

    def _keyword():
        if not mv_index:
            return "keyword", {}
        scores = {}
        kw_idx = mv_index["kw_index"]
        for query in queries:
            kw = extract_keywords_simple(query, top_k=8)
            if not kw.strip():
                continue
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

    sorted_ci   = sorted(rrf_total, key=lambda i: rrf_total[i], reverse=True)
    result      = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
    parallel_ms = int((time.time() - t_start) * 1000)

    if tracer:
        tracer.end("parallel_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개 × 4 채널 병렬",
                   output_summary=f"{len(result)}개 후보 ({parallel_ms}ms)",
                   decision=f"ThreadPoolExecutor RRF 통합")
    return result, parallel_ms


# =====================================================================
# LongContextReorder / Selective Context Phase 2
# =====================================================================

def reorder_lost_in_middle(chunks: list, scores: list) -> list:
    if len(chunks) <= 2:
        return chunks
    paired  = sorted(zip(scores or [0]*len(chunks), chunks), key=lambda x: x[0] or 0, reverse=True)
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


def selective_context_phase2(question: str, chunks: list,
                              dedup_threshold: float = DEDUP_THRESHOLD_DEFAULT,
                              max_sentences_per_chunk: int = 6,
                              tracer=None) -> tuple:
    tracer and tracer.start("selective_context")
    all_sents = []
    for ci, chunk in enumerate(chunks):
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        for s in sents:
            all_sents.append((ci, s))
    if not all_sents:
        return chunks, {"original_sents": 0, "after_dedup": 0, "ratio": 1.0}
    sent_texts = [s[1] for s in all_sents]
    sent_embs  = normalize(get_embeddings_cached(sent_texts))
    q_emb      = normalize(get_embeddings_cached([question]))[0]
    kept_indices, kept_embs = [], []
    for i, emb in enumerate(sent_embs):
        if not any(float(np.dot(emb, ke)) >= dedup_threshold for ke in kept_embs):
            kept_indices.append(i)
            kept_embs.append(emb)
    kept_sents = [(all_sents[i][0], all_sents[i][1], float(np.dot(sent_embs[i], q_emb)))
                  for i in kept_indices]
    chunk_sents: dict = {ci: [] for ci in range(len(chunks))}
    for ci, text, score in kept_sents:
        chunk_sents[ci].append((text, score))
    compressed = []
    for ci in range(len(chunks)):
        sents = sorted(chunk_sents[ci], key=lambda x: x[1], reverse=True)[:max_sentences_per_chunk]
        ordered = [s for s, _ in sents]
        compressed.append(" ".join(ordered) if ordered else chunks[ci][:200])
    orig_chars = sum(len(c) for c in chunks)
    comp_chars = sum(len(c) for c in compressed)
    stats = {"original_sents": len(all_sents), "after_dedup": len(kept_indices),
             "dedup_removed": len(all_sents) - len(kept_indices),
             "orig_chars": orig_chars, "comp_chars": comp_chars,
             "ratio": round(comp_chars / max(orig_chars, 1), 2)}
    if tracer:
        tracer.end("selective_context", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(chunks)}청크 / {len(all_sents)}문장",
                   output_summary=f"중복 {stats['dedup_removed']}개 제거 → {stats['ratio']:.0%}",
                   decision=f"Cross-chunk 코사인 중복 제거 (임계값 {dedup_threshold})")
    return compressed, stats


# =====================================================================
# Context Compression Phase 1 (임베딩 기반)
# =====================================================================

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
                      "ratio": round(len(comp_text) / max(len(chunk), 1), 2)})
    if tracer:
        orig_t = sum(s["original"] for s in stats)
        comp_t = sum(s["compressed"] for s in stats)
        tracer.end("context_compression", tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"{len(chunks)}개 청크 ({orig_t}자)",
                   output_summary=f"압축 후 {comp_t}자 (평균 {round(comp_t/max(orig_t,1),2):.0%})",
                   decision=f"문장 유사도 ≥ {min_sim}")
    return compressed, stats


# =====================================================================
# [NEW v23] Context Compression Phase 3 — LLM 추출형 압축
# =====================================================================

def compress_chunks_llm(question: str, chunks: list,
                         max_total_sentences: int = LLM_COMPRESS_MAX_SENTS,
                         tracer=None) -> tuple:
    """
    [v23] Phase 3 LLM 추출형 압축.
    Phase 1(임베딩 per-chunk), Phase 2(Cross-chunk 중복 제거)와 달리
    LLM이 모든 청크를 가로질러 질문에 가장 관련 있는 N개 문장을 직접 선택.
    """
    tracer and tracer.start("llm_compression")

    # 청크 → 문장 ID 목록 구성
    all_keys = []
    sent_map: dict = {}
    for ci, chunk in enumerate(chunks):
        sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', chunk) if len(s.strip()) > 15]
        if not sents:
            sents = [chunk[:300]]
        for si, sent in enumerate(sents):
            key = f"{ci}_{si}"
            all_keys.append(key)
            sent_map[key] = sent

    total_sents = len(all_keys)
    orig_chars  = sum(len(c) for c in chunks)

    # 문장 수가 max 이하면 압축 불필요
    if total_sents <= max_total_sentences:
        stats = {"total_sents": total_sents, "kept_sents": total_sents,
                 "ratio": 1.0, "orig_chars": orig_chars, "comp_chars": orig_chars}
        if tracer:
            tracer.end("llm_compression", tokens={"prompt": 0, "completion": 0, "total": 0},
                       input_summary=f"{total_sents}문장 ≤ {max_total_sentences} → 압축 불필요",
                       output_summary="패스", decision="임계치 이하")
        return chunks, stats

    numbered = "\n".join([f"{k}: {sent_map[k]}" for k in all_keys])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"아래 문장 목록에서 질문에 가장 관련 있는 {max_total_sentences}개 문장 ID를 선택하세요.\n"
                "반드시 쉼표로 구분된 ID만 출력하세요. 예: 0_0, 1_2, 2_1\n"
                "다른 텍스트는 절대 출력하지 마세요."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[문장 목록]\n{numbered}"}
        ]
    )

    raw = resp.choices[0].message.content.strip()
    selected_keys = [k.strip() for k in raw.split(",") if k.strip() in sent_map]
    if not selected_keys:
        selected_keys = all_keys[:max_total_sentences]

    # 선택된 문장을 청크 단위로 재구성
    chunk_kept: dict = {ci: [] for ci in range(len(chunks))}
    for key in selected_keys:
        parts = key.split("_", 1)
        if len(parts) == 2:
            try:
                ci = int(parts[0])
                if ci < len(chunks):
                    chunk_kept[ci].append(sent_map[key])
            except ValueError:
                pass

    compressed = []
    for ci in range(len(chunks)):
        if chunk_kept[ci]:
            compressed.append(" ".join(chunk_kept[ci]))
        else:
            compressed.append(chunks[ci])

    comp_chars = sum(len(c) for c in compressed)
    stats = {"total_sents": total_sents, "kept_sents": len(selected_keys),
             "ratio": round(comp_chars / max(orig_chars, 1), 2),
             "orig_chars": orig_chars, "comp_chars": comp_chars}

    if tracer:
        u = resp.usage
        tracer.end("llm_compression",
                   tokens={"prompt": u.prompt_tokens, "completion": u.completion_tokens, "total": u.total_tokens},
                   input_summary=f"{total_sents}문장 / {len(chunks)}청크",
                   output_summary=f"{len(selected_keys)}문장 유지 ({stats['ratio']:.0%})",
                   decision=f"LLM Phase 3 추출형 압축 (max={max_total_sentences})")
    return compressed, stats


# =====================================================================
# Tool-Augmented RAG (v21 — 수치 계산 특화)
# =====================================================================

def detect_calc_intent(question: str, chunks: list) -> bool:
    q_calc   = any(re.search(p, question) for p in CALC_PATTERNS)
    ctx_nums = any(re.search(r'\d{2,}', c) for c in chunks)
    return q_calc and ctx_nums


def _safe_eval(code: str, data: dict):
    safe_globals = {
        "__builtins__": {}, "math": math, "round": round, "abs": abs,
        "sum": sum, "max": max, "min": min, "len": len, "int": int, "float": float,
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
    if not detect_calc_intent(question, chunks):
        return None, None, None
    tracer and tracer.start("tool_extract")
    context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(chunks)])
    resp1 = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "문서에서 수치 계산에 필요한 데이터를 추출하고 Python 코드를 생성하세요.\n"
                "JSON 형식: {\"has_calculation\": bool, \"data\": {\"변수명\": 숫자값}, "
                "\"python_code\": \"result = ...\", \"explanation\": \"설명\"}"
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[문서]\n{context}"}
        ]
    )
    try:
        extracted = json.loads(resp1.choices[0].message.content)
    except Exception:
        extracted = {"has_calculation": False}
    tracer and tracer.end("tool_extract", tokens=_usage(resp1),
                           input_summary=question[:60],
                           output_summary=f"계산 필요: {extracted.get('has_calculation', False)}",
                           decision="수치 추출 + Python 코드 생성")
    if not extracted.get("has_calculation", False):
        return None, None, None
    python_code = extracted.get("python_code", "")
    calc_result = _safe_eval(python_code, extracted.get("data", {}))
    if calc_result is None:
        return None, python_code, None
    tracer and tracer.start("tool_answer")
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "문서 기반 어시스턴트. **📌 요약** / **📖 근거** / "
                "**🔢 계산 결과** / **✅ 결론** 구조. 계산 결과는 제공된 값을 그대로 쓰세요."
            )},
            {"role": "user", "content": (
                f"[참고 문서]\n{context}\n\n[질문]\n{question}\n\n"
                f"[Python 계산 결과]\n코드: {python_code}\n결과: {calc_result}\n"
                f"설명: {extracted.get('explanation', '')}"
            )}
        ]
    )
    answer = resp2.choices[0].message.content
    tracer and tracer.end("tool_answer", tokens=_usage(resp2),
                           input_summary=f"계산 결과: {calc_result}",
                           output_summary=f"Tool-Augmented 답변 {len(answer)}자",
                           decision="Python 계산 결과 주입 → 수치 환각 방지")
    return answer, python_code, str(calc_result)


# =====================================================================
# [NEW v23] Tool 클래스 계층
# =====================================================================

class Tool:
    name: str        = ""
    description: str = ""
    parameters: dict = {}

    def run(self, **kwargs) -> dict:
        raise NotImplementedError

    def to_openai_schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class CalculatorTool(Tool):
    name        = "calculator"
    description = "수학 수식을 계산합니다. Python 표현식을 사용하세요. 예: 1234 * 0.15, math.sqrt(256)"
    parameters  = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "계산할 Python 수학 표현식"}},
        "required": ["expression"]
    }

    def run(self, expression: str, **kwargs) -> dict:
        import ast
        safe_globals = {
            "__builtins__": {}, "math": math, "abs": abs, "round": round,
            "max": max, "min": min, "sum": sum, "len": len,
            "int": int, "float": float, "pow": pow,
        }
        try:
            tree   = ast.parse(expression, mode='eval')
            result = eval(compile(tree, '<expr>', 'eval'), safe_globals)
            return {"success": True, "expression": expression,
                    "result": result, "result_str": str(result)}
        except Exception as e:
            return {"success": False, "expression": expression,
                    "error": str(e), "result": None, "result_str": "계산 오류"}


class DateTimeTool(Tool):
    name        = "datetime_tool"
    description = "현재 날짜/시간 조회 또는 날짜 계산(차이, 더하기/빼기)을 수행합니다."
    parameters  = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["now", "diff_days", "add_days"],
                          "description": "now: 현재 시간, diff_days: 두 날짜 차이, add_days: 날짜 더하기"},
            "date1": {"type": "string", "description": "기준 날짜 YYYY-MM-DD"},
            "date2": {"type": "string", "description": "비교 날짜 YYYY-MM-DD (diff_days 전용)"},
            "days":  {"type": "integer", "description": "더하거나 뺄 일수 (add_days 전용)"},
        },
        "required": ["operation"]
    }

    def run(self, operation: str, date1: str = None, date2: str = None, days: int = None, **kwargs) -> dict:
        try:
            if operation == "now":
                now = datetime.now()
                return {"success": True, "result": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "year": now.year, "month": now.month, "day": now.day,
                        "weekday": now.strftime("%A")}
            elif operation == "diff_days" and date1 and date2:
                d1   = datetime.strptime(date1, "%Y-%m-%d")
                d2   = datetime.strptime(date2, "%Y-%m-%d")
                diff = (d2 - d1).days
                return {"success": True, "result": diff,
                        "result_str": f"{abs(diff)}일 차이 ({diff:+d}일)"}
            elif operation == "add_days" and date1 and days is not None:
                d1     = datetime.strptime(date1, "%Y-%m-%d")
                result = d1 + timedelta(days=days)
                return {"success": True, "result": result.strftime("%Y-%m-%d"),
                        "result_str": result.strftime("%Y년 %m월 %d일")}
            else:
                return {"success": False, "error": "잘못된 파라미터", "result": None, "result_str": "오류"}
        except Exception as e:
            return {"success": False, "error": str(e), "result": None, "result_str": "오류"}


class UnitConverterTool(Tool):
    name        = "unit_converter"
    description = "단위를 변환합니다. 길이(m/km/cm/ft/in), 무게(kg/g/lb/oz), 온도(C/F/K), 넓이, 부피 지원."
    parameters  = {
        "type": "object",
        "properties": {
            "value":     {"type": "number", "description": "변환할 숫자 값"},
            "from_unit": {"type": "string", "description": "원래 단위 (예: km, kg, C)"},
            "to_unit":   {"type": "string", "description": "변환할 단위 (예: mi, lb, F)"},
        },
        "required": ["value", "from_unit", "to_unit"]
    }

    _FACTORS = {
        # 길이 (m 기준)
        "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144,
        # 무게 (kg 기준)
        "kg": 1, "g": 0.001, "mg": 0.000001, "lb": 0.453592, "oz": 0.0283495, "t": 1000,
        # 넓이 (m² 기준)
        "m2": 1, "km2": 1e6, "cm2": 0.0001, "ha": 10000, "acre": 4046.86,
        # 부피 (L 기준)
        "l": 1, "ml": 0.001, "m3": 1000, "gal": 3.78541, "fl_oz": 0.0295735,
    }
    _TEMP_C = {"c", "celsius", "섭씨"}
    _TEMP_F = {"f", "fahrenheit", "화씨"}
    _TEMP_K = {"k", "kelvin", "켈빈"}

    def run(self, value: float, from_unit: str, to_unit: str, **kwargs) -> dict:
        fu, tu = from_unit.lower(), to_unit.lower()
        # 온도 특수 처리
        if fu in self._TEMP_C and tu in self._TEMP_F:
            result = value * 9 / 5 + 32
        elif fu in self._TEMP_F and tu in self._TEMP_C:
            result = (value - 32) * 5 / 9
        elif fu in self._TEMP_C and tu in self._TEMP_K:
            result = value + 273.15
        elif fu in self._TEMP_K and tu in self._TEMP_C:
            result = value - 273.15
        elif fu in self._FACTORS and tu in self._FACTORS:
            result = value * self._FACTORS[fu] / self._FACTORS[tu]
        else:
            return {"success": False, "error": f"지원하지 않는 단위: {from_unit} → {to_unit}",
                    "value": value, "from_unit": from_unit, "to_unit": to_unit,
                    "result": None, "result_str": "단위 오류"}
        result_r = round(result, 6)
        return {"success": True, "value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": result_r, "result_str": f"{value} {from_unit} = {result_r} {to_unit}"}


class WebSearchTool(Tool):
    name        = "web_search"
    description = "DuckDuckGo로 웹 검색을 수행합니다. 최신 정보, 실시간 데이터, 문서에 없는 정보 확인에 사용."
    parameters  = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "검색할 쿼리"}},
        "required": ["query"]
    }

    def run(self, query: str, **kwargs) -> dict:
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx 미설치 (pip install httpx)",
                    "query": query, "result": "httpx 필요", "source": "없음"}
        try:
            import urllib.parse
            url  = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            resp = httpx.get(url, timeout=TOOL_WEBSEARCH_TIMEOUT_SEC)
            data = resp.json()
            parts = []
            if data.get("AbstractText"):
                parts.append(data["AbstractText"])
            if data.get("Answer"):
                parts.append(f"답변: {data['Answer']}")
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    parts.append(topic["Text"])
            result = "\n".join(parts) if parts else "검색 결과 없음"
            return {"success": True, "query": query, "result": result, "source": "DuckDuckGo",
                    "abstract_source": data.get("AbstractSource", ""),
                    "abstract_url": data.get("AbstractURL", "")}
        except Exception as e:
            return {"success": False, "error": str(e), "query": query,
                    "result": "검색 실패", "source": "DuckDuckGo"}


# =====================================================================
# [NEW v23] ToolRegistry — OpenAI Function Calling 기반 도구 오케스트레이터
# =====================================================================

class ToolRegistry:
    def __init__(self, tools: list = None):
        self._tools: dict = {}
        self._call_log: list = []
        for tool in (tools or [CalculatorTool(), DateTimeTool(), UnitConverterTool(), WebSearchTool()]):
            self._tools[tool.name] = tool

    def get_openai_tools(self, enabled_names: list = None) -> list:
        if enabled_names is None:
            return [t.to_openai_schema() for t in self._tools.values()]
        return [t.to_openai_schema() for n, t in self._tools.items() if n in enabled_names]

    def execute(self, tool_name: str, args: dict) -> dict:
        tool = self._tools.get(tool_name)
        if not tool:
            result = {"success": False, "error": f"알 수 없는 도구: {tool_name}"}
        else:
            try:
                result = tool.run(**args)
            except Exception as e:
                result = {"success": False, "error": str(e)}
        self._call_log.append({"tool_name": tool_name, "args": args, "result": result,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        return result

    def get_call_log(self) -> list:
        return list(self._call_log)

    def clear_log(self):
        self._call_log.clear()

    def run_with_llm(self, question: str, context: str,
                     enabled_names: list = None, tracer=None) -> tuple:
        """OpenAI function calling으로 도구 선택 → 실행 → 최종 답변 생성."""
        tracer and tracer.start("tool_registry")
        tools_schema = self.get_openai_tools(enabled_names)
        if not tools_schema:
            tracer and tracer.end("tool_registry", input_summary=question[:60],
                                   output_summary="활성화된 도구 없음", decision="스킵")
            return None, []

        messages = [
            {"role": "system", "content": (
                "문서 기반 어시스턴트입니다. 질문에 답하기 위해 필요한 도구를 사용하세요.\n"
                "도구 결과를 활용해 정확하고 명확한 한국어 답변을 제공하세요."
            )},
            {"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"}
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            tracer and tracer.end("tool_registry", tokens=_usage(resp),
                                   input_summary=question[:60],
                                   output_summary="도구 호출 없음",
                                   decision="LLM이 도구 불필요 판단")
            return None, []

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
        })

        tool_calls_list = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = self.execute(tc.function.name, args)
            tool_calls_list.append({"tool_name": tc.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})

        resp2        = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        final_answer = resp2.choices[0].message.content

        if tracer:
            tool_names = [tc["tool_name"] for tc in tool_calls_list]
            total_tok  = {k: (resp.usage.__dict__.get(k, 0) or 0) + (resp2.usage.__dict__.get(k, 0) or 0)
                          for k in ["prompt_tokens", "completion_tokens", "total_tokens"]}
            tracer.end("tool_registry",
                       tokens={"prompt": total_tok["prompt_tokens"],
                               "completion": total_tok["completion_tokens"],
                               "total": total_tok["total_tokens"]},
                       input_summary=f"활성 도구: {enabled_names or 'all'}",
                       output_summary=f"도구 호출: {', '.join(tool_names)}",
                       decision=f"OpenAI function calling → {len(tool_calls_list)}개 실행")
        return final_answer, tool_calls_list


# =====================================================================
# [NEW v23] AsyncRAGEngine — asyncio.gather 기반 비동기 파이프라인
# =====================================================================

class AsyncRAGEngine:
    """
    [v23] asyncio.gather()로 4채널 검색을 동시 실행하고,
    AsyncOpenAI로 모든 LLM 호출을 non-blocking 처리.
    Streamlit 호환: run_sync()가 이벤트 루프 충돌을 자동 처리.
    """

    @staticmethod
    def run_sync(coro):
        """Streamlit(이미 루프 실행 중) 환경에서도 async 코루틴을 안전하게 실행."""
        try:
            asyncio.get_running_loop()
            # 루프가 이미 실행 중 → 별도 스레드에서 새 루프 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    # ── 비동기 LLM 헬퍼 ────────────────────────────────────────────

    async def _llm(self, messages: list, model: str = "gpt-4o-mini",
                   response_format=None, tools=None, tool_choice=None) -> object:
        kwargs = {"model": model, "messages": messages}
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return await async_client.chat.completions.create(**kwargs)

    async def _embed_async(self, texts: list) -> np.ndarray:
        return await asyncio.to_thread(get_embeddings_cached, texts)

    # ── 비동기 4채널 검색 ───────────────────────────────────────────

    async def _dense_async(self, queries, index, chunks, sources, mv_index, top_k):
        RRF_K, n_chunks = 60, len(chunks)
        scores = {}
        idx = mv_index["chunk_index"] if mv_index else index
        for query in queries:
            q_emb = await self._embed_async([query])
            q_emb = normalize(q_emb)
            _, indices = idx.search(q_emb, min(top_k, n_chunks))
            for rank, ci in enumerate(indices[0]):
                if 0 <= ci < n_chunks:
                    scores[ci] = scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        return "dense", scores

    async def _bm25_async(self, queries, chunks, use_bm25, top_k):
        if not use_bm25 or not BM25_AVAILABLE:
            return "bm25", {}
        return await asyncio.to_thread(self._bm25_sync, queries, chunks, top_k)

    @staticmethod
    def _bm25_sync(queries, chunks, top_k):
        RRF_K, n_chunks = 60, len(chunks)
        scores = {}
        bm25   = BM25Okapi([c.split() for c in chunks])
        for query in queries:
            arr = bm25.get_scores(query.split())
            for rank, ci in enumerate(np.argsort(arr)[::-1][:top_k]):
                if ci < n_chunks:
                    scores[int(ci)] = scores.get(int(ci), 0.0) + 1.0 / (RRF_K + rank + 1)
        return "bm25", scores

    async def _sentence_async(self, queries, mv_index, top_k):
        if not mv_index:
            return "sentence", {}
        RRF_K   = 60
        scores  = {}
        sent_idx, stc = mv_index["sent_index"], mv_index["sent_to_chunk"]
        n_s     = mv_index["n_sentences"]
        for query in queries:
            q_emb = await self._embed_async([query])
            q_emb = normalize(q_emb)
            _, sr  = sent_idx.search(q_emb, min(top_k * 2, n_s))
            seen   = {}
            for rank, si in enumerate(sr[0]):
                if 0 <= si < len(stc):
                    ci = stc[si]
                    if ci not in seen:
                        seen[ci] = rank
            for ci, rank in seen.items():
                scores[ci] = scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        return "sentence", scores

    async def _keyword_async(self, queries, mv_index, chunks, top_k):
        if not mv_index:
            return "keyword", {}
        RRF_K, n_chunks = 60, len(chunks)
        scores  = {}
        kw_idx  = mv_index["kw_index"]
        for query in queries:
            kw = extract_keywords_simple(query, top_k=8)
            if not kw.strip():
                continue
            kq_emb = await self._embed_async([kw])
            kq_emb = normalize(kq_emb)
            _, kr  = kw_idx.search(kq_emb, min(top_k, n_chunks))
            for rank, ci in enumerate(kr[0]):
                if 0 <= ci < n_chunks:
                    scores[int(ci)] = scores.get(int(ci), 0.0) + 1.0 / (RRF_K + rank + 1)
        return "keyword", scores

    async def retrieve_parallel_async(self, queries, index, chunks, sources,
                                       mv_index, use_bm25, top_k_per_query, tracer) -> tuple:
        tracer and tracer.start("async_parallel_search")
        t_start = time.time()

        results = await asyncio.gather(
            self._dense_async(queries, index, chunks, sources, mv_index, top_k_per_query),
            self._bm25_async(queries, chunks, use_bm25, top_k_per_query),
            self._sentence_async(queries, mv_index, top_k_per_query),
            self._keyword_async(queries, mv_index, chunks, top_k_per_query),
        )

        rrf_total: dict = {}
        for _, partial in results:
            for ci, sc in partial.items():
                rrf_total[ci] = rrf_total.get(ci, 0.0) + sc

        sorted_ci   = sorted(rrf_total, key=lambda i: rrf_total[i], reverse=True)
        result      = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
        parallel_ms = int((time.time() - t_start) * 1000)

        if tracer:
            tracer.end("async_parallel_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                       input_summary=f"쿼리 {len(queries)}개 × 4 채널 asyncio.gather",
                       output_summary=f"{len(result)}개 후보 ({parallel_ms}ms)",
                       decision="AsyncRAGEngine asyncio.gather RRF 통합")
        return result, parallel_ms

    # ── 비동기 LLM 단계들 ──────────────────────────────────────────

    async def _rewrite_async(self, question, n, tracer):
        cache_key = f"{question}||{n}"
        if cache_key in _rewrite_cache:
            return _rewrite_cache[cache_key]
        tracer and tracer.start("query_rewriting")
        resp = await self._llm([
            {"role": "system", "content": f"문서 검색 전문가. 의도 분해형 우선 총 {n}개 재작성. 번호·기호 없이 줄당 하나."},
            {"role": "user",   "content": f"원본 질문: {question}"}
        ])
        variants = [v.strip() for v in resp.choices[0].message.content.strip().split('\n') if v.strip()]
        queries  = [question] + variants[:n]
        _rewrite_cache[cache_key] = queries
        if tracer:
            tracer.end("query_rewriting", tokens=_usage(resp),
                       input_summary=f"원본: {question[:60]}",
                       output_summary=f"쿼리 {len(queries)}개 생성",
                       decision="의도 분해형 비동기")
        return queries

    async def _rerank_async(self, question, items, top_k, tracer):
        tracer and tracer.start("rerank")
        if not items:
            return [], {}
        chunks_text = "\n\n".join([f"[{i+1}] {item[0]}" for i, item in enumerate(items)])
        resp = await self._llm([
            {"role": "system", "content": "질문 대비 각 청크 관련성을 0~10점으로. 형식: 1: 8\n2: 3\n..."},
            {"role": "user",   "content": f"질문: {question}\n\n청크들:\n{chunks_text}"}
        ])
        all_scores = {}
        for line in resp.choices[0].message.content.strip().split('\n'):
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
        tracer and tracer.end("rerank", tokens=_usage(resp),
                               input_summary=f"{len(items)}개 후보",
                               output_summary=f"상위 {top_k}개",
                               decision="비동기 LLM 0~10 채점")
        return result, all_scores

    async def _generate_async(self, question, gen_chunks, final_sources, use_multidoc, tracer):
        if use_multidoc and gen_chunks:
            # Step1: 요약
            tracer and tracer.start("step1_summarize")
            chunks_text = "\n\n".join([f"[{i+1}]\n{c}" for i, c in enumerate(gen_chunks)])
            r1 = await self._llm([
                {"role": "system", "content": "질문 관련해 각 청크 2~3문장 요약. 형식: [1]: 요약\n[2]: 요약\n..."},
                {"role": "user",   "content": f"질문: {question}\n\n청크들:\n{chunks_text}"}
            ])
            summaries = {}
            for line in r1.choices[0].message.content.strip().split('\n'):
                m = re.match(r'\[(\d+)\]:\s*(.+)', line.strip())
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(gen_chunks):
                        summaries[idx] = m.group(2).strip()
            summaries_list = [summaries.get(i, gen_chunks[i][:120]+"...") for i in range(len(gen_chunks))]
            tracer and tracer.end("step1_summarize", tokens=_usage(r1),
                                   input_summary=f"{len(gen_chunks)}개 청크", output_summary="요약 완료",
                                   decision="비동기 청크 요약")

            # Step2: 관계 분석
            tracer and tracer.start("step2_analyze")
            summaries_text = "\n".join([f"[출처 {i+1} | {final_sources[i]}] {s}" for i, s in enumerate(summaries_list)])
            r2 = await self._llm([
                {"role": "system", "content": "청크 요약 구조화. **공통점**/**차이점**/**핵심 정보**/**불확실성** 형식."},
                {"role": "user",   "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
            ])
            analysis = r2.choices[0].message.content
            tracer and tracer.end("step2_analyze", tokens=_usage(r2),
                                   input_summary=f"{len(summaries_list)}개 요약", output_summary="분석 완료",
                                   decision="비동기 관계 분석")

            # Step3: 최종 답변
            tracer and tracer.start("step3_answer")
            numbered = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(gen_chunks)])
            r3 = await self._llm([
                {"role": "system", "content": "문서 기반 어시스턴트. **📌 요약** / **📖 근거** / **✅ 결론** 구조. 한국어."},
                {"role": "user",   "content": f"[요약]\n{summaries_text}\n\n[분석]\n{analysis}\n\n[원문]\n{numbered}\n\n[질문]\n{question}"}
            ])
            answer = r3.choices[0].message.content
            tracer and tracer.end("step3_answer", tokens=_usage(r3),
                                   input_summary="요약+분석+원문", output_summary=f"{len(answer)}자",
                                   decision="비동기 최종 답변")
            return answer, summaries_list, analysis, "multidoc_async"
        else:
            tracer and tracer.start("step3_answer")
            numbered = "\n\n".join([f"[출처 {i+1}]\n{c}" for i, c in enumerate(gen_chunks)])
            r = await self._llm([
                {"role": "system", "content": "문서 기반 어시스턴트. **📌 요약** / **📖 근거** / **✅ 결론** 구조. 한국어."},
                {"role": "user",   "content": f"[참고 문서]\n{numbered}\n\n[질문]\n{question}"}
            ])
            answer = r.choices[0].message.content
            tracer and tracer.end("step3_answer", tokens=_usage(r),
                                   input_summary="원문 직접", output_summary=f"{len(answer)}자",
                                   decision="비동기 단순 모드")
            return answer, [], "", "simple_async"

    async def _self_refine_async(self, question, gen_chunks, draft, tracer):
        tracer and tracer.start("critique")
        context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(gen_chunks)])
        r_crit = await self._llm([
            {"role": "system", "content": "RAG 답변 비판 전문가. **문제점**/**누락**/**개선 방향** 형식."},
            {"role": "user",   "content": f"[질문]\n{question}\n\n[문서]\n{context}\n\n[Draft]\n{draft}"}
        ])
        critique = r_crit.choices[0].message.content
        tracer and tracer.end("critique", tokens=_usage(r_crit),
                               input_summary=f"Draft {len(draft)}자", output_summary="비판 완료",
                               decision="비동기 Critique")

        tracer and tracer.start("refine")
        r_ref = await self._llm([
            {"role": "system", "content": "**📌 요약** / **📖 근거** / **✅ 결론** 구조로 Critique 반영 개선 답변 작성. 한국어."},
            {"role": "user",   "content": f"[질문]\n{question}\n\n[문서]\n{context}\n\n[Draft]\n{draft}\n\n[Critique]\n{critique}"}
        ])
        refined = r_ref.choices[0].message.content
        tracer and tracer.end("refine", tokens=_usage(r_ref),
                               input_summary=f"Draft+Critique", output_summary=f"{len(refined)}자",
                               decision="비동기 Refine")
        return refined, critique

    async def _evaluate_async(self, question, gen_chunks, answer, tracer):
        tracer and tracer.start("evaluation")
        context = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(gen_chunks)])
        resp = await self._llm([
            {"role": "system", "content": (
                "답변 품질 진단 전문가. 아래 형식으로만 출력하라.\n\n"
                "정확도: <1~5 정수>\n관련성: <1~5 정수>\n환각여부: <없음|부분적|있음>\n"
                "환각근거: <없으면 '없음'>\n신뢰도: <높음|보통|낮음>\n"
                "불일치_항목: <없으면 '없음'>\n누락_정보: <없으면 '없음'>\n개선_제안: <1문장>"
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[답변]\n{answer}"}
        ])
        result = {"정확도": 0, "관련성": 0, "환각여부": "알 수 없음", "환각근거": "",
                  "신뢰도": "보통", "불일치_항목": "없음", "누락_정보": "없음", "개선_제안": ""}
        for line in resp.choices[0].message.content.strip().split('\n'):
            if line.startswith("정확도:"):
                try: result["정확도"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
                except ValueError: pass
            elif line.startswith("관련성:"):
                try: result["관련성"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
                except ValueError: pass
            elif line.startswith("환각여부:"):    result["환각여부"]    = line.split(':',1)[1].strip()
            elif line.startswith("환각근거:"):    result["환각근거"]    = line.split(':',1)[1].strip()
            elif line.startswith("신뢰도:"):      result["신뢰도"]      = line.split(':',1)[1].strip()
            elif line.startswith("불일치_항목:"): result["불일치_항목"] = line.split(':',1)[1].strip()
            elif line.startswith("누락_정보:"):   result["누락_정보"]   = line.split(':',1)[1].strip()
            elif line.startswith("개선_제안:"):   result["개선_제안"]   = line.split(':',1)[1].strip()
        tracer and tracer.end("evaluation", tokens=_usage(resp),
                               input_summary="질문+문서+답변",
                               output_summary=f"정확도 {result['정확도']}/5 · 환각 {result['환각여부']}",
                               decision="비동기 구조화 품질 진단")
        return result

    async def run(self, question: str, eff: dict,
                  index, chunks: list, sources: list,
                  prefilter_n: int = 10, use_multidoc: bool = True,
                  num_rewrites: int = 3, use_session_cache: bool = True,
                  use_self_refine: bool = False, use_compression: bool = False,
                  mv_index: dict = None, auto_save_failure: bool = True,
                  gen_improvement_hint: bool = False, use_lim_reorder: bool = True,
                  use_selective_context: bool = False,
                  selective_dedup_thresh: float = DEDUP_THRESHOLD_DEFAULT,
                  use_tool_augment: bool = False, use_llm_compress: bool = False,
                  use_tool_registry: bool = False, enabled_tools: list = None,
                  user_id: str = "anonymous") -> dict:

        tracer = Tracer()

        # 답변 캐시 확인
        if use_session_cache:
            ans_key    = hashlib.md5(question.encode()).hexdigest()
            cached_ans = answer_cache.get(ans_key)
            if cached_ans:
                tracer.start("answer_cache_hit")
                tracer.end("answer_cache_hit", input_summary=question[:60],
                           output_summary="답변 캐시 히트", decision=f"TTL {ANSWER_CACHE_TTL_SEC//60}분")
                return {**_empty_result(tracer, question, eff, prefilter_n),
                        "answer": cached_ans["answer"], "draft_answer": cached_ans["answer"],
                        "evaluation": cached_ans["evaluation"],
                        "quality_report": cached_ans["quality_report"],
                        "mode": "answer_cache_hit", "cache_hit": "answer",
                        "user_id": user_id, "engine": "async"}

        # 쿼리 리라이팅
        queries = (await self._rewrite_async(question, num_rewrites, tracer)
                   if eff["use_query_rewrite"] else [question])

        # 쿼리 결과 캐시
        cache_hit   = None
        parallel_ms = None
        qr_cached   = query_result_cache.get(question, eff["use_bm25"], prefilter_n) if use_session_cache else None

        if qr_cached is not None:
            filtered  = qr_cached
            cache_hit = "query"
        else:
            candidates, parallel_ms = await self.retrieve_parallel_async(
                queries, index, chunks, sources,
                mv_index=mv_index, use_bm25=eff["use_bm25"],
                top_k_per_query=20, tracer=tracer
            )
            filtered = await asyncio.to_thread(
                prefilter_by_similarity, question, candidates, prefilter_n, tracer
            )
            if use_session_cache:
                query_result_cache.set(question, eff["use_bm25"], prefilter_n, filtered)

        # 리랭킹
        ndcg_k, sqr = None, None
        eff_top_k   = eff["top_k"]
        if eff["use_reranking"] and len(filtered) > eff_top_k:
            ranked, all_scores = await self._rerank_async(question, filtered, eff_top_k, tracer)
            ndcg_k = compute_ndcg(filtered, all_scores, k=eff_top_k)
            sqr    = compute_search_quality_report(ndcg_k, eff["use_bm25"], len(filtered))
        else:
            ranked = [(item[0], item[1], None) for item in filtered[:eff_top_k]]

        final_chunks  = [r[0] for r in ranked]
        final_sources = [r[1] for r in ranked]
        final_scores  = [r[2] for r in ranked]

        if use_lim_reorder and len(final_chunks) > 2:
            final_chunks = reorder_lost_in_middle(final_chunks, final_scores)

        gen_chunks         = final_chunks
        compression_stats  = None
        selective_stats    = None
        compress_stats_llm = None

        if use_selective_context and final_chunks:
            gen_chunks, selective_stats = await asyncio.to_thread(
                selective_context_phase2, question, final_chunks,
                selective_dedup_thresh, 6, tracer
            )
        elif use_compression and final_chunks:
            gen_chunks, compression_stats = await asyncio.to_thread(
                compress_chunks, question, final_chunks, 5, 0.25, tracer
            )

        # [NEW v23] LLM Phase 3 압축
        if use_llm_compress and gen_chunks:
            gen_chunks, compress_stats_llm = await asyncio.to_thread(
                compress_chunks_llm, question, gen_chunks, LLM_COMPRESS_MAX_SENTS, tracer
            )

        # Tool 처리
        tool_code             = None
        calc_result           = None
        tool_used             = False
        tool_registry_results = None
        tool_registry_calls   = []

        if use_tool_registry and gen_chunks:
            context_text = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(gen_chunks)])
            tr_answer, tr_calls = await asyncio.to_thread(
                tool_registry.run_with_llm, question, context_text, enabled_tools, tracer
            )
            if tr_answer:
                tool_registry_results = tr_answer
                tool_registry_calls   = tr_calls
                tool_used = True

        if not tool_used and use_tool_augment and gen_chunks:
            ta_answer, tool_code, calc_result = await asyncio.to_thread(
                tool_augmented_answer, question, gen_chunks, tracer
            )
            if ta_answer:
                tool_used = True
                answer    = ta_answer
                summaries, analysis = [], ""
                mode = "tool_augmented_async"

        if not tool_used:
            answer, summaries, analysis, mode = await self._generate_async(
                question, gen_chunks, final_sources, use_multidoc, tracer
            )

        # Self-Refinement
        draft_answer = answer
        critique     = None
        if use_self_refine and gen_chunks:
            answer, critique = await self._self_refine_async(question, gen_chunks, draft_answer, tracer)

        # 평가
        evaluation = await self._evaluate_async(question, gen_chunks, answer, tracer)
        hall       = evaluation.get("환각여부", "없음")
        hall_cause = None
        if hall != "없음":
            hall_cause = await asyncio.to_thread(
                analyze_hallucination_cause, question, gen_chunks, answer, hall, tracer
            )
        quality_report = build_quality_report(evaluation, hall_cause)

        if use_session_cache:
            ans_key = hashlib.md5(question.encode()).hexdigest()
            answer_cache.set(ans_key, {"answer": answer, "evaluation": evaluation,
                                       "quality_report": quality_report})

        failure_types, failure_saved = [], False
        if auto_save_failure:
            failure_types = classify_failure_types(evaluation, quality_report, sqr)
            if failure_types:
                hint = (await asyncio.to_thread(
                    generate_improvement_hint, question, gen_chunks, answer,
                    evaluation, failure_types, tracer
                ) if gen_improvement_hint else None)
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
            "cache_hit": cache_hit, "compression_stats": compression_stats,
            "selective_stats": selective_stats, "compress_stats_llm": compress_stats_llm,
            "failure_types": failure_types, "failure_saved": failure_saved,
            "parallel_ms": parallel_ms, "tool_code": tool_code,
            "calc_result": calc_result, "tool_used": tool_used,
            "tool_registry_results": tool_registry_results,
            "tool_registry_calls": tool_registry_calls,
            "user_id": user_id, "engine": "async",
        }


# =====================================================================
# [NEW v23] 전역 v23 인스턴스
# =====================================================================

async_rag_engine = AsyncRAGEngine()   # [NEW v23]
tool_registry    = ToolRegistry()     # [NEW v23]


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
            "검색_전략": {"dense_weight": 0.5, "bm25_weight": 0.5, "reranker_사용여부": True,
                          "reranker_모드": "heavy", "top_k": 3, "query_rewrite_필요": True,
                          "query_분해_필요": False, "recall_우선순위": True},
            "메타데이터_전략": {"메타데이터_필터_사용": False, "선호_출처": [], "시간_가중치": "없음"},
            "설명": f"라우팅 실패 → fallback: {str(e)}"
        }
    if tracer:
        s   = result.get("검색_전략", {})
        tok = _usage(resp_obj) if resp_obj else {"prompt": 0, "completion": 0, "total": 0}
        tracer.end("query_routing", tokens=tok,
                   input_summary=question[:60],
                   output_summary=f"의도: {result.get('의도','-')} | top_k: {s.get('top_k','-')}",
                   decision=result.get("설명", ""))
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
# 검색 파이프라인 단계들 (동기)
# =====================================================================

def rewrite_queries(original_query, n=3, tracer=None, use_session_cache=True):
    if use_session_cache:
        cache_key = f"{original_query}||{n}"
        if cache_key in _rewrite_cache:
            cached = _rewrite_cache[cache_key]
            if tracer:
                tracer.start("query_rewriting")
                tracer.end("query_rewriting", input_summary=f"원본: {original_query[:60]}",
                           output_summary=f"캐시 히트 → {len(cached)}개 재사용", decision="캐시 히트")
            return cached

    tracer and tracer.start("query_rewriting")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"문서 검색 전문가. 의도 분해형 우선 총 {n}개 재작성. 번호·기호 없이 줄당 하나."},
            {"role": "user",   "content": f"원본 질문: {original_query}"}
        ]
    )
    variants = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
    queries  = [original_query] + variants[:n]
    if use_session_cache:
        _rewrite_cache[f"{original_query}||{n}"] = queries
    if tracer:
        tracer.end("query_rewriting", tokens=_usage(response),
                   input_summary=f"원본: {original_query[:60]}",
                   output_summary=f"쿼리 {len(queries)}개 생성",
                   decision="의도 분해형 우선")
    return queries


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
    tracer and tracer.end("prefilter", tokens={"prompt": 0, "completion": 0, "total": 0},
                           input_summary=f"{len(items)}개 후보",
                           output_summary=f"상위 {len(result)}개 (컷오프: {cutoff})",
                           decision=f"코사인 ≥ {cutoff}")
    return result


def rerank_chunks(query, items, top_k=3, tracer=None):
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
    all_scores = {}
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
    tracer and tracer.end("rerank", tokens=_usage(response),
                           input_summary=f"{len(items)}개 후보",
                           output_summary=f"상위 {top_k}개 (점수: {', '.join([f'{s[2]:.1f}' for s in result])})",
                           decision="LLM 0~10 채점")
    return result, all_scores


# =====================================================================
# 답변 생성 단계들 (동기)
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
            if 0 <= idx < len(chunks):
                summaries[idx] = m.group(2).strip()
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
            {"role": "system", "content": "청크 요약 구조화. **공통점**/**차이점**/**핵심 정보**/**불확실성/누락 정보** 형식."},
            {"role": "user",   "content": f"질문: {question}\n\n청크 요약:\n{summaries_text}"}
        ]
    )
    result = response.choices[0].message.content
    tracer and tracer.end("step2_analyze", tokens=_usage(response),
                           input_summary=f"{len(summaries)}개 요약",
                           output_summary="분석 완료", decision="불확실성 분석")
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
    result = {"정확도": 0, "관련성": 0, "환각여부": "알 수 없음", "환각근거": "",
              "신뢰도": "보통", "불일치_항목": "없음", "누락_정보": "없음", "개선_제안": ""}
    for line in response.choices[0].message.content.strip().split('\n'):
        if line.startswith("정확도:"):
            try: result["정확도"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
            except ValueError: pass
        elif line.startswith("관련성:"):
            try: result["관련성"] = max(1, min(5, int(re.sub(r'[^0-9]','',line.split(':',1)[1].strip()))))
            except ValueError: pass
        elif line.startswith("환각여부:"):    result["환각여부"]    = line.split(':',1)[1].strip()
        elif line.startswith("환각근거:"):    result["환각근거"]    = line.split(':',1)[1].strip()
        elif line.startswith("신뢰도:"):      result["신뢰도"]      = line.split(':',1)[1].strip()
        elif line.startswith("불일치_항목:"): result["불일치_항목"] = line.split(':',1)[1].strip()
        elif line.startswith("누락_정보:"):   result["누락_정보"]   = line.split(':',1)[1].strip()
        elif line.startswith("개선_제안:"):   result["개선_제안"]   = line.split(':',1)[1].strip()
    tracer and tracer.end("evaluation", tokens=_usage(response),
                           input_summary="질문+문서+답변",
                           output_summary=f"정확도 {result['정확도']}/5 · 환각 {result['환각여부']}",
                           decision="구조화 품질 진단")
    return result


def analyze_hallucination_cause(question, context_chunks, answer, hall_type, tracer=None):
    if hall_type == "없음":
        return None
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
            if line.startswith(f"{key}:"):
                result[key] = line.split(':',1)[1].strip()
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
    overall = round(max(0.0, min(5.0, (acc + rel) / 2 - hall_penalty)), 2)
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
            {"role": "system", "content": "RAG 답변 심사 전문가. **문제점**/**누락**/**개선 방향** 형식. 없으면 '없음'."},
            {"role": "user",   "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft 답변]\n{draft}"}
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
            {"role": "system", "content": "**📌 요약** / **📖 근거** / **✅ 결론** 구조로 Critique 반영 개선 답변 작성. 한국어."},
            {"role": "user",   "content": f"[질문]\n{question}\n\n[참고 문서]\n{context}\n\n[Draft]\n{draft}\n\n[Critique]\n{critique}"}
        ]
    )
    result = response.choices[0].message.content
    tracer and tracer.end("refine", tokens=_usage(response),
                           input_summary=f"Draft {len(draft)}자 + Critique",
                           output_summary=f"Refined {len(result)}자",
                           decision="Critique 반영 최종 답변")
    return result


# =====================================================================
# 보조 함수
# =====================================================================

def _empty_result(tracer, question, eff, prefilter_n):
    return {
        "tracer": tracer, "queries": [question],
        "ranked": [], "final_chunks": [], "final_sources": [], "final_scores": [],
        "gen_chunks": [], "summaries": [], "analysis": "",
        "draft_answer": "", "critique": None, "mode": "",
        "hall_cause": None, "ndcg_k": None, "sqr": None,
        "eff": eff.copy(), "prefilter_n": prefilter_n,
        "cache_hit": None, "compression_stats": None, "selective_stats": None,
        "compress_stats_llm": None, "failure_types": [], "failure_saved": False,
        "parallel_ms": None, "tool_code": None, "calc_result": None, "tool_used": False,
        "tool_registry_results": None, "tool_registry_calls": [],
    }


def _retrieve_mv_sequential(queries, mv_index, chunks, sources, use_bm25, tracer):
    tracer and tracer.start("embedding_search")
    RRF_K, n_chunks = 60, mv_index["n_chunks"]
    rrf = {}
    for query in queries:
        q_emb = normalize(get_embeddings_cached([query]))
        _, cr = mv_index["chunk_index"].search(q_emb, min(20, n_chunks))
        for rank, ci in enumerate(cr[0]):
            if 0 <= ci < n_chunks:
                rrf[ci] = rrf.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
        _, sr = mv_index["sent_index"].search(q_emb, min(40, mv_index["n_sentences"]))
        seen  = {}
        for rank, si in enumerate(sr[0]):
            if 0 <= si < len(mv_index["sent_to_chunk"]):
                ci = mv_index["sent_to_chunk"][si]
                if ci not in seen:
                    seen[ci] = rank
        for ci, rank in seen.items():
            rrf[ci] = rrf.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)
    if use_bm25 and BM25_AVAILABLE:
        bm25 = BM25Okapi([c.split() for c in chunks])
        for query in queries:
            arr = bm25.get_scores(query.split())
            for rank, ci in enumerate(np.argsort(arr)[::-1][:20]):
                rrf[int(ci)] = rrf.get(int(ci), 0.0) + 1.0 / (RRF_K + rank + 1)
    sorted_ci = sorted(rrf, key=lambda i: rrf[i], reverse=True)
    result = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
    tracer and tracer.end("embedding_search", tokens={"prompt": 0, "completion": 0, "total": 0},
                           input_summary=f"쿼리 {len(queries)}개",
                           output_summary=f"Multi-Vector 순차 {len(result)}개",
                           decision="순차 Multi-Vector")
    return result


# =====================================================================
# 단일 파이프라인 실행 (동기) — v23 신규 파라미터 추가
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
                     use_parallel_search: bool = True,
                     use_lim_reorder: bool = True,
                     use_selective_context: bool = False,
                     selective_dedup_thresh: float = DEDUP_THRESHOLD_DEFAULT,
                     use_tool_augment: bool = False,
                     use_llm_compress: bool = False,       # [NEW v23]
                     use_tool_registry: bool = False,      # [NEW v23]
                     enabled_tools: list = None,           # [NEW v23]
                     use_async_engine: bool = False,       # [NEW v23]
                     user_id: str = "anonymous") -> dict:
    """
    [v23] 동기 RAG 파이프라인.
    use_async_engine=True 시 AsyncRAGEngine으로 위임.
    """
    if use_async_engine:
        return AsyncRAGEngine.run_sync(async_rag_engine.run(
            question, eff, index, chunks, sources,
            prefilter_n=prefilter_n, use_multidoc=use_multidoc,
            num_rewrites=num_rewrites, use_session_cache=use_session_cache,
            use_self_refine=use_self_refine, use_compression=use_compression,
            mv_index=mv_index, auto_save_failure=auto_save_failure,
            gen_improvement_hint=gen_improvement_hint,
            use_lim_reorder=use_lim_reorder,
            use_selective_context=use_selective_context,
            selective_dedup_thresh=selective_dedup_thresh,
            use_tool_augment=use_tool_augment,
            use_llm_compress=use_llm_compress,
            use_tool_registry=use_tool_registry,
            enabled_tools=enabled_tools,
            user_id=user_id,
        ))

    tracer = Tracer()

    if use_session_cache:
        ans_key    = hashlib.md5(question.encode("utf-8")).hexdigest()
        cached_ans = answer_cache.get(ans_key)
        if cached_ans:
            tracer.start("answer_cache_hit")
            tracer.end("answer_cache_hit", input_summary=question[:60],
                       output_summary="답변 캐시 히트", decision=f"TTL {ANSWER_CACHE_TTL_SEC//60}분")
            return {**_empty_result(tracer, question, eff, prefilter_n),
                    "answer": cached_ans["answer"], "draft_answer": cached_ans["answer"],
                    "evaluation": cached_ans["evaluation"],
                    "quality_report": cached_ans["quality_report"],
                    "mode": "answer_cache_hit", "cache_hit": "answer",
                    "user_id": user_id, "engine": "sync"}

    queries = (rewrite_queries(question, n=num_rewrites, tracer=tracer, use_session_cache=use_session_cache)
               if eff["use_query_rewrite"] else [question])

    cache_hit   = None
    parallel_ms = None
    qr_cached   = query_result_cache.get(question, eff["use_bm25"], prefilter_n) if use_session_cache else None

    if qr_cached is not None:
        filtered  = qr_cached
        cache_hit = "query"
        tracer.start("embedding_search")
        tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                   input_summary="쿼리 결과 캐시 히트",
                   output_summary=f"{len(filtered)}개 (캐시 재사용)",
                   decision=f"TTL {QUERY_CACHE_TTL_SEC//60}분 이내")
        tracer.start("prefilter")
        tracer.end("prefilter", input_summary="캐시", output_summary="캐시", decision="캐시 히트 스킵")
    else:
        if use_parallel_search:
            candidates, parallel_ms = retrieve_parallel(
                queries, index, chunks, sources,
                mv_index=mv_index, use_bm25=eff["use_bm25"],
                top_k_per_query=20, tracer=tracer
            )
        elif mv_index:
            candidates = _retrieve_mv_sequential(queries, mv_index, chunks, sources, eff["use_bm25"], tracer)
        else:
            candidates = _retrieve_hybrid_simple(queries, index, chunks, sources, eff["use_bm25"], tracer)
        filtered = prefilter_by_similarity(question, candidates, prefilter_n, tracer)
        if use_session_cache:
            query_result_cache.set(question, eff["use_bm25"], prefilter_n, filtered)

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

    if use_lim_reorder and len(final_chunks) > 2:
        final_chunks = reorder_lost_in_middle(final_chunks, final_scores)

    gen_chunks         = final_chunks
    compression_stats  = None
    selective_stats    = None
    compress_stats_llm = None

    if use_selective_context and final_chunks:
        gen_chunks, selective_stats = selective_context_phase2(
            question, final_chunks, dedup_threshold=selective_dedup_thresh, tracer=tracer
        )
    elif use_compression and final_chunks:
        gen_chunks, compression_stats = compress_chunks(question, final_chunks, tracer=tracer)

    # [NEW v23] LLM Phase 3 압축
    if use_llm_compress and gen_chunks:
        gen_chunks, compress_stats_llm = compress_chunks_llm(question, gen_chunks, tracer=tracer)

    tool_code             = None
    calc_result           = None
    tool_used             = False
    tool_registry_results = None
    tool_registry_calls   = []

    # [NEW v23] Tool Registry (function calling)
    if use_tool_registry and gen_chunks:
        context_text = "\n\n".join([f"[출처 {i+1}] {c}" for i, c in enumerate(gen_chunks)])
        tr_answer, tr_calls = tool_registry.run_with_llm(
            question, context_text, enabled_tools, tracer
        )
        if tr_answer:
            tool_registry_results = tr_answer
            tool_registry_calls   = tr_calls
            answer    = tr_answer
            tool_used = True
            summaries, analysis = [], ""
            mode = "tool_registry"

    # v21 Tool-Augmented (수치 계산 특화, registry 미사용 시 fallback)
    if not tool_used and use_tool_augment and gen_chunks:
        ta_answer, tool_code, calc_result = tool_augmented_answer(question, gen_chunks, tracer)
        if ta_answer:
            answer    = ta_answer
            tool_used = True
            summaries, analysis = [], ""
            mode = "tool_augmented"

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

    draft_answer = answer
    critique     = None
    if use_self_refine and gen_chunks:
        critique = critique_answer(question, gen_chunks, draft_answer, tracer)
        answer   = refine_answer(question, gen_chunks, draft_answer, critique, tracer)

    evaluation  = evaluate_answer(question, gen_chunks, answer, tracer)
    hall        = evaluation.get("환각여부", "없음")
    hall_cause  = None
    if hall != "없음":
        hall_cause = analyze_hallucination_cause(question, gen_chunks, answer, hall, tracer)
    quality_report = build_quality_report(evaluation, hall_cause)

    if use_session_cache:
        ans_key = hashlib.md5(question.encode("utf-8")).hexdigest()
        answer_cache.set(ans_key, {"answer": answer, "evaluation": evaluation,
                                   "quality_report": quality_report})

    failure_types, failure_saved = [], False
    if auto_save_failure:
        failure_types = classify_failure_types(evaluation, quality_report, sqr)
        if failure_types:
            hint = (generate_improvement_hint(question, gen_chunks, answer, evaluation, failure_types, tracer)
                    if gen_improvement_hint and gen_chunks else None)
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
        "cache_hit": cache_hit, "compression_stats": compression_stats,
        "selective_stats": selective_stats, "compress_stats_llm": compress_stats_llm,
        "failure_types": failure_types, "failure_saved": failure_saved,
        "parallel_ms": parallel_ms, "tool_code": tool_code,
        "calc_result": calc_result, "tool_used": tool_used,
        "tool_registry_results": tool_registry_results,
        "tool_registry_calls": tool_registry_calls,
        "user_id": user_id, "engine": "sync",
    }


def _retrieve_hybrid_simple(queries, index, chunks, sources, use_bm25, tracer):
    """병렬 검색 비활성화 시 폴백 — 기존 순차 hybrid 검색."""
    tracer and tracer.start("embedding_search")
    seen_dense, dense_items = set(), []
    for query in queries:
        q_emb = normalize(get_embeddings_cached([query]))
        _, indices = index.search(q_emb, 20)
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
        for i in np.argsort(scores_arr)[::-1][:20]:
            if i < len(chunks) and i not in seen_bm25:
                seen_bm25.add(i)
                bm25_items.append((chunks[i], sources[i] if sources else "알 수 없음"))
    RRF_K = 60
    rrf_scores, item_by_key = {}, {}
    def _key(item): return hashlib.md5((item[0][:120]+item[1]).encode()).hexdigest()
    for rank, item in enumerate(dense_items):
        k = _key(item); rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0/(RRF_K+rank+1); item_by_key[k] = item
    for rank, item in enumerate(bm25_items):
        k = _key(item); rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0/(RRF_K+rank+1); item_by_key[k] = item
    result = [item_by_key[k] for k in sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)]
    tracer and tracer.end("embedding_search", tokens={"prompt":0,"completion":0,"total":0},
                           input_summary=f"쿼리 {len(queries)}개",
                           output_summary=f"RRF {len(result)}개", decision="RRF(k=60)")
    return result


# =====================================================================
# Ablation
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
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_log(entry):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def build_log_entry(tracer, question, queries, ranked_items, answer, evaluation,
                    hall_cause, quality_report, search_quality_report=None, ndcg=None,
                    mode="", route_decision=None,
                    fallback_triggered=False, fallback_attempts=0, fallback_history=None,
                    self_refinement=None, dynamic_retrieval_profile=None,
                    cache_hit=None, compression_stats=None, mv_retrieval=False,
                    failure_types=None, failure_saved=False,
                    parallel_ms=None, tool_used=False, selective_stats=None,
                    compress_stats_llm=None, tool_registry_calls=None,   # [NEW v23]
                    engine="sync",                                         # [NEW v23]
                    user_id: str = "anonymous"):
    sqr = search_quality_report or {}
    return {
        "trace_id":      tracer.trace_id,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id":       user_id,
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
        "parallel_ms":        parallel_ms,
        "tool_used":          tool_used,
        "selective_stats":    selective_stats,
        "compress_stats_llm": compress_stats_llm,   # [NEW v23]
        "tool_registry_calls": tool_registry_calls or [],  # [NEW v23]
        "engine":             engine,                # [NEW v23]
        "spans":           tracer.spans, "total_tokens": tracer.total_tokens(),
        "total_latency_ms": tracer.total_latency_ms(), "bottleneck": tracer.bottleneck(),
        "mode": mode, "route_decision": route_decision,
        "embed_cache_size": embed_cache.size(),
        "embed_cache_hits": embed_cache.hits, "embed_cache_misses": embed_cache.misses,
    }


# =====================================================================
# process_rag_query() — 서버/외부에서 호출하는 단일 진입점
# =====================================================================

def process_rag_query(
    question: str,
    index,
    chunks: list,
    sources: list,
    user_id: str = "anonymous",
    auto_routing: bool = True,
    use_bm25: bool = True,
    use_reranking: bool = True,
    top_k: int = 3,
    use_query_rewrite: bool = True,
    num_rewrites: int = 3,
    prefilter_n: int = 10,
    use_multidoc: bool = True,
    mv_index: dict = None,
    use_parallel_search: bool = True,
    use_lim_reorder: bool = True,
    use_selective_context: bool = False,
    selective_dedup_thresh: float = DEDUP_THRESHOLD_DEFAULT,
    enable_fallback: bool = True,
    use_self_refine: bool = True,
    use_compression: bool = False,
    use_llm_compress: bool = False,
    use_tool_augment: bool = False,
    use_tool_registry: bool = False,
    enabled_tools: list = None,
    use_async_engine: bool = False,
    auto_save_failure: bool = True,
    gen_improvement_hint: bool = False,
    use_session_cache: bool = True,
    enable_dynamic_retrieval: bool = True,
) -> dict:
    """
    외부(서버/테스트)에서 호출하는 단일 진입점.
    Fallback 루프 포함. 로그 저장 포함.
    반환: {answer, evaluation, quality_report, tracer, mode, ...}
    """
    defaults = {"use_bm25": use_bm25, "use_reranking": use_reranking,
                "top_k": top_k, "use_query_rewrite": use_query_rewrite}
    eff           = defaults.copy()
    cur_multidoc  = use_multidoc
    cur_prefilter = prefilter_n
    route_decision        = None
    dynamic_profile_label = None

    if auto_routing:
        route_decision = route_query(question)
        eff = _apply_routing(route_decision, defaults)
        if enable_dynamic_retrieval:
            eff, cur_prefilter, cur_multidoc, dynamic_profile_label = apply_dynamic_retrieval(
                route_decision.get("의도","ambiguous"), eff, prefilter_n, use_multidoc
            )

    base_eff, base_prefilter   = eff.copy(), cur_prefilter
    final_result               = None
    fallback_history           = []
    fallback_triggered_flag    = False
    MAX_TRIES                  = MAX_RETRIES + 1

    for attempt in range(MAX_TRIES):
        attempt_eff, attempt_pf = (escalate_params(base_eff, base_prefilter, attempt)
                                   if attempt > 0 else (eff.copy(), cur_prefilter))
        result = run_rag_pipeline(
            question, attempt_eff, index, chunks, sources,
            prefilter_n=attempt_pf, use_multidoc=cur_multidoc,
            num_rewrites=num_rewrites,
            use_session_cache=use_session_cache,
            use_self_refine=use_self_refine,
            use_compression=use_compression,
            mv_index=mv_index,
            auto_save_failure=(auto_save_failure and attempt == 0),
            gen_improvement_hint=gen_improvement_hint,
            use_parallel_search=use_parallel_search,
            use_lim_reorder=use_lim_reorder,
            use_selective_context=use_selective_context,
            selective_dedup_thresh=selective_dedup_thresh,
            use_tool_augment=use_tool_augment,
            use_llm_compress=use_llm_compress,
            use_tool_registry=use_tool_registry,
            enabled_tools=enabled_tools,
            use_async_engine=use_async_engine,
            user_id=user_id,
        )

        evaluation = result["evaluation"]
        qr         = result["quality_report"]
        fallback_history.append({
            "attempt": attempt, "accuracy": evaluation.get("정확도",0),
            "hallucination": evaluation.get("환각여부","없음"),
            "overall_score": qr["overall_score"], "grade": qr["grade"],
            "tokens": result["tracer"].total_tokens()["total"],
        })

        fb_needed, _ = should_fallback(evaluation)
        if fb_needed and attempt < MAX_TRIES - 1 and enable_fallback:
            fallback_triggered_flag = True
        else:
            final_result = result
            break

    if final_result is None:
        final_result = result

    user_manager.record_usage(user_id)

    log_entry = build_log_entry(
        final_result["tracer"], question, final_result["queries"],
        final_result["ranked"], final_result["answer"],
        final_result["evaluation"], final_result["hall_cause"],
        final_result["quality_report"],
        search_quality_report=final_result["sqr"],
        ndcg=final_result["ndcg_k"],
        mode=final_result["mode"],
        route_decision=route_decision,
        fallback_triggered=fallback_triggered_flag,
        fallback_attempts=len(fallback_history) - 1,
        fallback_history=fallback_history,
        self_refinement=final_result.get("critique"),
        dynamic_retrieval_profile=dynamic_profile_label,
        cache_hit=final_result["cache_hit"],
        compression_stats=final_result["compression_stats"],
        mv_retrieval=(mv_index is not None),
        failure_types=final_result["failure_types"],
        failure_saved=final_result["failure_saved"],
        parallel_ms=final_result.get("parallel_ms"),
        tool_used=final_result.get("tool_used", False),
        selective_stats=final_result.get("selective_stats"),
        compress_stats_llm=final_result.get("compress_stats_llm"),
        tool_registry_calls=final_result.get("tool_registry_calls", []),
        engine=final_result.get("engine", "sync"),
        user_id=user_id,
    )
    save_log(log_entry)

    return {
        **final_result,
        "fallback_triggered": fallback_triggered_flag,
        "fallback_history":   fallback_history,
        "route_decision":     route_decision,
        "dynamic_profile":    dynamic_profile_label,
        "log_entry":          log_entry,
    }


def linkify_citations(text):
    return re.sub(
        r'\[출처 (\d+)\]',
        lambda m: f'<a href="#source-{m.group(1)}" style="color:#1976D2;font-weight:bold;text-decoration:none;">[출처 {m.group(1)}]</a>',
        text
    )
