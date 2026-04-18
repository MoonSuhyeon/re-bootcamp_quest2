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
import threading          # [NEW v22]
import statistics         # [NEW v22]
import base64             # [NEW v22]
from datetime import datetime, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE             = os.path.join(_BASE, "rag_eval_logs_v22.json")
EMBED_CACHE_FILE     = os.path.join(_BASE, "embed_cache_v22.pkl")
ANSWER_CACHE_FILE    = os.path.join(_BASE, "answer_cache_v22.json")
FAILURE_DATASET_FILE = os.path.join(_BASE, "failure_dataset_v22.json")
USERS_FILE           = os.path.join(_BASE, "rag_users_v22.json")      # [NEW v22]
USAGE_LOG_FILE       = os.path.join(_BASE, "rag_usage_v22.json")       # [NEW v22]

ANSWER_CACHE_TTL_SEC       = 1800
QUERY_CACHE_TTL_SEC        = 3600
FAILURE_THRESHOLD_ACCURACY = 3
PARALLEL_MAX_WORKERS       = 4
DEDUP_THRESHOLD_DEFAULT    = 0.85

# [NEW v22] 인증 / 모니터링 상수
RATE_LIMIT_PER_HOUR  = 20          # 사용자당 시간당 최대 쿼리 수
ALERT_ACCURACY_MIN   = 3.0         # 평균 정확도 < 3 → 경보
ALERT_HALL_MAX       = 0.30        # 환각 비율 > 30% → 경보
ALERT_LATENCY_P95_MS = 15_000      # P95 레이턴시 > 15s → 경보

CALC_PATTERNS = [
    r'\d+[,.]\d+', r'얼마', r'몇\s', r'합계', r'총\s', r'평균',
    r'비율', r'퍼센트', r'%', r'증가', r'감소', r'차이', r'계산',
    r'합산', r'곱하', r'나누', r'더하', r'빼'
]

st.set_page_config(page_title="RAG 챗봇 v22", page_icon="📚", layout="wide")


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


# =====================================================================
# [NEW v22] MetricsCollector — 실시간 성능 모니터링
# =====================================================================

class MetricsCollector:
    """
    [v22] 로그 기반 실시간 성능 지표 집계.
    P50 / P95 / P99 레이턴시, 정확도 트렌드, 환각 비율, 알림 생성.
    """

    @staticmethod
    def percentile(data: list, p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (len(sorted_data) - 1) * p / 100
        low, high = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        return sorted_data[low] + (sorted_data[high] - sorted_data[low]) * (idx - low)

    def compute_from_logs(self, logs: list) -> dict:
        if not logs:
            return {}

        latencies    = [l.get("total_latency_ms", 0) for l in logs]
        accuracies   = [l["evaluation"].get("정확도", 0) for l in logs if l.get("evaluation")]
        relevances   = [l["evaluation"].get("관련성", 0) for l in logs if l.get("evaluation")]
        hall_list    = [l["evaluation"].get("환각여부", "없음") for l in logs if l.get("evaluation")]
        tokens       = [l.get("total_tokens", {}).get("total", 0) for l in logs]
        fail_count   = sum(1 for l in logs if l.get("failure_saved"))
        cache_hits   = sum(1 for l in logs if l.get("cache_hit"))
        fallbacks    = sum(1 for l in logs if l.get("fallback_triggered"))
        tool_used    = sum(1 for l in logs if l.get("tool_used"))
        parallel_ms  = [l["parallel_ms"] for l in logs if l.get("parallel_ms")]

        hall_rate    = round(sum(1 for h in hall_list if h != "없음") / max(len(hall_list), 1), 4)

        # 24시간 / 7일 분리
        now      = datetime.now()
        day_ago  = now - timedelta(hours=24)
        week_ago = now - timedelta(days=7)
        logs_24h = [l for l in logs if self._parse_ts(l.get("timestamp","")) >= day_ago]
        logs_7d  = [l for l in logs if self._parse_ts(l.get("timestamp","")) >= week_ago]

        return {
            "total_queries":    len(logs),
            "queries_24h":      len(logs_24h),
            "queries_7d":       len(logs_7d),
            "latency_p50_ms":   round(self.percentile(latencies, 50)),
            "latency_p95_ms":   round(self.percentile(latencies, 95)),
            "latency_p99_ms":   round(self.percentile(latencies, 99)),
            "latency_avg_ms":   round(statistics.mean(latencies)) if latencies else 0,
            "latency_trend":    [l.get("total_latency_ms", 0) for l in logs[-30:]],
            "accuracy_avg":     round(statistics.mean(accuracies), 2) if accuracies else 0,
            "accuracy_trend":   accuracies[-30:],
            "relevance_avg":    round(statistics.mean(relevances), 2) if relevances else 0,
            "hallucination_rate": hall_rate,
            "hall_trend":       [1 if h != "없음" else 0 for h in hall_list[-30:]],
            "token_avg":        round(statistics.mean(tokens)) if tokens else 0,
            "token_total":      sum(tokens),
            "failure_count":    fail_count,
            "cache_hit_rate":   round(cache_hits / max(len(logs), 1), 4),
            "fallback_rate":    round(fallbacks / max(len(logs), 1), 4),
            "tool_used_count":  tool_used,
            "parallel_p95_ms":  round(self.percentile(parallel_ms, 95)) if parallel_ms else 0,
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
            alerts.append({
                "level": "error", "icon": "🔴",
                "message": f"평균 정확도 {stats['accuracy_avg']:.1f}/5 — 임계값 {ALERT_ACCURACY_MIN} 미달",
                "metric": "accuracy"
            })
        if stats.get("hallucination_rate", 0) > ALERT_HALL_MAX and stats.get("total_queries", 0) >= 5:
            alerts.append({
                "level": "warning", "icon": "🟡",
                "message": f"환각 비율 {stats['hallucination_rate']:.0%} — 임계값 {ALERT_HALL_MAX:.0%} 초과",
                "metric": "hallucination"
            })
        if stats.get("latency_p95_ms", 0) > ALERT_LATENCY_P95_MS:
            alerts.append({
                "level": "warning", "icon": "🟡",
                "message": f"P95 레이턴시 {stats['latency_p95_ms']:,}ms — 임계값 {ALERT_LATENCY_P95_MS:,}ms 초과",
                "metric": "latency"
            })
        if not alerts:
            alerts.append({
                "level": "ok", "icon": "🟢",
                "message": "모든 지표 정상",
                "metric": "all"
            })
        return alerts

    def export_prometheus(self, stats: dict) -> str:
        ts = int(time.time() * 1000)
        lines = [
            f"# HELP rag_queries_total 총 쿼리 수",
            f"# TYPE rag_queries_total counter",
            f'rag_queries_total {stats.get("total_queries", 0)} {ts}',
            f"# HELP rag_latency_p50_ms P50 레이턴시(ms)",
            f"# TYPE rag_latency_p50_ms gauge",
            f'rag_latency_p50_ms {stats.get("latency_p50_ms", 0)} {ts}',
            f"# HELP rag_latency_p95_ms P95 레이턴시(ms)",
            f"# TYPE rag_latency_p95_ms gauge",
            f'rag_latency_p95_ms {stats.get("latency_p95_ms", 0)} {ts}',
            f"# HELP rag_latency_p99_ms P99 레이턴시(ms)",
            f"# TYPE rag_latency_p99_ms gauge",
            f'rag_latency_p99_ms {stats.get("latency_p99_ms", 0)} {ts}',
            f"# HELP rag_accuracy_avg 평균 정확도(1~5)",
            f"# TYPE rag_accuracy_avg gauge",
            f'rag_accuracy_avg {stats.get("accuracy_avg", 0)} {ts}',
            f"# HELP rag_hallucination_rate 환각 발생 비율(0~1)",
            f"# TYPE rag_hallucination_rate gauge",
            f'rag_hallucination_rate {stats.get("hallucination_rate", 0)} {ts}',
            f"# HELP rag_cache_hit_rate 캐시 히트율(0~1)",
            f"# TYPE rag_cache_hit_rate gauge",
            f'rag_cache_hit_rate {stats.get("cache_hit_rate", 0)} {ts}',
            f"# HELP rag_failure_count 실패 케이스 저장 수",
            f"# TYPE rag_failure_count counter",
            f'rag_failure_count {stats.get("failure_count", 0)} {ts}',
            f"# HELP rag_token_avg 쿼리당 평균 토큰",
            f"# TYPE rag_token_avg gauge",
            f'rag_token_avg {stats.get("token_avg", 0)} {ts}',
        ]
        return "\n".join(lines)

    def export_json(self, stats: dict) -> bytes:
        export = dict(stats)
        export["exported_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        export.pop("latency_trend", None)
        export.pop("accuracy_trend", None)
        export.pop("hall_trend", None)
        return json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")


# =====================================================================
# [NEW v22] UserManager — 인증 / 사용자 관리 / Rate Limiting
# =====================================================================

class UserManager:
    """
    [v22] 사용자 인증 + 사용자별 로그 + Rate Limiting.
    - hashlib SHA-256 기반 비밀번호 해시 (운영 환경에서는 bcrypt 권장)
    - 기본 계정: admin / admin123 (role: admin), demo / demo123 (role: user)
    """

    def __init__(self, users_path: str, usage_path: str):
        self.users_path = users_path
        self.usage_path = usage_path
        self._ensure_defaults()

    # ── 내부 헬퍼 ──────────────────────────────────────────────

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
        if "admin" not in users:
            users["admin"] = {
                "password_hash": self._hash("admin123"),
                "role": "admin",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "display_name": "관리자"
            }
            changed = True
        if "demo" not in users:
            users["demo"] = {
                "password_hash": self._hash("demo123"),
                "role": "user",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "display_name": "데모 사용자"
            }
            changed = True
        if changed:
            self._save_users(users)

    # ── 공개 API ──────────────────────────────────────────────

    def verify_login(self, username: str, password: str) -> bool:
        users = self._load_users()
        u = users.get(username)
        if not u:
            return False
        return u.get("password_hash") == self._hash(password)

    def get_role(self, username: str) -> str:
        users = self._load_users()
        return users.get(username, {}).get("role", "user")

    def get_display_name(self, username: str) -> str:
        users = self._load_users()
        return users.get(username, {}).get("display_name", username)

    def list_users(self) -> list:
        users = self._load_users()
        return [
            {"username": k, "role": v.get("role","user"),
             "display_name": v.get("display_name", k),
             "created_at": v.get("created_at","-")}
            for k, v in users.items()
        ]

    def create_user(self, username: str, password: str, role: str = "user",
                    display_name: str = "") -> tuple:
        if not username.strip() or not password.strip():
            return False, "사용자명과 비밀번호를 입력하세요."
        if len(password) < 6:
            return False, "비밀번호는 최소 6자 이상이어야 합니다."
        users = self._load_users()
        if username in users:
            return False, f"이미 존재하는 사용자명: {username}"
        users[username] = {
            "password_hash": self._hash(password),
            "role": role,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "display_name": display_name or username
        }
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
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if username not in usage:
            usage[username] = []
        usage[username].append(ts)
        usage[username] = usage[username][-500:]  # 최근 500개만 보관
        self._save_usage(usage)

    def check_rate_limit(self, username: str) -> tuple:
        """(ok: bool, remaining: int) 반환"""
        users = self._load_users()
        role  = users.get(username, {}).get("role", "user")
        if role == "admin":
            return True, RATE_LIMIT_PER_HOUR  # admin 무제한

        usage  = self._load_usage()
        now    = datetime.now()
        cutoff = now - timedelta(hours=1)
        recent = [
            t for t in usage.get(username, [])
            if MetricsCollector._parse_ts(t) >= cutoff
        ]
        remaining = max(0, RATE_LIMIT_PER_HOUR - len(recent))
        return remaining > 0, remaining

    def get_user_stats(self, username: str) -> dict:
        usage  = self._load_usage()
        all_ts = usage.get(username, [])
        now    = datetime.now()
        h1     = [t for t in all_ts if MetricsCollector._parse_ts(t) >= now - timedelta(hours=1)]
        d1     = [t for t in all_ts if MetricsCollector._parse_ts(t) >= now - timedelta(days=1)]
        return {
            "total_queries":   len(all_ts),
            "queries_1h":      len(h1),
            "queries_24h":     len(d1),
            "rate_limit":      RATE_LIMIT_PER_HOUR,
            "remaining_1h":    max(0, RATE_LIMIT_PER_HOUR - len(h1)),
        }


# =====================================================================
# 실패 분류 / 힌트 (v20~)
# =====================================================================

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
metrics_collector  = MetricsCollector()                   # [NEW v22]
user_manager       = UserManager(USERS_FILE, USAGE_LOG_FILE)  # [NEW v22]


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
# 병렬 검색 (v21~)
# =====================================================================

def retrieve_parallel(queries: list, index, chunks: list, sources: list,
                      mv_index: dict = None, use_bm25: bool = True,
                      top_k_per_query: int = 20, tracer=None) -> tuple:
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

    sorted_ci   = sorted(rrf_total, key=lambda i: rrf_total[i], reverse=True)
    result      = [(chunks[i], sources[i] if sources else "알 수 없음") for i in sorted_ci]
    parallel_ms = int((time.time() - t_start) * 1000)

    if tracer:
        tracer.end("parallel_search",
                   tokens={"prompt": 0, "completion": 0, "total": 0},
                   input_summary=f"쿼리 {len(queries)}개 × 4 채널 병렬",
                   output_summary=f"{len(result)}개 후보 ({parallel_ms}ms)",
                   decision=f"ThreadPoolExecutor(workers={PARALLEL_MAX_WORKERS}) RRF 통합")
    return result, parallel_ms


# =====================================================================
# LongContextReorder (v21~)
# =====================================================================

def reorder_lost_in_middle(chunks: list, scores: list) -> list:
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
# Selective Context Phase 2 (v21~)
# =====================================================================

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
        is_dup = any(float(np.dot(emb, ke)) >= dedup_threshold for ke in kept_embs)
        if not is_dup:
            kept_indices.append(i)
            kept_embs.append(emb)

    kept_sents = [(all_sents[i][0], all_sents[i][1],
                   float(np.dot(sent_embs[i], q_emb)))
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
# Tool-Augmented RAG (v21~)
# =====================================================================

def detect_calc_intent(question: str, chunks: list) -> bool:
    q_calc   = any(re.search(p, question) for p in CALC_PATTERNS)
    ctx_nums = any(re.search(r'\d{2,}', c) for c in chunks)
    return q_calc and ctx_nums


def _safe_eval(code: str, data: dict):
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
# 단일 파이프라인 실행 (v22)
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
                     user_id: str = "anonymous") -> dict:   # [NEW v22]
    """
    [v22] RAG 파이프라인.
    신규: user_id 파라미터 → 로그에 사용자 정보 포함, Rate Limiting 통합.
    """
    tracer = Tracer()

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
                "user_id": user_id,
            }

    queries = rewrite_queries(question, n=num_rewrites, tracer=tracer,
                               use_session_cache=use_session_cache) if eff["use_query_rewrite"] else [question]

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
                   decision=f"TTL {QUERY_CACHE_TTL_SEC//60}분 이내 캐시")
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
            candidates = retrieve_hybrid(queries, index, chunks, sources,
                                         use_bm25=eff["use_bm25"], tracer=tracer)
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

    compression_stats = None
    selective_stats   = None
    gen_chunks        = final_chunks

    if use_selective_context and final_chunks:
        gen_chunks, selective_stats = selective_context_phase2(
            question, final_chunks,
            dedup_threshold=selective_dedup_thresh, tracer=tracer
        )
    elif use_compression and final_chunks:
        gen_chunks, compression_stats = compress_chunks(question, final_chunks, tracer=tracer)

    tool_code   = None
    calc_result = None
    tool_used   = False

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
        answer_cache.set(ans_key, {"answer": answer, "evaluation": evaluation, "quality_report": quality_report})

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
        "user_id": user_id,  # [NEW v22]
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
                    user_id: str = "anonymous"):   # [NEW v22]
    sqr = search_quality_report or {}
    return {
        "trace_id":      tracer.trace_id,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id":       user_id,  # [NEW v22]
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
    ("logged_in", False), ("current_user", "anonymous"),    # [NEW v22]
    ("user_role", "user"),                                   # [NEW v22]
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# [NEW v22] 로그인 게이트 — 인증되지 않으면 앱 진입 차단
# =====================================================================

if not st.session_state.logged_in:
    st.title("🔐 RAG 챗봇 v22 — 로그인")
    st.markdown("---")

    col_login, col_info = st.columns([1, 1])
    with col_login:
        st.subheader("로그인")
        with st.form("login_form"):
            username = st.text_input("사용자명", placeholder="username")
            password = st.text_input("비밀번호", type="password", placeholder="password")
            submitted = st.form_submit_button("로그인", use_container_width=True, type="primary")

        if submitted:
            if user_manager.verify_login(username, password):
                st.session_state.logged_in    = True
                st.session_state.current_user = username
                st.session_state.user_role    = user_manager.get_role(username)
                st.rerun()
            else:
                st.error("사용자명 또는 비밀번호가 올바르지 않습니다.")

    with col_info:
        st.subheader("기본 계정 안내")
        st.info(
            "**관리자 계정**\n"
            "- 사용자명: `admin`\n"
            "- 비밀번호: `admin123`\n\n"
            "**데모 계정**\n"
            "- 사용자명: `demo`\n"
            "- 비밀번호: `demo123`"
        )
        st.markdown("---")
        st.caption(
            "v22 주요 기능:\n"
            "- 사용자 인증 + Rate Limiting\n"
            "- 실시간 모니터링 (P50/P95/P99)\n"
            "- 경보 시스템 (정확도/환각/레이턴시)\n"
            "- Prometheus 메트릭 내보내기\n"
            "- FastAPI 서버 코드 미리보기"
        )

    st.stop()


# =====================================================================
# 탭 레이아웃 (v22: 탭 3개 추가)
# =====================================================================
tab_chat, tab_trace, tab_agent, tab_ablation, tab_search, tab_failure, tab_v21, \
tab_monitor, tab_users, tab_api = st.tabs([
    "💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation",
    "🔍 검색 품질", "🚨 실패 데이터셋", "⚡ v21 분석",
    "📊 모니터링",            # [NEW v22]
    "👤 사용자 관리",          # [NEW v22]
    "🔌 API 미리보기",         # [NEW v22]
])


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("📚 RAG 챗봇 v22")

    # [NEW v22] 사용자 정보 표시
    st.markdown("---")
    role_badge  = "👑 관리자" if st.session_state.user_role == "admin" else "👤 사용자"
    disp_name   = user_manager.get_display_name(st.session_state.current_user)
    st.caption(f"{role_badge} **{disp_name}** ({st.session_state.current_user})")

    u_stats = user_manager.get_user_stats(st.session_state.current_user)
    ok, remaining = user_manager.check_rate_limit(st.session_state.current_user)
    if st.session_state.user_role != "admin":
        st.caption(f"  시간당 잔여 쿼리: {remaining}/{RATE_LIMIT_PER_HOUR}")
        if not ok:
            st.warning("시간당 쿼리 한도에 도달했습니다.")
    st.caption(f"  누적 쿼리: {u_stats['total_queries']}건")

    if st.button("로그아웃", use_container_width=True):
        st.session_state.logged_in    = False
        st.session_state.current_user = "anonymous"
        st.session_state.user_role    = "user"
        st.rerun()

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
    st.caption("⚡ [v21] 병렬 검색")
    enable_parallel = st.toggle("병렬 검색 (ThreadPoolExecutor)", value=True,
                                 help=f"Dense / BM25 / 문장 / 키워드 4채널 동시 검색 (workers={PARALLEL_MAX_WORKERS})")
    enable_lim      = st.toggle("LongContextReorder", value=True,
                                 help="관련성 높은 청크를 프롬프트 시작·끝에 배치 → Lost in the Middle 방지")

    st.markdown("---")
    st.caption("🔬 [v21] Selective Context Phase 2")
    enable_selective  = st.toggle("Cross-chunk 중복 제거", value=False)
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
    st.caption("v22: 모니터링 · 인증 · API 미리보기")


# =====================================================================
# TAB 1 — 챗봇
# =====================================================================
with tab_chat:
    st.title("💬 문서 기반 챗봇 v22")
    if st.session_state.index is None:
        st.info("사이드바에서 문서를 업로드하고 처리해주세요.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        # [NEW v22] Rate Limit 체크
        ok, remaining = user_manager.check_rate_limit(st.session_state.current_user)
        if not ok and st.session_state.user_role != "admin":
            st.warning(f"시간당 쿼리 한도({RATE_LIMIT_PER_HOUR}회)에 도달했습니다. 잠시 후 다시 시도해 주세요.")
            st.stop()

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
                    with st.spinner("🧭 쿼리 라우팅 중..."):
                        route_decision = route_query(prompt)
                    eff = _apply_routing(route_decision, defaults)
                    if enable_dynamic_retrieval:
                        eff, cur_prefilter, cur_multidoc, dynamic_profile_label = apply_dynamic_retrieval(
                            route_decision.get("의도","ambiguous"), eff, prefilter_n, use_multidoc
                        )

                base_eff, base_prefilter = eff.copy(), cur_prefilter
                final_result = None
                attempt_num  = 0
                fallback_history, fallback_triggered_flag = [], False

                for attempt in range(MAX_RETRIES + 1):
                    attempt_num = attempt + 1
                    attempt_eff, attempt_pf = (escalate_params(base_eff, base_prefilter, attempt)
                                               if attempt > 0 else (eff.copy(), cur_prefilter))

                    status_ph = st.empty()
                    status_ph.info(f"⏳ 시도 {attempt_num}/{MAX_RETRIES+1} 실행 중...")

                    with st.spinner(f"RAG 파이프라인 실행 중 (시도 {attempt_num})..."):
                        result = run_rag_pipeline(
                            prompt, attempt_eff,
                            st.session_state.index, st.session_state.chunks,
                            st.session_state.chunk_sources,
                            prefilter_n=attempt_pf, use_multidoc=cur_multidoc,
                            num_rewrites=num_rewrites,
                            use_session_cache=(enable_answer_cache or enable_query_cache),
                            use_self_refine=enable_self_refine,
                            use_compression=enable_compression,
                            mv_index=st.session_state.mv_index,
                            auto_save_failure=(enable_failure_save and attempt == 0),
                            gen_improvement_hint=enable_hint_gen,
                            use_parallel_search=enable_parallel,
                            use_lim_reorder=enable_lim,
                            use_selective_context=enable_selective,
                            selective_dedup_thresh=selective_thresh,
                            use_tool_augment=enable_tool_augment,
                            user_id=st.session_state.current_user,  # [NEW v22]
                        )

                    evaluation   = result["evaluation"]
                    quality_report = result["quality_report"]
                    grade        = quality_report["grade"]
                    gc           = {"A":"🟢","B":"🔵","C":"🟡","D":"🟠","F":"🔴"}.get(grade,"⚪")
                    hall_txt     = evaluation.get("환각여부","없음")
                    fail_tag     = f" 🚨" if result["failure_saved"] else ""
                    tool_tag     = " 🔧Tool" if result.get("tool_used") else ""

                    fallback_history.append({
                        "attempt": attempt, "trigger": None if attempt == 0 else "fallback",
                        "accuracy": evaluation.get("정확도",0), "hallucination": hall_txt,
                        "overall_score": quality_report["overall_score"], "grade": grade,
                        "tokens": result["tracer"].total_tokens()["total"],
                    })

                    fb_needed, fb_reason = should_fallback(evaluation)
                    if fb_needed and attempt < MAX_RETRIES and enable_fallback:
                        fallback_triggered_flag = True
                        status_ph.warning(f"⚠️ 시도 {attempt_num} 완료{fail_tag}{tool_tag} | 정확도 {evaluation.get('정확도',0)}/5 · 환각 {hall_txt} · 등급 {grade} → Fallback 예정")
                    else:
                        final_result = result
                        status_ph.success(f"✅ 시도 {attempt_num} 완료{fail_tag}{tool_tag} | {gc} {grade} | 정확도 {evaluation.get('정확도',0)}/5 · 환각 {hall_txt}")
                        break

                if final_result is None:
                    final_result = result

                # [NEW v22] 사용량 기록
                user_manager.record_usage(st.session_state.current_user)

                if auto_evaluate:
                    log_entry = build_log_entry(
                        final_result["tracer"], prompt, final_result["queries"], final_result["ranked"],
                        final_result["answer"], final_result["evaluation"],
                        final_result["hall_cause"], final_result["quality_report"],
                        search_quality_report=final_result["sqr"],
                        ndcg=final_result["ndcg_k"],
                        mode=final_result["mode"],
                        route_decision=route_decision,
                        fallback_triggered=fallback_triggered_flag,
                        fallback_attempts=attempt_num - 1,
                        fallback_history=fallback_history,
                        self_refinement=final_result.get("critique"),
                        dynamic_retrieval_profile=dynamic_profile_label,
                        cache_hit=final_result["cache_hit"],
                        compression_stats=final_result["compression_stats"],
                        mv_retrieval=(st.session_state.mv_index is not None),
                        failure_types=final_result["failure_types"],
                        failure_saved=final_result["failure_saved"],
                        parallel_ms=final_result.get("parallel_ms"),
                        tool_used=final_result.get("tool_used", False),
                        selective_stats=final_result.get("selective_stats"),
                        user_id=st.session_state.current_user,  # [NEW v22]
                    )
                    save_log(log_entry)

                response        = final_result["answer"]
                final_chunks    = final_result["final_chunks"]
                final_sources   = final_result["final_sources"]
                final_scores    = final_result["final_scores"]
                final_eff       = final_result["eff"]

                if final_result["failure_saved"]:
                    ftype_str = " · ".join(final_result["failure_types"])
                    st.error(f"🚨 실패 케이스 저장됨 — {ftype_str} | 🚨 실패 데이터셋 탭에서 확인/내보내기 가능")

                if final_result.get("tool_used"):
                    st.info(f"🔧 Tool-Augmented 답변 | 계산 결과: `{final_result.get('calc_result')}`")

                st.markdown(linkify_citations(response), unsafe_allow_html=True)

                if final_result.get("critique"):
                    with st.expander("✏️ Self-Refinement (Draft → Critique → Refined)"):
                        t1, t2 = st.columns(2)
                        with t1:
                            st.caption("Draft 답변")
                            st.write(final_result["draft_answer"])
                        with t2:
                            st.caption("Critique")
                            st.write(final_result["critique"])

                if fallback_triggered_flag:
                    with st.expander(f"🔄 Fallback 내역 ({attempt_num}회)"):
                        for h in fallback_history:
                            is_best = h["overall_score"] == quality_report["overall_score"]
                            st.markdown(f"**{'🏆 채택' if is_best else '📋'} 시도 {h['attempt']+1}** — `{h.get('trigger') or '최초'}`")
                            c1,c2,c3,c4,c5 = st.columns(5)
                            c1.metric("정확도",h['accuracy']); c2.metric("환각",h['hallucination'])
                            c3.metric("종합",h['overall_score']); c4.metric("등급",h['grade']); c5.metric("토큰",h['tokens'])
                            if h["attempt"] < len(fallback_history)-1: st.divider()

                if final_result.get("sqr"):
                    sqr = final_result["sqr"]
                    lc  = {"excellent":"🟢","good":"🔵","fair":"🟡","poor":"🔴"}.get(sqr["quality_label"],"⚪")
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
                        for i, q in enumerate(final_result["queries"]):
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
        tool_count  = sum(1 for l in logs if l.get("tool_used"))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("총 트레이스",     len(logs))
        c2.metric("평균 응답",       f"{sum(total_lat_all)/len(total_lat_all)/1000:.1f}s")
        c3.metric("평균 토큰",       f"{int(sum(total_tokens_all)/len(total_tokens_all)):,}")
        c4.metric("🔄 Fallback",    f"{fb_count}회")
        c5.metric("⚡ 병렬 검색",   f"{len(par_logs)}건")
        c6.metric("🔧 Tool 사용",   f"{tool_count}건")

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
            uid_tag  = f" | 👤{log.get('user_id','?')}" if log.get("user_id") else ""  # [NEW v22]
            fb_tag   = f" | 🔄 {log.get('fallback_attempts',0)}회" if log.get("fallback_triggered") else ""
            sr_tag   = " | ✏️" if log.get("self_refinement") else ""
            ch_tag   = f" | ⚡{log['cache_hit']}" if log.get("cache_hit") else ""
            fl_tag   = " | 🚨" if log.get("failure_saved") else ""
            par_tag  = f" | ⚡{log['parallel_ms']}ms" if log.get("parallel_ms") else ""
            tl_tag   = " | 🔧Tool" if log.get("tool_used") else ""

            with st.expander(
                f"[{ts}] {q}... | ⏱ {total_ms/1000:.1f}s | 🔤 {tok.get('total',0):,} | {intent}"
                + (f" | NDCG {nd:.3f}" if nd else "") + uid_tag + fb_tag + sr_tag + ch_tag + fl_tag + par_tag + tl_tag
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
                              "결정":s["decision"][:60]} for s in spans]
                st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)


# =====================================================================
# TAB 3 — 에이전트 분석
# =====================================================================
with tab_agent:
    st.title("🧠 에이전트 분석")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        eval_logs = [l for l in logs if l.get("evaluation")]
        acc_list  = [l["evaluation"].get("정확도",0) for l in eval_logs]
        rel_list  = [l["evaluation"].get("관련성",0) for l in eval_logs]
        hall_list = [l["evaluation"].get("환각여부","없음") for l in eval_logs]

        if acc_list:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("총 쿼리",       len(logs))
            c2.metric("평균 정확도",   f"{sum(acc_list)/len(acc_list):.2f}/5")
            c3.metric("평균 관련성",   f"{sum(rel_list)/len(rel_list):.2f}/5")
            c4.metric("환각 비율",     f"{sum(1 for h in hall_list if h!='없음')/len(hall_list):.1%}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("정확도 분포")
                st.bar_chart(Counter(acc_list))
            with col2:
                st.subheader("환각 분포")
                st.bar_chart(Counter(hall_list))

        fail_logs = [l for l in logs if l.get("failure_saved")]
        if fail_logs:
            st.markdown("---")
            st.subheader("🚨 실패 케이스 분석")
            all_ftypes = []
            for l in fail_logs: all_ftypes.extend(l.get("failure_types",[]))
            ftype_counts = Counter(all_ftypes)
            fc1,fc2,fc3,fc4,fc5 = st.columns(5)
            fc1.metric("총 실패",      len(fail_logs))
            fc2.metric("낮은 정확도",  ftype_counts.get("low_accuracy",0))
            fc3.metric("환각",         ftype_counts.get("hallucination",0))
            fc4.metric("누락 정보",    ftype_counts.get("incomplete_answer",0))
            fc5.metric("검색 실패",    ftype_counts.get("retrieval_failure",0))
            if ftype_counts:
                st.bar_chart(dict(ftype_counts))

        st.markdown("---")
        st.subheader("쿼리별 상세")
        for log in reversed(logs[-20:]):
            ev  = log.get("evaluation", {})
            qr  = log.get("quality_report", {})
            ts  = log.get("timestamp","")
            q   = log.get("question","")[:60]
            uid = log.get("user_id","?")
            with st.expander(f"[{ts}] {q}... | 👤{uid}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("정확도",   ev.get("정확도","-"))
                    st.metric("관련성",   ev.get("관련성","-"))
                    st.metric("환각여부", ev.get("환각여부","-"))
                    if log.get("failure_saved"):
                        st.warning(f"🚨 실패 → {', '.join(log.get('failure_types',[]))}")
                with col_b:
                    st.metric("종합점수", qr.get("overall_score","-"))
                    st.metric("등급",     qr.get("grade","-"))
                    st.metric("모드",     log.get("mode","-"))


# =====================================================================
# TAB 4 — Ablation Study
# =====================================================================
with tab_ablation:
    st.title("🧬 Ablation Study")
    st.markdown("6가지 파이프라인 구성을 같은 질문에 실행해 성능을 비교합니다.")

    ablation_q = st.text_input("Ablation 질문", placeholder="비교할 질문을 입력하세요...")
    run_all = st.button("전체 Ablation 실행", type="primary",
                        disabled=(st.session_state.index is None or not ablation_q))

    if run_all and ablation_q and st.session_state.index is not None:
        results = []
        prog = st.progress(0)
        for i, cfg in enumerate(ABLATION_CONFIGS):
            with st.spinner(f"실행 중: {cfg['name']}..."):
                r = run_single_config(ablation_q, cfg, st.session_state.index,
                                      st.session_state.chunks, st.session_state.chunk_sources,
                                      top_k=top_k, prefilter_n=prefilter_n)
                results.append(r)
            prog.progress((i+1)/len(ABLATION_CONFIGS))
        st.session_state.ablation_results = results

    if st.session_state.ablation_results:
        import pandas as pd
        results = st.session_state.ablation_results
        df = pd.DataFrame([{
            "구성": r["config_name"], "정확도": r["accuracy"], "관련성": r["relevance"],
            "환각": r["hallucination"], "종합": r["overall_score"], "등급": r["grade"],
            "레이턴시(ms)": r["latency_ms"], "토큰": r["total_tokens"],
        } for r in results])
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("종합 점수 비교")
            st.bar_chart({r["config_name"]: r["overall_score"] for r in results})
        with col2:
            st.subheader("레이턴시 비교 (ms)")
            st.bar_chart({r["config_name"]: r["latency_ms"] for r in results})

        for r in results:
            if r.get("error"):
                st.error(f"❌ {r['config_name']}: {r['error']}")
            else:
                with st.expander(f"{r['config_name']} — 등급 {r['grade']} | 정확도 {r['accuracy']}/5"):
                    st.markdown(r["answer"])


# =====================================================================
# TAB 5 — 검색 품질
# =====================================================================
with tab_search:
    st.title("🔍 검색 품질 분석")
    logs = load_logs()
    sqr_logs = [l for l in logs if l.get("search_quality_report")]
    if not sqr_logs:
        st.info("검색 품질 데이터가 없습니다. 리랭킹을 활성화한 후 쿼리를 실행해주세요.")
    else:
        ndcg_vals = [l["search_quality_report"]["ndcg_prefilter"] for l in sqr_logs]
        gain_vals = [l["search_quality_report"]["reranker_gain"]  for l in sqr_logs]
        labels    = [l["search_quality_report"]["quality_label"]  for l in sqr_logs]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("평균 NDCG",       f"{sum(ndcg_vals)/len(ndcg_vals):.4f}")
        c2.metric("평균 Reranker Gain", f"{sum(gain_vals)/len(gain_vals):.4f}")
        c3.metric("excellent 비율",  f"{sum(1 for l in labels if l=='excellent')/len(labels):.1%}")
        c4.metric("poor 비율",       f"{sum(1 for l in labels if l=='poor')/len(labels):.1%}")

        st.subheader("NDCG 추이")
        st.line_chart(ndcg_vals)

        import pandas as pd
        rows = []
        for l in sqr_logs:
            sqr = l["search_quality_report"]
            rows.append({
                "시간": l.get("timestamp","")[:16],
                "질문": l.get("question","")[:40],
                "NDCG": round(sqr["ndcg_prefilter"],4),
                "Gain": round(sqr["reranker_gain"],4),
                "품질": sqr["quality_label"],
                "BM25": "✅" if sqr.get("use_bm25") else "❌",
                "실패 저장": "🚨" if l.get("failure_saved") else "",
                "사용자": l.get("user_id","?"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =====================================================================
# TAB 6 — 실패 데이터셋
# =====================================================================
with tab_failure:
    st.title("🚨 실패 데이터셋")
    all_failures = failure_dataset.get_all()

    if not all_failures:
        st.info("저장된 실패 케이스가 없습니다.")
    else:
        ftype_counts = Counter(ft for e in all_failures for ft in e.get("failure_types",[]))

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("총 실패 케이스",  len(all_failures))
        c2.metric("낮은 정확도",     ftype_counts.get("low_accuracy",0))
        c3.metric("환각 발생",       ftype_counts.get("hallucination",0))
        c4.metric("누락 정보",       ftype_counts.get("incomplete_answer",0))
        c5.metric("검색 실패",       ftype_counts.get("retrieval_failure",0))
        if ftype_counts:
            st.bar_chart(dict(ftype_counts))

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️ Fine-tune JSONL 내보내기 (OpenAI 형식)",
                               data=failure_dataset.export_finetune_jsonl(),
                               file_name=f"failure_finetune_{ts_now}.jsonl",
                               mime="application/jsonl")
        with col_exp2:
            st.download_button("⬇️ 문제 분석 JSON 내보내기",
                               data=failure_dataset.export_problems_json(),
                               file_name=f"failure_problems_{ts_now}.json",
                               mime="application/json")

        st.markdown("---")
        filter_type = st.selectbox("실패 유형 필터",
            ["전체","낮은 정확도","환각","누락 정보","검색 실패","낮은 관련성"])
        sort_by     = st.selectbox("정렬 기준", ["최신순","정확도 낮은순","오래된순"])

        type_map = {"낮은 정확도":"low_accuracy","환각":"hallucination",
                    "누락 정보":"incomplete_answer","검색 실패":"retrieval_failure","낮은 관련성":"low_relevance"}
        shown = all_failures if filter_type == "전체" else \
                [e for e in all_failures if type_map.get(filter_type,"") in e.get("failure_types",[])]
        if sort_by == "정확도 낮은순":
            shown = sorted(shown, key=lambda e: (e.get("evaluation") or {}).get("정확도",5))
        elif sort_by == "오래된순":
            shown = list(shown)
        else:
            shown = list(reversed(shown))

        for entry in shown:
            ev    = entry.get("evaluation") or {}
            ftags = " · ".join(entry.get("failure_types",[]))
            with st.expander(f"[{entry.get('id','')}] {entry.get('question','')[:60]}... | 🚨 {ftags}"):
                inner_t = st.tabs(["📋 개요","💬 답변","📄 청크","💡 개선 힌트"])
                with inner_t[0]:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("정확도",    ev.get("정확도","-"))
                    c2.metric("환각여부",  ev.get("환각여부","-"))
                    c3.metric("관련성",    ev.get("관련성","-"))
                    st.markdown(f"**실패 유형**: `{ftags}`")
                    st.markdown(f"**시간**: {entry.get('timestamp','')}")
                with inner_t[1]:
                    st.write(entry.get("answer",""))
                with inner_t[2]:
                    for i, (chunk, src) in enumerate(zip(entry.get("chunks",[]), entry.get("sources",[]))):
                        st.caption(f"[청크 {i+1}] {src}")
                        st.write(chunk)
                        st.divider()
                with inner_t[3]:
                    hint = entry.get("improvement_hint")
                    if hint:
                        st.markdown(hint)
                    else:
                        if st.button(f"지금 생성 ({entry.get('id','')})", key=f"hint_{entry.get('id','')}"):
                            with st.spinner("LLM 개선 힌트 생성 중..."):
                                new_hint = generate_improvement_hint(
                                    entry["question"], entry.get("chunks",[]),
                                    entry["answer"], ev, entry.get("failure_types",[])
                                )
                            entry["improvement_hint"] = new_hint
                            failure_dataset._save()
                            st.markdown(new_hint)


# =====================================================================
# TAB 7 — v21 분석
# =====================================================================
with tab_v21:
    st.title("⚡ v21 분석 — 병렬 검색 · Selective Context · Tool-Augmented")
    logs = load_logs()
    if not logs:
        st.info("아직 로그가 없습니다.")
    else:
        par_logs  = [l for l in logs if l.get("parallel_ms")]
        sel_logs  = [l for l in logs if l.get("selective_stats")]
        tool_logs = [l for l in logs if l.get("tool_used")]

        st.subheader("⚡ 병렬 검색 속도 분석")
        if par_logs:
            par_times = [l["parallel_ms"] for l in par_logs]
            total_ms  = [l.get("total_latency_ms",1) for l in par_logs]
            ratios    = [round(p/max(t,1)*100,1) for p,t in zip(par_times,total_ms)]
            p50       = round(MetricsCollector.percentile(par_times, 50))
            p95       = round(MetricsCollector.percentile(par_times, 95))

            pc1,pc2,pc3,pc4 = st.columns(4)
            pc1.metric("평균 병렬 검색", f"{sum(par_times)//len(par_times)}ms")
            pc2.metric("P50",            f"{p50}ms")
            pc3.metric("P95",            f"{p95}ms")
            pc4.metric("평균 비율",      f"{sum(ratios)/len(ratios):.1f}%")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption("병렬 검색 시간 추이 (ms)")
                st.line_chart(par_times)
            with col_s2:
                st.caption("전체 레이턴시 대비 비율 (%)")
                st.line_chart(ratios)
        else:
            st.caption("병렬 검색 사용 데이터 없음.")

        st.markdown("---")
        st.subheader("🔬 Selective Context Phase 2 — Cross-chunk 중복 제거")
        if sel_logs:
            dedup_rates = [round((l["selective_stats"].get("dedup_removed",0)/
                                  max(l["selective_stats"].get("original_sents",1),1))*100,1) for l in sel_logs]
            comp_ratios = [round((1-l["selective_stats"].get("ratio",1))*100,1) for l in sel_logs]
            sc1,sc2,sc3 = st.columns(3)
            sc1.metric("평균 중복 제거율",   f"{sum(dedup_rates)/len(dedup_rates):.1f}%")
            sc2.metric("평균 컨텍스트 압축", f"{sum(comp_ratios)/len(comp_ratios):.1f}%")
            sc3.metric("적용 건수",          f"{len(sel_logs)}건")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption("중복 제거율 추이 (%)")
                st.line_chart(dedup_rates)
            with col_s2:
                st.caption("컨텍스트 압축율 추이 (%)")
                st.line_chart(comp_ratios)
        else:
            st.caption("Selective Context 사용 데이터 없음.")

        st.markdown("---")
        st.subheader("🔧 Tool-Augmented RAG — 수치 환각 분석")
        if tool_logs:
            tool_hall  = [l["evaluation"].get("환각여부","없음") for l in tool_logs if l.get("evaluation")]
            notool_logs= [l for l in logs if not l.get("tool_used") and l.get("evaluation")]
            notool_hall= [l["evaluation"].get("환각여부","없음") for l in notool_logs]
            tool_hall_rate   = round(sum(1 for h in tool_hall  if h!="없음")/max(len(tool_hall),1)*100,1)
            notool_hall_rate = round(sum(1 for h in notool_hall if h!="없음")/max(len(notool_hall),1)*100,1)
            ta1,ta2,ta3 = st.columns(3)
            ta1.metric("Tool 사용 건수",      f"{len(tool_logs)}건")
            ta2.metric("Tool 사용 시 환각율",  f"{tool_hall_rate}%", delta_color="inverse")
            ta3.metric("일반 생성 환각율",     f"{notool_hall_rate}%", delta_color="inverse")
        else:
            st.caption("Tool-Augmented 사용 데이터 없음.")

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
        """)


# =====================================================================
# [NEW v22] TAB 8 — 📊 모니터링
# =====================================================================
with tab_monitor:
    st.title("📊 실시간 모니터링 대시보드")
    logs = load_logs()

    if not logs:
        st.info("아직 로그가 없습니다. 쿼리를 실행하면 지표가 나타납니다.")
    else:
        stats  = metrics_collector.compute_from_logs(logs)
        alerts = metrics_collector.get_alerts(stats)

        # ── 알림 배너 ────────────────────────────────────────────
        st.subheader("🔔 알림 상태")
        for alert in alerts:
            if alert["level"] == "error":
                st.error(f"{alert['icon']} {alert['message']}")
            elif alert["level"] == "warning":
                st.warning(f"{alert['icon']} {alert['message']}")
            else:
                st.success(f"{alert['icon']} {alert['message']}")

        st.markdown("---")

        # ── 핵심 지표 ────────────────────────────────────────────
        st.subheader("📈 핵심 지표")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("총 쿼리",        stats.get("total_queries",0))
        m2.metric("24시간 쿼리",     stats.get("queries_24h",0))
        m3.metric("평균 정확도",     f"{stats.get('accuracy_avg',0):.2f}/5")
        m4.metric("환각 비율",       f"{stats.get('hallucination_rate',0):.1%}")
        m5.metric("캐시 히트율",     f"{stats.get('cache_hit_rate',0):.1%}")
        m6.metric("실패 케이스",     stats.get("failure_count",0))

        st.markdown("---")

        # ── 레이턴시 퍼센타일 ────────────────────────────────────
        st.subheader("⏱ 레이턴시 퍼센타일")
        lc1,lc2,lc3,lc4 = st.columns(4)
        lc1.metric("평균",   f"{stats.get('latency_avg_ms',0):,}ms")
        lc2.metric("P50",    f"{stats.get('latency_p50_ms',0):,}ms")
        lc3.metric("P95",    f"{stats.get('latency_p95_ms',0):,}ms",
                   delta=None if stats.get('latency_p95_ms',0) <= ALERT_LATENCY_P95_MS
                         else f"+{stats['latency_p95_ms']-ALERT_LATENCY_P95_MS:,}ms", delta_color="inverse")
        lc4.metric("P99",    f"{stats.get('latency_p99_ms',0):,}ms")

        if stats.get("latency_trend"):
            st.caption("레이턴시 추이 (최근 30건, ms)")
            st.line_chart(stats["latency_trend"])

        st.markdown("---")

        # ── 정확도 / 환각 트렌드 ─────────────────────────────────
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.subheader("📊 정확도 트렌드")
            if stats.get("accuracy_trend"):
                st.line_chart(stats["accuracy_trend"])
            st.metric("평균 관련성", f"{stats.get('relevance_avg',0):.2f}/5")

        with col_a2:
            st.subheader("🧠 환각 트렌드 (0=없음, 1=있음)")
            if stats.get("hall_trend"):
                st.line_chart(stats["hall_trend"])
            st.metric("Fallback 비율", f"{stats.get('fallback_rate',0):.1%}")

        st.markdown("---")

        # ── 토큰 / 기타 ──────────────────────────────────────────
        st.subheader("🔤 토큰 사용량")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("총 토큰",     f"{stats.get('token_total',0):,}")
        tc2.metric("쿼리당 평균", f"{stats.get('token_avg',0):,}")
        tc3.metric("Tool 사용",   f"{stats.get('tool_used_count',0)}건")

        # ── 사용자별 쿼리 분포 ───────────────────────────────────
        st.markdown("---")
        st.subheader("👤 사용자별 쿼리 분포")
        user_counts = Counter(l.get("user_id","anonymous") for l in logs)
        if user_counts:
            st.bar_chart(dict(user_counts))

        # ── 내보내기 ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📤 메트릭 내보내기")
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            prom_txt = metrics_collector.export_prometheus(stats)
            st.download_button(
                "⬇️ Prometheus 형식으로 내보내기",
                data=prom_txt.encode("utf-8"),
                file_name=f"rag_metrics_{ts_now}.prom",
                mime="text/plain",
                help="Grafana / Prometheus에 직접 연결 가능한 형식"
            )
        with col_e2:
            st.download_button(
                "⬇️ JSON 형식으로 내보내기",
                data=metrics_collector.export_json(stats),
                file_name=f"rag_metrics_{ts_now}.json",
                mime="application/json"
            )

        # ── 알림 임계값 안내 ─────────────────────────────────────
        with st.expander("⚙️ 알림 임계값 설정 기준"):
            st.markdown(f"""
| 지표 | 임계값 | 현재 값 |
|------|--------|---------|
| 평균 정확도 | < {ALERT_ACCURACY_MIN} → 경보 | {stats.get('accuracy_avg',0):.2f}/5 |
| 환각 비율 | > {ALERT_HALL_MAX:.0%} → 경보 | {stats.get('hallucination_rate',0):.1%} |
| P95 레이턴시 | > {ALERT_LATENCY_P95_MS:,}ms → 경보 | {stats.get('latency_p95_ms',0):,}ms |

임계값을 변경하려면 코드 상단 `ALERT_*` 상수를 수정하세요.
            """)


# =====================================================================
# [NEW v22] TAB 9 — 👤 사용자 관리 (admin only)
# =====================================================================
with tab_users:
    st.title("👤 사용자 관리")

    if st.session_state.user_role != "admin":
        st.warning("관리자 권한이 필요합니다.")
    else:
        # ── 사용자 목록 ──────────────────────────────────────────
        st.subheader("등록된 사용자")
        users_list = user_manager.list_users()

        import pandas as pd
        user_rows = []
        for u in users_list:
            u_stats = user_manager.get_user_stats(u["username"])
            ok, rem = user_manager.check_rate_limit(u["username"])
            user_rows.append({
                "사용자명": u["username"],
                "표시 이름": u["display_name"],
                "권한": u["role"],
                "총 쿼리": u_stats["total_queries"],
                "1시간 쿼리": u_stats["queries_1h"],
                "잔여 (1h)": rem,
                "가입일": u["created_at"][:10] if u.get("created_at") else "-",
            })
        st.dataframe(pd.DataFrame(user_rows), use_container_width=True)

        # ── 신규 사용자 생성 ─────────────────────────────────────
        st.markdown("---")
        st.subheader("신규 사용자 생성")
        with st.form("create_user_form"):
            col_u1, col_u2, col_u3 = st.columns(3)
            with col_u1:
                new_username = st.text_input("사용자명")
            with col_u2:
                new_password = st.text_input("비밀번호 (6자 이상)", type="password")
            with col_u3:
                new_role     = st.selectbox("권한", ["user", "admin"])
            new_display  = st.text_input("표시 이름 (선택)")
            create_btn   = st.form_submit_button("사용자 생성", type="primary")

        if create_btn:
            ok, msg = user_manager.create_user(new_username, new_password, new_role, new_display)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        # ── 사용자 삭제 ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("사용자 삭제")
        del_candidates = [u["username"] for u in users_list if u["username"] != "admin"]
        if del_candidates:
            del_user = st.selectbox("삭제할 사용자 선택", del_candidates)
            if st.button(f"'{del_user}' 삭제", type="secondary"):
                ok, msg = user_manager.delete_user(del_user)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("삭제 가능한 사용자가 없습니다.")

        # ── 비밀번호 변경 ────────────────────────────────────────
        st.markdown("---")
        st.subheader("비밀번호 변경")
        with st.form("change_pw_form"):
            target_user = st.selectbox("대상 사용자", [u["username"] for u in users_list])
            new_pw      = st.text_input("새 비밀번호 (6자 이상)", type="password")
            pw_btn      = st.form_submit_button("비밀번호 변경")

        if pw_btn:
            ok, msg = user_manager.change_password(target_user, new_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        # ── Rate Limiting 현황 ───────────────────────────────────
        st.markdown("---")
        st.subheader("⏱ Rate Limiting 현황")
        st.info(f"현재 설정: 사용자당 시간당 최대 **{RATE_LIMIT_PER_HOUR}** 쿼리 (관리자 무제한)")
        st.caption("`RATE_LIMIT_PER_HOUR` 상수를 수정해 조정하세요.")

        logs = load_logs()
        now  = datetime.now()
        cutoff_1h = now - timedelta(hours=1)
        for u in users_list:
            u_logs_1h = [l for l in logs
                         if l.get("user_id") == u["username"]
                         and MetricsCollector._parse_ts(l.get("timestamp","")) >= cutoff_1h]
            rate = len(u_logs_1h)
            pct  = min(rate / max(RATE_LIMIT_PER_HOUR, 1), 1.0)
            col_n, col_bar = st.columns([1,3])
            col_n.caption(f"**{u['username']}** ({u['role']})")
            col_bar.progress(pct, text=f"{rate}/{RATE_LIMIT_PER_HOUR if u['role']!='admin' else '∞'} 쿼리/h")


# =====================================================================
# [NEW v22] TAB 10 — 🔌 API 미리보기
# =====================================================================
with tab_api:
    st.title("🔌 FastAPI 서버 코드 미리보기")
    st.markdown("""
현재 구조(Streamlit 단일 파일)를 **FastAPI 백엔드 + React 프론트엔드**로 분리하는 방법을 안내합니다.

v22에서는 코드 미리보기와 구조 안내를 제공합니다.
실제 배포는 `rag_app_v22_server.py`로 분리해 실행하세요.
    """)

    st.subheader("📐 분리 아키텍처")
    st.markdown("""
```
┌─────────────────────────────────────────────────────┐
│                   클라이언트                          │
│  React / Streamlit / Mobile App / 3rd-party          │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS REST
┌──────────────────────▼──────────────────────────────┐
│              FastAPI 백엔드 (v22_server.py)           │
│  POST /query      → run_rag_pipeline()               │
│  GET  /metrics    → MetricsCollector.export_json()   │
│  POST /auth/login → UserManager.verify_login()       │
│  GET  /users      → UserManager.list_users() [admin] │
│  GET  /logs       → load_logs() [admin]              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           RAG 코어 (현재 rag_app_v22.py)              │
│  run_rag_pipeline / MetricsCollector / UserManager   │
└─────────────────────────────────────────────────────┘
```
    """)

    st.markdown("---")

    api_tabs = st.tabs([
        "FastAPI 서버 코드",
        "API 엔드포인트 명세",
        "인증 플로우",
        "Prometheus 연동"
    ])

    with api_tabs[0]:
        st.subheader("FastAPI 서버 기본 구조")
        st.code('''
# rag_app_v22_server.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import jwt, hashlib
from datetime import datetime, timedelta

# 현재 파일의 핵심 함수들을 임포트
# from rag_core import run_rag_pipeline, MetricsCollector, UserManager

SECRET_KEY = "your-secret-key-here"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="RAG API v22", version="22.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── 모델 ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    use_parallel: bool = True
    use_lim_reorder: bool = True
    use_tool_augment: bool = False

class QueryResponse(BaseModel):
    answer: str
    evaluation: dict
    quality_report: dict
    latency_ms: int
    user_id: str
    trace_id: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


# ── 인증 ─────────────────────────────────────────────
def create_access_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                           detail="유효하지 않은 토큰")


# ── 엔드포인트 ───────────────────────────────────────
@app.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    if not user_manager.verify_login(form.username, form.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                           detail="잘못된 사용자명 또는 비밀번호")
    role  = user_manager.get_role(form.username)
    token = create_access_token(form.username, role)
    return {"access_token": token, "token_type": "bearer", "role": role}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest,
                current_user: dict = Depends(get_current_user)):
    ok, remaining = user_manager.check_rate_limit(current_user["username"])
    if not ok and current_user["role"] != "admin":
        raise HTTPException(status_code=429,
                           detail=f"Rate limit 초과. 잔여: {remaining}")

    result = run_rag_pipeline(
        question=req.question,
        eff={"use_bm25": True, "use_reranking": True,
             "top_k": 3, "use_query_rewrite": True},
        index=index, chunks=chunks, sources=sources,
        prefilter_n=10, use_multidoc=True,
        use_parallel_search=req.use_parallel,
        use_lim_reorder=req.use_lim_reorder,
        use_tool_augment=req.use_tool_augment,
        user_id=current_user["username"]
    )
    user_manager.record_usage(current_user["username"])

    return QueryResponse(
        answer=result["answer"],
        evaluation=result["evaluation"],
        quality_report=result["quality_report"],
        latency_ms=result["tracer"].total_latency_ms(),
        user_id=current_user["username"],
        trace_id=result["tracer"].trace_id
    )


@app.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    logs  = load_logs()
    stats = MetricsCollector().compute_from_logs(logs)
    return stats


@app.get("/metrics/prometheus")
async def prometheus_metrics(current_user: dict = Depends(get_current_user)):
    from fastapi.responses import PlainTextResponse
    logs  = load_logs()
    stats = MetricsCollector().compute_from_logs(logs)
    return PlainTextResponse(MetricsCollector().export_prometheus(stats))


@app.get("/users", dependencies=[Depends(get_current_user)])
async def list_users(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한 필요")
    return user_manager.list_users()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "22.0.0",
            "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        ''', language="python")

    with api_tabs[1]:
        st.subheader("API 엔드포인트 명세")
        import pandas as pd
        endpoints = [
            {"메서드": "POST", "경로": "/auth/login",        "인증": "없음",   "설명": "JWT 토큰 발급"},
            {"메서드": "POST", "경로": "/query",             "인증": "Bearer", "설명": "RAG 파이프라인 실행"},
            {"메서드": "GET",  "경로": "/metrics",           "인증": "Bearer", "설명": "JSON 형식 메트릭"},
            {"메서드": "GET",  "경로": "/metrics/prometheus", "인증": "Bearer", "설명": "Prometheus 형식 메트릭"},
            {"메서드": "GET",  "경로": "/users",             "인증": "admin",  "설명": "사용자 목록 (관리자)"},
            {"메서드": "GET",  "경로": "/health",            "인증": "없음",   "설명": "서버 상태 확인"},
        ]
        st.dataframe(pd.DataFrame(endpoints), use_container_width=True)

        st.subheader("요청/응답 예시")
        st.code('''
# 로그인
curl -X POST http://localhost:8000/auth/login \\
  -d "username=demo&password=demo123"

# → {"access_token": "eyJhb...", "token_type": "bearer", "role": "user"}

# 쿼리
curl -X POST http://localhost:8000/query \\
  -H "Authorization: Bearer eyJhb..." \\
  -H "Content-Type: application/json" \\
  -d \'{"question": "매출 증가율은?", "use_tool_augment": true}\'

# → {"answer": "...", "evaluation": {...}, "latency_ms": 2341, ...}

# 메트릭
curl http://localhost:8000/metrics/prometheus \\
  -H "Authorization: Bearer eyJhb..."
        ''', language="bash")

    with api_tabs[2]:
        st.subheader("JWT 기반 인증 플로우")
        st.markdown("""
```
클라이언트                     FastAPI                    UserManager
    │                             │                            │
    │── POST /auth/login ─────────▶                            │
    │   {username, password}      │── verify_login() ─────────▶
    │                             │                            │
    │                             │◀─ True / False ────────────│
    │                             │                            │
    │                             │  create_access_token()     │
    │                             │  (JWT, 30분 유효)           │
    │                             │                            │
    │◀─ {access_token, role} ─────│                            │
    │                             │                            │
    │── POST /query ──────────────▶                            │
    │   Authorization: Bearer JWT │                            │
    │                             │── check_rate_limit() ──────▶
    │                             │◀─ (ok, remaining) ─────────│
    │                             │                            │
    │                             │  run_rag_pipeline()        │
    │                             │  record_usage()            │
    │                             │                            │
    │◀─ {answer, evaluation, ...} ─│                            │
```
        """)

    with api_tabs[3]:
        st.subheader("Grafana + Prometheus 연동 가이드")
        st.markdown("""
#### 1. Prometheus 설정 (`prometheus.yml`)
```yaml
scrape_configs:
  - job_name: 'rag_api_v22'
    scrape_interval: 30s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
    bearer_token: 'YOUR_API_TOKEN'
```

#### 2. 주요 메트릭
| Prometheus 메트릭 | 설명 |
|---|---|
| `rag_queries_total` | 총 쿼리 수 (Counter) |
| `rag_latency_p50_ms` | P50 레이턴시 (Gauge) |
| `rag_latency_p95_ms` | P95 레이턴시 (Gauge) |
| `rag_accuracy_avg` | 평균 정확도 1~5 (Gauge) |
| `rag_hallucination_rate` | 환각 비율 0~1 (Gauge) |
| `rag_cache_hit_rate` | 캐시 히트율 0~1 (Gauge) |
| `rag_failure_count` | 실패 케이스 수 (Counter) |

#### 3. Grafana 대시보드 패널 예시
- **정확도 트렌드**: `rate(rag_accuracy_avg[5m])`
- **환각 비율 경보**: `rag_hallucination_rate > 0.3`
- **P95 SLA 모니터링**: `rag_latency_p95_ms > 15000`
        """)

        st.info(
            "**SigNoz** 또는 **Datadog** 사용 시 OpenTelemetry 트레이싱을 추가하면\n"
            "각 RAG 단계(검색/리랭킹/생성/평가)별 분산 트레이싱이 가능합니다."
        )
