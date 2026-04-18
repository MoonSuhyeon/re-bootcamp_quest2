# client_app.py — v23 Streamlit 프론트엔드
# server_api.py (FastAPI) 에 HTTP 요청만 수행 — 로컬 RAG 로직 없음

from __future__ import annotations

import os
import json
import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

# =====================================================================
# 서버 URL
# =====================================================================

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

# =====================================================================
# API 헬퍼
# =====================================================================

def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path: str, params: dict = None, timeout: int = 30) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=_headers(), params=params, timeout=timeout)
        if r.status_code == 401:
            st.session_state["token"] = ""
            st.session_state["logged_in"] = False
            st.rerun()
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"서버 연결 실패: {API_BASE}")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API 오류: {detail}")
        return None
    except Exception as e:
        st.error(f"요청 오류: {e}")
        return None


def api_post(path: str, json_data: dict = None, files=None, data=None,
             timeout: int = 120) -> dict | None:
    try:
        kwargs: dict = {"headers": _headers(), "timeout": timeout}
        if files:
            kwargs["files"] = files
            if data:
                kwargs["data"] = data
        elif json_data is not None:
            kwargs["json"] = json_data
        r = requests.post(f"{API_BASE}{path}", **kwargs)
        if r.status_code == 401:
            st.session_state["token"] = ""
            st.session_state["logged_in"] = False
            st.rerun()
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"서버 연결 실패: {API_BASE}")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API 오류: {detail}")
        return None
    except Exception as e:
        st.error(f"요청 오류: {e}")
        return None


def api_delete(path: str, timeout: int = 30) -> dict | None:
    try:
        r = requests.delete(f"{API_BASE}{path}", headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"삭제 오류: {e}")
        return None

# =====================================================================
# 세션 초기화
# =====================================================================

def init_session():
    defaults = {
        "logged_in": False,
        "token": "",
        "username": "",
        "role": "user",
        "chat_history": [],
        "last_result": None,
        "upload_done": False,
        "upload_filename": "",
        "upload_chunks": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =====================================================================
# 로그인 화면
# =====================================================================

def render_login():
    st.title("🔐 RAG App v23")
    st.caption("LLM Compression · ToolRegistry · AsyncRAGEngine")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("로그인")
        username = st.text_input("사용자명", key="login_user")
        password = st.text_input("비밀번호", type="password", key="login_pass")

        if st.button("로그인", use_container_width=True, type="primary"):
            if not username or not password:
                st.warning("사용자명과 비밀번호를 입력하세요")
                return
            try:
                r = requests.post(
                    f"{API_BASE}/auth/login",
                    data={"username": username, "password": password},
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state["token"]     = data["access_token"]
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = data["username"]
                    st.session_state["role"]      = data["role"]
                    st.rerun()
                elif r.status_code == 401:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다")
                else:
                    st.error(f"로그인 실패: {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(f"서버({API_BASE})에 연결할 수 없습니다")
            except Exception as e:
                st.error(f"오류: {e}")

        health = None
        try:
            health = requests.get(f"{API_BASE}/health", timeout=3).json()
        except Exception:
            pass
        if health:
            st.caption(f"서버 상태: {'✅ 온라인' if health.get('status') == 'ok' else '⚠️ 오류'} · "
                       f"인덱스: {'준비됨' if health.get('index_ready') else '없음'}")
        else:
            st.caption(f"서버({API_BASE}) 오프라인")

# =====================================================================
# 사이드바
# =====================================================================

def render_sidebar() -> dict:
    """사이드바 렌더링 → 설정 dict 반환"""
    with st.sidebar:
        st.title("⚙️ RAG v23 설정")

        user_info = api_get("/auth/me")
        if user_info:
            st.info(f"👤 {user_info['username']} ({user_info['role']})\n"
                    f"오늘 {user_info.get('usage_today', 0)}건 / 이번 시간 {user_info.get('usage_this_hour', 0)}건")

        if st.button("로그아웃"):
            st.session_state.update({"logged_in": False, "token": "", "username": "", "role": "user"})
            st.rerun()

        st.divider()

        # ── 문서 업로드 ──────────────────────────────────────────────
        st.subheader("📄 문서 업로드")
        uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"])
        chunk_size = st.slider("청크 크기", 200, 1000, 500, 50)
        overlap    = st.slider("오버랩", 0, 200, 50, 10)
        build_mv   = st.checkbox("Multi-Vector 인덱스", value=True)

        if uploaded and st.button("📤 업로드 & 인덱싱"):
            with st.spinner("업로드 중..."):
                result = api_post(
                    "/docs/upload",
                    files={"file": (uploaded.name, uploaded.read(), "application/pdf")},
                    data={"chunk_size": chunk_size, "overlap": overlap, "build_mv": str(build_mv).lower()},
                    timeout=180,
                )
                if result:
                    st.success(f"완료: {result['chunk_count']}청크")
                    st.session_state["upload_done"]     = True
                    st.session_state["upload_filename"] = result["filename"]
                    st.session_state["upload_chunks"]   = result["chunk_count"]

        health = api_get("/health")
        if health:
            if health.get("index_ready"):
                st.success(f"인덱스 준비됨 ({health['chunk_count']}청크)")
            else:
                st.warning("인덱스 없음 — PDF를 업로드하세요")

        st.divider()

        # ── 검색 설정 ─────────────────────────────────────────────────
        st.subheader("🔍 검색 설정")
        mode      = st.selectbox("검색 모드", ["hybrid", "dense", "bm25"])
        top_k     = st.slider("Top-K", 1, 20, 5)
        prefilter = st.slider("Pre-filter N", 10, 100, 30, 5)
        dedup     = st.slider("중복 제거 임계값", 0.5, 1.0, 0.85, 0.05)

        st.subheader("🛠 파이프라인 옵션")
        use_rewrite     = st.checkbox("쿼리 재작성", value=True)
        use_self_refine = st.checkbox("Self-Refinement", value=True)
        use_multidoc    = st.checkbox("Multi-Doc Fusion", value=True)
        use_compression = st.checkbox("Context Compression (임베딩)", value=True)
        auto_routing    = st.checkbox("자동 라우팅", value=True)

        st.subheader("[NEW v23] 고급 옵션")
        use_llm_compress  = st.checkbox("LLM Context Compression (Phase 3)", value=False,
                                         help="+1 LLM 호출 — LLM이 문장 단위로 핵심 선별")
        use_tool_registry = st.checkbox("Tool Registry (함수 호출)", value=False,
                                         help="계산기 · 날짜 · 단위 변환 · 웹검색")
        enabled_tools: list[str] = []
        if use_tool_registry:
            enabled_tools = st.multiselect(
                "활성화 도구",
                ["calculator", "datetime", "unit_converter", "web_search"],
                default=["calculator", "datetime"],
            )
        use_async_engine = st.checkbox("Async Engine (병렬 검색)", value=False,
                                        help="asyncio.gather() 4-way 병렬 검색")

        st.subheader("🚨 실패 데이터셋")
        auto_save_failure   = st.checkbox("실패 케이스 자동 저장", value=True)
        gen_improvement_hint = st.checkbox("개선 힌트 자동 생성 (+1 LLM)", value=False)

        failures = api_get("/failures", params={"limit": 1})
        if failures:
            st.caption(f"저장된 실패 케이스: {failures['total']}건")

    return {
        "mode": mode,
        "top_k": top_k,
        "prefilter_n": prefilter,
        "dedup_threshold": dedup,
        "use_rewrite": use_rewrite,
        "use_self_refine": use_self_refine,
        "use_multidoc": use_multidoc,
        "use_compression": use_compression,
        "auto_routing": auto_routing,
        "use_llm_compress": use_llm_compress,
        "use_tool_registry": use_tool_registry,
        "enabled_tools": enabled_tools,
        "use_async_engine": use_async_engine,
        "auto_save_failure": auto_save_failure,
        "gen_improvement_hint": gen_improvement_hint,
    }

# =====================================================================
# TAB 1 — 챗봇
# =====================================================================

def render_chat_tab(cfg: dict):
    st.header("💬 RAG 챗봇")

    # 채팅 히스토리 표시
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(4)
                cols[0].metric("지연(ms)", f"{meta.get('latency_ms', 0):.0f}")
                eval_ = meta.get("evaluation") or {}
                cols[1].metric("정확도", f"{eval_.get('정확도', '-')}/5")
                cols[2].metric("등급", (meta.get("quality_report") or {}).get("grade", "-"))
                cols[3].metric("모드", meta.get("mode_used", "-"))

                if meta.get("failure_types"):
                    st.error(f"🚨 실패 저장됨: {' · '.join(meta['failure_types'])}")
                if meta.get("tool_calls"):
                    with st.expander("🔧 Tool 호출 내역"):
                        for tc in meta["tool_calls"]:
                            st.json(tc)
                if meta.get("sources"):
                    with st.expander("📄 출처"):
                        for s in set(meta["sources"]):
                            st.caption(s)

    # 입력
    question = st.chat_input("질문을 입력하세요...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = api_post("/chat", json_data={
                    "question": question,
                    **cfg,
                }, timeout=180)

            if result:
                st.write(result["answer"])
                meta = {k: result.get(k) for k in
                        ["latency_ms", "mode_used", "evaluation", "quality_report",
                         "failure_types", "failure_saved", "tool_calls", "sources", "async_used"]}

                cols = st.columns(4)
                cols = st.columns(4)
                cols[0].metric("지연(ms)", f"{meta.get('latency_ms', 0):.0f}")
                eval_ = meta.get("evaluation") or {}
                cols[1].metric("정확도", f"{eval_.get('정확도', '-')}/5")
                cols[2].metric("등급", (meta.get("quality_report") or {}).get("grade", "-"))
                cols[3].metric("모드", meta.get("mode_used", "-"))

                if meta.get("failure_types"):
                    st.error(f"🚨 실패 케이스 저장됨: {' · '.join(meta['failure_types'])}")
                if meta.get("async_used"):
                    st.caption("⚡ Async Engine 사용")
                if meta.get("tool_calls"):
                    with st.expander("🔧 Tool 호출 내역"):
                        for tc in meta["tool_calls"]:
                            st.json(tc)
                if meta.get("sources"):
                    with st.expander("📄 출처"):
                        for s in set(meta["sources"]):
                            st.caption(s)

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": meta,
                })
                st.session_state["last_result"] = result
            else:
                st.error("답변 생성 실패")

    if st.session_state["chat_history"]:
        if st.button("대화 초기화"):
            st.session_state["chat_history"] = []
            st.rerun()

# =====================================================================
# TAB 2 — 평가 로그
# =====================================================================

def render_logs_tab():
    st.header("📋 평가 로그")

    col1, col2 = st.columns([3, 1])
    with col2:
        limit = st.number_input("표시 개수", 10, 200, 50)
    with col1:
        if st.button("로그 초기화", type="secondary"):
            if st.session_state.get("role") == "admin":
                result = api_delete("/logs")
                if result:
                    st.success(result["message"])
                    st.rerun()
            else:
                st.error("관리자 권한이 필요합니다")

    data = api_get("/logs", params={"limit": limit})
    if not data:
        return

    logs = data.get("items", [])
    st.caption(f"총 {data['total']}건 중 {len(logs)}건 표시")

    if not logs:
        st.info("로그가 없습니다")
        return

    rows = []
    for log in logs:
        eval_ = log.get("evaluation") or {}
        qr    = log.get("quality_report") or {}
        rows.append({
            "시각":    log.get("timestamp", "")[:19],
            "질문":    log.get("question", "")[:60],
            "정확도":  eval_.get("accuracy_score", "-"),
            "관련성":  eval_.get("relevance_score", "-"),
            "등급":    qr.get("grade", "-"),
            "지연(ms)": log.get("latency_ms", 0),
            "모드":    log.get("mode", "-"),
            "실패":    "🚨" if log.get("failure_saved") else "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 정확도 추이 차트
    acc_values = [log.get("evaluation", {}).get("accuracy_score") for log in logs
                  if log.get("evaluation", {}).get("accuracy_score")]
    if acc_values:
        fig = px.line(y=acc_values, title="정확도 추이", labels={"y": "정확도", "index": "쿼리 번호"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 3 — 메트릭 / 모니터링
# =====================================================================

def render_metrics_tab():
    st.header("📊 메트릭 & 모니터링")

    data = api_get("/metrics")
    if not data:
        return

    # 기본 지표
    cols = st.columns(4)
    cols[0].metric("총 쿼리", data.get("total_queries", 0))
    cols[1].metric("평균 정확도", f"{data.get('avg_accuracy', 0):.2f}")
    cols[2].metric("총 실패 케이스", data.get("failure", {}).get("total", 0))
    cols[3].metric("청크 수", data.get("chunk_count", 0))

    # 지연시간
    latency = data.get("latency", {})
    if latency:
        st.subheader("⏱ 응답 지연 (ms)")
        lat_cols = st.columns(4)
        lat_cols[0].metric("P50", f"{latency.get('p50', 0):.0f}")
        lat_cols[1].metric("P95", f"{latency.get('p95', 0):.0f}")
        lat_cols[2].metric("P99", f"{latency.get('p99', 0):.0f}")
        lat_cols[3].metric("평균", f"{latency.get('mean', 0):.0f}")

    # 실패 유형 분포
    failure_by_type = data.get("failure", {}).get("by_type", {})
    if any(v > 0 for v in failure_by_type.values()):
        st.subheader("🚨 실패 유형 분포")
        labels = {
            "low_accuracy":    "낮은 정확도",
            "hallucination":   "환각",
            "incomplete_answer": "누락 정보",
            "retrieval_failure": "검색 실패",
            "low_relevance":   "낮은 관련성",
        }
        df = pd.DataFrame([
            {"유형": labels.get(k, k), "건수": v}
            for k, v in failure_by_type.items() if v > 0
        ])
        fig = px.bar(df, x="유형", y="건수", title="실패 유형별 건수")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # 헬스
    health = api_get("/health")
    if health:
        st.subheader("🩺 서버 상태")
        st.json(health)

# =====================================================================
# TAB 4 — 실패 데이터셋
# =====================================================================

def render_failure_tab():
    st.header("🚨 실패 데이터셋")

    # 내보내기 버튼
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬇️ Fine-tune JSONL"):
            try:
                r = requests.get(f"{API_BASE}/failures/export/jsonl", headers=_headers(), timeout=30)
                r.raise_for_status()
                filename = f"failure_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                st.download_button("다운로드", r.content, file_name=filename, mime="application/jsonlines")
            except Exception as e:
                st.error(f"다운로드 오류: {e}")
    with col2:
        if st.button("⬇️ 문제 분석 JSON"):
            try:
                r = requests.get(f"{API_BASE}/failures/export/json", headers=_headers(), timeout=30)
                r.raise_for_status()
                filename = f"failure_problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button("다운로드", r.content, file_name=filename, mime="application/json")
            except Exception as e:
                st.error(f"다운로드 오류: {e}")
    with col3:
        if st.button("🗑️ 데이터셋 초기화"):
            if st.session_state.get("role") == "admin":
                result = api_delete("/failures")
                if result:
                    st.success(result["message"])
                    st.rerun()
            else:
                st.error("관리자 권한 필요")

    st.divider()

    # 필터
    filter_type = st.selectbox(
        "실패 유형 필터",
        ["전체", "low_accuracy", "hallucination", "incomplete_answer",
         "retrieval_failure", "low_relevance"],
    )
    params = {"limit": 50}
    if filter_type != "전체":
        params["failure_type"] = filter_type

    data = api_get("/failures", params=params)
    if not data:
        return

    items = data.get("items", [])
    st.caption(f"총 {data['total']}건 중 {len(items)}건")

    if not items:
        st.info("저장된 실패 케이스가 없습니다")
        return

    for item in items:
        eval_ = item.get("evaluation") or {}
        types_str = " · ".join(item.get("failure_types", []))
        title = f"🚨 [{item.get('timestamp', '')[:16]}] {item.get('question', '')[:50]}... | {types_str}"
        with st.expander(title):
            inner = st.tabs(["📋 개요", "💬 답변", "📄 청크", "💡 개선 힌트"])

            with inner[0]:
                st.write(f"**질문:** {item.get('question', '')}")
                st.write(f"**실패 유형:** {types_str}")
                acc = eval_.get("accuracy_score", "-")
                rel = eval_.get("relevance_score", "-")
                hall = eval_.get("hallucination", "-")
                st.write(f"**정확도:** {acc}/5 · **관련성:** {rel}/5 · **환각:** {hall}")
                ri = item.get("retrieval_info", {})
                if ri:
                    st.write(f"**검색 모드:** {ri.get('mode','-')} · **NDCG:** {ri.get('ndcg','-')} · "
                             f"**품질:** {ri.get('quality_label','-')}")

            with inner[1]:
                st.write(item.get("answer", ""))

            with inner[2]:
                for i, (chunk, src) in enumerate(
                    zip(item.get("chunks", []), item.get("sources", [])), 1
                ):
                    with st.expander(f"청크 {i} — {src}"):
                        st.text(chunk)

            with inner[3]:
                hint = item.get("improvement_hint")
                if hint:
                    st.markdown(hint)
                else:
                    st.info("개선 힌트가 없습니다 (서버에서 gen_improvement_hint=True 로 재실행)")

# =====================================================================
# TAB 5 — 관리자 패널
# =====================================================================

def render_admin_tab():
    if st.session_state.get("role") != "admin":
        st.warning("관리자 전용 탭입니다")
        return

    st.header("👑 관리자 패널")

    # ── 사용자 목록 ───────────────────────────────────────────────────
    st.subheader("사용자 관리")
    users_data = api_get("/users")
    if users_data:
        users = users_data.get("users", [])
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True)

    # ── 신규 사용자 등록 ──────────────────────────────────────────────
    with st.expander("신규 사용자 등록"):
        new_user = st.text_input("사용자명", key="new_user_name")
        new_pass = st.text_input("비밀번호", type="password", key="new_user_pass")
        new_role = st.selectbox("역할", ["user", "admin"], key="new_user_role")
        if st.button("등록"):
            result = api_post("/auth/register",
                              json_data={"username": new_user, "password": new_pass, "role": new_role})
            if result:
                st.success(result["message"])
                st.rerun()

    # ── 사용자 삭제 ───────────────────────────────────────────────────
    with st.expander("사용자 삭제"):
        del_user = st.text_input("삭제할 사용자명", key="del_user")
        if st.button("삭제", type="secondary"):
            result = api_delete(f"/users/{del_user}")
            if result:
                st.success(result["message"])
                st.rerun()

    st.divider()

    # ── 인덱스 / 캐시 초기화 ─────────────────────────────────────────
    st.subheader("시스템 초기화")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ 인덱스 초기화"):
            result = api_delete("/docs/reset")
            if result:
                st.success(result["message"])
    with col2:
        if st.button("🗑️ 로그 초기화"):
            result = api_delete("/logs")
            if result:
                st.success(result["message"])
    with col3:
        if st.button("🗑️ 실패 데이터셋 초기화"):
            result = api_delete("/failures")
            if result:
                st.success(result["message"])

# =====================================================================
# TAB 6 — API 정보
# =====================================================================

def render_api_info_tab():
    st.header("🔌 API 정보")

    st.info(f"FastAPI 서버: **{API_BASE}**")
    st.markdown(f"[Swagger UI]({API_BASE}/docs) | [ReDoc]({API_BASE}/redoc)")

    st.subheader("주요 엔드포인트")
    endpoints = [
        ("POST", "/auth/login",           "로그인 — JWT 토큰 발급"),
        ("POST", "/docs/upload",          "PDF 업로드 → 인덱싱"),
        ("POST", "/chat",                 "RAG 질의응답"),
        ("GET",  "/metrics",              "성능 메트릭 조회"),
        ("GET",  "/logs",                 "평가 로그 조회"),
        ("GET",  "/failures",             "실패 케이스 조회"),
        ("GET",  "/failures/export/jsonl","Fine-tune JSONL 내보내기"),
        ("GET",  "/users",                "사용자 목록 (관리자)"),
        ("GET",  "/health",               "헬스체크"),
    ]
    df = pd.DataFrame(endpoints, columns=["Method", "Path", "설명"])
    st.dataframe(df, use_container_width=True)

    st.subheader("[NEW v23] 파라미터")
    st.code("""\
POST /chat
{
  "question": "...",
  "use_llm_compress": false,    # LLM Context Compression (Phase 3)
  "use_tool_registry": false,   # ToolRegistry (함수 호출)
  "enabled_tools": ["calculator", "datetime", "unit_converter", "web_search"],
  "use_async_engine": false,    # AsyncRAGEngine (asyncio.gather)
  ...
}
""", language="json")

    st.subheader("서버 연결 설정")
    new_base = st.text_input("API Base URL", value=API_BASE)
    st.caption("환경 변수 RAG_API_BASE 또는 위 필드에서 변경 후 재시작")

# =====================================================================
# 메인
# =====================================================================

def main():
    st.set_page_config(
        page_title="RAG App v23",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()

    if not st.session_state["logged_in"]:
        render_login()
        return

    cfg = render_sidebar()

    tabs = st.tabs([
        "💬 챗봇",
        "📋 평가 로그",
        "📊 메트릭",
        "🚨 실패 데이터셋",
        "👑 관리자",
        "🔌 API 정보",
    ])

    with tabs[0]:
        render_chat_tab(cfg)
    with tabs[1]:
        render_logs_tab()
    with tabs[2]:
        render_metrics_tab()
    with tabs[3]:
        render_failure_tab()
    with tabs[4]:
        render_admin_tab()
    with tabs[5]:
        render_api_info_tab()


if __name__ == "__main__":
    main()
