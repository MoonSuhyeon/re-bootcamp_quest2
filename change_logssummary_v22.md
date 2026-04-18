# rag_app_v22.py 변동사항 요약

## 핵심 변경: 모니터링 시스템 + 인증/사용자 관리 + FastAPI 서버화 미리보기

v21의 병렬 검색 / LongContextReorder / Tool-Augmented RAG 위에,
**실시간 지표 수집·알림**, **로그인 게이트·사용자별 쿼터 관리**,
**FastAPI 서버 전환 코드 미리보기**를 추가.

> 핵심 방향: "단일 Streamlit 앱 → 운영 가능한 멀티유저 서비스"로의 전환점

---

## 최종 스택

| 구성 | v21 (rag_app_v21.py) | v22 (rag_app_v22.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Multi-Vector + BM25 RRF + Parallel | 동일 |
| Context 압축 | Phase 1 임베딩 + Phase 2 Cross-chunk 코사인 중복 제거 | 동일 |
| LongContextReorder | `reorder_lost_in_middle()` | 동일 |
| Tool-Augmented RAG | `detect_calc_intent()` + `_safe_eval()` | 동일 |
| 실패 데이터셋 | `FailureDataset` + JSONL 내보내기 | 동일 |
| **모니터링** | (없음) | **`MetricsCollector` — P50/P95/P99 + 알림 + Prometheus 내보내기** |
| **인증** | (없음) | **`UserManager` — 로그인 게이트 + 역할 기반 접근** |
| **사용자 관리** | (없음) | **사용자 생성/삭제/비밀번호 변경 + 시간당 쿼터** |
| **FastAPI 미리보기** | (없음) | **JWT 인증 서버 코드 + 엔드포인트 명세 + Prometheus 연동 가이드** |

---

## 변경 사항

### 1. 신규 import 및 상수

```python
import threading          # [NEW v22]
import statistics         # [NEW v22]
import base64             # [NEW v22]

USERS_FILE            = os.path.join(_BASE, "rag_users_v22.json")
USAGE_LOG_FILE        = os.path.join(_BASE, "rag_usage_v22.json")
RATE_LIMIT_PER_HOUR   = 20
ALERT_ACCURACY_MIN    = 3.0
ALERT_HALL_MAX        = 0.30
ALERT_LATENCY_P95_MS  = 15_000
```

---

### 2. `MetricsCollector` — 지표 수집 클래스 (신규)

```python
class MetricsCollector:
    @staticmethod
    def percentile(data: list, p: float) -> float
    def compute_from_logs(self, logs: list) -> dict
    def get_alerts(self, stats: dict) -> list
    def export_prometheus(self, stats: dict) -> str
    def export_json(self, stats: dict) -> bytes
    @staticmethod
    def _parse_ts(ts_str: str) -> datetime
```

#### `compute_from_logs()` 반환 필드

| 필드 | 설명 |
|------|------|
| `total_queries` | 전체 쿼리 수 |
| `queries_24h` | 최근 24h 쿼리 수 |
| `latency_p50/p95/p99_ms` | 응답 시간 백분위수 (ms) |
| `accuracy_avg` | 평균 정확도 |
| `hallucination_rate` | 환각 발생 비율 |
| `token_avg` | 평균 토큰 사용량 |
| `failure_count` | 실패 케이스 수 |
| `cache_hit_rate` | 캐시 히트율 |
| `user_query_dist` | 사용자별 쿼리 분포 dict |

#### `get_alerts()` 알림 기준

| 알림 | 기준 |
|------|------|
| accuracy_low | 평균 정확도 < 3.0 |
| hallucination_high | 환각률 > 30% |
| latency_p95_high | P95 응답 시간 > 15,000ms |

#### `export_prometheus()` — Prometheus 텍스트 형식

```
# HELP rag_queries_total Total number of RAG queries
rag_queries_total <value>
rag_latency_p50_ms <value>
rag_latency_p95_ms <value>
rag_latency_p99_ms <value>
rag_accuracy_avg <value>
rag_hallucination_rate <value>
rag_cache_hit_rate <value>
rag_failure_count_total <value>
```

---

### 3. `UserManager` — 사용자 관리 클래스 (신규)

```python
class UserManager:
    def __init__(self, users_path: str, usage_path: str)
    def _hash(self, password: str) -> str          # hashlib SHA-256
    def _ensure_defaults(self)                     # admin/admin123, demo/demo123 자동 생성
    def verify_login(self, username, password) -> bool
    def get_role(self, username) -> str            # "admin" | "user"
    def get_display_name(self, username) -> str
    def list_users(self) -> list
    def create_user(self, username, password, role="user", display_name="") -> tuple
    def delete_user(self, username) -> tuple       # admin 계정 삭제 불가
    def change_password(self, username, new_password) -> tuple
    def record_usage(self, username)
    def check_rate_limit(self, username) -> tuple[bool, int]  # admin: 무제한
    def get_user_stats(self, username) -> dict
```

#### 기본 계정 (최초 실행 시 자동 생성)

| 계정 | 비밀번호 | 역할 |
|------|----------|------|
| admin | admin123 | 관리자 |
| demo | demo123 | 사용자 |

#### `get_user_stats()` 반환 필드
- `total_queries`, `queries_1h`, `queries_24h`
- `rate_limit` (시간당 허용), `remaining_1h` (남은 횟수)

---

### 4. 전역 인스턴스 추가

```python
metrics_collector = MetricsCollector()
user_manager      = UserManager(USERS_FILE, USAGE_LOG_FILE)
```

---

### 5. 로그인 게이트 (신규)

탭 레이아웃 이전에 배치. 미인증 상태면 `st.stop()` 으로 앱 전체 차단.

```python
if not st.session_state.logged_in:
    # 로그인 폼 표시 (username + password)
    # 기본 계정 안내 expander
    # 인증 실패 시 st.stop()
```

세션 상태 추가:
```python
("logged_in",    False),
("current_user", "anonymous"),
("user_role",    "user"),
```

---

### 6. 사이드바 변화

#### 사용자 정보 섹션 (신규)
```
👑 관리자  또는  👤 사용자   ← 역할 뱃지
홍길동 (admin)              ← 표시 이름 + 사용자명
남은 쿼리: 18/20 (1시간)
총 쿼리: 42회
[로그아웃 버튼]
⚠️ 시간당 쿼리 한도 도달 시 경고 표시
```

---

### 7. 챗봇 탭 — Rate Limit 적용 (신규)

```python
# 파이프라인 실행 전
ok, remaining = user_manager.check_rate_limit(st.session_state.current_user)
if not ok and st.session_state.user_role != "admin":
    st.warning(f"시간당 쿼리 한도({RATE_LIMIT_PER_HOUR}회)에 도달했습니다.")
    st.stop()

# 파이프라인 실행 후
user_manager.record_usage(st.session_state.current_user)
```

---

### 8. `run_rag_pipeline()` — user_id 파라미터 추가

```python
def run_rag_pipeline(...,
    user_id: str = "anonymous") -> dict:  # [NEW v22]
# 반환 dict 추가 필드: "user_id": user_id
```

---

### 9. `build_log_entry()` — user_id 필드 추가

```python
def build_log_entry(...,
    user_id: str = "anonymous"):   # [NEW v22]
# 로그 엔트리 추가 필드: "user_id": user_id
```

---

### 10. 탭 레이아웃 변화

```python
tab_chat, tab_trace, tab_agent, tab_ablation, tab_search, tab_failure, tab_v21, \
tab_monitor, tab_users, tab_api = st.tabs([
    "💬 챗봇", "🔬 트레이싱", "🧠 에이전트 분석", "🧬 Ablation",
    "🔍 검색 품질", "🚨 실패 데이터셋", "⚡ v21 분석",
    "📊 모니터링",          # NEW v22
    "👤 사용자 관리",        # NEW v22
    "🔌 API 미리보기",       # NEW v22
])
```

---

### 11. [NEW] TAB 8 — 📊 모니터링

#### 알림 배너
```
🚨 [error]   평균 정확도가 기준(3.0) 이하입니다
⚠️ [warning] 환각률이 30%를 초과했습니다
⚠️ [warning] P95 응답 시간이 15초를 초과했습니다
✅ [success] 모든 지표가 정상 범위입니다
```

#### 핵심 지표 (6개 메트릭)
```
총 쿼리 수 | 24h 쿼리 | 평균 정확도 | 환각률 | 캐시 히트율 | 실패 케이스
```

#### 응답 시간 백분위수
```
평균 응답시간 | P50 | P95 | P99
(delta: P95 > 10,000ms 시 빨간색)
```

#### 차트
- 응답 시간 추이 (최근 30건 line chart)
- 정확도 추이 + 관련성 평균 (line chart)
- 환각률 추이 + Fallback 비율 (line chart)

#### 토큰 사용량
```
총 토큰 | 평균 토큰 | Tool-Augmented 사용 건수
```

#### 사용자별 쿼리 분포
- 사용자명별 bar chart

#### 내보내기
```
[⬇️ Prometheus 형식 내보내기]  → rag_metrics_YYYYMMDD_HHMMSS.txt
[⬇️ JSON 내보내기]             → rag_metrics_YYYYMMDD_HHMMSS.json
```

#### 알림 임계값 테이블 (expander)
```
| 지표 | 현재값 | 임계값 | 상태 |
```

---

### 12. [NEW] TAB 9 — 👤 사용자 관리 (관리자 전용)

비관리자 접근 시: `st.warning("관리자만 접근 가능합니다")`

#### 사용자 목록 테이블
```
사용자명 | 역할 | 전체 쿼리 | 1h 쿼리 | 남은 횟수 | 생성일
```

#### 사용자 생성 폼
```
사용자명 / 비밀번호 / 역할(user|admin) / 표시이름
[사용자 생성] 버튼
```

#### 사용자 삭제
```
삭제할 사용자 선택 dropdown (admin 제외)
[삭제] 버튼
```

#### 비밀번호 변경
```
대상 사용자 / 새 비밀번호
[변경] 버튼
```

#### Rate Limit 현황
```
각 사용자별 progress bar: 사용 N/20회 (1시간)
```

---

### 13. [NEW] TAB 10 — 🔌 API 미리보기

#### 아키텍처 다이어그램
```
Client → FastAPI (JWT) → RAG Core → OpenAI API
                      → FAISS Vector DB
                      → Cache Layer
Prometheus ← /metrics/prometheus
Grafana ← Prometheus
```

#### 내부 4탭

**① FastAPI 서버 코드**
- `pip install fastapi uvicorn python-jose python-multipart` 안내
- 전체 서버 코드 (code_editor):
  - `oauth2_scheme` + `create_access_token()` + `get_current_user()`
  - `POST /auth/login` — JWT 토큰 발급
  - `POST /query` — RAG 쿼리 (인증 필요)
  - `GET /metrics` — JSON 지표
  - `GET /metrics/prometheus` — Prometheus 텍스트
  - `GET /users` — 사용자 목록 (관리자 전용)
  - `GET /health` — 헬스 체크

**② API 엔드포인트 명세**
- pandas DataFrame 테이블:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /auth/login | ❌ | JWT 토큰 발급 |
| POST | /query | ✅ | RAG 쿼리 실행 |
| GET | /metrics | ✅ | JSON 지표 |
| GET | /metrics/prometheus | ✅ | Prometheus 형식 |
| GET | /users | ✅ Admin | 사용자 목록 |
| GET | /health | ❌ | 헬스 체크 |

- curl 예시 (login / query / metrics)

**③ 인증 플로우**
```
JWT 인증 시퀀스:
1. POST /auth/login → access_token (JWT)
2. 이후 요청: Authorization: Bearer <token>
3. FastAPI: token 검증 → username 추출 → 권한 확인
토큰 만료: 30분 (기본값)
```

**④ Prometheus 연동**
- `prometheus.yml` scrape config 예시
- 수집 메트릭 목록
- Grafana 패널 예시 (응답시간/정확도/환각률)

---

## LLM 호출 횟수

| 상황 | v21 | v22 |
|------|-----|-----|
| 로그인 전 | 동일 | 앱 차단 (0회) |
| 일반 쿼리 | 동일 | 동일 |
| Rate limit 초과 | 동일 | 앱 차단 (0회) |
| 모니터링/사용자관리 탭 | 동일 | 0회 (계산만) |

> v22는 LLM 호출 횟수 변화 없음. MetricsCollector는 기존 로그에서 계산, UserManager는 JSON 파일 읽기/쓰기.

---

## 파일 변경

| 항목 | v21 | v22 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v21.json` | `rag_eval_logs_v22.json` |
| EMBED_CACHE_FILE | `embed_cache_v21.pkl` | `embed_cache_v22.pkl` |
| ANSWER_CACHE_FILE | `answer_cache_v21.json` | `answer_cache_v22.json` |
| FAILURE_DATASET_FILE | `failure_dataset_v21.json` | `failure_dataset_v22.json` |
| USERS_FILE | (없음) | `rag_users_v22.json` (신규) |
| USAGE_LOG_FILE | (없음) | `rag_usage_v22.json` (신규) |

---

## 운영화 로드맵

```
v1~v10:  파이프라인 구축 (임베딩, FAISS, BM25, Self-Refinement)
v11~v19: 평가·최적화·캐시 (8개 필드 평가, QueryResultCache, AnswerCache)
v20:     실패 → 학습 루프 (FailureDataset, Fine-tune JSONL)
v21:     성능 한 끗 (병렬 검색, LongContextReorder, Tool-Augmented RAG)
v22:     운영 준비 (모니터링, 인증, FastAPI 미리보기) ← 현재
```

### v22 핵심 원리

```
[인증] 로그인 게이트 → 사용자별 rate limit → 쿼리 실행
    ↓
[기록] user_id 포함 로그 저장
    ↓
[모니터링] MetricsCollector.compute_from_logs()
    → P50/P95/P99 / 정확도 / 환각률 / 캐시히트율
    → 임계값 초과 시 알림 배너
    → Prometheus 형식 내보내기 → Grafana 연동 가능

[FastAPI 미리보기]
    → Streamlit 앱 코드를 그대로 임포트
    → JWT 인증 래핑
    → 동일 RAG 코어 재사용
```

이것이 "개인 실험 앱 → 운영 가능한 멀티유저 서비스"로의 전환점:
- 누가 언제 얼마나 쿼리했는지 추적
- 시스템 건강 지표 실시간 모니터링
- Prometheus/Grafana 표준 모니터링 스택 연동 준비
- FastAPI 전환 시 코드 재사용 경로 확보
