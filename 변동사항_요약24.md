# v24 변동사항 요약

## v23 → v24 핵심 변경점

### 1. config.py 분리 (설정 중앙화)
- **신규 파일**: `config.py`
- 하드코딩된 모든 상수·경로·모델명을 `config.py` 한 곳에서 관리
- 환경변수로 오버라이드 가능 (`.env` 파일 지원)
- `rag_engine.py`에서 `from config import ...` 로 import — 더 이상 파일 안에 상수 없음
- 관리 대상: `LLM_MODEL`, `EMBED_MODEL`, 경로 6개, TTL 2개, 임계값 7개, JWT 설정 3개, API 서버 설정 2개

### 2. Service Layer 분리 (라우터 구조화)
- **신규 파일**: `deps.py`, `routers/__init__.py`, `routers/auth.py`, `routers/docs.py`, `routers/chat.py`, `routers/metrics.py`, `routers/admin.py`
- v23의 단일 `server_api.py` (~400줄) → `server_api.py` (얇은 앱 팩토리) + 5개 라우터로 분리
- 공유 의존성(`INDEX_STATE`, JWT 유틸)은 `deps.py`에 통합

#### 라우터 구조

| 라우터 | prefix | 주요 엔드포인트 |
|--------|--------|----------------|
| `auth.py` | `/auth` | POST /login, POST /register, GET /me |
| `docs.py` | `/docs` | POST /upload, DELETE /reset |
| `chat.py` | `/chat` | POST / (RAG 파이프라인) |
| `metrics.py` | — | GET /metrics, GET /logs, GET /logs/{trace_id}, GET /failures, GET /failures/export/* |
| `admin.py` | `/users` | GET /, POST /, DELETE /{username}, POST /{username}/reset-password, GET /{username}/usage |

## 파일 구성 비교

| v23 | v24 |
|-----|-----|
| `rag_engine.py` (상수 내장) | `rag_engine.py` + `config.py` |
| `server_api.py` (단일 파일) | `server_api.py` + `deps.py` + `routers/` (5개) |
| `client_app.py` | `client_app.py` (동일) |

## 기능 변경 없음
- RAG 파이프라인 로직, 평가 지표, 실패 데이터셋, 메트릭 수집, JWT 인증 등 **모든 기능은 v23과 동일**
- 구조 리팩토링만 수행 (테스트·운영 용이성 향상)

## 포트폴리오 관점 개선 포인트
- 실무에서 FastAPI 프로젝트의 표준 구조(`routers/`, `deps.py`, `config.py`)를 따름
- 환경변수 기반 설정 → 배포 환경(dev/prod)별 모델 교체 가능
- 관심사 분리로 단위 테스트 작성 용이
