# v26 변동사항 요약

## v25 → v26 핵심 변경점

### 테마: "실시간 응답 + 정량 검증"
답변이 나가는 순간(스트리밍)과 그 답변의 품질(RAGAS)을 하나의 파이프라인으로 연결합니다.

---

### 1. Streaming 응답 (SSE)   [NEW]

`POST /chat/stream` 엔드포인트가 추가됩니다. Server-Sent Events(SSE) 형식으로 토큰을 실시간 전송합니다.

**파이프라인:**
```
검색 (동기) → 스트리밍 생성 (yield token) → RAGAS 로그 저장 → done 이벤트
```

**SSE 이벤트 형식:**
```json
data: {"type": "token",  "content": "답변 토큰"}
data: {"type": "done",   "trace_id": "abc123", "latency_ms": 1200, "sources": [...]}
data: {"type": "error",  "content": "오류 메시지"}
```

**새 rag_engine.py 함수:**
| 함수 | 역할 |
|------|------|
| `retrieve_for_streaming(question, eff, ...)` | 검색만 수행 → (chunks, sources) 반환 |
| `stream_generate_answer(question, chunks)` | OpenAI stream=True 로 토큰 yield |
| `build_ragas_log_entry(...)` | RAGAS 평가용 로그 엔트리 생성 |
| `save_ragas_log(entry)` | `ragas_log_v26.json`에 저장 |

**Streamlit 클라이언트:**
- 채팅 탭에 `⚡ 스트리밍 응답` 토글 추가
- 토글 ON 시: `POST /chat/stream` → 토큰 수신 → 실시간 `placeholder.markdown()` 업데이트

---

### 2. RAGAS Evaluation   [NEW]

**평가 흐름:**
```
스트리밍 응답 완료 → ragas_log_v26.json 자동 저장
          ↓
python evaluate_ragas.py   (오프라인 실행)
          ↓
ragas_result_v26.json   (점수 출력 + 저장)
```

**evaluate_ragas.py (독립 실행 도구):**
```bash
python evaluate_ragas.py               # 전체 로그 평가
python evaluate_ragas.py --last 20     # 최근 20개만
python evaluate_ragas.py --log custom.json
```

**RAGAS 메트릭 (ground_truth 없이 가능한 3가지):**
| 메트릭 | 설명 |
|--------|------|
| Faithfulness | 답변이 검색 문서에 근거하는 정도 (환각 탐지) |
| Answer Relevancy | 답변이 질문에 적절한 정도 |
| Context Precision | 검색된 문서가 실제로 유용한 비율 |

---

### 3. config.py 신규 상수

| 상수 | 값 |
|------|----|
| `RAGAS_LOG_FILE` | `ragas_log_v26.json` |

---

## 파일 변경 요약

| 파일 | 변경 내용 |
|------|-----------|
| `rag_engine.py` | `retrieve_for_streaming`, `stream_generate_answer`, `build_ragas_log_entry`, `save_ragas_log` 추가 |
| `routers/chat.py` | `POST /chat/stream` SSE 엔드포인트 추가; `StreamRequest` 모델 추가 |
| `client_app.py` | 채팅 탭에 스트리밍 토글 + SSE 소비 로직 추가 |
| `evaluate_ragas.py` | **신규** 독립 실행 평가 스크립트 |
| `config.py` | `RAGAS_LOG_FILE` 추가, 파일명 v26으로 업데이트 |
| `server_api.py` | 버전 26.0.0으로 업데이트 |

## 추가 패키지 필요

```bash
pip install ragas datasets
```

## 포트폴리오 관점

- **스트리밍**: "속도"를 사용자가 직접 체감할 수 있는 데모 포인트
- **RAGAS**: Faithfulness/Answer Relevancy/Context Precision 수치로 시스템 신뢰도를 **객관적으로 증명**
- 두 기능이 하나의 데이터 흐름(`저장 → 평가`)으로 연결되어 "측정 가능한 RAG 시스템" 스토리 완성
