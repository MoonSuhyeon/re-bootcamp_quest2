# rag_app_v20.py 변동사항 요약

## 핵심 변경: 실패 케이스 저장 + 데이터셋화 + Auto Improvement

v19의 캐시/압축/멀티벡터 위에,
**실패 케이스를 자동으로 분류·저장**하고
**fine-tune 데이터셋 / 청크 개선 힌트로 내보내는**
"로그 → 데이터셋화 → 개선" 루프를 추가.

> 여기서 진짜 차이: 단순 로그가 아니라 **재사용 가능한 학습 데이터**로 전환

---

## 최종 스택

| 구성 | v19 (rag_app_v19.py) | v20 (rag_app_v20.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 방식 | Multi-Vector + BM25 RRF | 동일 |
| 평가 시스템 | 구조화 8개 필드 | 동일 |
| 캐시 전략 | QueryResultCache + AnswerCache | 동일 |
| Context Compression | 임베딩 유사도 문장 추출 | 동일 |
| Self-Refinement | Draft → Critique → Refine | 동일 |
| Fallback | 평가 기반 자동 재시도 | 동일 |
| **실패 감지** | (없음) | **`classify_failure_types()` 자동 분류** |
| **실패 저장** | 일반 로그에 포함 | **`FailureDataset` 전용 파일 분리** |
| **개선 힌트** | (없음) | **`generate_improvement_hint()` LLM 분석** |
| **데이터셋 내보내기** | (없음) | **JSONL(fine-tune) + JSON(문제 분석)** |

---

## 변경 사항

### 1. 파일 상수 추가

```python
FAILURE_DATASET_FILE      = os.path.join(_BASE, "failure_dataset_v20.json")
FAILURE_THRESHOLD_ACCURACY = 3   # accuracy_score <= 3 → 실패 기준
```

---

### 2. `FailureDataset` — 실패 케이스 전용 저장소 (신규)

```python
class FailureDataset:
    def add(self, entry: dict)
    def get_all(self) -> list
    def get_by_type(self, failure_type: str) -> list
    def size(self) -> int
    def clear(self)
    def export_finetune_jsonl(self) -> bytes   # OpenAI fine-tuning 형식
    def export_problems_json(self) -> bytes    # 문제 분석 요약 JSON
```

- `failure_dataset_v20.json` 파일 기반 영구 저장
- `export_finetune_jsonl()` — OpenAI fine-tune API 직접 사용 가능한 JSONL
- `export_problems_json()` — 질문·실패유형·개선힌트 요약 JSON

#### Fine-tune JSONL 스키마
```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "[참고 문서]\n...\n\n[질문]\n..."},
    {"role": "assistant", "content": "(improvement_hint 또는 draft answer)"}
  ],
  "_meta": {
    "failure_types": [...],
    "evaluation": {...},
    "timestamp": "..."
  }
}
```

---

### 3. `classify_failure_types()` — 실패 유형 자동 분류 (신규)

```python
def classify_failure_types(evaluation: dict, quality_report: dict,
                            sqr: dict = None) -> list[str]:
```

| 유형 | 기준 |
|------|------|
| `low_accuracy` | 정확도 ≤ 3 |
| `low_relevance` | 관련성 ≤ 3 |
| `hallucination` | 환각여부 = "부분적" 또는 "있음" |
| `incomplete_answer` | 누락_정보 ≠ "없음" |
| `retrieval_failure` | sqr 품질 = "poor" 또는 "fair" |

- 여러 유형이 동시에 해당 가능 (다중 태그)
- LLM 추가 호출 없음 — 기존 평가 결과만 사용

---

### 4. `generate_improvement_hint()` — LLM 개선 힌트 생성 (신규)

```python
def generate_improvement_hint(question, chunks, answer, evaluation,
                               failure_types, tracer=None) -> str:
```

- 실패 유형 + 평가 결과 + 답변 → LLM 1회 추가 호출
- 4가지 관점으로 분석:

```
**실패 원인 분석**
- (구체적 원인)

**청크 개선 제안**
- (어떤 정보가 청크에 추가/수정되어야 하는지)

**쿼리 개선 제안**
- (더 나은 검색을 위한 쿼리 전략)

**개선된 답변 방향**
- (어떤 내용이 포함되어야 하는지)
```

- 사이드바 "개선 힌트 자동 생성" 토글 OFF 시 LLM 호출 생략
- 실패 데이터셋 탭에서 개별 케이스에 대해 사후 생성도 가능

---

### 5. `build_failure_entry()` — 실패 케이스 빌더 (신규)

```python
def build_failure_entry(question, answer, chunks, sources,
                         evaluation, quality_report,
                         failure_types, improvement_hint=None,
                         ndcg=None, sqr=None,
                         mode="", mv_retrieval=False) -> dict:
```

```python
# 저장 스키마
{
    "id":               str (UUID 12자),
    "timestamp":        str,
    "question":         str,
    "answer":           str,
    "chunks":           list[str],       # 생성에 사용된 청크 (압축 후)
    "sources":          list[str],
    "evaluation":       dict,            # 8개 필드 평가
    "quality_report":   dict,            # overall_score + grade + issues
    "failure_types":    list[str],       # 다중 태그
    "improvement_hint": str | None,      # LLM 개선 힌트
    "retrieval_info": {
        "ndcg":          float | None,
        "quality_label": str | None,
        "mode":          str,
        "mv_retrieval":  bool,
    }
}
```

---

### 6. `run_rag_pipeline()` — 실패 자동 저장 파라미터 추가

```python
def run_rag_pipeline(...,
    auto_save_failure: bool = True,
    gen_improvement_hint: bool = False) -> dict:
```

#### 실패 저장 실행 시점 (파이프라인 내)
```
[8] 평가 완료
    ↓
[NEW v20] 실패 분류 (classify_failure_types)
    ↓ failure_types 비어있으면 → 저장 생략
    ↓ failure_types 있으면
      → gen_improvement_hint=True 시 LLM 힌트 생성
      → build_failure_entry() 구성
      → failure_dataset.add() 저장
```

반환 dict 추가 필드:
```python
{
    "failure_types": list[str],  # 실패 유형 목록
    "failure_saved": bool,       # 실패 케이스 저장 여부
}
```

**Fallback 재시도는 저장 제외** (`auto_save_failure=False` 전달) — 최초 시도만 저장

---

### 7. `build_log_entry()` — 신규 필드 추가

```python
def build_log_entry(...,
    failure_types=None,
    failure_saved=False):
```

```python
# 신규 로그 필드
{
    "failure_types": ["low_accuracy", "hallucination", ...],
    "failure_saved": True | False,
}
```

---

### 8. 사이드바 변화

#### 🚨 실패 데이터셋 설정 섹션 (신규)
```
🚨 실패 데이터셋 설정
├─ 실패 케이스 자동 저장 토글 (기본값: ON)
│    정확도 ≤ 3 또는 환각/검색 실패 시 자동 저장
├─ 개선 힌트 자동 생성 토글 (기본값: OFF, +1 LLM 호출)
│    실패 시 청크/쿼리/답변 개선 방향을 LLM으로 자동 분석
└─ 저장된 실패 케이스: N건
```

추가 버튼:
```
[실패 데이터셋 초기화]  ← failure_dataset_v20.json 삭제
```

---

### 9. 챗봇 탭 변화

#### 🚨 실패 저장 배너 (신규)
```
🚨 실패 케이스 저장됨 — 낮은 정확도 · 환각 | 🚨 실패 데이터셋 탭에서 확인/내보내기 가능
```

#### 시도 완료 status에 🚨 태그 추가
```
⚠️ 시도 1 완료 🚨 | 정확도 2/5 · 환각 있음 · 등급 F → Fallback 예정
```

---

### 10. 트레이싱 탭 변화

#### 요약 지표 6번째 → `🚨 실패 저장 N건`

#### 트레이스 expander 제목에 `🚨` 태그 추가

---

### 11. 에이전트 분석 탭 변화

#### 🚨 실패 케이스 분석 섹션 (신규)
```
총 실패 케이스 N건 | 낮은 정확도 N건 | 환각 N건 | 누락 정보 N건 | 검색 실패 N건
실패 유형 분포 bar chart
```

#### per-log expander에 실패 정보 추가
```
🚨 실패 저장: low_accuracy · hallucination  ← 실패 유형 표시
col_b: 🚨 실패 → "저장됨" metric
```

---

### 12. 검색 품질 탭 데이터 테이블

`실패 저장` 컬럼 추가 (`🚨` 표시)

---

### 13. [NEW] TAB 6 — 🚨 실패 데이터셋 탭 (완전 신규)

#### 요약 대시보드
```
총 실패 케이스 | 낮은 정확도 | 환각 발생 | 누락 정보 | 검색 실패
실패 유형 분포 bar chart
```

#### 필터 + 정렬
```
실패 유형 필터: 전체 / 낮은 정확도 / 환각 / 누락 정보 / 검색 실패 / 낮은 관련성
정렬 기준: 최신순 / 정확도 낮은순 / 오래된순
```

#### 내보내기 버튼
```
[⬇️ Fine-tune JSONL 내보내기 (OpenAI 형식)]
    → failure_finetune_YYYYMMDD_HHMMSS.jsonl
    → OpenAI fine-tuning API 직접 사용 가능

[⬇️ 문제 분석 JSON 내보내기]
    → failure_problems_YYYYMMDD_HHMMSS.json
    → 질문·실패유형·평가·개선힌트 요약
```

#### 케이스별 상세 (4탭 내부 탭)
```
📋 개요    → 질문 / 실패 유형 / 평가 점수 / 검색 정보 / 품질 리포트
💬 답변    → 실패한 최종 답변 전문
📄 청크    → 생성에 사용된 청크 원문 (출처 포함)
💡 개선 힌트 → LLM 생성 힌트 (없으면 "지금 생성" 버튼)
```

---

## LLM 호출 횟수

| 상황 | v19 | v20 |
|------|-----|-----|
| 실패 저장 OFF | 동일 | 동일 |
| 실패 저장 ON, 힌트 생성 OFF | 동일 | 동일 (분류는 LLM 없음) |
| 실패 저장 ON, 힌트 생성 ON, 실패 발생 시 | 동일 | **+1회** (improvement_hint) |
| 실패 저장 ON, 힌트 생성 ON, 성공 시 | 동일 | 동일 |

> 실패 분류(`classify_failure_types`)는 LLM 호출 없음 — 기존 평가 결과만 활용.
> 개선 힌트는 기본값 OFF — 사용자가 명시적으로 ON 해야 추가 비용 발생.

---

## 파일 변경

| 항목 | v19 | v20 |
|------|-----|-----|
| LOG_FILE | `rag_eval_logs_v19.json` | `rag_eval_logs_v20.json` |
| EMBED_CACHE_FILE | `embed_cache_v19.pkl` | `embed_cache_v20.pkl` |
| ANSWER_CACHE_FILE | `answer_cache_v19.json` | `answer_cache_v20.json` |
| FAILURE_DATASET_FILE | (없음) | `failure_dataset_v20.json` (신규) |

---

## Auto Improvement 루프 전체 그림

```
[실시간] 챗봇 답변 생성
    ↓
[평가] 정확도 / 관련성 / 환각 / 누락_정보 자동 평가
    ↓
[분류] classify_failure_types() — LLM 없이 기준치 비교
    ↓ 실패 기준 충족
[저장] FailureDataset.add() → failure_dataset_v20.json
    ↓ (선택, 힌트 생성 ON 시)
[분석] generate_improvement_hint() — LLM 1회
        ↓
        청크 개선 제안 + 쿼리 개선 + 답변 방향

[나중에] 실패 데이터셋 탭
    ├─ 실패 유형별 필터 + 분포 확인
    ├─ JSONL 내보내기 → OpenAI fine-tuning 입력
    └─ JSON 내보내기 → 청크/파이프라인 개선 작업 입력
```

### 핵심 원리
```
로그 (단순 기록)
    ↓ classify_failure_types()
실패 케이스 (유형 태그 + 청크 보존)
    ↓ export_finetune_jsonl()
Fine-tune 데이터셋 (학습 가능한 형태)
    ↓ OpenAI fine-tuning
더 나은 모델 → 검색 실패 케이스 개선 → 청크 재구성
```

이것이 "단순 RAG → 자기 개선 시스템"으로의 전환점:
- **v1~v10**: 파이프라인 구축
- **v11~v19**: 평가·최적화·캐시
- **v20**: **실패 → 학습 → 개선 루프**
