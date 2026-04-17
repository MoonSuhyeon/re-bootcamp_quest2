# rag_app_v11.py 변동사항 요약

## 핵심 변경: v10 버그 수정 2건 — 평가 점수 클램핑 + 출처 앵커 링크

v10의 기능은 그대로 유지하면서 사용자가 발견한 두 가지 버그를 수정.

---

## 최종 스택
| 구성 | v10 (rag_app_v10.py) | v11 (rag_app_v11.py) |
|------|----------------------|----------------------|
| 임베딩 | text-embedding-3-small | 동일 |
| LLM | gpt-4o-mini | 동일 |
| 검색 파이프라인 | 의도분해 + Pre-filter + Rerank | 동일 |
| 멀티문서 추론 | Step1→2→3 | 동일 |
| Tracing / Agent Analysis | Tracer 클래스 | 동일 |
| **평가 점수 범위** | 클램핑 없음 → 9/5 등 비정상 출력 가능 | **max(1, min(5, raw)) 클램핑 + 프롬프트 강제** |
| **[출처 N] 링크** | st.markdown → 파란 링크처럼 보이나 작동 안 함 | **HTML 앵커 링크로 변환, DOM 타겟 항상 렌더링** |

---

## 변경 사항

### 1. 평가 점수 클램핑 (`evaluate_answer`) — Bug Fix

#### 문제
v10에서 LLM이 `정확도: 9`, `관련성: 9` 등 1~5 범위를 초과하는 값을 반환할 수 있었음.
파싱 코드에 범위 제한이 없어 그대로 저장·표시되었음.

```python
# v10 (버그): 클램핑 없음
result["정확도"] = int(line.split(':')[1].strip())  # 9가 그대로 통과
result["관련성"] = int(line.split(':')[1].strip())
```

#### 수정
두 가지 방어 레이어 추가:

**① 프롬프트 명시적 제한** — LLM이 처음부터 5 초과 값을 내지 않도록 유도
```python
"【중요】정확도와 관련성은 반드시 1~5 사이의 정수여야 합니다. "
"5를 초과하는 숫자(6, 7, 8, 9, 10 등)는 절대 사용하지 마세요. "
"최고점은 5입니다."
```

**② 파싱 시 max/min 클램핑** — LLM이 규칙을 어겨도 코드가 차단
```python
# v11 (수정): 숫자 추출 후 1~5 강제
raw = int(re.sub(r'[^0-9]', '', line.split(':', 1)[1].strip()))
result["정확도"] = max(1, min(5, raw))
result["관련성"] = max(1, min(5, raw))
```

`re.sub(r'[^0-9]', '', ...)` 추가로 `"4점"`, `" 4 "` 등 비정상 포맷도 안전하게 처리.

---

### 2. `[출처 N]` 출처 링크 수정 — Bug Fix

#### 문제
v10에서 답변의 `[출처 1]`, `[출처 2]` 등이 파란 링크 스타일로 표시되었지만,
클릭해도 아무 동작 없이 깨진 링크 상태였음.

**원인 1**: `st.markdown(response)` 사용 시, `[텍스트]`만 있고 `(URL)` 없는 패턴은
Markdown의 "shortcut reference link"로 해석되어 파란색으로 렌더링되지만 실제 href 없음.

**원인 2**: 출처 원문이 `st.expander("📄 출처 원문")` 안에 있어
expander가 닫힌 상태에서는 DOM에 타겟 요소 자체가 없음.

#### 수정

**① `linkify_citations()` 헬퍼 함수 신규 추가**
```python
def linkify_citations(text: str) -> str:
    """[출처 N]을 <a href="#source-N"> HTML 앵커 링크로 변환"""
    return re.sub(
        r'\[출처 (\d+)\]',
        lambda m: (
            f'<a href="#source-{m.group(1)}" '
            f'style="color:#1976D2;font-weight:bold;text-decoration:none;">'
            f'[출처 {m.group(1)}]</a>'
        ),
        text
    )
```

**② 답변 렌더링 변경**
```python
# v10 (버그): 마크다운이 [출처 N]을 깨진 링크로 처리
st.markdown(response)

# v11 (수정): HTML 링크로 변환 후 unsafe_allow_html=True로 렌더링
st.markdown(linkify_citations(response), unsafe_allow_html=True)
```

**③ 출처 원문 — expander 밖으로 이동 + id 앵커 삽입**
```python
# v10 (버그): expander 안에 있어 닫힌 상태면 DOM에 없음
with st.expander("📄 출처 원문"):
    for i, (c, src, sc) in enumerate(...):
        st.caption(f"[출처 {i+1}] **{src}**")
        st.write(c)

# v11 (수정): expander 밖에 항상 렌더링 + id 앵커 태그 삽입
st.markdown("**📄 출처 원문**")
for i, (c, src, sc) in enumerate(...):
    st.markdown(f'<div id="source-{i+1}"></div>', unsafe_allow_html=True)
    st.caption(f"[출처 {i+1}] **{src}**")
    st.write(c)
```

`<div id="source-1">` → 답변의 `<a href="#source-1">` 클릭 시 해당 위치로 스크롤.

---

## 변경 범위 요약

| 변경 위치 | v10 | v11 |
|-----------|-----|-----|
| `evaluate_answer()` 프롬프트 | 범위 제한 없음 | "1~5 사이, 5 초과 금지" 명시 |
| `evaluate_answer()` 파싱 | `int(...)` 직접 | `max(1, min(5, int(re.sub(...))))` |
| 신규 함수 | 없음 | `linkify_citations(text)` 추가 |
| 답변 렌더링 | `st.markdown(response)` | `st.markdown(linkify_citations(response), unsafe_allow_html=True)` |
| 출처 원문 표시 위치 | `st.expander` 안 | expander 밖 (항상 DOM에 존재) |
| 출처 앵커 타겟 | 없음 | `<div id="source-N">` 삽입 |
| LOG_FILE | `rag_eval_logs_v10.json` | `rag_eval_logs_v11.json` |
| 사이드바 캡션 | `v10: Full Tracing + Agent Analysis` | `v11: 평가 점수 클램핑 + 출처 앵커 링크 수정` |

---

## LLM 호출 횟수
v10과 동일 (변경 없음)

| 단계 | v11 |
|------|-----|
| 쿼리 리라이팅 | 1 |
| Rerank | 1 |
| Step1 요약 | 1 |
| Step2 분석 | 1 |
| Step3 답변 | 1 |
| 평가 | 1 |
| 환각 원인 분석 | +1 (환각 감지 시에만) |
| **합계** | **6~7회** |
