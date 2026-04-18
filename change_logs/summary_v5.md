# rag_app_v5.py 변동사항 요약

## 핵심 변경: 청킹 품질 개선 (Overlap + Semantic Chunking)

v4의 검색/리랭킹은 유지하면서, 청크 자체의 품질을 높여 Retrieval 성능 근본 개선.

---

## 최종 스택
| 구성 | v4 (rag_app_v4.py) | v5 (rag_app_v5.py) |
|------|---------------------|---------------------|
| 임베딩 | text-embedding-3-small | text-embedding-3-small (동일) |
| LLM | gpt-4o-mini | gpt-4o-mini (동일) |
| 벡터 DB | FAISS - 코사인 유사도 | FAISS - 코사인 유사도 (동일) |
| 청킹 방식 | 문단/문장 단위 (overlap 없음) | 문단/문장 + Overlap **또는** 의미 기반(Semantic) + Overlap |
| Overlap | 없음 | 100자 (기본값, 슬라이더 조절) |
| 쿼리 처리 | 리라이팅 | 리라이팅 (동일) |
| 검색 후처리 | LLM 리랭킹 | LLM 리랭킹 (동일) |

---

## 변경 사항

### 1. Overlap 청킹 추가 (`chunk_text_with_overlap`)
v3/v4의 문단/문장 경계 청킹에 Overlap 로직 추가.
```python
# 변경 전 (v3/v4: overlap 없음)
def chunk_text(text, chunk_size=500):
    # 문단 → 문장 순서로 청크 생성, 끝

# 변경 후 (v5: 생성된 청크에 이전 청크 끝부분을 앞에 붙임)
def chunk_text_with_overlap(text, chunk_size=500, overlap=100):
    raw_chunks = ...  # 기존과 동일한 문단/문장 기반 청크 생성
    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]  # 이전 청크 마지막 100자
        overlapped.append(prev_tail + " " + raw_chunks[i])
```
**문제:** 청크 경계에 걸쳐 있는 정보(예: 앞 청크 끝 문장 + 뒷 청크 첫 문장이 하나의 맥락)는 어느 쪽으로 검색해도 불완전하게 검색됨.
**해결:** 이전 청크의 끝부분을 다음 청크 앞에 중복 포함시켜 경계 손실 방지.

---

### 2. 의미 기반(Semantic) 청킹 추가 (`chunk_text_semantic`)
문장 임베딩의 코사인 유사도로 주제 전환 지점을 자동 감지해 청크 경계를 설정.
```python
def chunk_text_semantic(text, chunk_size=500, overlap=100):
    # 1. 문장 분리
    sentences = re.split(r'(?<=[.!?。])\s+', text)

    # 2. 문장 임베딩 계산
    embeddings = normalize(get_embeddings(sentences))

    # 3. 인접 문장 간 코사인 유사도 계산
    similarities = [np.dot(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    # 4. 유사도가 낮은 지점(평균 - 0.5*표준편차 이하)을 청크 경계로 설정
    threshold = mean_sim - 0.5 * std_sim
    breakpoints = [i+1 for i, sim in enumerate(similarities) if sim < threshold]

    # 5. 경계로 나눈 그룹을 chunk_size에 맞게 묶고 overlap 적용
```
**문제:** 문단/문장 기반 청킹은 텍스트 구조(빈 줄)에만 의존해 의미가 다른 문장이 같은 청크에 묶이거나, 의미적으로 이어지는 문장이 다른 청크로 분리될 수 있음.
**해결:** 임베딩 유사도로 실제 주제 전환 지점을 찾아 의미 단위로 청크를 구성. 구조에 의존하지 않고 내용 기반으로 분리.

---

### 3. 청킹 설정 UI 추가
사이드바에서 청킹 방식·크기·overlap을 직접 조절 가능.
```
- 청킹 방식: [문단/문장 + Overlap] / [의미 기반(Semantic) + Overlap] 라디오 선택
- 청크 크기: 슬라이더 (200~1000, 기본 500)
- Overlap 크기: 슬라이더 (0~200, 기본 100)
```

---

## 청킹 방식 비교
| 항목 | 문단/문장 + Overlap | 의미 기반 + Overlap |
|------|---------------------|----------------------|
| 처리 속도 | 빠름 (임베딩 불필요) | 느림 (문장 수만큼 임베딩 필요) |
| 경계 기준 | 빈 줄 / 문장 부호 | 임베딩 유사도 |
| 구조 있는 문서 | 적합 | 보통 |
| 구조 없는 긴 글 | 보통 | 적합 |
| Overlap | 있음 | 있음 |

---

## 검색 파이프라인 (v5 전체)
```
[문서 업로드]
  → 텍스트 추출
  → 청킹 (문단/문장+Overlap 또는 Semantic+Overlap)
  → FAISS 인덱싱

[질문 입력]
  → [1단계] 쿼리 리라이팅 (1+n개 쿼리)
  → [2단계] 멀티 쿼리 임베딩 검색 (후보 청크 수집)
  → [3단계] LLM 배치 리랭킹 (상위 top_k 선택)
  → [4단계] LLM 답변 생성
```

## 성능 개선 기대 포인트
- Overlap → 청크 경계의 맥락 손실 방지
- Semantic Chunking → 의미 단위 청크로 각 청크의 임베딩 품질 향상
- 청크 품질 자체가 올라가면 리랭킹 전 후보군의 품질도 함께 향상
