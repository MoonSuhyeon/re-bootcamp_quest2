# re-bootcamp_quest2
# 📚 Advanced RAG System (v2 ~ v14 Evolution)

이 프로젝트는 단순한 RAG 구현이 아니라  
검색 품질 → 추론 구조 → 관측 가능성 → 자동 평가 → 실험 기반 개선(Ablation)까지 확장된  
엔드투엔드 RAG 엔지니어링 시스템입니다.

---

# System Evolution

각 버전은 “RAG의 한 문제”를 해결하는 방향으로 확장되었습니다.

---

## 🔹 v2: OpenAI API 기반 전환
- Embedding + GPT를 OpenAI API로 통합
- 로컬 모델 → API 기반 확장

---

## 🔹 v3: 기본 검색 품질 개선
- 코사인 유사도 기반 검색
- 문단/문장 단위 청킹

---

## 🔹 v4: 검색 파이프라인 고도화
- Query Rewriting 추가
- LLM 기반 Reranking 도입
- 검색 정확도 개선

---

## 🔹 v5: Chunking 품질 개선
- Overlap chunking
- Semantic chunking
- 문맥 단절 문제 해결

---

## 🔹 v6: 답변 품질 제어
- 근거 인용 (Source attribution)
- 확신도 기반 출력
- 구조화된 답변 포맷

---

## 🔹 v7: Multi-document Reasoning
- 3단계 추론 파이프라인
- 문서 간 비교 및 종합 reasoning 추가

---

## 🔹 v8: Retrieval 안정성 & Recall 개선
- 의도 분해 Query Rewriting
- Embedding Pre-filter
- (chunk, source) 튜플 구조로 출처 정확성 보장

---

## 🔹 v9: Observability Layer
- LLM 자동 평가 (accuracy / relevance / hallucination)
- 로그 저장 시스템
- Arize-style 대시보드 UI

---

## 🔹 v10: Full Tracing & AI Observability
- Span 기반 Tracing 시스템
- Token usage / latency 측정
- Bottleneck detection
- Hallucination root-cause 분석
- Arize Phoenix 수준 관측성 구현

---

## 🔹 v11: 안정성 버그 수정
- 평가 점수 클램핑 (1~5 보장)
- [출처 N] 클릭 가능한 앵커 링크 구현

---

## 🔹 v12: Hybrid Retrieval + Evaluation
- Dense + BM25 + RRF Hybrid Search
- Embedding Cache 시스템
- NDCG@k 기반 검색 품질 평가

---

## 🔹 v13: Query Routing Engine
- 질문 의도 자동 분류 (6 types)
- 검색 전략 동적 선택
- Retrieval policy 자동화

---

## 🔹 v14: Ablation Study Framework
- 구성 요소별 파이프라인 자동 실험
- Query Rewrite / BM25 / Rerank 기여도 분석
- RAG 시스템을 “실험 가능한 구조”로 전환

