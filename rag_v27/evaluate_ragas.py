#!/usr/bin/env python3
# evaluate_ragas.py — RAGAS 평가 독립 실행 도구   [NEW v26]
#
# 사용법:
#   python evaluate_ragas.py                  # 기본 로그 파일 사용
#   python evaluate_ragas.py --log custom.json
#   python evaluate_ragas.py --last 20        # 최근 20개만 평가
#
# 필요 패키지:
#   pip install ragas datasets openai python-dotenv

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# =====================================================================
# 설정
# =====================================================================

_BASE          = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG    = os.path.join(_BASE, "ragas_log_v26.json")
RESULT_FILE    = os.path.join(_BASE, "ragas_result_v26.json")


# =====================================================================
# 로그 로드
# =====================================================================

def load_ragas_logs(log_file: str) -> list:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"로그 파일 없음: {log_file}\n"
                                f"/chat/stream 엔드포인트를 먼저 사용하세요.")
    with open(log_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# RAGAS 평가
# =====================================================================

def run_ragas_evaluation(logs: list) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError:
        raise ImportError(
            "ragas/datasets 패키지가 없습니다.\n"
            "pip install ragas datasets"
        )

    if not logs:
        raise ValueError("평가할 로그가 없습니다.")

    dataset = Dataset.from_dict({
        "question": [l["question"] for l in logs],
        "answer":   [l["answer"]   for l in logs],
        "contexts": [l["contexts"] for l in logs],
    })

    print(f"평가 중... ({len(logs)}개 쿼리, 메트릭: faithfulness · answer_relevancy · context_precision)")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result


# =====================================================================
# 결과 출력 및 저장
# =====================================================================

def print_report(result, logs: list):
    scores = dict(result)
    print("\n" + "=" * 60)
    print("  RAGAS 평가 결과")
    print("=" * 60)
    print(f"  총 쿼리 수        : {len(logs)}")
    print(f"  Faithfulness      : {scores.get('faithfulness', 0):.4f}")
    print(f"  Answer Relevancy  : {scores.get('answer_relevancy', 0):.4f}")
    print(f"  Context Precision : {scores.get('context_precision', 0):.4f}")

    avg = sum([
        scores.get("faithfulness",      0),
        scores.get("answer_relevancy",  0),
        scores.get("context_precision", 0),
    ]) / 3
    print(f"  ─────────────────────────────────────")
    print(f"  종합 평균         : {avg:.4f}")
    print("=" * 60 + "\n")

    # 최근 5개 샘플 출력
    print("[ 샘플 쿼리 (최대 5개) ]")
    for i, log in enumerate(logs[-5:]):
        print(f"  [{i+1}] {log['question'][:60]}")
        print(f"       → {log['answer'][:80].strip()}...")
        print()


def save_result(result, logs: list):
    scores = dict(result)
    export = {
        "evaluated_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query_count":      len(logs),
        "faithfulness":     round(scores.get("faithfulness",      0), 4),
        "answer_relevancy": round(scores.get("answer_relevancy",  0), 4),
        "context_precision": round(scores.get("context_precision", 0), 4),
        "avg_score":        round(sum([
            scores.get("faithfulness",      0),
            scores.get("answer_relevancy",  0),
            scores.get("context_precision", 0),
        ]) / 3, 4),
        "per_query": [
            {
                "trace_id": l.get("trace_id", ""),
                "question": l["question"],
                "latency_ms": l.get("latency_ms", 0),
            }
            for l in logs
        ],
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {RESULT_FILE}")


# =====================================================================
# 엔트리포인트
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="RAGAS 평가 도구")
    parser.add_argument("--log",  default=DEFAULT_LOG, help="RAGAS 로그 파일 경로")
    parser.add_argument("--last", type=int, default=0,  help="최근 N개만 평가 (0=전체)")
    args = parser.parse_args()

    print(f"로그 파일: {args.log}")
    logs = load_ragas_logs(args.log)
    print(f"총 {len(logs)}개 로그 로드됨")

    if args.last > 0:
        logs = logs[-args.last:]
        print(f"최근 {args.last}개만 평가")

    result = run_ragas_evaluation(logs)
    print_report(result, logs)
    save_result(result, logs)


if __name__ == "__main__":
    main()
