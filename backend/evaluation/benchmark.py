"""
Compare ML vs heuristic performance across all components.

Usage:
    python -m evaluation.benchmark
    python -m evaluation.benchmark --quick
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.evaluation.benchmark")

TEST_CONTRACT = """
MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is entered into as of June 1, 2025
by and between Acme Corporation ("Client") and Beta Solutions Inc. ("Provider").

1. SERVICES. Provider shall perform the services described in each Statement of Work.
   Provider shall maintain commercial general liability insurance of $2,000,000.

2. PAYMENT. Client shall pay all invoices within 30 calendar days of receipt.
   Late payments shall accrue interest at 1.5% per month.

3. CONFIDENTIALITY. Provider shall hold all Confidential Information in confidence
   for a period of 5 years. This obligation shall survive termination.

4. INTELLECTUAL PROPERTY. All work product created under this Agreement shall be
   owned exclusively by Client. Provider assigns all rights, title, and interest.

5. LIMITATION OF LIABILITY. In no event shall either party's aggregate liability
   exceed $1,000,000. This limitation does not apply to indemnification obligations.

6. TERMINATION. Either party may terminate this Agreement upon 90 days written
   notice. Either party may terminate for cause immediately upon written notice.

7. INDEMNIFICATION. Provider shall indemnify and hold harmless Client from all
   third-party claims arising from Provider's negligence.

8. GOVERNING LAW. This Agreement shall be governed by the laws of the State
   of New York.
"""


def benchmark_segmentation():
    from app.segmentation.clause_segmenter import ClauseSegmenter

    segmenter = ClauseSegmenter()
    start = time.time()
    clauses = segmenter.segment(TEST_CONTRACT)
    elapsed = time.time() - start

    return {
        "component": "Clause Segmentation",
        "method": "Regex + spaCy fallback",
        "count": len(clauses),
        "time_ms": round(elapsed * 1000, 1),
    }


def benchmark_classification():
    from app.classification import ClauseClassifier

    classifier = ClauseClassifier()
    clauses = TEST_CONTRACT.split("\n\n")

    start = time.time()
    results = []
    for clause in clauses:
        clause = clause.strip()
        if clause:
            result = classifier.classify_best(clause)
            results.append(result)
    elapsed = time.time() - start

    return {
        "component": "Clause Classification",
        "method": "Keyword + embedding fusion",
        "count": len(results),
        "time_ms": round(elapsed * 1000, 1),
        "details": [
            {"clause": c[:50], "predicted": r[0], "confidence": round(r[1], 3)}
            for c, r in zip(clauses, results)
        ],
    }


def benchmark_extraction():
    from app.extraction import AttributeExtractor

    extractor = AttributeExtractor()
    clauses = TEST_CONTRACT.split("\n\n")

    start = time.time()
    total_attrs = 0
    for clause in clauses:
        clause = clause.strip()
        if clause:
            attrs = extractor.extract(clause, "Termination")
            total_attrs += len(attrs)
    elapsed = time.time() - start

    return {
        "component": "Attribute Extraction",
        "method": "Regex + semantic fallback",
        "attributes_found": total_attrs,
        "time_ms": round(elapsed * 1000, 1),
    }


def benchmark_outlier_detection():
    from app.outlier.outlier_detector import OutlierDetector

    detector = OutlierDetector()
    start = time.time()
    result = detector.detect_outliers(
        clause={"clause_text": "This agreement may be terminated upon 90 days notice.", "clause_index": 0, "clause_type": "Termination"},
        attributes={"notice_days": 90},
        clause_type="Termination",
    )
    elapsed = time.time() - start

    return {
        "component": "Outlier Detection",
        "method": "Statistical (percentile-based)",
        "outliers_found": len(result),
        "time_ms": round(elapsed * 1000, 1),
    }


def benchmark_risk_engine():
    from app.risk import RiskEngine

    engine = RiskEngine()

    findings = [
        {"clause_type": "Termination", "rule_name": "Termination for Convenience", "points": 5},
        {"clause_type": "Payment Terms", "rule_name": "Late Payment Penalty", "points": 10},
        {"clause_type": "Liability", "rule_name": "Liability Cap Below Threshold", "points": 15},
    ]

    start = time.time()
    score = engine.calculate_risk_score(findings)
    level = engine.get_risk_level(score)
    elapsed = time.time() - start

    return {
        "component": "Risk Scoring",
        "method": "Rule-based (22 rules)",
        "score": score,
        "level": level,
        "time_ms": round(elapsed * 1000, 1),
    }


def benchmark_contract_type():
    from app.risk.contract_type_detector import detect_contract_type

    start = time.time()
    result = detect_contract_type(TEST_CONTRACT)
    elapsed = time.time() - start

    return {
        "component": "Contract Type Detection",
        "method": "Keyword scoring",
        "detected_type": result.get("contract_type"),
        "confidence": round(result.get("confidence", 0), 3),
        "time_ms": round(elapsed * 1000, 1),
    }


def benchmark_embedding():
    from app.embeddings import EmbeddingService

    svc = EmbeddingService()

    texts = [clause.strip() for clause in TEST_CONTRACT.split("\n\n") if clause.strip()]

    start = time.time()
    embeddings = svc.encode(texts)
    elapsed = time.time() - start

    return {
        "component": "Embedding Generation",
        "method": "Sentence-transformers or hashing",
        "texts_encoded": len(texts),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "time_ms": round(elapsed * 1000, 1),
    }


def run_all(quick=False):
    benchmarks = [
        benchmark_segmentation,
        benchmark_classification,
        benchmark_extraction,
        benchmark_outlier_detection,
        benchmark_risk_engine,
        benchmark_contract_type,
    ]

    if not quick:
        benchmarks.insert(0, benchmark_embedding)

    results = []
    for bench_fn in benchmarks:
        try:
            r = bench_fn()
            results.append(r)
            logger.info(f"  {r['component']:<35} {r['method']:<30} {r.get('time_ms', 0):>8.1f}ms")
        except Exception as e:
            logger.warning(f"  {bench_fn.__name__} failed: {e}")

    total_time = sum(r.get("time_ms", 0) for r in results)
    logger.info(f"\n  {'TOTAL':<35} {'':<30} {total_time:>8.1f}ms")

    output_path = Path(__file__).resolve().parent.parent / "models" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "benchmarks": results,
            "total_time_ms": total_time,
            "test_contract_length": len(TEST_CONTRACT),
        }, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML vs heuristic performance")
    parser.add_argument("--quick", action="store_true", help="Skip embedding benchmark")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Running ML pipeline benchmark...")
    results = run_all(quick=args.quick)
    logger.info(f"Benchmark results saved to models/benchmark_results.json")


if __name__ == "__main__":
    main()
