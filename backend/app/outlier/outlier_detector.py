import logging
from typing import Dict, List, Optional

import numpy as np

from app.benchmarking import BenchmarkingEngine
from app.classification import ClauseClassifier
from app.config import settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class OutlierDetector:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        classifier: Optional[ClauseClassifier] = None,
        benchmarking_engine: Optional[BenchmarkingEngine] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.classifier = classifier or ClauseClassifier(self.embedding_service)
        self.benchmarking = benchmarking_engine or BenchmarkingEngine(
            self.embedding_service, self.classifier
        )

    def detect_outliers(self, clause: dict, attributes: dict, clause_type: str) -> List[Dict]:
        outliers = []

        benchmarks = self.benchmarking.benchmark_attributes(
            clause, attributes, clause_type
        )

        for bench in benchmarks:
            outlier_findings = self._evaluate_outlier(bench, clause, clause_type)
            if outlier_findings:
                outliers.append(outlier_findings)

        peers = self.benchmarking.retrieve_similar_peers(
            clause.get("clause_text", ""), clause_type, top_k=10
        )
        if peers:
            clause_emb = self.embedding_service.encode_single(
                clause.get("clause_text", "")
            )
            avg_similarity = np.mean([p.get("similarity_score", 0) for p in peers])
            if avg_similarity < 0.5:
                outliers.append(
                    {
                        "type": "semantic_outlier",
                        "attribute": "clause_similarity",
                        "severity": "Medium",
                        "contract_value": f"{avg_similarity:.2f} avg similarity",
                        "market_median": 0.75,
                        "percentile_rank": 5.0,
                        "z_score": -2.5,
                        "explanation": (
                            "This clause is semantically dissimilar to typical clauses of the same type "
                            "in the reference contract corpus. The language or structure may be unusual."
                        ),
                    }
                )

        return outliers

    def _evaluate_outlier(self, benchmark: dict, clause: dict, clause_type: str) -> Optional[Dict]:
        percentile = benchmark.get("percentile_rank", 50)
        z_score = benchmark.get("z_score", 0)

        is_outlier = False
        severity = "Low"
        explanation = ""

        if percentile >= 95 or z_score >= 1.96:
            is_outlier = True
            severity = "High"
            explanation = self._generate_outlier_explanation(
                benchmark, "significantly above", "above the 95th percentile"
            )
        elif percentile >= 90 or z_score >= 1.645:
            is_outlier = True
            severity = "Medium"
            explanation = self._generate_outlier_explanation(
                benchmark, "above", "above the 90th percentile"
            )
        elif percentile <= 5 or z_score <= -1.96:
            is_outlier = True
            severity = "Medium"
            explanation = self._generate_outlier_explanation(
                benchmark, "significantly below", "below the 5th percentile"
            )
        elif abs(z_score) >= 2.0:
            is_outlier = True
            severity = "High"
            explanation = self._generate_outlier_explanation(
                benchmark, "significantly deviating from", "beyond 2 standard deviations from the mean"
            )

        if is_outlier:
            return {
                "type": "statistical_outlier",
                "attribute": benchmark.get("attribute", "unknown"),
                "severity": severity,
                "contract_value": benchmark.get("contract_value", "N/A"),
                "market_median": benchmark.get("market_median"),
                "market_p95": benchmark.get("market_p95"),
                "market_p5": benchmark.get("market_p5"),
                "percentile_rank": benchmark.get("percentile_rank"),
                "z_score": benchmark.get("z_score"),
                "peer_count": benchmark.get("peer_count"),
                "clause_type": clause_type,
                "explanation": explanation,
            }

        return None

    def _generate_outlier_explanation(
        self, benchmark: dict, direction: str, statistical_ref: str
    ) -> str:
        attr = benchmark.get("attribute", "attribute")
        contract_val = benchmark.get("contract_value", "N/A")
        median_val = benchmark.get("market_median", "N/A")
        pct = benchmark.get("percentile_rank", 50)
        z = benchmark.get("z_score", 0)

        attr_display = attr.replace("_", " ").title()

        return (
            f"{attr_display} value ({contract_val}) is {direction} market norms "
            f"(median: {median_val}, percentile rank: {pct}%, z-score: {z}). "
            f"This contract's value is {statistical_ref} of similar contracts, "
            f"indicating a potentially unusual or non-standard term."
        )
