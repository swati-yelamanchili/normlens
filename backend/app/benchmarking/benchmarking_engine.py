import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

from app.benchmarking.market_data import MarketNormDatabase
from app.classification import ClauseClassifier
from app.config import settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class BenchmarkingEngine:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        classifier: Optional[ClauseClassifier] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.classifier = classifier or ClauseClassifier(self.embedding_service)
        self.market_data = MarketNormDatabase()

    def benchmark_attributes(
        self,
        clause: dict,
        attributes: dict,
        clause_type: str,
    ) -> List[Dict]:
        benchmarks = []
        market_norms = self.market_data.get_norms_for_type(clause_type)

        for attr_key, attr_value in attributes.items():
            if attr_value is None:
                continue

            numeric_value = self._to_numeric(attr_value)
            if numeric_value is None:
                continue

            norm = market_norms.get(attr_key)
            if norm is None:
                continue

            peer_values = np.array(norm.get("values", []))
            if len(peer_values) == 0:
                continue

            percentile = scipy_stats.percentileofscore(peer_values, numeric_value, kind="mean")

            mean_val = float(np.mean(peer_values))
            std_val = float(np.std(peer_values)) if len(peer_values) > 1 else 1.0
            z_score = (numeric_value - mean_val) / std_val if std_val > 0 else 0.0

            benchmark_result = {
                "clause_type": clause_type,
                "attribute": attr_key,
                "contract_value": str(attr_value),
                "numeric_value": numeric_value,
                "market_median": float(np.median(peer_values)),
                "market_mean": mean_val,
                "market_std": std_val,
                "market_p5": float(np.percentile(peer_values, 5)),
                "market_p25": float(np.percentile(peer_values, 25)),
                "market_p75": float(np.percentile(peer_values, 75)),
                "market_p95": float(np.percentile(peer_values, 95)),
                "percentile_rank": round(float(percentile), 1),
                "z_score": round(float(z_score), 2),
                "peer_count": len(peer_values),
            }
            benchmarks.append(benchmark_result)

        return benchmarks

    def retrieve_similar_peers(
        self, clause_text: str, clause_type: str, top_k: int = 10
    ) -> List[Dict]:
        clause_emb = self.embedding_service.encode_single(clause_text)
        peers = self.market_data.get_similar_clauses(
            clause_type, clause_emb, top_k=top_k
        )
        return peers

    def _to_numeric(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
