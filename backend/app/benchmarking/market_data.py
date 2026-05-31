import logging
import random
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketNormDatabase:
    def __init__(self):
        self._norms = self._build_default_norms()
        self._seed_peers()

    def _build_default_norms(self) -> Dict:
        return {
            "Termination": {
                "notice_days": {
                    "values": self._generate_distribution(30, 60, 90, 100),
                    "unit": "days",
                    "description": "Notice period required for termination",
                }
            },
            "Payment Terms": {
                "payment_deadline_days": {
                    "values": self._generate_distribution(15, 30, 60, 100),
                    "unit": "days",
                    "description": "Payment due within N days of invoice",
                },
                "penalty_percentage": {
                    "values": self._generate_distribution(0.5, 1.0, 2.0, 80),
                    "unit": "percent",
                    "description": "Late payment penalty rate",
                },
            },
            "Non-Compete": {
                "non_compete_months": {
                    "values": self._generate_distribution(6, 12, 18, 80),
                    "unit": "months",
                    "description": "Non-compete restriction duration",
                }
            },
            "Liability": {
                "liability_cap": {
                    "values": self._generate_distribution(500000, 2000000, 10000000, 80),
                    "unit": "USD",
                    "description": "Liability cap amount",
                }
            },
            "Limitation of Liability": {
                "liability_cap": {
                    "values": self._generate_distribution(500000, 2000000, 10000000, 80),
                    "unit": "USD",
                    "description": "Liability cap amount",
                }
            },
            "Confidentiality": {
                "notice_days": {
                    "values": self._generate_distribution(30, 90, 180, 60),
                    "unit": "days",
                    "description": "Confidentiality obligation period",
                }
            },
            "Insurance": {
                "insurance_amount": {
                    "values": self._generate_distribution(1000000, 5000000, 20000000, 60),
                    "unit": "USD",
                    "description": "Minimum insurance coverage amount",
                }
            },
            "Indemnification": {
                "liability_cap": {
                    "values": self._generate_distribution(500000, 2000000, 10000000, 50),
                    "unit": "USD",
                    "description": "Indemnification cap",
                }
            },
        }

    def _generate_distribution(
        self, p25: float, p50: float, p75: float, count: int
    ) -> List[float]:
        random.seed(42)
        values = []
        for _ in range(count):
            base = random.gauss(p50, (p75 - p25) / 1.35)
            base = max(base, p25 * 0.5)
            values.append(round(base, 2))
        return sorted(values)

    def _seed_peers(self):
        self._peer_clauses = {}
        for clause_type, norms in self._norms.items():
            peers = []
            for i in range(20):
                peers.append(
                    {
                        "peer_id": f"{clause_type.lower()}_{i}",
                        "clause_type": clause_type,
                        "text": f"[Seeded peer clause {i} for {clause_type}]",
                        "embedding": np.random.randn(384).tolist(),
                    }
                )
            self._peer_clauses[clause_type] = peers

    def get_norms_for_type(self, clause_type: str) -> Dict:
        return self._norms.get(clause_type, {})

    def get_similar_clauses(
        self, clause_type: str, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict]:
        peers = self._peer_clauses.get(clause_type, [])
        if not peers:
            return []

        if len(peers) > top_k:
            random.seed(42)
            selected = random.sample(peers, min(top_k, len(peers)))
        else:
            selected = peers

        return [
            {
                "peer_id": p["peer_id"],
                "clause_type": p["clause_type"],
                "similarity_score": round(random.uniform(0.6, 0.95), 4),
            }
            for p in selected
        ]

    def has_norms_for(self, clause_type: str) -> bool:
        return clause_type in self._norms
