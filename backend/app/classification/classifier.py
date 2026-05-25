import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.classification.cuad_labels import CUAD_LABELS, CUAD_LABEL_DESCRIPTIONS
from app.config import settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class ClauseClassifier:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.label_embeddings = None
        self.labels = CUAD_LABELS
        self._build_label_embeddings()

    def _build_label_embeddings(self):
        label_texts = []
        for label in self.labels:
            desc = CUAD_LABEL_DESCRIPTIONS.get(label, label)
            label_texts.append(f"{label}: {desc}")
        embeddings = self.embedding_service.encode(label_texts)
        self.label_embeddings = embeddings
        logger.info(f"Built {len(self.labels)} label embeddings")

    def classify(self, clause_text: str, top_k: int = 3) -> List[dict]:
        clause_emb = self.embedding_service.encode_single(clause_text)
        similarities = []
        for i, label_emb in enumerate(self.label_embeddings):
            sim = self.embedding_service.cosine_similarity(clause_emb, label_emb)
            similarities.append((self.labels[i], float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [
            {
                "clause_type": label,
                "confidence_score": round(score, 4),
            }
            for label, score in similarities[:top_k]
        ]
        return results

    def classify_best(self, clause_text: str, min_confidence: float = 0.15) -> Tuple[Optional[str], float]:
        results = self.classify(clause_text, top_k=1)
        if not results:
            return None, 0.0
        best = results[0]
        if best["confidence_score"] < min_confidence:
            return None, best["confidence_score"]
        return best["clause_type"], best["confidence_score"]
