import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def similarity_matrix(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        a_norm = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        b_norm = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm == 0:
            return 0.0
        return float(dot / norm)
