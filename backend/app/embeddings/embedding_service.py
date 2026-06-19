import logging
import hashlib
import re
import uuid
from typing import Dict, List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class HashingEmbeddingModel:
    """Small deterministic embedding fallback for local development and tests."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        return np.array([self._encode_one(text) for text in texts], dtype=np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.model = self._load_model()
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding backend ready: %s (%s dimensions)", self.model_name, self.dimension)

    def _load_model(self):
        if settings.embedding_backend == "hashing":
            self.model_name = "hashing"
            return HashingEmbeddingModel(settings.embedding_dim)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            if settings.embedding_backend == "sentence-transformers":
                raise
            logger.warning(
                "sentence-transformers is not installed; using deterministic hashing embeddings. "
                "Install backend/requirements-ml.txt and set EMBEDDING_BACKEND=sentence-transformers "
                "for semantic embeddings."
            )
            self.model_name = "hashing"
            return HashingEmbeddingModel(settings.embedding_dim)

        logger.info("Loading sentence-transformers model: %s", self.model_name)
        return SentenceTransformer(self.model_name)

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

    def store_in_vector_db(
        self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]
    ) -> List[str]:
        embedding_ids = []
        try:
            from app.embeddings.vector_store import get_vector_store

            vs = get_vector_store()
            emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
            embedding_ids = [str(uuid.uuid4()) for _ in texts]
            vs.add_embeddings(
                embedding_ids=embedding_ids,
                embeddings=emb_list,
                metadatas=metadatas,
                documents=texts,
            )
        except Exception as e:
            logger.warning("Failed to store embeddings in vector DB: %s", e)
        return embedding_ids

    def delete_from_vector_db(self, embedding_ids: List[str]):
        if not embedding_ids:
            return
        try:
            from app.embeddings.vector_store import get_vector_store

            vs = get_vector_store()
            vs.delete_embeddings(embedding_ids)
        except Exception as e:
            logger.warning("Failed to delete embeddings from vector DB: %s", e)
