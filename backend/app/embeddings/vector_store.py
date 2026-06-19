import logging
import math
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)


COLLECTION_NAME = "clause_embeddings"
_use_chromadb = True


class VectorStore:
    def __init__(self):
        self._collection = None
        self._client = None
        self._available = False
        self._memory_store: Dict[str, Dict] = {}
        self._init_client()

    def _init_client(self):
        global _use_chromadb
        if not _use_chromadb:
            return
        try:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    chroma_client_auth_provider=None,
                ),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info(
                "ChromaDB connected: %s:%s, collection: %s",
                settings.chroma_host, settings.chroma_port, COLLECTION_NAME,
            )
        except Exception as e:
            logger.warning(
                "ChromaDB unavailable (%s). Vector search will use in-memory fallback.",
                e,
            )
            _use_chromadb = False
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def using_chromadb(self) -> bool:
        return self._available

    def add_embeddings(
        self,
        embedding_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        documents: Optional[List[str]] = None,
    ):
        if not embedding_ids:
            return
        if not self._available:
            for idx, embedding_id in enumerate(embedding_ids):
                self._add_memory(
                    embedding_id,
                    embeddings[idx],
                    metadatas[idx] if idx < len(metadatas) else {},
                    documents[idx] if documents and idx < len(documents) else None,
                )
            return
        try:
            self.collection.add(
                ids=embedding_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        except Exception as e:
            logger.warning("Failed to add embeddings to ChromaDB: %s", e)
            for idx, embedding_id in enumerate(embedding_ids):
                self._add_memory(
                    embedding_id,
                    embeddings[idx],
                    metadatas[idx] if idx < len(metadatas) else {},
                    documents[idx] if documents and idx < len(documents) else None,
                )

    def add_single(
        self,
        embedding_id: str,
        embedding: List[float],
        metadata: Dict,
        document: Optional[str] = None,
    ):
        if not self._available:
            self._add_memory(embedding_id, embedding, metadata, document)
            return
        try:
            self.collection.add(
                ids=[embedding_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document] if document else None,
            )
        except Exception as e:
            logger.warning("Failed to add single embedding: %s", e)
            self._add_memory(embedding_id, embedding, metadata, document)

    @property
    def collection(self):
        if self._collection is None and _use_chromadb:
            self._init_client()
        return self._collection

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self._available:
            return self._memory_search(query_embedding, top_k, where)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
            output = []
            if not results["ids"]:
                return output
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "score": results["distances"][0][i] if results["distances"] else 0.0,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else None,
                })
            return output
        except Exception as e:
            logger.warning("ChromaDB search failed: %s", e)
            return self._memory_search(query_embedding, top_k, where)

    def delete_embeddings(self, embedding_ids: List[str]):
        if not embedding_ids:
            return
        for embedding_id in embedding_ids:
            self._memory_store.pop(embedding_id, None)
        if not self._available:
            return
        try:
            self.collection.delete(ids=embedding_ids)
        except Exception as e:
            logger.warning("Failed to delete embeddings: %s", e)

    def delete_by_contract(self, contract_id: str):
        self._memory_store = {
            embedding_id: item
            for embedding_id, item in self._memory_store.items()
            if item["metadata"].get("contract_id") != contract_id
        }
        if not self._available:
            return
        try:
            self.collection.delete(where={"contract_id": contract_id})
        except Exception as e:
            logger.warning("Failed to delete contract embeddings: %s", e)

    def get_by_contract(self, contract_id: str) -> List[Dict]:
        if not self._available:
            return [
                {
                    "id": embedding_id,
                    "metadata": item["metadata"],
                    "document": item["document"],
                }
                for embedding_id, item in self._memory_store.items()
                if item["metadata"].get("contract_id") == contract_id
            ]
        try:
            results = self.collection.get(where={"contract_id": contract_id})
            output = []
            if not results["ids"]:
                return output
            for i in range(len(results["ids"])):
                output.append({
                    "id": results["ids"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "document": results["documents"][i] if results["documents"] else None,
                })
            return output
        except Exception as e:
            logger.warning("Failed to get contract embeddings: %s", e)
            return []

    def count(self) -> int:
        if not self._available:
            return len(self._memory_store)
        try:
            return self.collection.count()
        except Exception:
            return len(self._memory_store)

    def health_check(self) -> bool:
        return self._available or self._memory_store is not None

    def _add_memory(
        self,
        embedding_id: str,
        embedding: List[float],
        metadata: Dict,
        document: Optional[str],
    ):
        self._memory_store[embedding_id] = {
            "embedding": list(embedding),
            "metadata": dict(metadata or {}),
            "document": document,
        }

    def _memory_search(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict],
    ) -> List[Dict]:
        scored = []
        for embedding_id, item in self._memory_store.items():
            metadata = item["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            distance = 1.0 - self._cosine_similarity(query_embedding, item["embedding"])
            scored.append(
                {
                    "id": embedding_id,
                    "score": distance,
                    "metadata": metadata,
                    "document": item["document"],
                }
            )
        scored.sort(key=lambda result: result["score"])
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)


_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
