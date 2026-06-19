import logging
import re
from typing import Dict, List, Optional

import numpy as np

from app.classification import ClauseClassifier
from app.config import settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

INTENT_KEYWORDS = {
    "notice_period": ["notice period", "notice days", "notice of termination", "written notice", "notice requirement"],
    "liability": ["liability", "liable", "indemnify", "indemnification", "damages", "losses"],
    "termination": ["termination", "terminate", "terminated", "terminating", "right to end"],
    "payment": ["payment", "pay", "paid", "invoice", "fee", "fees", "compensation", "amount due"],
    "confidentiality": ["confidential", "confidentiality", "non-disclosure", "proprietary", "trade secret"],
    "non_compete": ["non-compete", "non compete", "noncompetition", "restrictive covenant", "competition"],
    "governing_law": ["governing law", "governed by", "choice of law", "jurisdiction", "venue"],
    "arbitration": ["arbitration", "arbitrate", "arbitral", "dispute resolution", "binding arbitration"],
    "insurance": ["insurance", "insured", "coverage", "policy", "indemnify and hold harmless"],
    "indemnification": ["indemnification", "indemnify", "hold harmless", "defend"],
    "assignment": ["assignment", "assign", "assigns", "assignee", "assignor", "transfer"],
    "data_protection": ["data protection", "data privacy", "personal data", "gdpr", "privacy"],
}


class SearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        classifier: ClauseClassifier,
    ):
        self.embedding_service = embedding_service
        self.classifier = classifier

    def search(
        self, query: str, clauses: List[dict], top_k: int = 5,
        contract_id: Optional[str] = None,
    ) -> List[Dict]:
        query = query.strip()
        if not query:
            return []

        intent_clauses = self._detect_intent_filter(query)
        if intent_clauses:
            filtered = [
                c
                for c in clauses
                if c.get("clause_type")
                and c["clause_type"].lower().replace(" ", "_") in intent_clauses
            ]
            if filtered:
                clauses = filtered

        exact_results = self._exact_match_search(query, clauses)
        if exact_results:
            return exact_results[:top_k]

        # Try ChromaDB semantic search first
        chroma_results = self._chroma_search(query, top_k, contract_id)
        if chroma_results:
            return chroma_results

        # Fallback to in-memory embedding search
        return self._fallback_embedding_search(query, clauses, top_k)

    def _chroma_search(
        self, query: str, top_k: int, contract_id: Optional[str] = None,
    ) -> Optional[List[Dict]]:
        try:
            from app.embeddings.vector_store import get_vector_store

            vs = get_vector_store()
            if vs.count() == 0:
                return None

            query_emb = self.embedding_service.encode_single(query)
            where = {"contract_id": contract_id} if contract_id else None
            results = vs.search(
                query_embedding=query_emb.tolist(),
                top_k=top_k,
                where=where,
            )
            if not results:
                return None

            return [
                {
                    "clause_index": int(r["metadata"].get("clause_index", 0)),
                    "clause_type": r["metadata"].get("clause_type", ""),
                    "clause_text": (r.get("document") or "")[:500],
                    "relevance_score": round(1.0 - r["score"], 4),
                }
                for r in results
            ]
        except Exception as e:
            logger.debug("ChromaDB search failed, falling back: %s", e)
            return None

    def _fallback_embedding_search(self, query: str, clauses: List[dict], top_k: int) -> List[Dict]:
        query_emb = self.embedding_service.encode_single(query)
        results = []

        for clause in clauses:
            clause_text = clause.get("clause_text", "")
            clause_emb = self.embedding_service.encode_single(clause_text)
            sim = self.embedding_service.cosine_similarity(query_emb, clause_emb)
            results.append(
                {
                    "clause_index": clause.get("clause_index"),
                    "clause_title": clause.get("clause_title"),
                    "clause_type": clause.get("clause_type"),
                    "clause_text": clause_text[:500],
                    "page_number": clause.get("page_number"),
                    "relevance_score": round(float(sim), 4),
                }
            )

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def _detect_intent_filter(self, query: str) -> List[str]:
        query_lower = query.lower()
        matched_intents = []
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched_intents.append(intent)
                    break
        return matched_intents

    def _exact_match_search(self, query: str, clauses: List[dict]) -> List[Dict]:
        query_lower = query.lower().strip()
        results = []

        for clause in clauses:
            text_lower = clause.get("clause_text", "").lower()
            title_lower = (clause.get("clause_title") or "").lower()
            clause_type_lower = (clause.get("clause_type") or "").lower()
            combined = f"{text_lower} {title_lower} {clause_type_lower}"

            query_terms = query_lower.split()

            if any(term in combined for term in query_terms):
                match_count = sum(1 for t in query_terms if t in combined)
                relevance = match_count / max(len(query_terms), 1)
                # Boost if all terms found
                if all(term in combined for term in query_terms):
                    relevance = max(relevance, 0.9)

                results.append(
                    {
                        "clause_index": clause.get("clause_index"),
                        "clause_title": clause.get("clause_title"),
                        "clause_type": clause.get("clause_type"),
                        "clause_text": clause.get("clause_text", "")[:500],
                        "page_number": clause.get("page_number"),
                        "relevance_score": round(relevance, 4),
                    }
                )

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results
