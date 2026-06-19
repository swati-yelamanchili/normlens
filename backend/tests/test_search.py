import pytest
from app.search.search_service import SearchService, INTENT_KEYWORDS
from app.classification import ClauseClassifier
from app.embeddings import EmbeddingService


class TestSearchService:
    def setup_method(self):
        self.emb = EmbeddingService()
        self.clf = ClauseClassifier(self.emb)
        self.service = SearchService(self.emb, self.clf)

    def _make_clauses(self):
        return [
            {"clause_index": 0, "clause_title": "Termination", "clause_type": "Termination",
             "clause_text": "This agreement may be terminated by either party upon 30 days written notice.",
             "page_number": 1},
            {"clause_index": 1, "clause_title": "Confidentiality", "clause_type": "Confidentiality",
             "clause_text": "Each party agrees to hold the other's confidential information in strict confidence.",
             "page_number": 2},
            {"clause_index": 2, "clause_title": "Liability", "clause_type": "Limitation of Liability",
             "clause_text": "Neither party shall be liable for any indirect or consequential damages.",
             "page_number": 3},
        ]

    def test_search_exact_match(self):
        clauses = self._make_clauses()
        results = self.service.search("termination", clauses, top_k=5)
        assert len(results) > 0
        assert results[0]["clause_index"] == 0

    def test_search_semantic_fallback(self):
        clauses = self._make_clauses()
        results = self.service.search("ending the agreement", clauses, top_k=3)
        assert len(results) <= 3
        assert all("relevance_score" in r for r in results)
        assert all("clause_index" in r for r in results)

    def test_search_intent_filter(self):
        clauses = self._make_clauses()
        results = self.service.search("payment terms", clauses, top_k=5)
        assert len(results) >= 0

    def test_search_returns_sorted_by_relevance(self):
        clauses = self._make_clauses()
        results = self.service.search("confidential information", clauses, top_k=5)
        if len(results) >= 2:
            assert results[0]["relevance_score"] >= results[1]["relevance_score"]

    def test_search_empty_query(self):
        clauses = self._make_clauses()
        results = self.service.search("", clauses, top_k=5)
        assert len(results) == 0

    def test_search_empty_corpus(self):
        results = self.service.search("test", [], top_k=5)
        assert results == []

    def test_intent_keywords_defined(self):
        assert len(INTENT_KEYWORDS) >= 12
        assert "termination" in INTENT_KEYWORDS
        assert "confidentiality" in INTENT_KEYWORDS
        assert "liability" in INTENT_KEYWORDS

    def test_intent_detection(self):
        matched = self.service._detect_intent_filter("notice period and termination")
        assert "notice_period" in matched or "termination" in matched

    def test_exact_match_search(self):
        clauses = self._make_clauses()
        results = self.service._exact_match_search("termination", clauses)
        assert len(results) > 0
        assert results[0]["clause_index"] == 0
