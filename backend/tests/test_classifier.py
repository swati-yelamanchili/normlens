import pytest
from app.classification.classifier import ClauseClassifier
from app.classification.cuad_labels import CUAD_LABELS


class TestClauseClassifier:
    def test_labels_defined(self):
        assert len(CUAD_LABELS) >= 13

    def test_classify_termination(self):
        classifier = ClauseClassifier()
        text = "This agreement may be terminated by either party upon 30 days written notice."
        results = classifier.classify(text, top_k=3)
        assert len(results) <= 3
        assert all("clause_type" in r for r in results)
        assert all("confidence_score" in r for r in results)

    def test_classify_best(self):
        classifier = ClauseClassifier()
        text = "This agreement shall be governed by the laws of Delaware."
        clause_type, confidence = classifier.classify_best(text)
        assert isinstance(clause_type, (str, type(None)))
        assert 0 <= confidence <= 1
