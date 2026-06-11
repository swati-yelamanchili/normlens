import pytest
from app.segmentation.clause_segmenter import ClauseSegmenter


class TestClauseSegmenter:
    def setup_method(self):
        self.segmenter = ClauseSegmenter()

    def test_simple_clause_detection(self):
        text = "1. Termination\nThis agreement may be terminated.\n\n2. Payment\nPayment shall be made."
        clauses = self.segmenter.segment(text)
        assert len(clauses) == 2
        assert clauses[0]["clause_title"] == "Termination"
        assert clauses[1]["clause_title"] == "Payment"

    def test_fallback_segmentation(self):
        text = "This is a long block of text.\n\nThis is another block."
        clauses = self.segmenter._fallback_segment(text)
        assert len(clauses) >= 1

    def test_empty_text(self):
        clauses = self.segmenter.segment("")
        assert clauses == []
