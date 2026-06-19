import pytest
from app.services.nlp_service import SpacyNLP, get_nlp_service


class TestSpacyNLP:
    def setup_method(self):
        try:
            self.nlp = get_nlp_service()
        except Exception:
            pytest.skip("spaCy not available")

    def test_available(self):
        if not self.nlp.available:
            pytest.skip("spaCy model not loaded")

    def test_extract_money_values(self):
        if not self.nlp.available:
            pytest.skip("spaCy not available")
        text = "The liability cap shall be $5,000,000 USD."
        values = self.nlp.extract_money_values(text)
        assert len(values) > 0
        assert 5000000 in values or 5000000.0 in values

    def test_extract_date_references(self):
        if not self.nlp.available:
            pytest.skip("spaCy not available")
        text = "This agreement shall be effective from January 1, 2024."
        dates = self.nlp.extract_date_references(text)
        assert len(dates) > 0

    def test_get_sentences(self):
        if not self.nlp.available:
            pytest.skip("spaCy not available")
        text = "First sentence. Second sentence. Third sentence."
        sentences = self.nlp.get_sentences(text)
        assert len(sentences) == 3

    def test_extract_org_names(self):
        if not self.nlp.available:
            pytest.skip("spaCy not available")
        text = "Acme Corporation and Beta Inc. entered into an agreement."
        orgs = self.nlp.extract_org_names(text)
        assert len(orgs) > 0

    def test_extract_entities(self):
        if not self.nlp.available:
            pytest.skip("spaCy not available")
        text = "The agreement is governed by the laws of Delaware."
        entities = self.nlp.extract_entities(text)
        assert len(entities) > 0
