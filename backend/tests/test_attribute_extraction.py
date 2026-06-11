import pytest
from app.extraction.attribute_extractor import AttributeExtractor


class TestAttributeExtractor:
    def setup_method(self):
        self.extractor = AttributeExtractor()

    def test_notice_period_extraction(self):
        text = "Either party may terminate with 180 days prior written notice."
        result = self.extractor.extract(text)
        assert result.get("notice_days") == 180

    def test_no_notice_period(self):
        text = "This agreement shall be governed by the laws of Delaware."
        result = self.extractor.extract(text)
        assert result.get("notice_days") is None

    def test_unlimited_liability_detection(self):
        text = "Liability shall not be limited in any way."
        result = self.extractor.extract(text)
        assert result.get("has_unlimited_liability") is True

    def test_liability_cap_extraction(self):
        text = "Aggregate liability shall not exceed $5,000,000."
        result = self.extractor.extract(text)
        assert result.get("liability_cap") is not None

    def test_non_compete_duration(self):
        text = "The non-compete period shall be 24 months."
        result = self.extractor.extract(text)
        assert result.get("non_compete_months") == 24

    def test_payment_deadline(self):
        text = "Payment shall be made within 30 days of invoice receipt."
        result = self.extractor.extract(text)
        assert result.get("payment_deadline_days") == 30

    def test_governing_law(self):
        text = "This agreement shall be governed by the laws of the State of New York."
        result = self.extractor.extract(text)
        assert result.get("governing_law") is not None
