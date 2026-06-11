import pytest
from app.parsers.pdf_parser import PDFParser
from app.parsers.docx_parser import DOCXParser


class TestPDFParser:
    def test_clean_text(self):
        parser = PDFParser()
        text = "Hello   World\n\n\n\nTest"
        result = parser._clean_text(text)
        assert "Hello World" in result
        assert "\n\n" in result
        assert "\n\n\n" not in result


class TestDOCXParser:
    def test_clean_text(self):
        parser = DOCXParser()
        text = "Hello\u2018World\u2019"
        result = parser._clean_text(text)
        assert "'" in result
        assert "\u2018" not in result
