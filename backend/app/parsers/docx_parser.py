import io
import logging
import re

from docx import Document

logger = logging.getLogger(__name__)


class DOCXParser:
    def parse(self, file_bytes: bytes, filename: str) -> dict:
        doc = Document(io.BytesIO(file_bytes))
        full_text = []
        pages = []

        current_section = ""
        current_page = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            full_text.append(text)

            pages.append(
                {
                    "page_number": current_page,
                    "text": text,
                }
            )

        combined = "\n".join(full_text)
        cleaned = self._clean_text(combined)

        return {
            "text": cleaned,
            "pages": pages,
            "page_count": 1,
            "file_type": "docx",
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\u2018|\u2019", "'", text)
        text = re.sub(r"\u201c|\u201d", '"', text)
        text = re.sub(r"\u2013|\u2014", "-", text)
        text = re.sub(r"http\S+", "[URL]", text)
        return text.strip()
