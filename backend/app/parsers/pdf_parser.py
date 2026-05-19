import io
import logging
import re
from typing import Optional

import pdfplumber
import fitz

logger = logging.getLogger(__name__)


class PDFParser:
    def parse(self, file_bytes: bytes, filename: str) -> dict:
        pages = []
        full_text = []

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append(
                        {
                            "page_number": i + 1,
                            "text": text.strip(),
                        }
                    )
                    full_text.append(text)
        except Exception as e:
            logger.warning(f"pdfplumber failed for {filename}: {e}. Trying PyMuPDF.")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                text = page.get_text()
                pages.append(
                    {
                        "page_number": i + 1,
                        "text": text.strip(),
                    }
                )
                full_text.append(text)
            doc.close()

        combined = "\n".join(full_text)
        cleaned = self._clean_text(combined)

        return {
            "text": cleaned,
            "pages": pages,
            "page_count": len(pages),
            "file_type": "pdf",
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\u2018|\u2019", "'", text)
        text = re.sub(r"\u201c|\u201d", '"', text)
        text = re.sub(r"\u2013|\u2014", "-", text)
        text = re.sub(r"http\S+", "[URL]", text)
        return text.strip()
