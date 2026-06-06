import logging
import re
from typing import List

logger = logging.getLogger(__name__)

CLAUSE_HEADER_PATTERNS = [
    r"^\d+\.\s+([A-Z][A-Za-z\s\-/]+)[.](?:\s|$)",
    r"^\d+\.\s+([A-Z][A-Za-z\s\-/]+)$",
    r"^\d+\.\d+\s+([A-Z][A-Za-z\s\-/]+)[.](?:\s|$)",
    r"^\d+\.\d+\s+([A-Z][A-Za-z\s\-/]+)$",
    r"^Section\s+\d+[\.\d]*\s*[–\-—]\s*([A-Z][A-Za-z\s\-/]+)$",
    r"^Section\s+\d+[\.\d]*[:.]?\s*([A-Z][A-Za-z\s\-/]+)$",
    r"^Article\s+[IVXLCDM]+\s*[–\-—:.]?\s*([A-Z][A-Za-z\s\-/]+)$",
    r"^ARTICLE\s+\d+[\.\d]*\s*[–\-—:.]?\s*([A-Z][A-Za-z\s\-/]+)$",
    r"^([A-Z][A-Z\s\-/]+)$",
    r"<clause_header>([A-Za-z\s\-/]+)</clause_header>",
]

STANDALONE_NUMBER_PATTERNS = [
    r"^\d+\.$",
    r"^\d+\.\d+$",
    r"^\([a-z]\)$",
    r"^[A-Z]\.$",
]

MARKER_SEPARATORS = [
    r"^\-{5,}$",
    r"^\*{5,}$",
    r"^_{5,}$",
]


class ClauseSegmenter:
    def segment(self, text: str, pages: list = None) -> List[dict]:
        lines = text.split("\n")
        clauses = []
        current_clause_lines = []
        current_title = None
        current_start = 0

        pending_standalone_number = False
        standalone_number_line = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            is_header = False
            matched_title = None
            is_standalone_header = False

            if pending_standalone_number:
                pending_standalone_number = False
                upper_match = re.match(r"^([A-Z][A-Za-z\s\-/]+)$", stripped)
                if upper_match:
                    matched_title = upper_match.group(1).strip()
                    is_header = True
                    is_standalone_header = True
                else:
                    current_clause_lines.append(standalone_number_line)
                    current_clause_lines.append(stripped)
                    continue

            if not is_header:
                for pattern in CLAUSE_HEADER_PATTERNS:
                    m = re.match(pattern, stripped)
                    if m:
                        matched_title = m.group(1).strip()
                        is_header = True
                        break

            if not is_header:
                for pattern in STANDALONE_NUMBER_PATTERNS:
                    if re.match(pattern, stripped):
                        pending_standalone_number = True
                        standalone_number_line = stripped
                        break

            if is_header:
                if current_clause_lines:
                    clause_text = "\n".join(current_clause_lines).strip()
                    if len(clause_text) > 20:
                        page_num = self._estimate_page(
                            current_start, len(lines), pages
                        )
                        clauses.append(
                            {
                                "clause_index": len(clauses),
                                "clause_title": current_title,
                                "clause_text": clause_text,
                                "page_number": page_num,
                            }
                        )
                current_clause_lines = [standalone_number_line] if is_standalone_header else []
                current_clause_lines.append(stripped)
                current_title = matched_title
                current_start = i
            elif not pending_standalone_number:
                current_clause_lines.append(stripped)

        if current_clause_lines:
            clause_text = "\n".join(current_clause_lines).strip()
            if len(clause_text) > 20:
                page_num = self._estimate_page(current_start, len(lines), pages)
                clauses.append(
                    {
                        "clause_index": len(clauses),
                        "clause_title": current_title,
                        "clause_text": clause_text,
                        "page_number": page_num,
                    }
                )

        if not clauses:
            clauses = self._fallback_segment(text)

        return clauses

    def _fallback_segment(self, text: str) -> List[dict]:
        blocks = re.split(r"\n{2,}", text)
        clauses = []
        for i, block in enumerate(blocks):
            stripped = block.strip()
            if len(stripped) < 20:
                continue
            clauses.append(
                {
                    "clause_index": i,
                    "clause_title": None,
                    "clause_text": stripped,
                    "page_number": 1,
                }
            )
        return clauses

    def _estimate_page(self, line_index: int, total_lines: int, pages: list) -> int:
        if not pages:
            return 1
        ratio = line_index / max(total_lines, 1)
        est_page = max(1, round(ratio * len(pages)))
        return est_page
