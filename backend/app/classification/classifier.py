import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.classification.cuad_labels import CUAD_LABELS, CUAD_LABEL_DESCRIPTIONS
from app.config import settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


CUAD_KEYWORD_PATTERNS = {
    "Termination": [
        r"\bterminat(?:e|ion|ing)\b",
        r"\bcancel\b",
        r"\bearly\s+termination\b",
        r"\bnotice\s+of\s+termination\b",
        r"\bterminat(?:e|ion)\s+(?:for\s+)?(?:cause|convenience|breach)\b",
    ],
    "Payment Terms": [
        r"\bpayment\b",
        r"\binvoice\b",
        r"\bpayable\b",
        r"\bdue\s+within\b",
        r"\bbilling\b",
        r"\bnet\s+\d+\b",
    ],
    "Liability": [
        r"\bliab(?:le|ility)\b",
        r"\bindemnif\b",
    ],
    "Limitation of Liability": [
        r"\b(?:limit|cap)\s*(?:of|on|to)\s+liab\w*\b",
        r"\bcapped?\s*(?:at|to)\s",
        r"\bexclusive\s+remedy\b",
        r"\baggregate\s+liab\b",
    ],
    "Confidentiality": [
        r"\bconfidential\b",
        r"\bnon.?disclosure\b",
        r"\bproprietary\b",
        r"\btrade\s+secret\b",
        r"\bdeemed\s+confidential\b",
        r"\bconfidential\s+information\b",
    ],
    "Non-Compete": [
        r"\bnon.?compete\b",
        r"\bnon.?competition\b",
        r"\brestrictive\s+covenant\b",
        r"\bcompete\s+with\b",
    ],
    "Intellectual Property": [
        r"\bintellectual\s+property\b",
        r"\bcopyright\b",
        r"\bpatent\b",
        r"\btrademark\b",
        r"\bownership\s+of\b",
        r"\bwork\s+(?:made\s+)?for\s+hire\b",
    ],
    "Indemnification": [
        r"\bindemnif\b",
        r"\bhold\s+harmless\b",
        r"\bdefend\b",
    ],
    "Assignment": [
        r"\bassign\b",
        r"\bassignment\b",
        r"\bdelegat\w+\b",
    ],
    "Governing Law": [
        r"\bgovern(?:ed|ing)?\s+(?:by|law)\b",
        r"\bconstrued\s+(?:in\s+accordance\s+with|under)\s+(?:the\s+)?laws?\s+of\b",
        r"\bjurisdiction\b",
        r"\bvenue\b",
        r"\bchoice\s+of\s+law\b",
        r"\bdispute\s+resolution\b",
    ],
    "Arbitration": [
        r"\barbitrat\w+\b",
        r"\bbinding\s+arbitration\b",
    ],
    "Insurance": [
        r"\binsurance\b",
        r"\binsured\b",
        r"\bcoverage\b",
    ],
    "Data Protection": [
        r"\bdata\s+protection\b",
        r"\bdata\s+privacy\b",
        r"\bgdpr\b",
        r"\bpersonal\s+data\b",
    ],
    "Force Majeure": [
        r"\bforce\s+majeure\b",
        r"\bact\s+of\s+god\b",
        r"\bbeyond\s+(?:.*?\s+)?control\b",
    ],
    "Warranty": [
        r"\bwarrant\w+\b",
        r"\bguarantee\b",
    ],
    "Representations and Warranties": [
        r"\brepresent(?:s|ation|ations)\b",
    ],
    "Entire Agreement": [
        r"\bentire\s+agreement\b",
        r"\bsupersed\w+\b",
    ],
    "Amendments": [
        r"\bamend\w+\b",
    ],
    "Notices": [
        r"\bnotice\s+(?:shall\s+)?be\s+(?:given|provided|sent)\b",
        r"\b(?:written|prior)\s+notice\s+(?:shall|must|will|may)\b",
        r"\bnotice\s+(?:address|requirement|procedure)\b",
    ],
    "Severability": [
        r"\bseverab\w+\b",
    ],
    "Waiver": [
        r"\bwaiv\w+\b",
    ],
    "Survival": [
        r"\bsurviv\w+\b",
        r"\bshall\s+survive\b",
    ],
    "Counterparts": [
        r"\bcounterpart\b",
    ],
    "Definitions": [
        r"\bdefined\s+terms?\b",
        r"\bherein\b",
        r"\bhereunder\b",
    ],
    "Expenses": [
        r"\bexpense\b",
        r"\bcost\b",
    ],
    "Publicity": [
        r"\bpublicity\b",
        r"\bpress\s+release\b",
    ],
    "Subcontracting": [
        r"\bsubcontract\b",
    ],
}


class ClauseClassifier:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.label_embeddings = None
        self.labels = CUAD_LABELS
        self._build_label_embeddings()

    def _build_label_embeddings(self):
        label_texts = []
        for label in self.labels:
            desc = CUAD_LABEL_DESCRIPTIONS.get(label, label)
            label_texts.append(f"{label}: {desc}")
        embeddings = self.embedding_service.encode(label_texts)
        self.label_embeddings = embeddings
        logger.info(f"Built {len(self.labels)} label embeddings")

    def classify(self, clause_text: str, top_k: int = 3) -> List[dict]:
        clause_emb = self.embedding_service.encode_single(clause_text)
        similarities = []
        for i, label_emb in enumerate(self.label_embeddings):
            sim = self.embedding_service.cosine_similarity(clause_emb, label_emb)
            similarities.append((self.labels[i], float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [
            {
                "clause_type": label,
                "confidence_score": round(score, 4),
            }
            for label, score in similarities[:top_k]
        ]
        return results

    def classify_best(self, clause_text: str, min_confidence: float = 0.15) -> Tuple[Optional[str], float]:
        keyword_type, keyword_score = self._keyword_classify(clause_text)

        embedding_result = self.classify(clause_text, top_k=3)
        embedding_type = None
        embedding_score = 0.0
        embedding_top3 = []
        if embedding_result:
            embedding_type = embedding_result[0]["clause_type"]
            embedding_score = embedding_result[0]["confidence_score"]
            embedding_top3 = [r["clause_type"] for r in embedding_result]

        if keyword_type and keyword_score >= 0.4:
            if keyword_type in embedding_top3:
                return keyword_type, max(keyword_score, embedding_score)
            if embedding_score < 0.6:
                return keyword_type, keyword_score

        if embedding_type and embedding_score >= min_confidence:
            return embedding_type, embedding_score

        if keyword_type and keyword_score >= 0.15:
            return keyword_type, keyword_score

        return embedding_type or None, embedding_score

    def _keyword_classify(self, clause_text: str) -> Tuple[Optional[str], float]:
        best_type = None
        best_score = 0.0
        for label, patterns in CUAD_KEYWORD_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, clause_text, re.IGNORECASE):
                    matches += 1
            if matches > 0:
                score = min(1.0, matches / max(len(patterns), 1) * 3)
                if score > best_score:
                    best_score = score
                    best_type = label
        if best_type and best_score >= 0.1:
            return best_type, round(best_score, 4)
        return None, 0.0
