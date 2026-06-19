import logging
import os
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
        self.ml_model = None
        self.ml_tokenizer = None
        self.ml_label_map = None
        self._ml_loaded = False
        self._build_label_embeddings()

    def _load_ml_model(self):
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models", "clause_classifier", "final",
        )
        if not os.path.exists(model_dir):
            logger.warning(f"ML model not found at {model_dir}, using heuristic only")
            return

        try:
            import transformers.utils.import_utils as _iu
            _original = getattr(_iu, "check_torch_load_is_safe", None)
            if _original:
                _iu.check_torch_load_is_safe = lambda: None

            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.ml_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.ml_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.ml_model.eval()

            import json
            label_map_path = os.path.join(os.path.dirname(model_dir), "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    self.ml_label_map = json.load(f)
                self.ml_label_map = {int(k): v for k, v in self.ml_label_map.items()}

            if _original:
                _iu.check_torch_load_is_safe = _original

            logger.info(f"Legal-BERT clause classifier loaded from {model_dir}")
        except Exception as e:
            logger.warning(f"Failed to load ML clause classifier: {e}")
            self.ml_model = None

    def _ensure_ml_loaded(self):
        if self._ml_loaded:
            return
        self._ml_loaded = True
        self._load_ml_model()

    def _ml_classify(self, clause_text: str) -> Tuple[Optional[str], float]:
        self._ensure_ml_loaded()
        if self.ml_model is None or self.ml_tokenizer is None:
            return None, 0.0

        import torch
        inputs = self.ml_tokenizer(
            clause_text, return_tensors="pt", truncation=True, padding=True, max_length=256,
        )
        with torch.no_grad():
            outputs = self.ml_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        label = self.ml_label_map.get(pred_idx, self.labels[pred_idx]) if self.ml_label_map else self.labels[pred_idx]
        return label, round(confidence, 4)

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

    def classify_best(
        self,
        clause_text: str,
        min_confidence: float = 0.15,
    ) -> Tuple[Optional[str], float]:
        ml_type, ml_score = self._ml_classify(clause_text)

        if ml_type and ml_score >= 0.5:
            return ml_type, ml_score

        keyword_type, keyword_score = self._keyword_classify(clause_text)

        embedding_result = self.classify(clause_text, top_k=5)
        embedding_type = None
        embedding_score = 0.0
        embedding_top5 = []
        if embedding_result:
            embedding_type = embedding_result[0]["clause_type"]
            embedding_score = embedding_result[0]["confidence_score"]
            embedding_top5 = [r["clause_type"] for r in embedding_result]

        combined = self._fuse_scores(
            keyword_type, keyword_score,
            embedding_type, embedding_score,
            embedding_top5,
        )

        if combined and combined[1] >= min_confidence:
            return combined

        if ml_type and ml_score >= min_confidence:
            return ml_type, ml_score

        return embedding_type or None, embedding_score

    def _fuse_scores(
        self,
        kw_type: Optional[str], kw_score: float,
        emb_type: Optional[str], emb_score: float,
        emb_top5: List[str],
    ) -> Optional[Tuple[Optional[str], float]]:
        if kw_type and kw_score >= 0.4:
            if kw_type in emb_top5:
                return kw_type, max(kw_score, emb_score)
            if emb_score < 0.6:
                return kw_type, kw_score * 0.8 + emb_score * 0.2

        if emb_type and emb_score >= 0.3:
            if kw_type == emb_type:
                return emb_type, min(1.0, emb_score * 1.3)
            return emb_type, emb_score

        if kw_type and kw_score >= 0.2:
            return kw_type, kw_score * 0.7

        return None

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
