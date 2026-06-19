import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SpacyNLP:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            self._available = True
            logger.info("spaCy model '%s' loaded successfully", self.model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                self.model_name, self.model_name,
            )
            self._available = False
        except ImportError:
            logger.warning(
                "spaCy is not installed. Install with: pip install spacy"
            )
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def extract_entities(self, text: str) -> List[Dict]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities

    def extract_money_values(self, text: str) -> List[float]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        values = []
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                try:
                    clean = ent.text.replace("$", "").replace(",", "").replace("USD", "").strip()
                    values.append(float(clean))
                except (ValueError, TypeError):
                    pass
        return values

    def extract_date_references(self, text: str) -> List[str]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]

    def extract_org_names(self, text: str) -> List[str]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    def extract_law_references(self, text: str) -> List[str]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        refs = []
        for ent in doc.ents:
            if ent.label_ == "LAW":
                refs.append(ent.text)
        for ent in doc.ents:
            if ent.label_ == "GPE" and any(
                kw in text.lower() for kw in ["law", "jurisdiction", "govern"]
            ):
                refs.append(ent.text)
        return refs

    def get_sentences(self, text: str) -> List[str]:
        if not self._available or not self.nlp:
            import re
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def get_noun_chunks(self, text: str) -> List[str]:
        if not self._available or not self.nlp:
            return []
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]


_nlp_instance: Optional[SpacyNLP] = None


def get_nlp_service() -> SpacyNLP:
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = SpacyNLP()
    return _nlp_instance
