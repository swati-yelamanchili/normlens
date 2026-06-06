import logging
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


GENERAL = None

EXTRACTION_GATES = {
    "notice_days": {"Termination", "Term", "Cancellation"},
    "payment_deadline_days": {"Payment Terms", "Payment", "Compensation", "Billing", GENERAL},
    "liability_cap": {"Liability", "Limitation of Liability", "Indemnification", "Indemnity", GENERAL},
    "non_compete_months": {"Non-Compete", "Noncompete", "Non Competition", "Non-Competition"},
    "contract_duration_months": {"Term", "Termination", "Agreement", GENERAL},
    "penalty_percentage": {"Payment Terms", "Payment", "Late Payment", "Billing", GENERAL},
    "renewal_notice_days": {"Renewal", "Term"},
    "insurance_amount": {"Insurance"},
    "has_unlimited_liability": {"Liability", "Limitation of Liability", "Indemnification", "Indemnity", GENERAL},
    "has_one_sided_termination": {"Termination", "Term", "Cancellation"},
    "arbitration_location": {"Arbitration", "Dispute Resolution", "Governing Law"},
    "governing_law": {"Governing Law", "Dispute Resolution", "Arbitration", GENERAL},
}

SEMANTIC_DETECT = {
    "has_unlimited_liability": {
        "questions": [
            "Does this clause have unlimited liability?",
            "Is liability uncapped or unlimited in this clause?",
        ],
        "positive_refs": [
            "liability is unlimited",
            "no cap on liability",
            "liability is not limited",
            "uncapped liability",
        ],
        "negative_refs": [
            "liability is capped at a specific amount",
            "liability is limited to",
            "liability shall not exceed",
        ],
    },
    "has_one_sided_termination": {
        "questions": [
            "Can only one party terminate this agreement?",
            "Is termination one-sided or unilateral?",
        ],
        "positive_refs": [
            "only Company can terminate",
            "one party has sole termination rights",
            "unilateral termination right",
            "Company may terminate at its sole discretion",
        ],
        "negative_refs": [
            "either party may terminate",
            "both parties can terminate",
            "mutual termination rights",
        ],
    },
}

SEMANTIC_EXTRACT = {
    "notice_days": [
        "How many days notice is required for termination?",
        "What is the notice period for termination?",
    ],
    "payment_deadline_days": [
        "How many days after invoice must payment be made?",
        "What is the payment term in days?",
    ],
    "non_compete_months": [
        "How long does the non-compete last?",
        "What is the duration of the non-compete restriction?",
    ],
}


class AttributeExtractor:
    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self._semantic_embeddings = {}

    def extract(self, clause_text: str, clause_type: Optional[str] = None) -> Dict:
        attributes = {}

        if self._type_matches(clause_type, EXTRACTION_GATES["notice_days"]):
            val = self._extract_notice_period(clause_text)
            if val is None:
                val = self._semantic_extract_int("notice_days", clause_text)
            if val is not None:
                attributes["notice_days"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["payment_deadline_days"]):
            val = self._extract_payment_deadline(clause_text)
            if val is None:
                val = self._semantic_extract_int("payment_deadline_days", clause_text)
            if val is not None:
                attributes["payment_deadline_days"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["liability_cap"]):
            liability_cap = self._extract_liability_cap(clause_text)
            if liability_cap is not None:
                attributes["liability_cap"] = liability_cap

        if self._type_matches(clause_type, EXTRACTION_GATES["non_compete_months"]):
            val = self._extract_non_compete_duration(clause_text)
            if val is None:
                val = self._semantic_extract_int("non_compete_months", clause_text)
            if val is not None:
                attributes["non_compete_months"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["contract_duration_months"]):
            contract_duration_months = self._extract_contract_duration(clause_text)
            if contract_duration_months is not None:
                attributes["contract_duration_months"] = contract_duration_months

        if self._type_matches(clause_type, EXTRACTION_GATES["penalty_percentage"]):
            penalty_percentage = self._extract_penalty_percentage(clause_text)
            if penalty_percentage is not None:
                attributes["penalty_percentage"] = penalty_percentage

        if self._type_matches(clause_type, EXTRACTION_GATES["renewal_notice_days"]):
            renewal_term_days = self._extract_renewal_term(clause_text)
            if renewal_term_days is not None:
                attributes["renewal_notice_days"] = renewal_term_days

        if self._type_matches(clause_type, EXTRACTION_GATES["insurance_amount"]):
            insurance_amount = self._extract_insurance_amount(clause_text)
            if insurance_amount is not None:
                attributes["insurance_amount"] = insurance_amount

        if self._type_matches(clause_type, EXTRACTION_GATES["has_unlimited_liability"]):
            val = self._detect_unlimited_liability(clause_text)
            if not val:
                val = self._semantic_detect("has_unlimited_liability", clause_text)
            attributes["has_unlimited_liability"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["has_one_sided_termination"]):
            val = self._detect_one_sided_termination(clause_text)
            if not val:
                val = self._semantic_detect("has_one_sided_termination", clause_text)
            attributes["has_one_sided_termination"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["arbitration_location"]):
            arbitration_location = self._extract_arbitration_location(clause_text)
            if arbitration_location:
                attributes["arbitration_location"] = arbitration_location

        if self._type_matches(clause_type, EXTRACTION_GATES["governing_law"]):
            governing_law = self._extract_governing_law(clause_text)
            if governing_law:
                attributes["governing_law"] = governing_law

        return attributes

    def _type_matches(self, clause_type: Optional[str], allowed_types: set) -> bool:
        if clause_type is None:
            return GENERAL in allowed_types
        return any(
            allowed is not None and allowed.lower() == clause_type.lower()
            for allowed in allowed_types
        )

    ONE_SIDED_DENIAL_PATTERNS = [
        r"(?:other|second|one)\s+party\s+(?:gets|has|shall\s+have)\s+no\s+right\s+to\s+(?:terminate|cancel)",
        r"no\s+(?:right|ability|option)\s+to\s+(?:terminate|cancel)",
        r"may\s+not\s+(?:terminate|cancel)",
        r"cannot?\s+(?:terminate|cancel)",
    ]

    def _semantic_detect(self, attr_name: str, text: str) -> bool:
        if not self.embedding_service:
            return False
        config = SEMANTIC_DETECT.get(attr_name)
        if not config:
            return False
        try:
            text_emb = self.embedding_service.encode_single(text)
            pos_embs = self.embedding_service.encode(config["positive_refs"])
            neg_embs = self.embedding_service.encode(config["negative_refs"])
            pos_sim = max(
                self.embedding_service.cosine_similarity(text_emb, pe) for pe in pos_embs
            )
            neg_sim = max(
                self.embedding_service.cosine_similarity(text_emb, ne) for ne in neg_embs
            )
            if pos_sim > 0.3:
                if pos_sim > neg_sim:
                    return True
                if attr_name == "has_one_sided_termination" and (pos_sim - neg_sim) > -0.1:
                    if any(re.search(p, text, re.IGNORECASE) for p in self.ONE_SIDED_DENIAL_PATTERNS):
                        return True
            return False
        except Exception as e:
            logger.warning(f"Semantic detection failed for {attr_name}: {e}")
            return False

    def _semantic_extract_int(self, attr_name: str, text: str) -> Optional[int]:
        if not self.embedding_service:
            return None
        questions = SEMANTIC_EXTRACT.get(attr_name)
        if not questions:
            return None
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if not sentences:
                return None
            sent_embs = self.embedding_service.encode(sentences)
            best_score = 0.0
            best_sent = None
            for q in questions:
                q_emb = self.embedding_service.encode_single(q)
                for i, s_emb in enumerate(sent_embs):
                    score = self.embedding_service.cosine_similarity(q_emb, s_emb)
                    if score > best_score:
                        best_score = score
                        best_sent = sentences[i]
            if best_sent and best_score > 0.4:
                val = self._extract_first_int(best_sent, [
                    r"(\d+)\s*(?:month|year)s?",
                    r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
                ])
                if val is not None:
                    return self._months_from_years_if_needed(val, best_sent)
            return None
        except Exception as e:
            logger.warning(f"Semantic extraction failed for {attr_name}: {e}")
            return None

    def _months_from_years_if_needed(self, val: int, text: str) -> int:
        if re.search(r"(\d+)\s*years?", text):
            return val * 12
        return val

    def _extract_notice_period(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:upon|after|following|by|give|given)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice",
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice\s*(?:of\s+)?(?:termination|cancel)",
            r"(?:terminat(?:e|ion|ed|ing)|cancel)\s+(?:\w+\s+){0,5}?(?:requires|shall\s+require|may\s+be\s+made|may\s+be\s+effected|may\s+be\s+given)\s*(?:by\s+)?(?:written\s+)?notice\s*(?:of\s+)?(?:not\s+less\s+than\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"notice\s*(?:period\s+)?(?:of\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+to\s+)?(?:termination|cancel)",
            r"(?:within|not\s+less\s+than)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice\s*(?:of\s+)?(?:termination|cancel)",
            r"(?:terminat(?:e|ion|ed|ing)|cancel)\s+(?:\w+\s+){0,15}?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s+notice",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_payment_deadline(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:within|within\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:of|after|from)\s*(?:receipt|invoice|billing)",
            r"(?:payable|due|payment)\s*(?:within|in)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"net\s+(\d+)(?:\s*(?:day|calendar\s*day|business\s*day)s?)?\s*(?:from|after)",
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s+(?:prior|after|from)\s+(?:payment|receipt|invoice)",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_liability_cap(self, text: str) -> Optional[str]:
        patterns = [
            r"liability\s*(?:shall\s+)?(?:be\s+)?limited\s+to\s+(?:an\s+amount\s+(?:equal\s+)?to\s+)?(\$?[\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"aggregate\s*liability\s*(?:shall\s+)?(?:not\s+)?exceed\s+(\$?[\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"cap\s*(?:on\s+)?liability\s*(?:of|is)\s+(\$?[\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"liability\s*cap\s*(?:of|is)\s+(\$?[\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
        ]
        return self._extract_first_match(text, patterns)

    def _extract_non_compete_duration(self, text: str) -> Optional[int]:
        patterns = [
            r"non.?compete\s*(?:period|term|duration|restriction)?\s*(?:shall\s+be|is|of|for)?\s*(?:a\s+(?:period|duration)\s+of\s+)?(\d+)\s*(?:month|year)s?",
            r"(\d+)\s*(?:month|year)s?\s*(?:non.?compete|noncompete)",
            r"(?:for\s+(?:a\s+(?:period|duration)\s+of\s+)?(\d+)\s*(?:month|year)s?)\s+.*?(?:non.?compete|non.?competition|not\s+.*?compete)",
            r"(?:not\s+.*?compete|non.?compete|non.?competition)\s+.*?(?:for\s+(?:a\s+(?:period|duration)\s+of\s+)?(\d+)\s*(?:month|year)s?)",
            r"(?:compete|competition)\s+.*?(?:for\s+(?:a\s+(?:period|duration)\s+of\s+)?(\d+)\s*(?:month|year)s?)",
        ]
        val = self._extract_first_int(text, patterns)
        return val

    def _extract_contract_duration(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:initial\s+)?(?:term|duration)\s*(?:of|:)?\s*(\d+)\s*(?:month|year)s?",
            r"(?:this\s+)?(?:agreement|contract)\s*(?:shall\s+)?(?:be\s+)?(?:for|effective\s+for|continues?\s+for)\s*(?:a\s+(?:period|term)\s+of\s+)?(\d+)\s*(?:month|year)s?",
            r"(?:starts?\s+on|begins?\s+on|commences?\s+on)\s+.*?(?:and\s+)?(?:continues?\s+for|lasts?\s+for|runs?\s+for)\s*(?:a\s+)?(?:period|term)\s+of\s+(\d+)\s*(?:month|year)s?",
        ]
        val = self._extract_first_int(text, patterns)
        if val:
            return self._months_from_years_if_needed(val, text)
        return val

    def _extract_penalty_percentage(self, text: str) -> Optional[float]:
        patterns = [
            r"(\d+(?:\.\d+)?)\%\s*(?:per\s+)?(?:annum|month|year|day|week|late|penalty|interest)",
            r"(?:late|penalty|interest)\s*(?:fee|charge|rate)?\s*(?:of|at)\s*(\d+(?:\.\d+)?)\%",
        ]
        return self._extract_first_float(text, patterns)

    def _extract_renewal_term(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:renewal|renew)\s*(?:notice|period)?\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"(?:prior\s+)?(?:written\s+)?notice\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+to\s+)?(?:renewal|renew)",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_insurance_amount(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:commercial\s+general\s+liability|general\s+liability|professional\s+liability|workers?\s*compensation|cyber|property|auto|umbrella|excess)\s+(?:insurance|coverage)\s*(?:of|in\s+an\s+amount|limits|coverage)\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"(?:insurance\s+(?:limits|coverage|amount))\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
        ]
        return self._extract_first_match(text, patterns)

    def _detect_unlimited_liability(self, text: str) -> bool:
        patterns = [
            r"(?:unlimited\s+liability|no\s+limitation\s+of\s+liability|without\s+limit)",
            r"liability\s+(?:is\s+|shall\s+be\s+)?(?:unlimited|not\s+(?:be\s+)?limited)",
            r"(?:not\s+(?:be\s+)?limited|shall\s+not\s+be\s+limited)\s*(?:in\s+any\s+way|in\s+any\s+manner)?",
            r"no\s+(?:cap|limit|limitation)\s+on\s+(?:liability|our\s+liability)",
            r"nothing\s+in\s+this\s+(?:section|agreement|clause|provision)\s+(?:limits|shall\s+limit|limits?|shall\s+be\s+deemed\s+to\s+limit)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_one_sided_termination(self, text: str) -> bool:
        text_lower = text.lower()
        mutual = bool(re.search(r'\b(?:either\s+party|both\s+parties|mutual|each\s+party|neither\s+party)\b', text_lower))
        patterns = [
            r"(?:reserves\s+the\s+right|shall\s+have\s+the\s+right)\s*(?:unilaterally\s+)?to\s+terminate",
            r"terminat(?:ion|e)\s*(?:rights?\s+)?(?:are\s+)?(?:materially\s+)?(?:unilateral|one.sided|sole\s+discretion|sole\s+option)",
        ]
        specific_one_sided = any(re.search(p, text, re.IGNORECASE) for p in patterns)
        if specific_one_sided:
            return True
        if mutual:
            return False
        denial_patterns = [
            r"(?:other|second|one)\s+party\s+(?:gets|has|shall\s+have|has\s+no)\s+no\s+right\s+to\s+(?:terminate|cancel)",
            r"no\s+(?:right|ability|option)\s+to\s+(?:terminate|cancel)",
            r"may\s+not\s+(?:terminate|cancel)",
            r"(?:is\s+)?not\s+entitled\s+to\s+(?:terminate|cancel)",
            r"cannot?\s+(?:terminate|cancel)",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in denial_patterns):
            return True
        party_patterns = [
            r"(?:company|client|licensor|licensee|employer|employee|contractor)\s+(?:may|can)\s+(?:terminate|cancel|end)",
            r"(?:may|can)\s+be\s+(?:terminated|ended)\s+by\s+(?:company|client|licensor|licensee|employer|employee|contractor)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in party_patterns)

    def _extract_arbitration_location(self, text: str) -> Optional[str]:
        patterns = [
            r"arbitration\s*(?:shall\s+be\s+)?(?:held|conducted|administered)\s*(?:in|at)\s+([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$|pursuant|according|by)",
            r"(?:seat|place|location|venue)\s*(?:of\s+)?arbitration\s*(?:shall\s+be\s+)?(?:in|at)\s+([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$|pursuant|according|by)",
        ]
        return self._extract_first_match(text, patterns)

    def _extract_governing_law(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:governed\s+by|governing\s+law|construed\s+in\s+accordance\s+with)\s+(?:the\s+(?:laws\s+of\s+)?)?([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$)",
            r"(?:this\s+)?(?:agreement|contract)\s*(?:shall\s+)?be\s+governed\s+by\s+(?:the\s+(?:laws\s+of\s+)?)?([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$)",
        ]
        return self._extract_first_match(text, patterns)

    def _extract_first_int(self, text: str, patterns: list) -> Optional[int]:
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_first_float(self, text: str, patterns: list) -> Optional[float]:
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_first_match(self, text: str, patterns: list) -> Optional[str]:
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                val = re.sub(r"\s+", " ", val)
                return val
        return None
