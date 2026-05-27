import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AttributeExtractor:
    def extract(self, clause_text: str, clause_type: Optional[str] = None) -> Dict:
        attributes = {}

        notice_days = self._extract_notice_period(clause_text)
        if notice_days is not None:
            attributes["notice_days"] = notice_days

        payment_days = self._extract_payment_deadline(clause_text)
        if payment_days is not None:
            attributes["payment_deadline_days"] = payment_days

        liability_cap = self._extract_liability_cap(clause_text)
        if liability_cap is not None:
            attributes["liability_cap"] = liability_cap

        non_compete_months = self._extract_non_compete_duration(clause_text)
        if non_compete_months is not None:
            attributes["non_compete_months"] = non_compete_months

        contract_duration_months = self._extract_contract_duration(clause_text)
        if contract_duration_months is not None:
            attributes["contract_duration_months"] = contract_duration_months

        penalty_percentage = self._extract_penalty_percentage(clause_text)
        if penalty_percentage is not None:
            attributes["penalty_percentage"] = penalty_percentage

        renewal_term_days = self._extract_renewal_term(clause_text)
        if renewal_term_days is not None:
            attributes["renewal_notice_days"] = renewal_term_days

        insurance_amount = self._extract_insurance_amount(clause_text)
        if insurance_amount is not None:
            attributes["insurance_amount"] = insurance_amount

        has_unlimited_liability = self._detect_unlimited_liability(clause_text)
        attributes["has_unlimited_liability"] = has_unlimited_liability

        has_one_sided_termination = self._detect_one_sided_termination(clause_text)
        attributes["has_one_sided_termination"] = has_one_sided_termination

        arbitration_location = self._extract_arbitration_location(clause_text)
        if arbitration_location:
            attributes["arbitration_location"] = arbitration_location

        governing_law = self._extract_governing_law(clause_text)
        if governing_law:
            attributes["governing_law"] = governing_law

        return attributes

    def _extract_notice_period(self, text: str) -> Optional[int]:
        patterns = [
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?notice",
            r"notice\s*(?:period\s+)?(?:of\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"(?:within|not\s+less\s+than)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice",
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*notice\s*(?:period|requirement)?",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_payment_deadline(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:within|within\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:of|after|from)\s*(?:receipt|invoice|billing)",
            r"(?:payable|due|payment)\s*(?:within|in)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"net\s+(\d+)",
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:payment\s+)?terms?",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_liability_cap(self, text: str) -> Optional[str]:
        patterns = [
            r"liability\s*(?:shall\s+)?(?:be\s+)?limited\s+to\s+(?:an\s+amount\s+(?:equal\s+)?to\s+)?([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"aggregate\s*liability\s*(?:shall\s+)?(?:not\s+)?exceed\s+([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"cap\s*(?:on\s+)?liability\s*(?:of|is)\s+([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"liability\s*cap\s*(?:of|is)\s+([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"not\s+exceed\s+([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)\s*(?:in\s+)?(?:aggregate\s+)?liability",
        ]
        return self._extract_first_match(text, patterns)

    def _extract_non_compete_duration(self, text: str) -> Optional[int]:
        patterns = [
            r"(\d+)\s*(?:month|year)s?\s*(?:period|term|duration|restriction|non.?compete|noncompete)",
            r"non.?compete\s*(?:period|term|duration|restriction)?\s*(?:of|for|is)?\s*(?:a\s+period\s+of\s+)?(\d+)\s*(?:month|year)s?",
            r"(?:restricted|covenant|restriction)\s*(?:period|term)?\s*(?:of|for)?\s*(\d+)\s*(?:month|year)s?",
        ]
        val = self._extract_first_int(text, patterns)
        if val:
            m = re.search(r"(?:(\d+)\s*years?)", text)
            if not any(p.search(r"(\d+)\s*years?") for p in patterns):
                m2 = re.search(r"(\d+)\s*years?", text)
                if m2:
                    return int(m2.group(1)) * 12
        return val

    def _extract_contract_duration(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:term|duration|period)\s*(?:of|is|shall\s+be)?\s*(?:a\s+period\s+of\s+)?(\d+)\s*(?:month|year)s?",
            r"(?:initial\s+)?(?:term|period)\s*(?:of|:)?\s*(\d+)\s*(?:month|year)s?",
            r"(?:this\s+)?(?:agreement|contract)\s*(?:shall\s+)?(?:be\s+)?(?:for|effective\s+for)\s*(?:a\s+period\s+of\s+)?(\d+)\s*(?:month|year)s?",
        ]
        val = self._extract_first_int(text, patterns)
        if val:
            if re.search(r"(\d+)\s*years?", text):
                m = re.search(r"(\d+)\s*years?", text)
                if m:
                    return int(m.group(1)) * 12
        return val

    def _extract_penalty_percentage(self, text: str) -> Optional[float]:
        patterns = [
            r"(\d+(?:\.\d+)?)\%\s*(?:per\s+)?(?:annum|month|year|day|week|late|penalty|interest)",
            r"(?:late|penalty|interest)\s*(?:fee|charge|rate)?\s*(?:of|at)\s*(\d+(?:\.\d+)?)\%",
            r"(?:monthly|annual|per\s+annum)\s*(?:late|penalty|interest|charge)?\s*(?:rate|fee)?\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\%",
        ]
        return self._extract_first_float(text, patterns)

    def _extract_renewal_term(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:renewal|renew)\s*(?:notice|period)?\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?",
            r"(?:prior\s+)?(?:written\s+)?notice\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+to\s+)?(?:renewal|renew)",
            r"(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice\s*(?:of\s+)?(?:renewal|non.?renewal)",
        ]
        return self._extract_first_int(text, patterns)

    def _extract_insurance_amount(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:commercial\s+general\s+liability|general\s+liability|professional\s+liability|workers?\s*compensation|cyber|property|auto|liability|umbrella|excess)\s*(?:insurance|coverage)?\s*(?:of|in\s+an\s+amount|limits|coverage)\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
            r"(?:insurance\s+(?:limits|coverage|amount))\s*(?:of|:)?\s*(?:not\s+less\s+than\s+)?([\d,]+(?:\s*(?:million|billion|thousand|M|K))?\s*(?:USD|EUR|GBP|INR|dollars|euros|pounds)?)",
        ]
        return self._extract_first_match(text, patterns)

    def _detect_unlimited_liability(self, text: str) -> bool:
        patterns = [
            r"(?:unlimited\s+liability|no\s+limitation\s+of\s+liability|liability\s+(?:is\s+)?not\s+limited|without\s+limit)",
            r"(?:not\s+(?:be\s+)?limited|shall\s+not\s+be\s+limited)\s*(?:in\s+any\s+way|in\s+any\s+manner)?",
            r"(?:expressly\s+)?(?:exclude|exclusion)\s*(?:or\s+)?(?:limit|limitation)\s*(?:of\s+)?(?:any\s+)?liability",
            r"liability\s*(?:for|arising\s+from)\s*(?:fraud|gross\s+negligence|willful\s+misconduct|death|personal\s+injury|breach\s+of\s+confidentiality|breach\s+of\s+ip)\s*(?:shall\s+)?not\s+(?:be\s+)?limited",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_one_sided_termination(self, text: str) -> bool:
        patterns = [
            r"(?:either\s+party|neither\s+party|party\s+a\s+may|party\s+b\s+may|company\s+may|client\s+may|provider\s+may)",
            r"terminat(?:ion|e)\s*(?:rights?\s+)?(?:are\s+)?(?:materially\s+)?(?:unilateral|one.sided|sole\s+discretion|sole\s+option)",
            r"(?:reserves\s+the\s+right|shall\s+have\s+the\s+right)\s*(?:unilaterally\s+)?to\s+terminate",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_arbitration_location(self, text: str) -> Optional[str]:
        patterns = [
            r"arbitration\s*(?:shall\s+be\s+)?(?:held|conducted|administered)\s*(?:in|at)\s+([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$|pursuant|according|by)",
            r"(?:seat|place|location|venue)\s*(?:of\s+)?arbitration\s*(?:shall\s+be\s+)?(?:in|at)\s+([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$|pursuant|according|by)",
            r"arbitration\s*(?:under\s+)?(?:the\s+)?(?:rules\s+of\s+)?(?:the\s+)?([A-Z][A-Za-z\s]+?)(?:\.|,|;|$)",
        ]
        return self._extract_first_match(text, patterns)

    def _extract_governing_law(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:governed\s+by|governing\s+law|construed\s+in\s+accordance\s+with)\s+(?:the\s+(?:laws\s+of\s+)?)?([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$)",
            r"(?:this\s+)?(?:agreement|contract)\s*(?:shall\s+)?be\s+governed\s+by\s+(?:the\s+(?:laws\s+of\s+)?)?([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$)",
            r"laws\s+of\s+([A-Z][A-Za-z\s,]+?)(?:\.|,|;|$)",
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
