import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


GENERAL = None

EXTRACTION_GATES = {
    "notice_days": {"Termination", "Term", "Cancellation", GENERAL},
    "payment_deadline_days": {"Payment Terms", "Payment", "Compensation", "Billing", GENERAL},
    "liability_cap": {"Liability", "Limitation of Liability", "Indemnification", "Indemnity", GENERAL},
    "non_compete_months": {"Non-Compete", "Noncompete", "Non Competition", "Non-Competition", GENERAL},
    "contract_duration_months": {"Term", "Termination", "Agreement", GENERAL},
    "penalty_percentage": {"Payment Terms", "Payment", "Late Payment", "Billing", GENERAL},
    "renewal_notice_days": {"Renewal", "Term"},
    "insurance_amount": {"Insurance"},
    "has_unlimited_liability": {"Liability", "Limitation of Liability", "Indemnification", "Indemnity", GENERAL},
    "has_one_sided_termination": {"Termination", "Term", "Cancellation"},
    # IP attributes
    "ip_ownership_transfer": {"Intellectual Property", GENERAL},
    "ip_license_back": {"Intellectual Property", GENERAL},
    "ip_indemnification": {"Intellectual Property", "Indemnification", "Indemnity", GENERAL},
    "pre_existing_ip_acknowledged": {"Intellectual Property", GENERAL},
    "ip_work_for_hire": {"Intellectual Property", GENERAL},
    "ip_exclusive_license": {"Intellectual Property", GENERAL},
    "ip_perpetual_license": {"Intellectual Property", GENERAL},
    "ip_copyright_assignment": {"Intellectual Property", GENERAL},
    # Data attributes
    "data_ownership_defined": {"Data Ownership", "Data Protection", GENERAL},
    "data_usage_restricted": {"Data Ownership", "Data Protection", GENERAL},
    "data_deletion_obligation": {"Data Ownership", "Data Protection", GENERAL},
    "data_derived_ownership": {"Data Ownership", "Data Protection", GENERAL},
    "data_analytics_rights": {"Data Ownership", "Data Protection", GENERAL},
    "data_resale_rights": {"Data Ownership", "Data Protection", GENERAL},
    "data_sharing_rights": {"Data Ownership", "Data Protection", GENERAL},
    # Confidentiality attributes
    "confidentiality_broad_definition": {"Confidentiality", GENERAL},
    "confidentiality_standard_exclusions": {"Confidentiality", GENERAL},
    "confidentiality_return_obligation": {"Confidentiality", GENERAL},
    "confidentiality_mutual": {"Confidentiality", GENERAL},
    "confidentiality_duration_years": {"Confidentiality", GENERAL},
    "confidentiality_permitted_disclosures": {"Confidentiality", GENERAL},
    # Security attributes
    "security_measures_defined": {"Security Obligations", "Data Protection", GENERAL},
    "security_breach_notification": {"Security Obligations", "Data Protection", GENERAL},
    "security_audit_rights": {"Security Obligations", "Data Protection", GENERAL},
    "security_standard_compliant": {"Security Obligations", "Data Protection", GENERAL},
    "security_encryption_required": {"Security Obligations", "Data Protection", GENERAL},
    "security_access_control": {"Security Obligations", "Data Protection", GENERAL},
    "security_incident_response": {"Security Obligations", "Data Protection", GENERAL},
    "breach_notification_days": {"Security Obligations", "Data Protection", GENERAL},
    # Indemnification / LoL attributes
    "indemnification_mutual": {"Indemnification", "Indemnity", GENERAL},
    "indemnification_survival_years": {"Indemnification", "Indemnity", "Survival", GENERAL},
    "indemnification_third_party_claims": {"Indemnification", "Indemnity", GENERAL},
    "indemnification_defense_obligations": {"Indemnification", "Indemnity", GENERAL},
    "lol_exclusions_present": {"Limitation of Liability", "Liability", GENERAL},
    "lol_mutual": {"Limitation of Liability", "Liability", GENERAL},
    "lol_excludes_ip": {"Limitation of Liability", "Liability", GENERAL},
    "lol_carveout_confidentiality": {"Limitation of Liability", "Liability", GENERAL},
    "lol_carveout_fraud": {"Limitation of Liability", "Liability", GENERAL},
    "lol_carveout_gross_negligence": {"Limitation of Liability", "Liability", GENERAL},
    "lol_carveout_death_injury": {"Limitation of Liability", "Liability", GENERAL},
    # General
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

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_ownership_transfer"]):
            attributes["ip_ownership_transfer"] = self._detect_ip_ownership_transfer(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_license_back"]):
            attributes["ip_license_back"] = self._detect_ip_license_back(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_indemnification"]):
            attributes["ip_indemnification"] = self._detect_ip_indemnification(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["pre_existing_ip_acknowledged"]):
            attributes["pre_existing_ip_acknowledged"] = self._detect_pre_existing_ip(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_ownership_defined"]):
            attributes["data_ownership_defined"] = self._detect_data_ownership(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_usage_restricted"]):
            attributes["data_usage_restricted"] = self._detect_data_usage_restrictions(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_deletion_obligation"]):
            attributes["data_deletion_obligation"] = self._detect_data_deletion_obligation(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_broad_definition"]):
            attributes["confidentiality_broad_definition"] = self._detect_confidentiality_broad(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_standard_exclusions"]):
            attributes["confidentiality_standard_exclusions"] = self._detect_confidentiality_exclusions(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_return_obligation"]):
            attributes["confidentiality_return_obligation"] = self._detect_confidentiality_return(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_measures_defined"]):
            attributes["security_measures_defined"] = self._detect_security_measures(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_breach_notification"]):
            attributes["security_breach_notification"] = self._detect_security_breach_notification(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_audit_rights"]):
            attributes["security_audit_rights"] = self._detect_security_audit_rights(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["indemnification_mutual"]):
            attributes["indemnification_mutual"] = self._detect_indemnification_mutual(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["indemnification_survival_years"]):
            val = self._extract_indemnification_survival(clause_text)
            if val is not None:
                attributes["indemnification_survival_years"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_exclusions_present"]):
            attributes["lol_exclusions_present"] = self._detect_lol_exclusions(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_mutual"]):
            attributes["lol_mutual"] = self._detect_lol_mutual(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_excludes_ip"]):
            attributes["lol_excludes_ip"] = self._detect_lol_excludes_ip(clause_text)

        # New IP attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["ip_work_for_hire"]):
            attributes["ip_work_for_hire"] = self._detect_ip_work_for_hire(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_exclusive_license"]):
            attributes["ip_exclusive_license"] = self._detect_ip_exclusive_license(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_perpetual_license"]):
            attributes["ip_perpetual_license"] = self._detect_ip_perpetual_license(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["ip_copyright_assignment"]):
            attributes["ip_copyright_assignment"] = self._detect_ip_copyright_assignment(clause_text)

        # New Data attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["data_derived_ownership"]):
            attributes["data_derived_ownership"] = self._detect_data_derived_ownership(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_analytics_rights"]):
            attributes["data_analytics_rights"] = self._detect_data_analytics_rights(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_resale_rights"]):
            attributes["data_resale_rights"] = self._detect_data_resale_rights(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["data_sharing_rights"]):
            attributes["data_sharing_rights"] = self._detect_data_sharing_rights(clause_text)

        # New Confidentiality attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_mutual"]):
            attributes["confidentiality_mutual"] = self._detect_confidentiality_mutual(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_duration_years"]):
            val = self._extract_confidentiality_duration(clause_text)
            if val is not None:
                attributes["confidentiality_duration_years"] = val

        if self._type_matches(clause_type, EXTRACTION_GATES["confidentiality_permitted_disclosures"]):
            attributes["confidentiality_permitted_disclosures"] = self._detect_confidentiality_permitted_disclosures(clause_text)

        # New Security attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["security_standard_compliant"]):
            attributes["security_standard_compliant"] = self._detect_security_standard_compliant(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_encryption_required"]):
            attributes["security_encryption_required"] = self._detect_security_encryption_required(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_access_control"]):
            attributes["security_access_control"] = self._detect_security_access_control(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["security_incident_response"]):
            attributes["security_incident_response"] = self._detect_security_incident_response(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["breach_notification_days"]):
            val = self._extract_breach_notification_days(clause_text)
            if val is not None:
                attributes["breach_notification_days"] = val

        # New Indemnification attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["indemnification_third_party_claims"]):
            attributes["indemnification_third_party_claims"] = self._detect_indemnification_third_party(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["indemnification_defense_obligations"]):
            attributes["indemnification_defense_obligations"] = self._detect_indemnification_defense(clause_text)

        # New LoL carve-out attributes
        if self._type_matches(clause_type, EXTRACTION_GATES["lol_carveout_confidentiality"]):
            attributes["lol_carveout_confidentiality"] = self._detect_lol_carveout_confidentiality(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_carveout_fraud"]):
            attributes["lol_carveout_fraud"] = self._detect_lol_carveout_fraud(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_carveout_gross_negligence"]):
            attributes["lol_carveout_gross_negligence"] = self._detect_lol_carveout_gross_negligence(clause_text)

        if self._type_matches(clause_type, EXTRACTION_GATES["lol_carveout_death_injury"]):
            attributes["lol_carveout_death_injury"] = self._detect_lol_carveout_death_injury(clause_text)

        # spaCy NER enrichment for money values, dates, and law references
        self._enrich_with_ner(clause_text, clause_type, attributes)

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
            r"(?:upon|after|following|by|give|given|with)\s+(\d+)\s*(?:day|business\s*day|calendar\s*day)s?\s*(?:prior\s+)?(?:written\s+)?notice",
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

    def _detect_ip_ownership_transfer(self, text: str) -> bool:
        patterns = [
            r"(?:shall\s+be\s+)?(?:the\s+)?(?:sole\s+and\s+)?(?:exclusive\s+)?(?:property|owner|ownership)\s+(?:of|shall\s+(?:be\s+)?vest\s+in)",
            r"(?:assigns?\s+|hereby\s+assigns?\s+)(?:all\s+)?(?:right,\s+)?title\s+(?:and\s+)?interest\s+in\s+and\s+to",
            r"(?:shall\s+)?own\s+(?:all\s+)?(?:right,\s+)?title\s+(?:and\s+)?interest",
            r"(?:all\s+)?intellectual\s+property\s+(?:rights\s+)?(?:shall\s+)?(?:be\s+)?(?:the\s+)?(?:sole\s+and\s+)?exclusive\s+(?:property|ownership)",
            r"(?:ownership|title)\s+(?:to|of)\s+(?:any\s+)?(?:and\s+all\s+)?intellectual\s+property\s+(?:shall\s+)?(?:be\s+)?(?:vested\s+in|transferred\s+to|assigned\s+to)",
            r"(?:shall\s+)?own\s+(?:all\s+)?(?:intellectual\s+property|ip|rights)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_ip_license_back(self, text: str) -> bool:
        patterns = [
            r"(?:retains?\s+a\s+|hereby\s+retains?\s+|is\s+(?:hereby\s+)?granted\s+a\s+)(?:non.?exclusive|perpetual|irrevocable|royalty.?free|worldwide)",
            r"(?:grants?\s+|hereby\s+grants?\s+)(?:back\s+)?(?:to\s+)?(?:the\s+)?(?:contributing|providing|disclosing|licensor)\s+party\s+a\s+(?:non.?exclusive|perpetual|irrevocable|royalty.?free)",
            r"(?:license\s+back|grant.back|retained\s+license)",
            r"(?:shall\s+be\s+)?(?:deemed\s+to\s+)?(?:have\s+)?(?:been\s+)?granted\s+(?:a\s+)?(?:non.?exclusive|perpetual|irrevocable)\s+license",
            r"(?:non.?exclusive|perpetual|irrevocable|royalty.?free)\s+license\s+(?:to|back)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_ip_indemnification(self, text: str) -> bool:
        patterns = [
            r"(?:indemnify|hold\s+harmless)\s+(?:.*?)?(?:against|from|for)\s+(?:any|all)\s+(?:claim|action|liability|lawsuit|proceeding).*?(?:infring|violat|misappropri)",
            r"(?:indemnify|indemnification)\s+(?:against|from|for)\s+(?:any|all)\s+(?:claim|action|liability).*?(?:intellectual\s+property|patent|copyright|trademark|trade\s+secret)",
            r"(?:infringement|violation|misappropriation)\s+(?:of|claim)\s+(?:any\s+)?(?:third.?party\s+)?(?:rights|ip|intellectual\s+property)\s+(?:shall\s+)?(?:be\s+)?(?:defended|indemnified)",
            r"(?:ip\s+)?indemnif(?:y|ication)\s+(?:for|against)\s+(?:infringement|violation)",
            r"(?:indemnify|indemnification)\s+.*?(?:intellectual\s+property|patent|copyright|trademark)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_pre_existing_ip(self, text: str) -> bool:
        patterns = [
            r"(?:pre.?existing|preexisting|background|prior)\s+(?:intellectual\s+property|ip|technology|materials|work)",
            r"(?:existing\s+)?(?:intellectual\s+property|ip)\s+(?:owned|developed)\s+(?:prior\s+to|before|previously)",
            r"(?:retains?\s+)?(?:all\s+)?(?:right|ownership)\s+(?:in|to)\s+(?:its?\s+)?(?:pre.?existing|preexisting|background)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_ownership(self, text: str) -> bool:
        patterns = [
            r"(?:data|information)\s+(?:shall\s+)?(?:be\s+)?(?:owned\s+by|is\s+the\s+(?:sole\s+)?(?:and\s+exclusive\s+)?property\s+of)",
            r"(?:ownership|title)\s+(?:of|to)\s+(?:the\s+)?(?:data|information)\s+(?:shall\s+)?(?:be\s+)?(?:vested\s+in|retained\s+by)",
            r"(?:all\s+)?(?:data|information)\s+(?:shall\s+)?(?:remain\s+)?(?:the\s+)?(?:sole\s+and\s+)?exclusive\s+(?:property|ownership)",
            r"(?:each\s+)?party\s+(?:shall\s+)?(?:own|retain)\s+(?:all\s+)?(?:right|title|interest)\s+(?:in|to)\s+(?:its?\s+)?(?:data|information)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_usage_restrictions(self, text: str) -> bool:
        patterns = [
            r"(?:shall\s+)?(?:not\s+)?(?:use|process|disclose|share|transfer)\s+(?:the\s+)?(?:data|information)\s+(?:for\s+(?:any\s+)?(?:purpose|reason)\s+(?:other|beyond)|except\s+(?:as\s+)?(?:permitted|necessary))",
            r"(?:may\s+(?:only|solely|exclusively)|limited\s+to|restricted\s+to)\s+(?:use|process)\s+(?:the\s+)?(?:data|information)\s+(?:for\s+)?(?:the\s+)?(?:purpose|scope)",
            r"(?:data|information)\s+(?:shall\s+)?(?:only|not\s+be)\s+used\s+(?:for|in\s+connection\s+with)\s+(?:the\s+)?(?:purpose|agreement|service)",
            r"(?:no\s+right|no\s+license|no\s+ownership)\s+(?:is\s+)?granted\s+(?:to|in)\s+(?:the\s+)?(?:data|information)\s+(?:other\s+than|beyond|except)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_deletion_obligation(self, text: str) -> bool:
        patterns = [
            r"(?:shall\s+)?(?:delete|destroy|return|erase|remove)\s+(?:all\s+)?(?:copies\s+of\s+)?(?:the\s+)?(?:data|information)\s+(?:upon\s+|after\s+|following\s+|within\s+.*?\s+of\s+)(?:termination|expir|cancel|completion|conclusion)",
            r"(?:upon|after|following)\s+(?:termination|expir|cancel|completion)\s+(?:of\s+this\s+)?(?:agreement|contract)\s*[,;]?\s*(?:the\s+)?(?:data|information)\s+(?:shall\s+)?be\s+(?:deleted|destroyed|returned|erased)",
            r"(?:obligat|require)?(?:to\s+)?(?:delete|destroy|return|erase)\s+(?:any\s+)?(?:and\s+all\s+)?(?:data|information|confidential)",
            r"(?:certify|certification)\s+(?:in\s+)?writing\s+(?:that\s+)?(?:the\s+)?(?:data|information)\s+(?:has\s+)?(?:been\s+)?(?:deleted|destroyed|erased)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_confidentiality_broad(self, text: str) -> bool:
        patterns = [
            r"(?:any\s+(?:and\s+all\s+)?)(?:information|data|communication|material|document)",
            r"(?:all\s+)?(?:information|data|communication|material|document)\s+(?:of\s+)?(?:whatsoever|whatsoever\s+nature|in\s+any\s+form)",
            r"any\s+(?:and\s+all\s+)?(?:oral|written|electronic|visual)\s+(?:information|data|communication)",
            r"any\s+(?:and\s+all\s+)?information\s+(?:relating|pertaining)\s+to\s+(?:the\s+)?(?:business|company|agreement|relationship)",
            r"without\s+(?:limiting|restricting)\s+(?:the\s+)?(?:generality|foregoing)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_confidentiality_exclusions(self, text: str) -> bool:
        patterns = [
            r"(?:exception|excluding|excluded|does\s+not\s+apply)\s+(?:to|for)",
            r"(?:public\s+(?:domain|knowledge)|publicly\s+available|becomes\s+public)",
            r"(?:independently\s+(?:developed|created)|developed\s+without\s+use)",
            r"(?:rightfully\s+(?:received|obtained)\s+(?:from\s+)?(?:a\s+)?third\s+part)",
            r"(?:required\s+(?:to\s+be\s+)?disclosed|required\s+by\s+(?:law|regulation|legal\s+process))",
            r"(?:written\s+)?consent|authorization\s+(?:of\s+)?(?:the\s+)?(?:disclosing|owner)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_confidentiality_return(self, text: str) -> bool:
        patterns = [
            r"(?:shall\s+)?(?:return|destroy|delete|surrender|deliver\s+up)\s+(?:all\s+)?(?:copies\s+of\s+)?(?:the\s+)?(?:confidential\s+)?(?:information|material|document)",
            r"(?:promptly|immediately|within\s+\d+\s+(?:day|business\s*day)s?)\s+(?:upon|after|following)\s+(?:the\s+)?(?:request|termination|expir)",
            r"(?:return|destroy|delete)\s+(?:any\s+)?(?:and\s+all\s+)?(?:confidential\s+)?(?:information|material|document)\s+(?:in\s+)?(?:its?\s+)?possession",
            r"(?:shall\s+)?(?:cease\s+all\s+use|certify\s+in\s+writing)\s+(?:of\s+)?(?:the\s+)?(?:confidential\s+)?(?:information|material)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_measures(self, text: str) -> bool:
        patterns = [
            r"(?:implement|maintain|adopt|establish)\s+(?:and\s+(?:maintain|enforce))?\s+(?:appropriate|reasonable|adequate|technical|organizational|administrative|physical)\s+(?:security|safeguard|measure|control|procedure)",
            r"(?:encryption|firewall|access\s+control|multi.?factor\s+auth|intrusion\s+detection|anti.?virus)",
            r"(?:security\s+(?:measure|standard|control|policy|procedure|practice|requirement))",
            r"(?:information\s+security|cyber.security|cybersecurity|data\s+security)\s+(?:program|framework|policy|standard)",
            r"(?:ISO\s*27001|SOC\s*2|NIST|PCI\s*DSS|HIPAA|GDPR)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_breach_notification(self, text: str) -> bool:
        patterns = [
            r"(?:notify|notification|notify\s+promptly)\s+(?:the\s+)?(?:other\s+)?party\s+(?:in\s+)?(?:the\s+)?(?:event|case|situation)\s+(?:of\s+)?(?:a\s+)?(?:security|data)\s+(?:breach|incident|compromise)",
            r"(?:security|data|breach|personal\s+data)\s+(?:breach|incident|compromise)\s+(?:notification|notice|report)",
            r"(?:shall\s+)?(?:promptly|immediately|within\s+\d+\s*(?:hour|day)s?)\s+notify",
            r"(?:breach|incident|unauthorized\s+access)\s+(?:of|to)\s+(?:the\s+)?(?:data|system|information)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_audit_rights(self, text: str) -> bool:
        patterns = [
            r"(?:right|entitled|permitted)\s+(?:to\s+)?(?:audit|inspect|review|examine|assess)\s+(?:the\s+)?(?:security|system|facility|control|data\s+center)",
            r"(?:audit|inspection|assessment)\s+(?:right|provision|clause|obligation)",
            r"(?:upon\s+)?(?:reasonable\s+)?(?:notice|request)\s+(?:conduct|perform|carry\s+out)\s+(?:an\s+)?(?:audit|inspection|examination)",
            r"(?:audit|inspect|verify)\s+(?:compliance|adherence)\s+(?:with|to)\s+(?:the\s+)?(?:security|this)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_indemnification_mutual(self, text: str) -> bool:
        patterns = [
            r"(?:each\s+party|both\s+parties)\s+(?:shall|agrees|will)\s+(?:indemnify|defend|hold\s+harmless)",
            r"(?:mutual|reciprocal|cross)\s+indemnif",
            r"(?:indemnif(?:y|ication|ies))\s+(?:by|from)\s+(?:each|both)\s+(?:party|parties)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_indemnification_survival(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:indemnification|indemnity|indemnify)\s+(?:obligation|liability)?\s*(?:shall|will)?\s*(?:survive|continue|remain\s+in\s+effect)\s+(?:for\s+)?(?:a\s+(?:period\s+of\s+)?)?(\d+)\s*(?:year|month)s?",
            r"(?:survival|survive)\s+(?:period|term)?\s*(?:of|:)?\s*(\d+)\s*(?:year|month)s?\s*(?:after|following|from|subsequent)",
        ]
        val = self._extract_first_int(text, patterns)
        if val:
            return self._months_from_years_if_needed(val, text)
        return None

    def _detect_lol_exclusions(self, text: str) -> bool:
        patterns = [
            r"(?:does\s+not\s+apply|excluded|exceptions|notwithstanding|nothing\s+in\s+this)",
            r"(?:shall\s+not\s+(?:apply\s+to|limit|exclude|preclude))",
            r"(?:death|personal\s+injury|bodily\s+injury|gross\s+negligence|willful\s+misconduct|fraud|intentional)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_lol_mutual(self, text: str) -> bool:
        patterns = [
            r"(?:each\s+party|both\s+parties|neither\s+party)\s+(?:shall|will|may)\s+(?:be\s+)?(?:liable|responsible|subject)",
            r"(?:mutual|reciprocal)\s+(?:limitation|cap|exclusion)",
            r"(?:limitation|cap|exclusion)\s+(?:of\s+)?liability\s+(?:applies|shall\s+apply)\s+(?:to|equally|mutually)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_lol_excludes_ip(self, text: str) -> bool:
        patterns = [
            r"(?:intellectual\s+property|ip|patent|copyright|trademark|trade\s+secret)\s+(?:infringement|violation|misappropriation)",
            r"(?:infringement|violation)\s+(?:of\s+)?(?:intellectual\s+property|patent|copyright|trademark|trade\s+secret)",
            r"(?:ip|intellectual\s+property)\s+(?:indemnif|claim)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # ---------------------------------------------------------------
    # New IP Detection Methods
    # ---------------------------------------------------------------

    def _detect_ip_work_for_hire(self, text: str) -> bool:
        patterns = [
            r"\bwork[\s\-]+for[\s\-]+hire\b",
            r"\bwork\s+made\s+for\s+hire\b",
            r"\bmade[\s\-]+for[\s\-]+hire\b",
            r"\bwork\s+product\b.*\bshall\s+be\b.*\bproperty\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_ip_exclusive_license(self, text: str) -> bool:
        exclusive_patterns = [
            r"\bexclusive\s+licen[cs]e\b",
            r"\bgrants?\s+(?:an?\s+)?exclusive\s+(?:right|licen[cs]e)\b",
        ]
        non_exclusive_patterns = [
            r"\bnon[\s\-]?exclusive\s+licen[cs]e\b",
            r"\bnon[\s\-]?exclusive\s+(?:right|grant)\b",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in exclusive_patterns):
            if not any(re.search(p, text, re.IGNORECASE) for p in non_exclusive_patterns):
                return True
        return False

    def _detect_ip_perpetual_license(self, text: str) -> bool:
        patterns = [
            r"\bperpetual\s+licen[cs]e\b",
            r"\bperpetual(?:,\s*irrevocable)?\s+(?:right|licen[cs]e|grant)\b",
            r"\birrevocable(?:,\s*perpetual)?\s+licen[cs]e\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_ip_copyright_assignment(self, text: str) -> bool:
        patterns = [
            r"\bhereby\s+assigns?\b.*\b(?:copyright|patent|trademark|intellectual\s+property)\b",
            r"\bassignment\s+of\s+(?:copyright|patent|trademark|all\s+(?:intellectual\s+property|ip))\b",
            r"\b(?:copyright|patent)\s+(?:is\s+hereby\s+)?assigned\b",
            r"\btransfers?\s+(?:all\s+)?(?:copyright|patent|ip|intellectual\s+property)\s+(?:rights?\s+)?to\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # ---------------------------------------------------------------
    # New Data Detection Methods
    # ---------------------------------------------------------------

    def _detect_data_derived_ownership(self, text: str) -> bool:
        patterns = [
            r"\bderived\s+data\b",
            r"\baggregate(?:d)?\s+data\b",
            r"\banonymized?\s+data\b",
            r"\bde[\s\-]?identified\s+data\b",
            r"\binferred\s+data\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_analytics_rights(self, text: str) -> bool:
        patterns = [
            r"\b(?:use|process|analyze|analyse)\s+(?:customer|user|client)\s+data\s+(?:for|to)\s+(?:analytics|statistical|benchmarking|improvement|training)\b",
            r"\banalytics?\s+(?:rights?|data|purposes?)\b",
            r"\buse\s+(?:of\s+)?(?:data|information)\s+(?:for|to\s+(?:improve|train|develop|enhance))\b",
            r"\btelemetry\b",
            r"\busage\s+data\b.*\b(?:may\s+use|collect|process)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_resale_rights(self, text: str) -> bool:
        patterns = [
            r"\b(?:sell|resell|license|sublicen[cs]e|transfer|share|disclose)\s+(?:customer|user|client)\s+(?:data|information)\s+(?:to|with)\s+third[\s\-]?part(?:y|ies)\b",
            r"\b(?:monetize|commercialize)\s+(?:customer|user|client)?\s*data\b",
            r"\bsell\s+(?:or\s+)?(?:license|transfer)\s+(?:the\s+)?data\b",
            r"\b(?:data|information)\s+(?:may\s+be\s+)?sold\b",
            r"\bdata\s+(?:broker|marketplace|exchange)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_data_sharing_rights(self, text: str) -> bool:
        patterns = [
            r"\bshare\s+(?:customer|user|client)?\s*data\s+with\s+third[\s\-]?part(?:y|ies)\b",
            r"\bdisclose\s+(?:customer|user|client)\s+(?:data|information)\s+to\s+(?:third[\s\-]?part(?:y|ies)|affiliates|partners)\b",
            r"\bdata\s+sharing\s+(?:agreement|arrangement|rights?)\b",
            r"\bsubprocessor[s]?\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # ---------------------------------------------------------------
    # New Confidentiality Detection Methods
    # ---------------------------------------------------------------

    def _detect_confidentiality_mutual(self, text: str) -> bool:
        patterns = [
            r"\b(?:each\s+party|both\s+parties|mutual(?:ly)?)\s+(?:shall|agrees?|will)\s+(?:keep|hold|treat|maintain)\s+(?:the\s+other(?:'s)?\s+)?(?:confidential|proprietary)\b",
            r"\bmutual(?:ly)?\s+confidential\b",
            r"\b(?:each\s+party|both\s+parties)\s+(?:is|are)\s+(?:a\s+)?(?:disclosing\s+and\s+receiving|receiving\s+and\s+disclosing)\s+party\b",
            r"\breciprocal\s+confidentiality\b",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            return True
        # One-sided signal: only Company's info is protected
        one_sided = [
            r"\b(?:company|client|licensor|employer)\s+(?:confidential|proprietary)\s+information\b",
        ]
        both_parties = [r"\beach\s+party\b", r"\bboth\s+parties\b", r"\bmutual\b"]
        if any(re.search(p, text, re.IGNORECASE) for p in one_sided):
            if not any(re.search(p, text, re.IGNORECASE) for p in both_parties):
                return False
        return True  # default assume mutual if unclear

    def _extract_confidentiality_duration(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:confidentiality|non[\s\-]?disclosure)\s+(?:obligation[s]?|restriction[s]?|period)\s+(?:shall\s+)?(?:survive|continue|remain\s+in\s+effect)\s+(?:for\s+)?(?:a\s+(?:period\s+of\s+)?)?(\d+)\s*(?:year|month)s?",
            r"(?:for\s+(?:a\s+period\s+of\s+)?)?(\d+)\s*(?:year|month)s?\s+(?:following|after|from)\s+(?:the\s+)?(?:term|termination|expiry|end)\s+(?:of\s+this\s+(?:agreement|nda))?",
            r"(?:term\s+of\s+)?(\d+)\s*(?:year|month)s?\s+from\s+(?:the\s+)?(?:date|effective\s+date)\s+(?:of\s+)?(?:this\s+agreement|disclosure)",
        ]
        val = self._extract_first_int(text, patterns)
        if val:
            return self._months_from_years_if_needed(val, text)
        return None

    def _detect_confidentiality_permitted_disclosures(self, text: str) -> bool:
        patterns = [
            r"\b(?:may\s+disclose|permitted\s+to\s+disclose|disclosure\s+permitted)\s+(?:to\s+)?(?:its?\s+)?(?:employees?|officers?|directors?|attorneys?|advisors?|representatives?)\b",
            r"\bneed[\s\-]+to[\s\-]+know\s+basis\b",
            r"\b(?:permitted|allowed)\s+disclosures?\b",
            r"\bdisclose\s+(?:to\s+)?(?:its?\s+)?(?:legal\s+counsel|accountants?|auditors?)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # ---------------------------------------------------------------
    # New Security Detection Methods
    # ---------------------------------------------------------------

    def _detect_security_standard_compliant(self, text: str) -> bool:
        patterns = [
            r"\bSOC\s*2\b",
            r"\bISO\s*27001\b",
            r"\bNIST\b",
            r"\bPCI[\s\-]?DSS\b",
            r"\bHIPAA\b",
            r"\bGDPR\b",
            r"\bFedRAMP\b",
            r"\bCIS\s+(?:benchmark|control)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_encryption_required(self, text: str) -> bool:
        patterns = [
            r"\bencrypt(?:ion|ed)\s+(?:in\s+transit|at\s+rest|of\s+(?:data|information))\b",
            r"\bdata\s+(?:shall\s+(?:be\s+)?)?encrypted\b",
            r"\b(?:require[sd]?|mandate[sd]?)\s+encryption\b",
            r"\bAES[\s\-]?(?:128|192|256)\b",
            r"\bTLS\s+(?:1\.[23])\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_access_control(self, text: str) -> bool:
        patterns = [
            r"\b(?:role[\s\-]+based\s+)?access\s+control[s]?\b",
            r"\bleast[\s\-]+privilege\b",
            r"\bmulti[\s\-]+factor\s+auth(?:entication)?\b",
            r"\bMFA\b",
            r"\b(?:single\s+sign[\s\-]+on|SSO)\b",
            r"\bprivileged\s+access\s+(?:management|control)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_security_incident_response(self, text: str) -> bool:
        patterns = [
            r"\bincident\s+response\s+(?:plan|procedure|policy|process)\b",
            r"\b(?:respond|response)\s+to\s+(?:security\s+)?incidents?\b",
            r"\bsecurity\s+(?:incident|event)\s+(?:management|handling|response)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_breach_notification_days(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:notify|notification|report)\s+(?:within|no\s+later\s+than)\s+(\d+)\s*(?:hour|day|business\s*day)s?",
            r"(\d+)[\s\-]*(?:hour|day|business\s*day)s?\s+(?:breach|incident|security)?\s*notification",
            r"notification\s+(?:within|no\s+later\s+than)\s+(\d+)\s*(?:hour|day)s?",
        ]
        val = self._extract_first_int(text, patterns)
        # Convert hours to day fraction (store as hours for clarity; or keep as-is)
        return val

    # ---------------------------------------------------------------
    # New Indemnification Detection Methods
    # ---------------------------------------------------------------

    def _detect_indemnification_third_party(self, text: str) -> bool:
        patterns = [
            r"\bthird[\s\-]+party\s+(?:claim[s]?|action[s]?|lawsuit[s]?|proceeding[s]?)\b",
            r"\bthird[\s\-]+party\s+(?:liability|loss(?:es)?|damage[s]?)\b",
            r"\bclaim[s]?\s+(?:brought|made|asserted)\s+by\s+(?:any\s+)?third[\s\-]+part(?:y|ies)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_indemnification_defense(self, text: str) -> bool:
        patterns = [
            r"\b(?:defend|defense)\s+(?:and\s+)?(?:indemnify|hold\s+harmless)\b",
            r"\b(?:indemnify[,]?\s+defend[,]?\s+and\s+hold\s+harmless)\b",
            r"\b(?:at\s+its\s+own\s+expense)\s+(?:defend|assume\s+the\s+defense)\b",
            r"\bobligation\s+to\s+defend\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # ---------------------------------------------------------------
    # New LoL Carve-out Detection Methods
    # ---------------------------------------------------------------

    def _detect_lol_carveout_confidentiality(self, text: str) -> bool:
        patterns = [
            r"\bconfidentiality\s+(?:breach|obligation|violation)\b.*\b(?:excluded|not\s+(?:subject|limited)|notwithstanding)\b",
            r"\b(?:excluded|notwithstanding|except)\b.*\bbreach\s+of\s+confidentiality\b",
            r"\b(?:confidential(?:ity)?|proprietary)\s+information\b.*\b(?:excluded\s+from|not\s+(?:capped|limited))\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_lol_carveout_fraud(self, text: str) -> bool:
        patterns = [
            r"\bfraud(?:ulent)?\b.*\b(?:excluded|not\s+(?:subject|limited)|notwithstanding)\b",
            r"\b(?:excluded|notwithstanding|except)\b.*\bfraud\b",
            r"\bwillful\s+(?:misconduct|misrepresentation)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_lol_carveout_gross_negligence(self, text: str) -> bool:
        patterns = [
            r"\bgross\s+negligence\b.*\b(?:excluded|not\s+(?:subject|limited)|notwithstanding)\b",
            r"\b(?:excluded|notwithstanding|except)\b.*\bgross\s+negligence\b",
            r"\bgross\s+negligence\s+(?:or|and)\s+willful\s+misconduct\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _detect_lol_carveout_death_injury(self, text: str) -> bool:
        patterns = [
            r"\b(?:death|personal\s+injury|bodily\s+injury)\b.*\b(?:excluded|not\s+(?:subject|limited)|notwithstanding)\b",
            r"\b(?:excluded|notwithstanding|except)\b.*\b(?:death|personal\s+injury)\b",
            r"\b(?:cannot|may\s+not)\s+(?:be\s+)?(?:limited|capped|excluded)\b.*\b(?:death|personal\s+injury)\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _enrich_with_ner(self, text: str, clause_type: Optional[str], attributes: Dict):
        try:
            from app.services.nlp_service import get_nlp_service
            nlp = get_nlp_service()
            if nlp and nlp.available:
                if clause_type in ("Governing Law", "Dispute Resolution", "Arbitration"):
                    law_refs = nlp.extract_law_references(text)
                    if law_refs and "governing_law" not in attributes:
                        attributes["governing_law"] = law_refs[0]
                    orgs = nlp.extract_org_names(text)
                    if orgs and "governing_law" not in attributes:
                        attributes["governing_law"] = orgs[0]
                if clause_type in ("Insurance", "Liability", "Indemnification", "Limitation of Liability"):
                    money_vals = nlp.extract_money_values(text)
                    if money_vals and "liability_cap" not in attributes and "insurance_amount" not in attributes:
                        if clause_type == "Insurance":
                            attributes["insurance_amount"] = money_vals[0]
                        else:
                            attributes["liability_cap"] = money_vals[0]
                if clause_type in ("Termination", "Confidentiality"):
                    dates = nlp.extract_date_references(text)
                    if dates and "notice_days" not in attributes:
                        import re as _re
                        for d in dates:
                            nums = _re.findall(r'\d+', d)
                            if nums:
                                attributes["notice_days"] = int(nums[0])
                                break
        except Exception as e:
            logger.debug("spaCy NER enrichment failed: %s", e)

        self._enrich_with_trained_ner(text, clause_type, attributes)

    _trained_ner_model = None

    def _enrich_with_trained_ner(self, text: str, clause_type: Optional[str], attributes: Dict):
        if AttributeExtractor._trained_ner_model is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", "attribute_ner",
            )
            if os.path.exists(model_dir):
                try:
                    import spacy
                    AttributeExtractor._trained_ner_model = spacy.load(model_dir)
                    logger.info(f"Trained NER model loaded from {model_dir}")
                except Exception as e:
                    logger.debug("Failed to load trained NER model: %s", e)
                    AttributeExtractor._trained_ner_model = False

        nlp = AttributeExtractor._trained_ner_model
        if not nlp or not isinstance(nlp, object) or nlp is False:
            return

        try:
            doc = nlp(text)
            import re as _re

            for ent in doc.ents:
                if ent.label_ == "NOTICE_DAYS" and "notice_days" not in attributes:
                    nums = _re.findall(r'\d+', ent.text)
                    if nums:
                        attributes["notice_days"] = int(nums[0])

                if ent.label_ == "LIABILITY_AMOUNT" and "liability_cap" not in attributes and "insurance_amount" not in attributes:
                    nums = _re.findall(r'[\d,]+(?:\.\d+)?', ent.text)
                    if nums:
                        val = int(nums[0].replace(",", ""))
                        if clause_type == "Insurance":
                            attributes["insurance_amount"] = val
                        else:
                            attributes["liability_cap"] = val

                if ent.label_ == "PAYMENT_DEADLINE" and "payment_deadline_days" not in attributes:
                    nums = _re.findall(r'\d+', ent.text)
                    if nums:
                        attributes["payment_deadline_days"] = int(nums[0])

                if ent.label_ == "DURATION" and "non_compete_months" not in attributes:
                    nums = _re.findall(r'\d+', ent.text)
                    if nums:
                        val = int(nums[0])
                        if "month" in ent.text.lower():
                            attributes["non_compete_months"] = val
                        elif "year" in ent.text.lower() or "years" in ent.text.lower():
                            attributes["non_compete_months"] = val * 12

                if ent.label_ == "MONEY":
                    nums = _re.findall(r'[\d,]+(?:\.\d+)?', ent.text)
                    if nums:
                        val = int(nums[0].replace(",", ""))
                        if clause_type == "Insurance" and "insurance_amount" not in attributes:
                            attributes["insurance_amount"] = val
                        elif clause_type in ("Liability", "Limitation of Liability", "Indemnification") and "liability_cap" not in attributes:
                            attributes["liability_cap"] = val

                if ent.label_ == "LAW" and "governing_law" not in attributes:
                    attributes["governing_law"] = ent.text

        except Exception as e:
            logger.debug("Trained NER enrichment failed: %s", e)

