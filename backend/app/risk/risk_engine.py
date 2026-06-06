import logging
import re
from typing import Dict, List, Optional

from app.risk.risk_rules import ALL_RULES

logger = logging.getLogger(__name__)


CLAUSE_TYPE_ALIASES = {
    "Termination": {"Termination", "Term", "Termination for Convenience", "Termination for Cause", "Cancellation", "Dismissal", "Early Termination"},
    "Governing Law": {"Governing Law", "Dispute Resolution", "Jurisdiction", "Venue", "Choice of Law", "Applicable Law", "Legal Compliance", "Law"},
    "Confidentiality": {"Confidentiality", "NDA", "Non-Disclosure", "Non Disclosure", "Proprietary Information", "Confidential Information", "Data Protection", "Privacy"},
}


CLAUSE_TITLE_PATTERNS = {
    "Termination": [r"\bterminat", r"\bcancel", r"\bdismissal"],
    "Governing Law": [r"\bgoverning\s+law", r"\bdispute", r"\bjuri", r"\bvenue", r"\bchoice\s+of\s+law", r"\bapplicable\s+law"],
    "Confidentiality": [r"\bconfiden", r"\bnon.?disclos", r"\bproprietary", r"\bndas?\b"],
}


CLAUSE_FULLTEXT_PATTERNS = {
    "Termination": [
        r"\b(?:this\s+)?agreement\s+(?:may\s+)?(?:be\s+)?terminated",
        r"\bright\s+(?:of\s+)?(?:either\s+)?party\s+to\s+terminate",
        r"\bterminat(?:e|ion)\s+(?:for\s+)?(?:cause|convenience|breach)",
        r"\bnotice\s+of\s+termination",
        r"\btermination\s+(?:rights?|notice|effect)",
        r"\bsection\s+\d+[\.\d]*\s*(?:early\s+)?termination",
    ],
    "Governing Law": [
        r"\bgovern(?:ed|ing)?\s+(?:by|law)",
        r"\bconstrued\s+(?:in\s+accordance\s+with|under)\s+(?:the\s+)?laws?\s+of",
        r"\b(?:this\s+)?(?:agreement|contract)\s+(?:shall\s+)?be\s+governed",
        r"\bchoice\s+of\s+law",
    ],
    "Confidentiality": [
        r"\bdeemed\s+confidential\b",
        r"\bconfidential\s+information\b",
        r"\bshall\s+(?:be\s+)?(?:held|kept|treated)\s+confidential",
        r"\binformation\s+.*?\bconfidential\b",
        r"\bnon.?disclosure",
        r"\bproprietary\s+information\b",
        r"\btrade\s+secrets?\b",
    ],
}


def _check_title_for_type(clauses: List[dict], clause_type: str) -> bool:
    patterns = CLAUSE_TITLE_PATTERNS.get(clause_type, [])
    if not patterns:
        return False
    for clause in clauses:
        title = clause.get("clause_title", "") or ""
        if any(re.search(p, title, re.IGNORECASE) for p in patterns):
            return True
    return False


def _scan_full_text(full_text: str, clause_type: str) -> bool:
    patterns = CLAUSE_FULLTEXT_PATTERNS.get(clause_type, [])
    if not patterns:
        return False
    return any(re.search(p, full_text, re.IGNORECASE) for p in patterns)


def _alias_matches(clause_types_found: List[str], target_type: str) -> bool:
    aliases = CLAUSE_TYPE_ALIASES.get(target_type, {target_type})
    for found in clause_types_found:
        for alias in aliases:
            if found.lower() == alias.lower():
                return True
    return False


class RiskEngine:
    def __init__(self):
        self.rules = ALL_RULES

    def evaluate_clause(self, clause: dict, attributes: dict) -> List[Dict]:
        findings = []
        for rule in self.rules:
            try:
                result = rule.evaluate(clause, attributes)
                if result:
                    findings.append(
                        {
                            "rule_id": result["rule_id"],
                            "risk_name": result["risk_name"],
                            "severity": result["severity"],
                            "points": result["points"],
                            "supporting_clause": clause.get("clause_text", "")[:500],
                            "extracted_value": result.get("extracted_value", ""),
                            "explanation": result["explanation"],
                            "clause_index": clause.get("clause_index"),
                            "clause_type": clause.get("clause_type"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.rule_id}: {e}")
        return findings

    def evaluate_missing_clauses(
        self, clauses: List[dict], clause_types_found: List[str], full_text: str = ""
    ) -> List[Dict]:
        findings = []
        required_clauses = [
            ("Confidentiality", "Medium", 15, "R004"),
            ("Governing Law", "Low", 5, "R007"),
            ("Termination", "Medium", 10, "R008"),
        ]
        for clause_type, severity, points, rule_id in required_clauses:
            present = _alias_matches(clause_types_found, clause_type)

            if not present:
                present = _check_title_for_type(clauses, clause_type)

            if not present and full_text:
                present = _scan_full_text(full_text, clause_type)

            if not present:
                rule_name_map = {
                    "Confidentiality": "Missing Confidentiality Clause",
                    "Governing Law": "Missing Governing Law Clause",
                    "Termination": "Missing Termination Clause",
                }
                findings.append(
                    {
                        "rule_id": rule_id,
                        "risk_name": rule_name_map.get(clause_type, f"Missing {clause_type}"),
                        "severity": severity,
                        "points": points,
                        "supporting_clause": "No matching clause found in contract",
                        "extracted_value": f"No {clause_type.lower()} clause",
                        "clause_index": None,
                        "clause_type": None,
                        "explanation": (
                            f"The contract does not contain a {clause_type.lower()} clause. "
                            f"This is a significant gap that could create legal uncertainty and risk."
                        ),
                    }
                )
        return findings

    def calculate_risk_score(self, findings: List[Dict]) -> int:
        return sum(f.get("points", 0) for f in findings)

    def get_risk_level(self, score: int) -> str:
        if score <= 25:
            return "Low"
        elif score <= 50:
            return "Moderate"
        elif score <= 75:
            return "High"
        else:
            return "Critical"
