import logging
import re
from typing import Dict, List, Optional

from app.risk.risk_rules import ALL_RULES
from app.risk.contract_type_detector import get_required_clauses_for_type

logger = logging.getLogger(__name__)


CLAUSE_TYPE_ALIASES = {
    "Termination": {"Termination", "Term", "Termination for Convenience", "Termination for Cause", "Cancellation", "Dismissal", "Early Termination"},
    "Governing Law": {"Governing Law", "Dispute Resolution", "Jurisdiction", "Venue", "Choice of Law", "Applicable Law", "Legal Compliance", "Law"},
    "Confidentiality": {"Confidentiality", "NDA", "Non-Disclosure", "Non Disclosure", "Proprietary Information", "Confidential Information", "Data Protection", "Privacy"},
    "Intellectual Property": {"Intellectual Property", "IP", "IP Rights", "IP Ownership", "Intellectual Property Rights", "Copyright", "Patent", "Trademark"},
    "Data Ownership": {"Data Ownership", "Data Rights", "Data Usage", "Data", "Data Processing", "Data Protection"},
    "Security Obligations": {"Security Obligations", "Security", "Information Security", "Data Security", "Cyber Security", "Data Protection", "IT Security"},
}

CLAUSE_TITLE_PATTERNS = {
    "Termination": [r"\bterminat", r"\bcancel", r"\bdismissal"],
    "Governing Law": [r"\bgoverning\s+law", r"\bdispute", r"\bjuri", r"\bvenue", r"\bchoice\s+of\s+law", r"\bapplicable\s+law"],
    "Confidentiality": [r"\bconfiden", r"\bnon.?disclos", r"\bproprietary", r"\bndas?\b"],
    "Intellectual Property": [r"\bintellectual\s+property", r"\bip\s+(rights|ownership)", r"\bcopyright", r"\bpatent", r"\btrademark"],
    "Data Ownership": [r"\bdata\s+(ownership|rights|usage)", r"\bdata\s+protection", r"\bdata\s+processing"],
    "Security Obligations": [r"\bsecurity\s+(obligation|measure|control|standard|requirement)", r"\binformation\s+security", r"\bdata\s+security", r"\bcyber.security"],
}

CLAUSE_FULLTEXT_PATTERNS = {
    "Termination": [
        r"\b(?:this\s+)?agreement\s+(?:may\s+)?(?:be\s+)?terminated",
        r"\bright\s+(?:of\s+)?(?:either\s+)?party\s+to\s+terminate",
        r"\bterminat(?:e|ion)\s+(?:for\s+)?(?:cause|convenience|breach)",
        r"\bnotice\s+of\s+termination",
    ],
    "Governing Law": [
        r"\bgovern(?:ed|ing)?\s+(?:by|law)",
        r"\bconstrued\s+(?:in\s+accordance\s+with|under)\s+(?:the\s+)?laws?\s+of",
        r"\bchoice\s+of\s+law",
    ],
    "Confidentiality": [
        r"\bdeemed\s+confidential\b",
        r"\bconfidential\s+information\b",
        r"\bshall\s+(?:be\s+)?(?:held|kept|treated)\s+confidential",
        r"\bnon.?disclosure",
        r"\bproprietary\s+information\b",
        r"\btrade\s+secrets?\b",
    ],
    "Intellectual Property": [
        r"\bintellectual\s+property\s+(?:rights|ownership|assignment)",
        r"\b(?:all\s+)?(?:right,\s+)?title\s+(?:and\s+)?interest\s+in\s+(?:and\s+to\s+)?(?:any\s+)?intellectual\s+property",
        r"\b(?:owns?|ownership)\s+(?:all\s+)?(?:intellectual\s+property|ip)",
        r"\b(?:shall\s+)?(?:be\s+)?(?:the\s+)?sole\s+(?:and\s+exclusive\s+)?(?:owner|property)",
        r"\bwork[\s\-]+for[\s\-]+hire\b",
        r"\bwork\s+made\s+for\s+hire\b",
        r"\blicen[cs]e\s+grant\b",
        r"\bgrants?\s+(?:a\s+)?(?:non.?exclusive|exclusive|perpetual)\s+licen[cs]e\b",
    ],
    "Data Ownership": [
        r"\b(?:data|information)\s+(?:shall\s+)?(?:be\s+)?(?:owned|the\s+property)",
        r"\b(?:ownership|title)\s+(?:of|to)\s+(?:the\s+)?(?:data|information)",
        r"\bdata\s+(?:processing|usage|sharing|deletion|retention)",
        r"\b(?:shall\s+)?(?:delete|return|destroy)\s+(?:any\s+)?(?:and\s+all\s+)?(?:data|information)",
        r"\bderived\s+data\b",
        r"\baggregate\s+data\b",
        r"\banonymized\s+data\b",
    ],
    "Security Obligations": [
        r"\b(?:implement|maintain|adopt)\s+(?:appropriate|reasonable|adequate)\s+(?:security|safeguard|measure)",
        r"\b(?:security|information\s+security|data\s+security)\s+(?:program|policy|standard|framework)",
        r"\b(?:encryption|firewall|access\s+control|intrusion\s+detection|incident\s+response)",
        r"\b(?:breach\s+notification|security\s+breach|data\s+breach|incident\s+response)",
        r"\b(?:SOC\s*2|ISO\s*27001|NIST|PCI\s*DSS|HIPAA)",
    ],
}

# Attribute keys that confirm clause presence (to eliminate false positives)
CLAUSE_PRESENCE_ATTRIBUTES = {
    "Confidentiality": [
        "confidentiality_broad_definition", "confidentiality_standard_exclusions",
        "confidentiality_return_obligation", "confidentiality_mutual",
        "confidentiality_duration_years", "confidentiality_permitted_disclosures",
    ],
    "Intellectual Property": [
        "ip_ownership_transfer", "ip_license_back", "ip_indemnification",
        "pre_existing_ip_acknowledged", "ip_work_for_hire", "ip_exclusive_license",
        "ip_copyright_assignment",
    ],
    "Data Ownership": [
        "data_ownership_defined", "data_usage_restricted", "data_deletion_obligation",
        "data_resale_rights", "data_analytics_rights", "data_sharing_rights",
        "data_derived_ownership",
    ],
    "Security Obligations": [
        "security_measures_defined", "security_breach_notification",
        "security_audit_rights", "security_standard_compliant",
        "security_encryption_required", "security_incident_response",
        "breach_notification_days",
    ],
}


def _check_title_for_type(clauses: List[dict], clause_type: str) -> bool:
    patterns = CLAUSE_TITLE_PATTERNS.get(clause_type, [])
    for clause in clauses:
        title = clause.get("clause_title", "") or ""
        if any(re.search(p, title, re.IGNORECASE) for p in patterns):
            return True
    return False


def _scan_full_text(full_text: str, clause_type: str) -> bool:
    patterns = CLAUSE_FULLTEXT_PATTERNS.get(clause_type, [])
    return any(re.search(p, full_text, re.IGNORECASE) for p in patterns)


def _alias_matches(clause_types_found: List[str], target_type: str) -> bool:
    aliases = CLAUSE_TYPE_ALIASES.get(target_type, {target_type})
    for found in clause_types_found:
        for alias in aliases:
            if found.lower() == alias.lower():
                return True
    return False


def _attribute_confirms_presence(classified_clauses: List[dict], clause_type: str) -> bool:
    """Check if any clause's extracted attributes confirm the clause type is present."""
    attr_keys = CLAUSE_PRESENCE_ATTRIBUTES.get(clause_type, [])
    if not attr_keys:
        return False
    for clause in classified_clauses:
        attrs = clause.get("attributes", {})
        for key in attr_keys:
            if key in attrs:
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
                    findings.append({
                        "rule_id": result["rule_id"],
                        "risk_name": result["risk_name"],
                        "severity": result["severity"],
                        "points": result["points"],
                        "supporting_clause": clause.get("clause_text", "")[:500],
                        "extracted_value": result.get("extracted_value", ""),
                        "explanation": result["explanation"],
                        "clause_index": clause.get("clause_index"),
                        "clause_type": clause.get("clause_type"),
                        "finding_category": result.get("finding_category", "Risky Provision"),
                        "clause_group": result.get("clause_group", "General"),
                        "negotiation_recommendation": result.get("negotiation_recommendation", ""),
                    })
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.rule_id}: {e}")
        return findings

    def evaluate_missing_clauses(
        self,
        clauses: List[dict],
        clause_types_found: List[str],
        full_text: str = "",
        contract_type: str = "General Commercial Agreement",
    ) -> List[Dict]:
        findings = []
        required_clauses = get_required_clauses_for_type(contract_type)

        # Always ensure Governing Law is checked with default severity
        if "Governing Law" not in required_clauses:
            required_clauses["Governing Law"] = ("Low", 5)

        missing_rule_ids = {
            "Confidentiality": "MC001",
            "Governing Law": "MC002",
            "Termination": "MC003",
            "Intellectual Property": "MC004",
            "Data Ownership": "MC005",
            "Security Obligations": "MC006",
        }

        missing_recommendations = {
            "Confidentiality": "Add a mutual confidentiality clause protecting both parties' proprietary information, with standard exclusions and return obligations.",
            "Governing Law": "Add a governing law and dispute resolution clause specifying jurisdiction, venue, and dispute process.",
            "Termination": "Add a termination clause defining termination rights for both parties, notice periods, and post-termination obligations.",
            "Intellectual Property": "Add an IP clause clarifying ownership of pre-existing IP, work product, and license grants.",
            "Data Ownership": "Add a data ownership clause clarifying who owns customer data, permitted usage, and deletion obligations.",
            "Security Obligations": "Add a security obligations clause specifying required security standards, controls, and breach notification obligations.",
        }

        for clause_type, (severity, points) in required_clauses.items():
            if clause_type not in missing_rule_ids:
                continue

            # Multi-layer presence check
            present = _alias_matches(clause_types_found, clause_type)

            if not present:
                present = _check_title_for_type(clauses, clause_type)

            if not present and full_text:
                present = _scan_full_text(full_text, clause_type)

            # Attribute-backed presence check (eliminates false positives)
            if not present:
                present = _attribute_confirms_presence(clauses, clause_type)

            if not present:
                rule_name_map = {
                    "Confidentiality": "Missing Confidentiality Clause",
                    "Governing Law": "Missing Governing Law Clause",
                    "Termination": "Missing Termination Clause",
                    "Intellectual Property": "Missing Intellectual Property Clause",
                    "Data Ownership": "Missing Data Ownership Clause",
                    "Security Obligations": "Missing Security Obligations Clause",
                }
                findings.append({
                    "rule_id": missing_rule_ids[clause_type],
                    "risk_name": rule_name_map.get(clause_type, f"Missing {clause_type}"),
                    "severity": severity,
                    "points": points,
                    "supporting_clause": "No matching clause found in contract",
                    "extracted_value": f"No {clause_type.lower()} clause detected",
                    "clause_index": None,
                    "clause_type": None,
                    "finding_category": "Missing Protection",
                    "clause_group": self._clause_type_to_group(clause_type),
                    "negotiation_recommendation": missing_recommendations.get(clause_type, ""),
                    "explanation": (
                        f"The contract does not contain a {clause_type.lower()} clause. "
                        f"For a {contract_type}, this is a {'critical' if severity in ('Critical', 'High') else 'notable'} gap "
                        f"that could create legal uncertainty and risk."
                    ),
                })
        return findings

    def _clause_type_to_group(self, clause_type: str) -> str:
        mapping = {
            "Confidentiality": "Confidentiality",
            "Governing Law": "General",
            "Termination": "Termination",
            "Intellectual Property": "IP",
            "Data Ownership": "Data",
            "Security Obligations": "Security",
        }
        return mapping.get(clause_type, "General")

    def deduplicate_findings(self, findings: List[Dict]) -> List[Dict]:
        """
        Merge duplicate findings with the same rule_id.
        Keeps highest severity and max points, accumulates supporting clauses.
        """
        severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        grouped: Dict[str, Dict] = {}

        for f in findings:
            key = f.get("rule_id") or f.get("risk_name", "unknown")
            if key not in grouped:
                grouped[key] = {
                    **f,
                    "supporting_clauses": [],
                    "clause_ids": [],
                }
                sc = f.get("supporting_clause", "")
                if sc and sc != "No matching clause found in contract":
                    grouped[key]["supporting_clauses"].append(sc)
                cid = f.get("clause_id")
                if cid:
                    grouped[key]["clause_ids"].append(cid)
            else:
                existing = grouped[key]
                new_sev = severity_order.get(f["severity"], 0)
                cur_sev = severity_order.get(existing["severity"], 0)
                if new_sev > cur_sev:
                    existing["severity"] = f["severity"]
                    existing["explanation"] = f["explanation"]
                # Always keep max points across all duplicates
                existing["points"] = max(existing["points"], f.get("points", 0))
                # Collect unique supporting clauses
                sc = f.get("supporting_clause", "")
                if sc and sc != "No matching clause found in contract" and sc not in existing["supporting_clauses"]:
                    existing["supporting_clauses"].append(sc)
                # Collect unique clause IDs
                cid = f.get("clause_id")
                if cid and cid not in existing["clause_ids"]:
                    existing["clause_ids"].append(cid)

        result = list(grouped.values())
        for r in result:
            if not r.get("supporting_clause") and r.get("supporting_clauses"):
                r["supporting_clause"] = r["supporting_clauses"][0]
        return result

    def calculate_risk_score(self, findings: List[Dict]) -> int:
        return sum(f.get("points", 0) for f in findings)

    def get_risk_level(self, score: int) -> str:
        if score <= 30:
            return "Low"
        elif score <= 70:
            return "Moderate"
        elif score <= 120:
            return "High"
        else:
            return "Critical"
