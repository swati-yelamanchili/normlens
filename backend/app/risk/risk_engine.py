import logging
from typing import Dict, List, Optional

from app.risk.risk_rules import ALL_RULES, _condition_missing_clause

logger = logging.getLogger(__name__)


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
        self, clauses: List[dict], clause_types_found: List[str]
    ) -> List[Dict]:
        findings = []
        required_clauses = [
            ("Confidentiality", "Medium", 15, "R004"),
            ("Governing Law", "Low", 5, "R007"),
            ("Termination", "Medium", 10, "R008"),
        ]
        for clause_type, severity, points, rule_id in required_clauses:
            if clause_type not in clause_types_found:
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
                        "explanation": (
                            f"The contract does not contain a {clause_type.lower()} clause. "
                            f"This is a significant gap that could create legal uncertainty and risk."
                        ),
                        "clause_index": None,
                        "clause_type": None,
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
