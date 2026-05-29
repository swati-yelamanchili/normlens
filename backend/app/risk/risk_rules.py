from typing import Dict, List, Optional


class RiskRuleSet:
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        category: str,
        severity: str,
        points: int,
        condition_fn,
        explanation_template: str,
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.points = points
        self.condition_fn = condition_fn
        self.explanation_template = explanation_template

    def evaluate(self, clause: dict, attributes: dict) -> Optional[dict]:
        result = self.condition_fn(clause, attributes)
        if result:
            return {
                "rule_id": self.rule_id,
                "risk_name": self.name,
                "severity": self.severity,
                "points": self.points,
                "explanation": self._generate_explanation(attributes, result),
                "extracted_value": result.get("extracted_value"),
            }
        return None

    def _generate_explanation(self, attributes: dict, result: dict) -> str:
        return self.explanation_template.format(
            **attributes,
            **result,
            extracted_value=result.get("extracted_value", "N/A"),
        )


def _condition_excessive_notice(clause: dict, attributes: dict) -> Optional[dict]:
    notice_days = attributes.get("notice_days")
    if notice_days is not None and notice_days > 90:
        return {
            "detected": True,
            "extracted_value": f"{notice_days} days",
            "threshold": "90 days",
            "actual": notice_days,
        }
    return None


def _condition_unlimited_liability(clause: dict, attributes: dict) -> Optional[dict]:
    if attributes.get("has_unlimited_liability"):
        return {
            "detected": True,
            "extracted_value": "Unlimited liability clause detected",
            "threshold": "Liability should be capped",
        }
    return None


def _condition_one_sided_termination(clause: dict, attributes: dict) -> Optional[dict]:
    if attributes.get("has_one_sided_termination"):
        return {
            "detected": True,
            "extracted_value": "One-sided termination detected",
            "threshold": "Termination should be mutual",
        }
    return None


def _condition_missing_confidentiality(clause: dict, attributes: dict) -> Optional[dict]:
    clause_type = clause.get("clause_type", "")
    if clause_type and clause_type != "Confidentiality":
        return None
    return None


def _condition_missing_clause(clause_type: str, clauses: list) -> Optional[dict]:
    found = any(
        c.get("clause_type") == clause_type for c in clauses
    )
    if not found:
        return {
            "detected": True,
            "extracted_value": f"No {clause_type.lower()} clause found",
        }
    return None


def _condition_excessive_non_compete(clause: dict, attributes: dict) -> Optional[dict]:
    months = attributes.get("non_compete_months")
    if months is not None and months > 12:
        return {
            "detected": True,
            "extracted_value": f"{months} months",
            "threshold": "12 months",
        }
    return None


def _condition_high_liability_cap(clause: dict, attributes: dict) -> Optional[dict]:
    cap = attributes.get("liability_cap")
    if cap:
        try:
            cap_clean = cap.replace(",", "").upper()
            if "MILLION" in cap_clean:
                val = float(cap_clean.split()[0]) * 1_000_000
            elif "BILLION" in cap_clean:
                val = float(cap_clean.split()[0]) * 1_000_000_000
            elif "THOUSAND" in cap_clean or "K" in cap_clean:
                val = float(cap_clean.split()[0]) * 1_000
            elif "M" in cap_clean and not cap_clean.startswith("M"):
                val = float(cap_clean.split()[0]) * 1_000_000
            else:
                val = float(cap_clean.split()[0])
            if val > 10_000_000:
                return {
                    "detected": True,
                    "extracted_value": cap,
                    "threshold": "$10,000,000",
                }
        except (ValueError, IndexError):
            pass
    return None


def _condition_long_payment_terms(clause: dict, attributes: dict) -> Optional[dict]:
    days = attributes.get("payment_deadline_days")
    if days is not None and days > 60:
        return {
            "detected": True,
            "extracted_value": f"{days} days",
            "threshold": "60 days",
        }
    return None


def _condition_no_data_protection(clause: dict, attributes: dict) -> Optional[dict]:
    return None


ALL_RULES: List[RiskRuleSet] = [
    RiskRuleSet(
        rule_id="R001",
        name="Excessive Notice Period",
        description="Notice period exceeds 90 days threshold",
        category="Termination Risk",
        severity="Medium",
        points=20,
        condition_fn=_condition_excessive_notice,
        explanation_template=(
            "Notice period of {extracted_value} exceeds the recommended threshold of {threshold}. "
            "Long notice periods may create operational inflexibility."
        ),
    ),
    RiskRuleSet(
        rule_id="R002",
        name="Unlimited Liability",
        description="Liability clause contains unlimited liability provisions",
        category="Liability Risk",
        severity="Critical",
        points=40,
        condition_fn=_condition_unlimited_liability,
        explanation_template=(
            "The contract includes unlimited liability provisions ({extracted_value}). "
            "This exposes the party to uncapped financial risk. Standard market practice is to cap liability."
        ),
    ),
    RiskRuleSet(
        rule_id="R003",
        name="One-Sided Termination",
        description="Termination rights are materially imbalanced",
        category="Termination Risk",
        severity="High",
        points=30,
        condition_fn=_condition_one_sided_termination,
        explanation_template=(
            "One-sided termination rights detected ({extracted_value}). "
            "Standard practice requires mutual termination rights for both parties."
        ),
    ),
    RiskRuleSet(
        rule_id="R004",
        name="Excessive Non-Compete Duration",
        description="Non-compete period exceeds 12 months",
        category="Non-Compete Risk",
        severity="Medium",
        points=20,
        condition_fn=_condition_excessive_non_compete,
        explanation_template=(
            "Non-compete duration of {extracted_value} exceeds standard market threshold of {threshold}. "
            "Extended non-compete periods may be commercially unreasonable and potentially unenforceable."
        ),
    ),
    RiskRuleSet(
        rule_id="R005",
        name="High Liability Cap",
        description="Liability cap exceeds $10M threshold",
        category="Liability Risk",
        severity="Medium",
        points=15,
        condition_fn=_condition_high_liability_cap,
        explanation_template=(
            "Liability cap of {extracted_value} exceeds standard threshold of {threshold}. "
            "High liability caps increase financial exposure."
        ),
    ),
    RiskRuleSet(
        rule_id="R006",
        name="Extended Payment Terms",
        description="Payment terms exceed 60 days",
        category="Payment Risk",
        severity="Low",
        points=10,
        condition_fn=_condition_long_payment_terms,
        explanation_template=(
            "Payment deadline of {extracted_value} exceeds standard threshold of {threshold}. "
            "Extended payment terms may impact cash flow."
        ),
    ),
]
