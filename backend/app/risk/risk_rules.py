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
        values = {**attributes, **result}
        values.setdefault("extracted_value", "N/A")
        return self.explanation_template.format(**values)


ALLOWED_CLAUSE_TYPES = {
    "excessive_notice": {"Termination", "Term"},
    "unlimited_liability": {"Liability", "Indemnification", "Indemnity"},
    "one_sided_termination": {"Termination"},
    "excessive_non_compete": {"Non-Compete", "Noncompete", "Non Competition"},
    "high_liability_cap": {"Liability", "Indemnification", "Indemnity"},
    "long_payment_terms": {"Payment Terms", "Payment"},
}


def _clause_type_in(clause: dict, allowed: set) -> bool:
    ct = clause.get("clause_type")
    if not ct:
        return False
    return any(ct.lower() == a.lower() for a in allowed)


def _condition_excessive_notice(clause: dict, attributes: dict) -> Optional[dict]:
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["excessive_notice"]):
        return None
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
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["unlimited_liability"]):
        return None
    if attributes.get("has_unlimited_liability"):
        return {
            "detected": True,
            "extracted_value": "Unlimited liability clause detected",
            "threshold": "Liability should be capped",
        }
    return None


def _condition_one_sided_termination(clause: dict, attributes: dict) -> Optional[dict]:
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["one_sided_termination"]):
        return None
    if attributes.get("has_one_sided_termination"):
        return {
            "detected": True,
            "extracted_value": "One-sided termination detected",
            "threshold": "Termination should be mutual",
        }
    return None


def _condition_excessive_non_compete(clause: dict, attributes: dict) -> Optional[dict]:
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["excessive_non_compete"]):
        return None
    months = attributes.get("non_compete_months")
    if months is not None and months > 12:
        return {
            "detected": True,
            "extracted_value": f"{months} months",
            "threshold": "12 months",
        }
    return None


_LIABILITY_MULTIPLIERS = [
    ("BILLION", 1_000_000_000),
    ("MILLION", 1_000_000),
    ("THOUSAND", 1_000),
    ("K", 1_000),
    ("M", 1_000_000),
]


def _parse_monetary_value(cap: str) -> Optional[float]:
    cleaned = cap.replace(",", "").replace("$", "").replace("USD", "").replace("EUR", "").replace("GBP", "").replace("INR", "").strip().upper()
    parts = cleaned.split()
    if not parts:
        return None
    for suffix, multiplier in _LIABILITY_MULTIPLIERS:
        if suffix in cleaned:
            try:
                idx = parts.index(suffix)
                if idx > 0:
                    return float(parts[idx - 1]) * multiplier
                return None
            except (ValueError, IndexError):
                continue
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _condition_high_liability_cap(clause: dict, attributes: dict) -> Optional[dict]:
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["high_liability_cap"]):
        return None
    cap = attributes.get("liability_cap")
    if cap:
        val = _parse_monetary_value(cap)
        if val is not None and val > 10_000_000:
            return {
                "detected": True,
                "extracted_value": cap,
                "threshold": "$10,000,000",
            }
    return None


def _condition_long_payment_terms(clause: dict, attributes: dict) -> Optional[dict]:
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["long_payment_terms"]):
        return None
    days = attributes.get("payment_deadline_days")
    if days is not None and days > 60:
        return {
            "detected": True,
            "extracted_value": f"{days} days",
            "threshold": "60 days",
        }
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
