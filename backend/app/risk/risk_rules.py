from typing import Dict, List, Optional


CLAUSE_GROUP_MAP = {
    "Termination Risk": "Termination",
    "Liability Risk": "Liability",
    "Indemnification Risk": "Indemnification",
    "IP Risk": "IP",
    "Data Risk": "Data",
    "Security Risk": "Security",
    "Confidentiality Risk": "Confidentiality",
    "Non-Compete Risk": "Termination",
    "Payment Risk": "Payment",
}


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
        finding_category: str = "Risky Provision",
        negotiation_template: str = "",
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.points = points
        self.condition_fn = condition_fn
        self.explanation_template = explanation_template
        self.finding_category = finding_category
        self.clause_group = CLAUSE_GROUP_MAP.get(category, "General")
        self.negotiation_template = negotiation_template

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
                "finding_category": self.finding_category,
                "clause_group": self.clause_group,
                "negotiation_recommendation": self.negotiation_template,
            }
        return None

    def _generate_explanation(self, attributes: dict, result: dict) -> str:
        values = {**attributes, **result}
        values.setdefault("extracted_value", "N/A")
        try:
            return self.explanation_template.format(**values)
        except KeyError:
            return self.explanation_template


ALLOWED_CLAUSE_TYPES = {
    "excessive_notice": {"Termination", "Term"},
    "unlimited_liability": {"Liability", "Indemnification", "Indemnity"},
    "one_sided_termination": {"Termination"},
    "excessive_non_compete": {"Non-Compete", "Noncompete", "Non Competition"},
    "high_liability_cap": {"Liability", "Indemnification", "Indemnity"},
    "long_payment_terms": {"Payment Terms", "Payment"},
    "one_sided_ip_assignment": {"Intellectual Property"},
    "no_ip_indemnification": {"Intellectual Property", "Indemnification", "Indemnity"},
    "unrestricted_data_usage": {"Data Ownership", "Data Protection"},
    "no_data_deletion": {"Data Ownership", "Data Protection"},
    "broad_confidentiality": {"Confidentiality"},
    "no_exclusions_confidentiality": {"Confidentiality"},
    "no_security_measures": {"Security Obligations", "Data Protection"},
    "no_breach_notification": {"Security Obligations", "Data Protection"},
    "one_sided_indemnification": {"Indemnification", "Indemnity"},
    "lol_missing_exclusions": {"Limitation of Liability", "Liability"},
    "lol_one_sided": {"Limitation of Liability", "Liability"},
    "work_for_hire_no_license": {"Intellectual Property"},
    "data_resale_rights": {"Data Ownership", "Data Protection"},
    "data_sharing_unrestricted": {"Data Ownership", "Data Protection"},
    "one_sided_confidentiality": {"Confidentiality"},
    "no_breach_notification_timeline": {"Security Obligations", "Data Protection"},
}


def _clause_type_in(clause: dict, allowed: set) -> bool:
    ct = clause.get("clause_type")
    if not ct:
        return False
    return any(ct.lower() == a.lower() for a in allowed)


def _condition_excessive_notice(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["excessive_notice"]):
        return None
    notice_days = attributes.get("notice_days")
    if notice_days is not None and notice_days > 90:
        return {"detected": True, "extracted_value": f"{notice_days} days", "threshold": "90 days", "actual": notice_days}
    return None


def _condition_unlimited_liability(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["unlimited_liability"]):
        return None
    if attributes.get("has_unlimited_liability"):
        return {"detected": True, "extracted_value": "Unlimited liability clause detected", "threshold": "Liability should be capped"}
    return None


def _condition_one_sided_termination(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["one_sided_termination"]):
        return None
    if attributes.get("has_one_sided_termination"):
        return {"detected": True, "extracted_value": "One-sided termination detected", "threshold": "Termination should be mutual"}
    return None


def _condition_excessive_non_compete(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["excessive_non_compete"]):
        return None
    months = attributes.get("non_compete_months")
    if months is not None and months > 12:
        return {"detected": True, "extracted_value": f"{months} months", "threshold": "12 months"}
    return None


_LIABILITY_MULTIPLIERS = [
    ("BILLION", 1_000_000_000), ("MILLION", 1_000_000),
    ("THOUSAND", 1_000), ("K", 1_000), ("M", 1_000_000),
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


def _condition_high_liability_cap(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["high_liability_cap"]):
        return None
    cap = attributes.get("liability_cap")
    if cap:
        val = _parse_monetary_value(cap)
        if val is not None and val > 10_000_000:
            return {"detected": True, "extracted_value": cap, "threshold": "$10,000,000"}
    return None


def _condition_long_payment_terms(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["long_payment_terms"]):
        return None
    days = attributes.get("payment_deadline_days")
    if days is not None and days > 60:
        return {"detected": True, "extracted_value": f"{days} days", "threshold": "60 days"}
    return None


def _condition_one_sided_ip_assignment(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["one_sided_ip_assignment"]):
        return None
    ip_transfer = attributes.get("ip_ownership_transfer", False)
    ip_license = attributes.get("ip_license_back", False)
    if ip_transfer and not ip_license:
        return {"detected": True, "extracted_value": "IP assigned without license back", "threshold": "IP assignments should include a license back"}
    return None


def _condition_no_ip_indemnification(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_ip_indemnification"]):
        return None
    ip_indemn = attributes.get("ip_indemnification", None)
    if ip_indemn is False:
        return {"detected": True, "extracted_value": "No IP indemnification clause", "threshold": "Contracts involving IP should include infringement indemnification"}
    return None


def _condition_unrestricted_data_usage(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["unrestricted_data_usage"]):
        return None
    data_owned = attributes.get("data_ownership_defined", None)
    data_restricted = attributes.get("data_usage_restricted", None)
    if data_owned is not None and not data_restricted:
        return {"detected": True, "extracted_value": "Data ownership defined but no usage restrictions", "threshold": "Data usage rights should be explicitly restricted"}
    if data_owned is False and data_restricted is False:
        return {"detected": True, "extracted_value": "No data ownership or usage restrictions", "threshold": "Data ownership and usage restrictions should be defined"}
    return None


def _condition_no_data_deletion(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_data_deletion"]):
        return None
    deletion = attributes.get("data_deletion_obligation", None)
    if deletion is False:
        return {"detected": True, "extracted_value": "No data deletion obligation", "threshold": "Data deletion or return obligations should be specified upon termination"}
    return None


def _condition_broad_confidentiality(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["broad_confidentiality"]):
        return None
    if attributes.get("confidentiality_broad_definition", False):
        return {"detected": True, "extracted_value": "Overly broad confidentiality definition", "threshold": "Confidential information definitions should be reasonably scoped"}
    return None


def _condition_no_exclusions_confidentiality(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_exclusions_confidentiality"]):
        return None
    has_exclusions = attributes.get("confidentiality_standard_exclusions", None)
    if has_exclusions is False:
        return {"detected": True, "extracted_value": "No standard exclusions from confidentiality", "threshold": "Should include standard exclusions (public info, independently developed, etc.)"}
    return None


def _condition_no_security_measures(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_security_measures"]):
        return None
    measures = attributes.get("security_measures_defined", None)
    if measures is False:
        return {"detected": True, "extracted_value": "No security measures defined", "threshold": "Contracts involving data processing should specify required security measures"}
    return None


def _condition_no_breach_notification(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_breach_notification"]):
        return None
    notification = attributes.get("security_breach_notification", None)
    if notification is False:
        return {"detected": True, "extracted_value": "No security breach notification requirement", "threshold": "Security breach notification obligations should be specified"}
    return None


def _condition_one_sided_indemnification(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["one_sided_indemnification"]):
        return None
    mutual = attributes.get("indemnification_mutual", None)
    if mutual is False:
        return {"detected": True, "extracted_value": "One-sided indemnification", "threshold": "Indemnification obligations should generally be mutual"}
    return None


def _condition_lol_missing_exclusions(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["lol_missing_exclusions"]):
        return None
    has_exclusions = attributes.get("lol_exclusions_present", None)
    if has_exclusions is False:
        return {"detected": True, "extracted_value": "No key exclusions from liability cap", "threshold": "LoL should exclude IP infringement, confidentiality breach, death/injury, fraud"}
    return None


def _condition_lol_one_sided(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["lol_one_sided"]):
        return None
    mutual = attributes.get("lol_mutual", None)
    if mutual is False:
        return {"detected": True, "extracted_value": "One-sided limitation of liability", "threshold": "Limitation of liability should apply equally to both parties"}
    return None


def _condition_work_for_hire_no_license(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["work_for_hire_no_license"]):
        return None
    wfh = attributes.get("ip_work_for_hire", False)
    license_back = attributes.get("ip_license_back", False)
    if wfh and not license_back:
        return {"detected": True, "extracted_value": "Work-for-hire without license back", "threshold": "Work-for-hire clauses should include a license back to the author"}
    return None


def _condition_data_resale_rights(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["data_resale_rights"]):
        return None
    if attributes.get("data_resale_rights", False):
        return {"detected": True, "extracted_value": "Provider may sell or license customer data", "threshold": "Customer data should not be sold or licensed to third parties"}
    return None


def _condition_data_sharing_unrestricted(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["data_sharing_unrestricted"]):
        return None
    if attributes.get("data_analytics_rights", False) and not attributes.get("data_usage_restricted", True):
        return {"detected": True, "extracted_value": "Provider has unrestricted analytics/sharing rights on customer data", "threshold": "Data analytics and sharing rights should be explicitly limited"}
    return None


def _condition_one_sided_confidentiality(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["one_sided_confidentiality"]):
        return None
    mutual = attributes.get("confidentiality_mutual", None)
    if mutual is False:
        return {"detected": True, "extracted_value": "One-sided confidentiality obligations", "threshold": "Confidentiality obligations should generally be mutual"}
    return None


def _condition_no_breach_notification_timeline(clause, attributes):
    if not _clause_type_in(clause, ALLOWED_CLAUSE_TYPES["no_breach_notification_timeline"]):
        return None
    has_notif = attributes.get("security_breach_notification", None)
    has_timeline = attributes.get("breach_notification_days", None)
    if has_notif is True and has_timeline is None:
        return {"detected": True, "extracted_value": "Breach notification exists but no specific timeline defined", "threshold": "Should specify a notification timeline (e.g., within 72 hours)"}
    return None


ALL_RULES: List[RiskRuleSet] = [
    RiskRuleSet(
        rule_id="R001", name="Excessive Notice Period",
        description="Notice period exceeds 90 days threshold",
        category="Termination Risk", severity="Medium", points=15,
        condition_fn=_condition_excessive_notice,
        explanation_template="Notice period of {extracted_value} exceeds the recommended threshold of {threshold}. Long notice periods may create operational inflexibility.",
        finding_category="Risky Provision",
        negotiation_template="Replace the {extracted_value} notice period with 30–60 days to align with market standard.",
    ),
    RiskRuleSet(
        rule_id="R002", name="Unlimited Liability",
        description="Liability clause contains unlimited liability provisions",
        category="Liability Risk", severity="Critical", points=50,
        condition_fn=_condition_unlimited_liability,
        explanation_template="The contract includes unlimited liability provisions ({extracted_value}). This exposes the party to uncapped financial risk. Standard market practice is to cap liability.",
        finding_category="Risky Provision",
        negotiation_template="Negotiate a mutual aggregate liability cap (e.g., fees paid in the prior 12 months) with carve-outs for fraud, IP infringement, and data breaches.",
    ),
    RiskRuleSet(
        rule_id="R003", name="One-Sided Termination",
        description="Termination rights are materially imbalanced",
        category="Termination Risk", severity="High", points=30,
        condition_fn=_condition_one_sided_termination,
        explanation_template="One-sided termination rights detected ({extracted_value}). Standard practice requires mutual termination rights for both parties.",
        finding_category="Risky Provision",
        negotiation_template="Add mutual termination rights — both parties should have the right to terminate for cause and for convenience with reasonable notice.",
    ),
    RiskRuleSet(
        rule_id="R004", name="Excessive Non-Compete Duration",
        description="Non-compete period exceeds 12 months",
        category="Non-Compete Risk", severity="Medium", points=15,
        condition_fn=_condition_excessive_non_compete,
        explanation_template="Non-compete duration of {extracted_value} exceeds standard market threshold of {threshold}. Extended non-compete periods may be commercially unreasonable.",
        finding_category="Risky Provision",
        negotiation_template="Reduce the non-compete duration to 12 months or less and narrow the scope to directly competing products/services.",
    ),
    RiskRuleSet(
        rule_id="R005", name="High Liability Cap",
        description="Liability cap exceeds $10M threshold",
        category="Liability Risk", severity="Medium", points=15,
        condition_fn=_condition_high_liability_cap,
        explanation_template="Liability cap of {extracted_value} exceeds standard threshold of {threshold}. High liability caps increase financial exposure.",
        finding_category="Risky Provision",
        negotiation_template="Consider negotiating the liability cap down to reflect the contract value (e.g., 1–2× annual fees).",
    ),
    RiskRuleSet(
        rule_id="R006", name="Extended Payment Terms",
        description="Payment terms exceed 60 days",
        category="Payment Risk", severity="Low", points=5,
        condition_fn=_condition_long_payment_terms,
        explanation_template="Payment deadline of {extracted_value} exceeds standard threshold of {threshold}. Extended payment terms may impact cash flow.",
        finding_category="Risky Provision",
        negotiation_template="Negotiate payment terms down to Net 30 or Net 45 to improve cash flow.",
    ),
    RiskRuleSet(
        rule_id="R007", name="One-Sided IP Assignment",
        description="IP ownership is assigned without a license back to the contributing party",
        category="IP Risk", severity="High", points=30,
        condition_fn=_condition_one_sided_ip_assignment,
        explanation_template="{extracted_value}. {threshold}. Without a license back, the contributing party loses all rights to use its own IP.",
        finding_category="Risky Provision",
        negotiation_template="Add a license-back provision granting the IP contributor a perpetual, royalty-free, non-exclusive license to use the assigned IP.",
    ),
    RiskRuleSet(
        rule_id="R008", name="No IP Indemnification",
        description="Intellectual property clause lacks indemnification for infringement claims",
        category="IP Risk", severity="Medium", points=15,
        condition_fn=_condition_no_ip_indemnification,
        explanation_template="{extracted_value}. {threshold}. This exposes the licensee to potential third-party IP claims without recourse.",
        finding_category="Risky Provision",
        negotiation_template="Add an IP indemnification clause requiring the IP owner to defend against third-party infringement claims.",
    ),
    RiskRuleSet(
        rule_id="R009", name="Unrestricted Data Usage",
        description="Data usage rights are not explicitly restricted to the contract purpose",
        category="Data Risk", severity="High", points=30,
        condition_fn=_condition_unrestricted_data_usage,
        explanation_template="{extracted_value}. {threshold}. Without restrictions, data may be used beyond the intended scope.",
        finding_category="Risky Provision",
        negotiation_template="Restrict data usage to the specific purpose of the agreement. Prohibit analytics, sublicensing, and resale of customer data.",
    ),
    RiskRuleSet(
        rule_id="R010", name="No Data Deletion Obligation",
        description="Data deletion or return obligation is not specified upon termination",
        category="Data Risk", severity="High", points=30,
        condition_fn=_condition_no_data_deletion,
        explanation_template="{extracted_value}. {threshold}. Without this obligation, sensitive data may persist indefinitely after the relationship ends.",
        finding_category="Risky Provision",
        negotiation_template="Add a data deletion obligation requiring the provider to delete all customer data within 30 days of contract termination, with written certification.",
    ),
    RiskRuleSet(
        rule_id="R011", name="Overly Broad Confidentiality Definition",
        description="Confidential information definition is overly broad",
        category="Confidentiality Risk", severity="Medium", points=15,
        condition_fn=_condition_broad_confidentiality,
        explanation_template="{extracted_value}. {threshold}. Overly broad definitions can make compliance impractical and create disputes.",
        finding_category="Risky Provision",
        negotiation_template="Narrow the definition of confidential information to specifically identified categories or marked/designated materials.",
    ),
    RiskRuleSet(
        rule_id="R012", name="Missing Standard Confidentiality Exclusions",
        description="Confidentiality clause lacks standard exclusions",
        category="Confidentiality Risk", severity="Low", points=5,
        condition_fn=_condition_no_exclusions_confidentiality,
        explanation_template="{extracted_value}. {threshold}. Without standard exclusions, protections may extend beyond what is commercially reasonable.",
        finding_category="Risky Provision",
        negotiation_template="Add standard exclusions: publicly available information, independently developed information, and information received from third parties without restriction.",
    ),
    RiskRuleSet(
        rule_id="R013", name="No Security Measures Defined",
        description="Security obligations clause does not specify concrete security measures",
        category="Security Risk", severity="Medium", points=15,
        condition_fn=_condition_no_security_measures,
        explanation_template="{extracted_value}. {threshold}. Without specified measures, data security obligations may be unenforceable.",
        finding_category="Risky Provision",
        negotiation_template="Specify required security standards (e.g., SOC 2 Type II, ISO 27001) and technical controls including encryption, access controls, and incident response.",
    ),
    RiskRuleSet(
        rule_id="R014", name="No Breach Notification Requirement",
        description="Security obligations clause lacks a breach notification requirement",
        category="Security Risk", severity="Medium", points=15,
        condition_fn=_condition_no_breach_notification,
        explanation_template="{extracted_value}. {threshold}. Without a notification obligation, breaches may go undetected and unaddressed.",
        finding_category="Risky Provision",
        negotiation_template="Add a breach notification clause requiring notification within 72 hours of discovering a security incident, with details of the breach and remediation steps.",
    ),
    RiskRuleSet(
        rule_id="R015", name="One-Sided Indemnification",
        description="Indemnification obligations are not mutual between the parties",
        category="Indemnification Risk", severity="High", points=30,
        condition_fn=_condition_one_sided_indemnification,
        explanation_template="{extracted_value}. {threshold}. One-sided indemnification creates an unbalanced allocation of risk.",
        finding_category="Risky Provision",
        negotiation_template="Negotiate mutual indemnification — both parties should indemnify each other for breaches of their respective representations and third-party claims arising from their conduct.",
    ),
    RiskRuleSet(
        rule_id="R016", name="Missing Key Liability Cap Exclusions",
        description="Limitation of liability clause does not specify key exclusions from the cap",
        category="Liability Risk", severity="Medium", points=15,
        condition_fn=_condition_lol_missing_exclusions,
        explanation_template="{extracted_value}. {threshold}. Without specified exclusions, critical risks may be inadvertently capped.",
        finding_category="Risky Provision",
        negotiation_template="Add liability cap carve-outs for: IP infringement, confidentiality breach, fraud, gross negligence, and death/personal injury.",
    ),
    RiskRuleSet(
        rule_id="R017", name="One-Sided Limitation of Liability",
        description="Limitation of liability applies only to one party",
        category="Liability Risk", severity="Medium", points=15,
        condition_fn=_condition_lol_one_sided,
        explanation_template="{extracted_value}. {threshold}. A one-sided limitation creates an imbalance in risk allocation.",
        finding_category="Risky Provision",
        negotiation_template="Negotiate a mutual limitation of liability that applies equally to both parties with the same cap amount.",
    ),
    RiskRuleSet(
        rule_id="R018", name="Work-for-Hire Without License Back",
        description="Work-for-hire clause exists without a license back to the author",
        category="IP Risk", severity="High", points=30,
        condition_fn=_condition_work_for_hire_no_license,
        explanation_template="{extracted_value}. {threshold}. Under work-for-hire, the author loses all IP rights without a license back.",
        finding_category="Risky Provision",
        negotiation_template="Add a license-back provision granting the creator a perpetual, royalty-free, non-exclusive license to use the work-for-hire output.",
    ),
    RiskRuleSet(
        rule_id="R019", name="Provider Data Resale Rights",
        description="Provider may sell or license customer data to third parties",
        category="Data Risk", severity="Critical", points=50,
        condition_fn=_condition_data_resale_rights,
        explanation_template="{extracted_value}. {threshold}. This allows the provider to monetize customer data without restriction.",
        finding_category="Risky Provision",
        negotiation_template="Add an explicit prohibition on the provider selling, licensing, or transferring customer data to third parties for commercial purposes.",
    ),
    RiskRuleSet(
        rule_id="R020", name="Uncontrolled Data Sharing / Analytics Rights",
        description="Provider has broad analytics or data sharing rights without restrictions",
        category="Data Risk", severity="High", points=30,
        condition_fn=_condition_data_sharing_unrestricted,
        explanation_template="{extracted_value}. {threshold}. Unrestricted analytics rights may expose customer data beyond the intended scope.",
        finding_category="Risky Provision",
        negotiation_template="Limit analytics to aggregated, anonymized data only. Require explicit consent for any use of identifiable customer data for analytics purposes.",
    ),
    RiskRuleSet(
        rule_id="R021", name="One-Sided Confidentiality",
        description="Confidentiality obligations apply to only one party",
        category="Confidentiality Risk", severity="Medium", points=15,
        condition_fn=_condition_one_sided_confidentiality,
        explanation_template="{extracted_value}. {threshold}. One-sided confidentiality puts one party at a disadvantage.",
        finding_category="Risky Provision",
        negotiation_template="Negotiate mutual confidentiality obligations so both parties are bound by the same restrictions.",
    ),
    RiskRuleSet(
        rule_id="R022", name="No Breach Notification Timeline",
        description="Breach notification exists but specifies no timeline",
        category="Security Risk", severity="Medium", points=15,
        condition_fn=_condition_no_breach_notification_timeline,
        explanation_template="{extracted_value}. {threshold}. Without a specific timeline, notification may be delayed.",
        finding_category="Risky Provision",
        negotiation_template="Specify a maximum notification period of 72 hours for critical breaches and 7 days for other security incidents.",
    ),
]
