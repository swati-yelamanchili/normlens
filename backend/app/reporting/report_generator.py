import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

CLAUSE_GROUP_ORDER = [
    "Termination", "Liability", "Indemnification",
    "IP", "Data", "Security", "Confidentiality", "General", "Payment",
]


class ReportGenerator:
    def generate(
        self,
        contract: dict,
        clauses: List[dict],
        classifications: List[dict],
        risk_findings: List[dict],
        outliers: List[dict],
        benchmarks: List[dict],
        total_risk_score: int,
        risk_level: str,
        contract_type: str = "General Commercial Agreement",
        contract_type_confidence: float = 0.0,
    ) -> dict:
        clause_summaries = []
        for clause in clauses:
            clause_summaries.append({
                "clause_index": clause.get("clause_index"),
                "clause_title": clause.get("clause_title"),
                "clause_type": clause.get("clause_type", "Unclassified"),
                "page_number": clause.get("page_number"),
                "text_preview": (clause.get("clause_text") or "")[:200],
            })

        risk_summaries = []
        for finding in risk_findings:
            risk_summaries.append({
                "risk_name": finding.get("risk_name"),
                "severity": finding.get("severity"),
                "points": finding.get("points"),
                "explanation": finding.get("explanation"),
                "supporting_clause": (finding.get("supporting_clause") or "")[:300],
                "supporting_clauses": finding.get("supporting_clauses", []),
                "extracted_value": finding.get("extracted_value"),
                "clause_index": finding.get("clause_index"),
                "finding_category": finding.get("finding_category", "Risky Provision"),
                "clause_group": finding.get("clause_group", "General"),
                "negotiation_recommendation": finding.get("negotiation_recommendation", ""),
            })

        # Category separation
        missing_protection = [f for f in risk_summaries if f.get("finding_category") == "Missing Protection"]
        risky_provision = [f for f in risk_summaries if f.get("finding_category") != "Missing Protection"]

        # Group by clause group
        findings_by_group: Dict[str, List[dict]] = {}
        for f in risk_summaries:
            group = f.get("clause_group", "General")
            findings_by_group.setdefault(group, []).append(f)

        # Sort groups by canonical order
        sorted_group_keys = sorted(
            findings_by_group.keys(),
            key=lambda k: CLAUSE_GROUP_ORDER.index(k) if k in CLAUSE_GROUP_ORDER else 99
        )
        findings_by_group_ordered = {k: findings_by_group[k] for k in sorted_group_keys}

        outlier_summaries = []
        for outlier in outliers:
            outlier_summaries.append({
                "type": outlier.get("type"),
                "attribute": outlier.get("attribute", "").replace("_", " ").title(),
                "severity": outlier.get("severity"),
                "contract_value": outlier.get("contract_value"),
                "market_median": outlier.get("market_median"),
                "market_p95": outlier.get("market_p95"),
                "percentile_rank": outlier.get("percentile_rank"),
                "z_score": outlier.get("z_score"),
                "explanation": outlier.get("explanation"),
            })

        severity_breakdown = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        category_breakdown = {"Missing Protection": 0, "Risky Provision": 0}
        for f in risk_findings:
            sev = f.get("severity", "Low")
            if sev in severity_breakdown:
                severity_breakdown[sev] += 1
            cat = f.get("finding_category", "Risky Provision")
            if cat in category_breakdown:
                category_breakdown[cat] += 1
            else:
                category_breakdown[cat] = 1

        report = {
            "contract_summary": {
                "filename": contract.get("original_filename"),
                "file_type": contract.get("file_type"),
                "page_count": contract.get("page_count"),
                "clause_count": len(clauses),
                "contract_type": contract_type,
                "contract_type_confidence": round(contract_type_confidence * 100),
            },
            "risk_summary": {
                "total_score": total_risk_score,
                "risk_level": risk_level,
                "finding_count": len(risk_findings),
                "outlier_count": len(outliers),
                "severity_breakdown": severity_breakdown,
                "category_breakdown": category_breakdown,
            },
            "clause_classifications": clause_summaries,
            "risk_findings": risk_summaries,
            "missing_protections": missing_protection,
            "risky_provisions": risky_provision,
            "findings_by_group": findings_by_group_ordered,
            "outlier_detections": outlier_summaries,
            "recommendations": self._generate_recommendations(risk_findings, outliers, risk_level),
            "methodology": {
                "classification": "Embedding similarity + k-NN using CUAD taxonomy",
                "attribute_extraction": "Deterministic rule-based extraction (regex/pattern matching)",
                "risk_detection": "Deterministic rule engine with configurable thresholds",
                "benchmarking": "Statistical comparison against peer contract corpus (CUAD/LEDGAR)",
                "outlier_detection": "Percentile analysis + z-score methods on peer distributions",
                "scoring": "Aggregated weighted risk points from detected findings",
                "contract_type_detection": "Keyword scoring across 10 contract type categories",
            },
        }
        return report

    def _generate_recommendations(
        self, risk_findings: List[dict], outliers: List[dict], risk_level: str
    ) -> List[str]:
        recommendations = []
        seen = set()

        if risk_level in ("High", "Critical"):
            recommendations.append(
                "The contract has significant risk findings that should be addressed before execution."
            )

        for finding in risk_findings:
            rec = finding.get("negotiation_recommendation", "")
            if rec and rec not in seen:
                seen.add(rec)
                recommendations.append(rec)

        # Fallback legacy recommendations for findings without negotiation_recommendation
        for finding in risk_findings:
            name = finding.get("risk_name", "")
            if finding.get("negotiation_recommendation"):
                continue
            msg = None
            if "Unlimited Liability" in name:
                msg = "Consider negotiating a liability cap to limit financial exposure."
            elif "One-Sided" in name and "Termination" in name:
                msg = "Review termination rights for mutuality and balance between parties."
            elif "Notice Period" in name:
                msg = "Evaluate if the notice period aligns with operational requirements."
            elif "Confidentiality" in name and "Missing" in name:
                msg = "A confidentiality clause should be added to protect proprietary information."
            elif "Non-Compete" in name:
                msg = "Review non-compete scope for commercial reasonableness and enforceability."
            if msg and msg not in seen:
                seen.add(msg)
                recommendations.append(msg)

        if not recommendations:
            recommendations.append("No material risk findings detected. Continue with standard review.")

        return recommendations
