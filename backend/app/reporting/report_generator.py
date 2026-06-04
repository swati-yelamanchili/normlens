import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


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
    ) -> dict:
        clause_summaries = []
        for clause in clauses:
            clause_type = clause.get("clause_type", "Unclassified")
            clause_summaries.append(
                {
                    "clause_index": clause.get("clause_index"),
                    "clause_title": clause.get("clause_title"),
                    "clause_type": clause_type,
                    "page_number": clause.get("page_number"),
                    "text_preview": (clause.get("clause_text") or "")[:200],
                }
            )

        risk_summaries = []
        for finding in risk_findings:
            risk_summaries.append(
                {
                    "risk_name": finding.get("risk_name"),
                    "severity": finding.get("severity"),
                    "points": finding.get("points"),
                    "explanation": finding.get("explanation"),
                    "supporting_clause": (finding.get("supporting_clause") or "")[:200],
                    "extracted_value": finding.get("extracted_value"),
                    "clause_index": finding.get("clause_index"),
                }
            )

        outlier_summaries = []
        for outlier in outliers:
            outlier_summaries.append(
                {
                    "type": outlier.get("type"),
                    "attribute": outlier.get("attribute", "").replace("_", " ").title(),
                    "severity": outlier.get("severity"),
                    "contract_value": outlier.get("contract_value"),
                    "market_median": outlier.get("market_median"),
                    "percentile_rank": outlier.get("percentile_rank"),
                    "z_score": outlier.get("z_score"),
                    "explanation": outlier.get("explanation"),
                }
            )

        severity_breakdown = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for f in risk_findings:
            sev = f.get("severity", "Low")
            if sev in severity_breakdown:
                severity_breakdown[sev] += 1

        report = {
            "contract_summary": {
                "filename": contract.get("original_filename"),
                "file_type": contract.get("file_type"),
                "page_count": contract.get("page_count"),
                "clause_count": len(clauses),
            },
            "risk_summary": {
                "total_score": total_risk_score,
                "risk_level": risk_level,
                "finding_count": len(risk_findings),
                "outlier_count": len(outliers),
                "severity_breakdown": severity_breakdown,
            },
            "clause_classifications": clause_summaries,
            "risk_findings": risk_summaries,
            "outlier_detections": outlier_summaries,
            "recommendations": self._generate_recommendations(
                risk_findings, outliers, risk_level
            ),
            "methodology": {
                "classification": "Embedding similarity + k-NN using CUAD taxonomy",
                "attribute_extraction": "Deterministic rule-based extraction (regex/pattern matching)",
                "risk_detection": "Deterministic rule engine with configurable thresholds",
                "benchmarking": "Statistical comparison against peer contract corpus (CUAD/LEDGAR)",
                "outlier_detection": "Percentile analysis + z-score methods on peer distributions",
                "scoring": "Aggregated weighted risk points from detected findings",
            },
        }

        return report

    def _generate_recommendations(
        self, risk_findings: List[dict], outliers: List[dict], risk_level: str
    ) -> List[str]:
        recommendations = []

        if risk_level in ("High", "Critical"):
            recommendations.append(
                "The contract has significant risk findings that should be addressed before execution."
            )

        for finding in risk_findings:
            name = finding.get("risk_name", "")
            if "Unlimited Liability" in name:
                recommendations.append(
                    "Consider negotiating a liability cap to limit financial exposure."
                )
            elif "One-Sided" in name:
                recommendations.append(
                    "Review termination rights for mutuality and balance between parties."
                )
            elif "Notice Period" in name:
                recommendations.append(
                    "Evaluate if the notice period aligns with operational requirements."
                )
            elif "Confidentiality" in name and "Missing" in name:
                recommendations.append(
                    "A confidentiality clause should be added to protect proprietary information."
                )
            elif "Non-Compete" in name:
                recommendations.append(
                    "Review non-compete scope for commercial reasonableness and enforceability."
                )

        if not recommendations:
            recommendations.append(
                "No material risk findings detected. Continue with standard review."
            )

        return recommendations
