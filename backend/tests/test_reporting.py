import pytest
from app.reporting.report_generator import ReportGenerator


class TestReportGenerator:
    def setup_method(self):
        self.generator = ReportGenerator()

    def _make_sample_finding(self, overrides=None):
        finding = {
            "risk_name": "Excessive Notice Period",
            "severity": "High",
            "points": 25,
            "explanation": "Notice period of 180 days exceeds market norms",
            "supporting_clause": "Either party may terminate with 180 days notice",
            "extracted_value": "180 days",
            "clause_index": 3,
            "finding_category": "Risky Provision",
            "clause_group": "Termination",
            "negotiation_recommendation": "Negotiate a shorter notice period",
        }
        if overrides:
            finding.update(overrides)
        return finding

    def test_generate_minimal(self):
        report = self.generator.generate(
            contract={"original_filename": "test.pdf", "file_type": "pdf", "page_count": "10"},
            clauses=[
                {"clause_index": 0, "clause_title": "Termination", "clause_type": "Termination",
                 "page_number": 1, "clause_text": "Termination clause text"}
            ],
            classifications=[{"clause_index": 0, "clause_type": "Termination"}],
            risk_findings=[],
            outliers=[],
            benchmarks=[],
            total_risk_score=0,
            risk_level="Low",
        )
        assert "contract_summary" in report
        assert "risk_summary" in report
        assert report["contract_summary"]["filename"] == "test.pdf"
        assert report["risk_summary"]["risk_level"] == "Low"
        assert report["risk_summary"]["total_score"] == 0

    def test_generate_with_findings(self):
        findings = [self._make_sample_finding()]
        report = self.generator.generate(
            contract={"original_filename": "test.pdf", "file_type": "pdf", "page_count": "5"},
            clauses=[],
            classifications=[],
            risk_findings=findings,
            outliers=[],
            benchmarks=[],
            total_risk_score=25,
            risk_level="Moderate",
        )
        assert report["risk_summary"]["finding_count"] == 1
        assert report["risk_summary"]["total_score"] == 25
        assert len(report["risk_findings"]) == 1
        assert report["risk_findings"][0]["risk_name"] == "Excessive Notice Period"

    def test_category_separation(self):
        findings = [
            self._make_sample_finding({"finding_category": "Risky Provision", "risk_name": "R001"}),
            self._make_sample_finding({"finding_category": "Missing Protection", "risk_name": "MC001"}),
        ]
        report = self.generator.generate(
            contract={"original_filename": "t.pdf", "file_type": "pdf", "page_count": "1"},
            clauses=[], classifications=[], risk_findings=findings,
            outliers=[], benchmarks=[], total_risk_score=20, risk_level="Moderate",
        )
        assert len(report["risky_provisions"]) == 1
        assert len(report["missing_protections"]) == 1
        assert report["risky_provisions"][0]["risk_name"] == "R001"
        assert report["missing_protections"][0]["risk_name"] == "MC001"

    def test_severity_breakdown(self):
        findings = [
            self._make_sample_finding({"severity": "Critical", "points": 40}),
            self._make_sample_finding({"severity": "High", "points": 25}),
            self._make_sample_finding({"severity": "Medium", "points": 15}),
            self._make_sample_finding({"severity": "Low", "points": 5}),
        ]
        report = self.generator.generate(
            contract={"original_filename": "t.pdf", "file_type": "pdf", "page_count": "1"},
            clauses=[], classifications=[], risk_findings=findings,
            outliers=[], benchmarks=[], total_risk_score=85, risk_level="High",
        )
        breakdown = report["risk_summary"]["severity_breakdown"]
        assert breakdown["Critical"] == 1
        assert breakdown["High"] == 1
        assert breakdown["Medium"] == 1
        assert breakdown["Low"] == 1

    def test_recommendations_from_findings(self):
        findings = [self._make_sample_finding()]
        report = self.generator.generate(
            contract={"original_filename": "t.pdf", "file_type": "pdf", "page_count": "1"},
            clauses=[], classifications=[], risk_findings=findings,
            outliers=[], benchmarks=[], total_risk_score=25, risk_level="Moderate",
        )
        assert len(report["recommendations"]) > 0
        assert "Negotiate a shorter notice period" in report["recommendations"]

    def test_findings_grouped_by_clause_group(self):
        findings = [
            self._make_sample_finding({"clause_group": "Termination", "risk_name": "R001"}),
            self._make_sample_finding({"clause_group": "Confidentiality", "risk_name": "R011"}),
        ]
        report = self.generator.generate(
            contract={"original_filename": "t.pdf", "file_type": "pdf", "page_count": "1"},
            clauses=[], classifications=[], risk_findings=findings,
            outliers=[], benchmarks=[], total_risk_score=20, risk_level="Moderate",
        )
        assert "Termination" in report["findings_by_group"]
        assert "Confidentiality" in report["findings_by_group"]

    def test_outlier_summaries(self):
        outliers = [
            {
                "type": "statistical_outlier",
                "attribute": "notice_days",
                "severity": "High",
                "contract_value": "180 days",
                "market_median": 45.0,
                "market_p95": 90.0,
                "percentile_rank": 99.0,
                "z_score": 3.5,
                "explanation": "Notice period is significantly above market norms",
            }
        ]
        report = self.generator.generate(
            contract={"original_filename": "t.pdf", "file_type": "pdf", "page_count": "1"},
            clauses=[], classifications=[], risk_findings=[],
            outliers=outliers, benchmarks=[], total_risk_score=0, risk_level="Low",
        )
        assert len(report["outlier_detections"]) == 1
        assert report["outlier_detections"][0]["attribute"] == "Notice Days"
