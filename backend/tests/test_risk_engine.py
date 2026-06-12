import pytest
from app.risk.risk_engine import RiskEngine
from app.risk.risk_rules import ALL_RULES


class TestRiskEngine:
    def setup_method(self):
        self.engine = RiskEngine()

    def test_rules_loaded(self):
        assert len(self.engine.rules) > 0

    def test_excessive_notice_period(self):
        clause = {"clause_text": "Termination with notice.", "clause_type": "Termination"}
        attributes = {"notice_days": 180}
        findings = self.engine.evaluate_clause(clause, attributes)
        rule_ids = [f["rule_id"] for f in findings]
        assert "R001" in rule_ids

    def test_normal_notice_period_no_finding(self):
        clause = {"clause_text": "Termination with notice.", "clause_type": "Termination"}
        attributes = {"notice_days": 30}
        findings = self.engine.evaluate_clause(clause, attributes)
        rule_ids = [f["rule_id"] for f in findings]
        assert "R001" not in rule_ids

    def test_unlimited_liability(self):
        clause = {"clause_text": "Unlimited liability clause.", "clause_type": "Liability"}
        attributes = {"has_unlimited_liability": True}
        findings = self.engine.evaluate_clause(clause, attributes)
        rule_ids = [f["rule_id"] for f in findings]
        assert "R002" in rule_ids

    def test_risk_score_calculation(self):
        findings = [
            {"points": 40},
            {"points": 20},
            {"points": 15},
        ]
        score = self.engine.calculate_risk_score(findings)
        assert score == 75

    def test_risk_levels(self):
        assert self.engine.get_risk_level(10) == "Low"
        assert self.engine.get_risk_level(30) == "Moderate"
        assert self.engine.get_risk_level(60) == "High"
        assert self.engine.get_risk_level(80) == "Critical"
