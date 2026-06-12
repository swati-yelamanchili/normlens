import pytest
from app.outlier.outlier_detector import OutlierDetector


class TestOutlierDetector:
    def test_evaluate_outlier_high_percentile(self):
        detector = OutlierDetector()
        benchmark = {
            "attribute": "notice_days",
            "contract_value": "180 days",
            "market_median": 45.0,
            "market_p95": 90.0,
            "percentile_rank": 99.0,
            "z_score": 3.5,
            "peer_count": 100,
        }
        clause = {"clause_text": "Test clause"}
        result = detector._evaluate_outlier(benchmark, clause, "Termination")
        assert result is not None
        assert result["severity"] == "High"

    def test_evaluate_outlier_normal_value(self):
        detector = OutlierDetector()
        benchmark = {
            "attribute": "notice_days",
            "contract_value": "45 days",
            "market_median": 45.0,
            "market_p95": 90.0,
            "percentile_rank": 50.0,
            "z_score": 0.0,
            "peer_count": 100,
        }
        clause = {"clause_text": "Test clause"}
        result = detector._evaluate_outlier(benchmark, clause, "Termination")
        assert result is None
