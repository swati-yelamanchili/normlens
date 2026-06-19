import pytest
from app.benchmarking.benchmarking_engine import BenchmarkingEngine
from app.benchmarking.market_data import MarketNormDatabase


class TestMarketNormDatabase:
    def setup_method(self):
        self.db = MarketNormDatabase()

    def test_has_market_data(self):
        assert self.db.has_norms_for("Termination")
        assert self.db.has_norms_for("Payment Terms")
        assert self.db.has_norms_for("Non-Compete")
        assert self.db.has_norms_for("Liability")

    def test_get_norms_for_known_type(self):
        norms = self.db.get_norms_for_type("Termination")
        assert "notice_days" in norms
        assert "values" in norms["notice_days"]
        assert len(norms["notice_days"]["values"]) > 0

    def test_get_norms_for_unknown_type(self):
        norms = self.db.get_norms_for_type("UnknownType")
        assert norms == {}

    def test_peer_clauses_returned(self):
        import numpy as np
        peers = self.db.get_similar_clauses("Termination", np.random.randn(384), top_k=5)
        assert len(peers) <= 5
        assert all("peer_id" in p for p in peers)
        assert all("similarity_score" in p for p in peers)


class TestBenchmarkingEngine:
    def setup_method(self):
        self.engine = BenchmarkingEngine()

    def test_benchmark_termination_notice(self):
        clause = {"clause_text": "30 days notice"}
        attributes = {"notice_days": 90}
        results = self.engine.benchmark_attributes(clause, attributes, "Termination")
        assert len(results) > 0
        bench = results[0]
        assert bench["attribute"] == "notice_days"
        assert bench["contract_value"] == "90"
        assert bench["market_median"] > 0
        assert bench["percentile_rank"] > 0
        assert bench["peer_count"] > 0

    def test_benchmark_payment_terms(self):
        clause = {"clause_text": "Net 30 payment"}
        attributes = {"payment_deadline_days": 30}
        results = self.engine.benchmark_attributes(clause, attributes, "Payment Terms")
        assert len(results) > 0
        assert any(r["attribute"] == "payment_deadline_days" for r in results)

    def test_benchmark_unknown_type_returns_empty(self):
        clause = {"clause_text": "Some clause"}
        attributes = {"notice_days": 30}
        results = self.engine.benchmark_attributes(clause, attributes, "UnknownType")
        assert results == []

    def test_benchmark_liability_cap(self):
        clause = {"clause_text": "Liability cap of $2M"}
        attributes = {"liability_cap": 5000000}
        results = self.engine.benchmark_attributes(clause, attributes, "Liability")
        assert len(results) > 0
        cap_result = [r for r in results if r["attribute"] == "liability_cap"]
        assert len(cap_result) > 0
        assert cap_result[0]["percentile_rank"] > 0

    def test_benchmark_non_compete(self):
        clause = {"clause_text": "12 month non-compete"}
        attributes = {"non_compete_months": 24}
        results = self.engine.benchmark_attributes(clause, attributes, "Non-Compete")
        assert len(results) > 0
        nc_result = [r for r in results if r["attribute"] == "non_compete_months"]
        assert len(nc_result) > 0
        assert nc_result[0]["z_score"] > 0

    def test_benckmark_skip_none_values(self):
        clause = {"clause_text": "Test clause"}
        attributes = {"notice_days": None}
        results = self.engine.benchmark_attributes(clause, attributes, "Termination")
        assert results == []

    def test_retrieve_similar_peers(self):
        peers = self.engine.retrieve_similar_peers("Test clause", "Termination", top_k=3)
        assert len(peers) <= 3
