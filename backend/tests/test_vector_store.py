import pytest
from app.embeddings.vector_store import VectorStore, get_vector_store


class TestVectorStore:
    def setup_method(self):
        try:
            self.store = get_vector_store()
        except Exception:
            pytest.skip("ChromaDB not available")

    def test_health_check(self):
        healthy = self.store.health_check()
        assert healthy is True

    def test_add_and_search(self):
        embedding = [0.1] * 384
        self.store.add_single(
            embedding_id="test_1",
            embedding=embedding,
            metadata={"contract_id": "test_contract", "clause_index": "0", "clause_type": "TestType"},
            document="This is a test clause",
        )
        results = self.store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
        )
        ids = [r["id"] for r in results]
        assert "test_1" in ids

    def test_search_with_where_filter(self):
        embedding = [0.2] * 384
        self.store.add_single(
            embedding_id="test_2",
            embedding=embedding,
            metadata={"contract_id": "filter_test", "clause_index": "1", "clause_type": "TestType"},
            document="Filtered clause",
        )
        results = self.store.search(
            query_embedding=[0.2] * 384,
            top_k=5,
            where={"contract_id": "filter_test"},
        )
        assert len(results) > 0
        assert all(r["metadata"]["contract_id"] == "filter_test" for r in results)

    def test_count(self):
        count = self.store.count()
        assert count >= 0

    def test_get_by_contract(self):
        results = self.store.get_by_contract("test_contract")
        assert len(results) > 0
