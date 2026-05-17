import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contract_id = Column(
        UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False
    )
    clause_id = Column(
        UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False
    )
    clause_type = Column(String(128), nullable=False)
    attribute = Column(String(128), nullable=False)
    contract_value = Column(String(64), nullable=True)
    market_median = Column(Float, nullable=True)
    market_mean = Column(Float, nullable=True)
    market_std = Column(Float, nullable=True)
    market_p5 = Column(Float, nullable=True)
    market_p25 = Column(Float, nullable=True)
    market_p75 = Column(Float, nullable=True)
    market_p95 = Column(Float, nullable=True)
    percentile_rank = Column(Float, nullable=True)
    z_score = Column(Float, nullable=True)
    peer_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
