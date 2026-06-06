import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class RiskRule(Base):
    __tablename__ = "risk_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(String(32), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(128), nullable=False)
    severity = Column(String(32), nullable=False)
    points = Column(Integer, nullable=False)
    conditions_json = Column(JSON, nullable=False)
    enabled = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RiskFinding(Base):
    __tablename__ = "risk_findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contract_id = Column(
        UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False
    )
    clause_id = Column(
        UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="SET NULL"), nullable=True
    )
    rule_id = Column(UUID(as_uuid=True), ForeignKey("risk_rules.id"), nullable=True)
    risk_name = Column(String(256), nullable=False)
    severity = Column(String(32), nullable=False)
    points = Column(Integer, nullable=False)
    supporting_clause = Column(Text, nullable=True)
    extracted_value = Column(String(256), nullable=True)
    explanation = Column(Text, nullable=False)
    benchmark_comparison = Column(Text, nullable=True)
    percentile_rank = Column(Float, nullable=True)
    peer_group_reference = Column(String(256), nullable=True)
    # New fields (nullable for backward compatibility)
    finding_category = Column(String(64), nullable=True)
    clause_group = Column(String(64), nullable=True)
    supporting_clauses_json = Column(JSON, nullable=True)
    negotiation_recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

