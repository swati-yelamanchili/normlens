import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Clause(Base):
    __tablename__ = "clauses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contract_id = Column(
        UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False
    )
    clause_index = Column(Integer, nullable=False)
    clause_title = Column(String(256), nullable=True)
    clause_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    clause_type = Column(String(128), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    attributes = Column(JSON, nullable=True)
    embedding_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
