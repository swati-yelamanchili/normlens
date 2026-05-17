import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class ContractStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    PARSED = "parsed"
    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    FAILED = "failed"


class Contract(Base):
    __tablename__ = "contracts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size_bytes = Column(String(32), nullable=False)
    status = Column(Enum(ContractStatus), default=ContractStatus.PENDING, nullable=False)
    text_content = Column(Text, nullable=True)
    page_count = Column(String(8), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
