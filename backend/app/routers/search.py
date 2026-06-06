import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.classification import ClauseClassifier
from app.database import get_db
from app.embeddings import EmbeddingService
from app.models.clause import Clause
from app.models.contract import Contract
from app.search import SearchService

router = APIRouter(prefix="/api/search", tags=["search"])

_embedding_service = None
_classifier = None
_search_service = None


def _get_search_service():
    global _embedding_service, _classifier, _search_service
    if _search_service is None:
        _embedding_service = EmbeddingService()
        _classifier = ClauseClassifier(_embedding_service)
        _search_service = SearchService(_embedding_service, _classifier)
    return _search_service


@router.get("/")
def search_clauses(
    q: str = Query(..., description="Search query"),
    contract_id: Optional[uuid.UUID] = None,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    query = db.query(Clause)

    if contract_id:
        contract = db.query(Contract).filter(Contract.id == contract_id).first()
        if not contract:
            raise HTTPException(status_code=404, detail="Contract not found")
        query = query.filter(Clause.contract_id == contract_id)

    clause_models = query.order_by(Clause.clause_index).all()
    if not clause_models:
        return {"results": [], "query": q, "total": 0}

    clauses = [
        {
            "clause_index": c.clause_index,
            "clause_title": c.clause_title,
            "clause_text": c.clause_text,
            "page_number": c.page_number,
            "clause_type": c.clause_type,
            "classification_confidence": c.classification_confidence,
            "attributes": c.attributes,
        }
        for c in clause_models
    ]

    results = _get_search_service().search(q, clauses, top_k=top_k)

    return {
        "query": q,
        "total": len(results),
        "results": results,
    }
