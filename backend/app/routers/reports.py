import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.report import AnalysisReport

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{contract_id}")
def get_report(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.contract_id == contract_id)
        .order_by(AnalysisReport.generated_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for this contract")
    return {
        "report_id": str(report.id),
        "contract_id": str(report.contract_id),
        "total_risk_score": report.total_risk_score,
        "risk_level": report.risk_level,
        "clause_count": report.clause_count,
        "risk_finding_count": report.risk_finding_count,
        "outlier_count": report.outlier_count,
        "report_data": report.report_data,
        "generated_at": report.generated_at.isoformat(),
    }


@router.get("/{contract_id}/summary")
def get_report_summary(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.contract_id == contract_id)
        .order_by(AnalysisReport.generated_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    data = report.report_data
    return {
        "contract_summary": data.get("contract_summary"),
        "risk_summary": data.get("risk_summary"),
        "recommendations": data.get("recommendations", []),
    }
