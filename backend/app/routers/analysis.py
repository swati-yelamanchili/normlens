import logging
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.benchmarking import BenchmarkingEngine
from app.classification import ClauseClassifier
from app.config import settings
from app.database import get_db
from app.embeddings import EmbeddingService
from app.extraction import AttributeExtractor
from app.models.benchmark import BenchmarkResult
from app.models.clause import Clause as ClauseModel
from app.models.contract import Contract, ContractStatus
from app.models.report import AnalysisReport
from app.models.risk import RiskFinding as RiskFindingModel
from app.outlier import OutlierDetector
from app.parsers import DOCXParser, PDFParser
from app.reporting import ReportGenerator
from app.risk import RiskEngine
from app.segmentation import ClauseSegmenter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

_embedding_service = None
_classifier = None
_extractor = None
_segmenter = None
_risk_engine = None
_benchmarking = None
_outlier_detector = None
_report_generator = None


def _get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ClauseClassifier(_get_embedding_service())
    return _classifier

def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = AttributeExtractor()
    return _extractor

def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        _segmenter = ClauseSegmenter()
    return _segmenter

def _get_risk_engine():
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine

def _get_benchmarking():
    global _benchmarking
    if _benchmarking is None:
        _benchmarking = BenchmarkingEngine(_get_embedding_service(), _get_classifier())
    return _benchmarking

def _get_outlier_detector():
    global _outlier_detector
    if _outlier_detector is None:
        _outlier_detector = OutlierDetector(_get_embedding_service(), _get_classifier(), _get_benchmarking())
    return _outlier_detector

def _get_report_generator():
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


@router.post("/analyze/{contract_id}")
def analyze_contract(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    file_path = os.path.join(settings.upload_dir, contract.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Contract file not found on disk")

    try:
        db.query(BenchmarkResult).filter(BenchmarkResult.contract_id == contract.id).delete()
        db.query(RiskFindingModel).filter(RiskFindingModel.contract_id == contract.id).delete()
        db.query(AnalysisReport).filter(AnalysisReport.contract_id == contract.id).delete()
        db.query(ClauseModel).filter(ClauseModel.contract_id == contract.id).delete()
        db.commit()

        contract.status = ContractStatus.PARSING
        db.commit()

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if contract.file_type == "pdf":
            parser = PDFParser()
        else:
            parser = DOCXParser()

        parsed = parser.parse(file_bytes, contract.filename)
        contract.text_content = parsed["text"]
        contract.page_count = str(parsed["page_count"])
        contract.status = ContractStatus.PARSED
        db.commit()

        contract.status = ContractStatus.SEGMENTING
        db.commit()

        raw_clauses = _get_segmenter().segment(parsed["text"], parsed.get("pages"))
        contract.status = ContractStatus.SEGMENTED
        db.commit()

        clause_models = []
        for clause_data in raw_clauses:
            clause_model = ClauseModel(
                contract_id=contract.id,
                clause_index=clause_data["clause_index"],
                clause_title=clause_data.get("clause_title"),
                clause_text=clause_data["clause_text"],
                page_number=clause_data.get("page_number"),
            )
            db.add(clause_model)
            clause_models.append(clause_model)
        db.commit()
        for cm in clause_models:
            db.refresh(cm)

        clause_by_index = {cm.clause_index: cm for cm in clause_models}

        contract.status = ContractStatus.CLASSIFYING
        db.commit()

        classified_clauses = []
        for clause_model in clause_models:
            clause_type, confidence = _get_classifier().classify_best(clause_model.clause_text)
            clause_model.clause_type = clause_type
            clause_model.classification_confidence = confidence
            db.commit()

            classified_clauses.append(
                {
                    "clause_index": clause_model.clause_index,
                    "clause_title": clause_model.clause_title,
                    "clause_text": clause_model.clause_text,
                    "page_number": clause_model.page_number,
                    "clause_type": clause_type,
                    "classification_confidence": confidence,
                }
            )

        contract.status = ContractStatus.CLASSIFIED
        db.commit()

        contract.status = ContractStatus.ANALYZING
        db.commit()

        all_findings = []
        all_outliers = []
        all_benchmarks = []
        clause_types_found = set()

        for clause_data in classified_clauses:
            clause_index = clause_data["clause_index"]
            current_cm = clause_by_index.get(clause_index)
            attr = _get_extractor().extract(clause_data["clause_text"], clause_data["clause_type"])
            clause_data["attributes"] = attr

            clause_type = clause_data.get("clause_type")
            if clause_type:
                clause_types_found.add(clause_type)

            findings = _get_risk_engine().evaluate_clause(clause_data, attr)
            for f in findings:
                f["clause_id"] = str(current_cm.id) if current_cm else None
            all_findings.extend(findings)

            if clause_type:
                outliers = _get_outlier_detector().detect_outliers(clause_data, attr, clause_type)
                all_outliers.extend(outliers)

                benchmarks = _get_benchmarking().benchmark_attributes(
                    clause_data, attr, clause_type
                )
                all_benchmarks.extend(benchmarks)

                for bench in benchmarks:
                    bench_model = BenchmarkResult(
                        contract_id=contract.id,
                        clause_id=current_cm.id if current_cm else None,
                        clause_type=clause_type,
                        attribute=bench["attribute"],
                        contract_value=bench.get("contract_value"),
                        market_median=bench.get("market_median"),
                        market_mean=bench.get("market_mean"),
                        market_std=bench.get("market_std"),
                        market_p5=bench.get("market_p5"),
                        market_p25=bench.get("market_p25"),
                        market_p75=bench.get("market_p75"),
                        market_p95=bench.get("market_p95"),
                        percentile_rank=bench.get("percentile_rank"),
                        z_score=bench.get("z_score"),
                        peer_count=bench.get("peer_count"),
                    )
                    db.add(bench_model)
                db.commit()

            if current_cm:
                current_cm.attributes = attr
                db.commit()

        missing_findings = _get_risk_engine().evaluate_missing_clauses(
            classified_clauses, list(clause_types_found)
        )
        all_findings.extend(missing_findings)

        for finding in all_findings:
            finding_model = RiskFindingModel(
                contract_id=contract.id,
                clause_id=finding.get("clause_id"),
                risk_name=finding.get("risk_name", "Unknown Risk"),
                severity=finding.get("severity", "Medium"),
                points=finding.get("points", 0),
                supporting_clause=finding.get("supporting_clause", "")[:1000],
                extracted_value=finding.get("extracted_value", ""),
                explanation=finding.get("explanation", "No explanation provided."),
            )
            db.add(finding_model)
        db.commit()

        total_score = _get_risk_engine().calculate_risk_score(all_findings)
        risk_level = _get_risk_engine().get_risk_level(total_score)

        report_data = _get_report_generator().generate(
            contract={
                "original_filename": contract.original_filename,
                "file_type": contract.file_type,
                "page_count": contract.page_count,
            },
            clauses=classified_clauses,
            classifications=classified_clauses,
            risk_findings=all_findings,
            outliers=all_outliers,
            benchmarks=all_benchmarks,
            total_risk_score=total_score,
            risk_level=risk_level,
        )

        report_model = AnalysisReport(
            contract_id=contract.id,
            total_risk_score=total_score,
            risk_level=risk_level,
            clause_count=len(classified_clauses),
            risk_finding_count=len(all_findings),
            outlier_count=len(all_outliers),
            report_data=report_data,
        )
        db.add(report_model)
        db.commit()

        contract.status = ContractStatus.ANALYZED
        db.commit()

        return {
            "contract_id": str(contract.id),
            "status": "completed",
            "total_risk_score": total_score,
            "risk_level": risk_level,
            "clause_count": len(classified_clauses),
            "finding_count": len(all_findings),
            "outlier_count": len(all_outliers),
        }

    except Exception as e:
        logger.exception(f"Analysis failed for contract {contract_id}")
        db.rollback()
        contract.status = ContractStatus.FAILED
        contract.error_message = str(e)[:500]
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/status/{contract_id}")
def get_analysis_status(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return {
        "contract_id": str(contract.id),
        "status": contract.status.value,
        "error_message": contract.error_message,
    }
