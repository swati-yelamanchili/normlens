import logging
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.contract import Contract, ContractStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/contracts", tags=["contracts"])

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload", status_code=201)
async def upload_contract(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_type = ext.replace(".", "")
    file_bytes = await file.read()

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.upload_dir, safe_filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    contract = Contract(
        filename=safe_filename,
        original_filename=file.filename or "unknown",
        file_type=file_type,
        file_size_bytes=str(len(file_bytes)),
        status=ContractStatus.PENDING,
    )

    db.add(contract)
    db.commit()
    db.refresh(contract)

    return {
        "id": str(contract.id),
        "filename": contract.original_filename,
        "file_type": contract.file_type,
        "status": contract.status.value,
        "message": "Contract uploaded successfully",
    }


@router.get("/")
def list_contracts(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    contracts = (
        db.query(Contract)
        .order_by(Contract.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": str(c.id),
            "filename": c.original_filename,
            "file_type": c.file_type,
            "status": c.status.value,
            "page_count": c.page_count,
            "created_at": c.created_at.isoformat(),
        }
        for c in contracts
    ]


@router.get("/{contract_id}")
def get_contract(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return {
        "id": str(contract.id),
        "filename": contract.original_filename,
        "file_type": contract.file_type,
        "status": contract.status.value,
        "page_count": contract.page_count,
        "text_preview": (contract.text_content or "")[:500],
        "created_at": contract.created_at.isoformat(),
        "updated_at": contract.updated_at.isoformat(),
    }


@router.get("/{contract_id}/download")
def download_contract(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    file_path = os.path.join(settings.upload_dir, contract.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Contract file not found on disk")
    media_type = "application/pdf" if contract.file_type == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=contract.original_filename,
    )


@router.delete("/{contract_id}")
def delete_contract(
    contract_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    file_path = os.path.join(settings.upload_dir, contract.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    db.delete(contract)
    db.commit()
    return {"message": "Contract deleted"}
