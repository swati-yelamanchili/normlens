import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.routers import analysis, contracts, reports, search

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Contract intelligence platform for clause classification, risk analysis, and statistical outlier detection.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(contracts.router)
    app.include_router(analysis.router)
    app.include_router(reports.router)
    app.include_router(search.router)

    @app.get("/api/health")
    def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    return app


app = create_app()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    
    # Auto-migrate missing columns for risk_findings table
    from sqlalchemy import text
    with engine.begin() as conn:
        for col, col_type in [
            ("finding_category", "VARCHAR"),
            ("clause_group", "VARCHAR"),
            ("supporting_clauses_json", "VARCHAR"),
            ("negotiation_recommendation", "VARCHAR")
        ]:
            try:
                conn.execute(text(f'ALTER TABLE risk_findings ADD COLUMN IF NOT EXISTS {col} {col_type}'))
                logger.info(f"Verified/added column {col}")
            except Exception as e:
                logger.warning(f"Could not check/add column {col}: {e}")
                
    logger.info("Database tables created / verified")
