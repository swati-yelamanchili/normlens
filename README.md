# NormLens

A contract intelligence platform that ingests contracts (PDF/DOCX), extracts clauses, classifies them using the CUAD taxonomy, detects risk patterns, benchmarks terms against market norms, and flags statistical outliers — all exposed through a REST API and a Next.js dashboard.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Frontend (Next.js 14, React 18, Tailwind, Recharts)  │
│  :3000                                                 │
└──────────────────────┬─────────────────────────────────┘
                       │ HTTP / JSON
┌──────────────────────▼─────────────────────────────────┐
│  Backend (FastAPI, SQLAlchemy, uvicorn)                 │
│  :8000                                                  │
│  ┌──────────── ──────────── ──────────── ──────────┐   │
│  │  routers/   │  services/        │  models/       │   │
│  │  contracts  │  extraction       │  Contract      │   │
│  │  analysis   │  embeddings       │  Clause        │   │
│  │  reports    │  classification   │  RiskFinding   │   │
│  │  search     │  rag              │  RiskRule      │   │
│  │             │  benchmarking     │  Benchmark     │   │
│  │             │  outlier          │  Report        │   │
│  │             │  reporting        │                 │   │
│  └──────────── ──────────── ──────────── ──────────┘   │
└──────┬──────────────────────────┬──────────────────────┘
       │                          │
┌──────▼──────┐          ┌───────▼────────┐
│  PostgreSQL │          │  ChromaDB      │
│  (metadata, │          │  (embeddings,  │
│   findings) │          │   similarity)  │
└─────────────┘          └────────────────┘
```

### Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Backend     | Python 3.11, FastAPI, SQLAlchemy    |
| Frontend    | Next.js 14, React 18, Tailwind CSS  |
| Database    | PostgreSQL 16 (Alembic migrations)  |
| Vector DB   | ChromaDB 0.5.3                      |
| Embeddings  | sentence-transformers (all-MiniLM-L6-v2) |
| ML          | scikit-learn, numpy, scipy          |
| PDF         | PyMuPDF, pdfplumber                 |
| NLP         | spaCy                               |

## Data Model

```
┌──────────────┐       ┌─────────────────┐
│   Contract   │1───many│     Clause      │
│──────────────│       │─────────────────│
│ id           │       │ id              │
│ filename     │       │ contract_id (FK)│
│ status       │       │ clause_index    │
│ text_content │       │ clause_title    │
│ file_type    │       │ clause_text     │
│ page_count   │       │ clause_type     │
│ error_msg    │       │ confidence      │
│ created_at   │       │ embedding_id    │
└──────┬───────┘       └────────┬────────┘
       │                        │
       │ 1                      │ 0..1
┌──────▼───────┐       ┌───────▼────────────┐
│ RiskFinding  │       │ BenchmarkResult    │
│──────────────│       │────────────────────│
│ id           │       │ id                 │
│ contract_id  │       │ contract_id (FK)   │
│ clause_id    │       │ clause_id (FK)     │
│ risk_name    │       │ clause_type        │
│ severity     │       │ attribute          │
│ points       │       │ market_median      │
│ percentile   │       │ percentile_rank    │
└──────────────┘       │ z_score            │
                        └────────────────────┘
       ▲                        ▲
┌──────┴───────┐       ┌────────┴──────────┐
│  RiskRule    │       │ AnalysisReport    │
│──────────────│       │───────────────────│
│ rule_id      │       │ contract_id (FK)  │
│ name         │       │ total_risk_score  │
│ category     │       │ risk_level        │
│ severity     │       │ clause_count      │
│ points       │       │ finding_count     │
│ conditions   │       │ report_data (JSON)│
│ enabled      │       └───────────────────┘
└──────────────┘
```

## API Endpoints

| Method | Path                         | Description              |
|--------|------------------------------|--------------------------|
| GET    | `/api/health`                | Health check             |
| POST   | `/api/contracts/upload`      | Upload PDF/DOCX          |
| GET    | `/api/contracts/`            | List all contracts       |
| GET    | `/api/contracts/{id}`        | Get contract details     |
| GET    | `/api/contracts/{id}/download` | Download original file |
| DELETE | `/api/contracts/{id}`        | Delete contract          |
| POST   | `/api/analysis/analyze/{id}` | Run full analysis pipleine |
| GET    | `/api/analysis/status/{id}`  | Get analysis status      |
| GET    | `/api/reports/{id}`          | Get full report          |
| GET    | `/api/reports/{id}/summary`  | Get report summary       |
| GET    | `/api/search/`               | Semantic clause search   |

## Analysis Pipeline

```
Upload ──► Parse ──► Segment ──► Embed ──► Classify ──► Extract ──► Risk ──► Benchmark ──► Outlier ──► Report
               │          │           │          │         │         │         │            │
          text/PDF   clause       vector    CUAD     regex/    rule     peer       z-score/
          extraction splitting   storage   k-NN     pattern  engine  comparison  percentile

```

1. **Parse** — Extract raw text from PDF (PyMuPDF + pdfplumber) or DOCX (python-docx)
2. **Segment** — Split text into clauses using NLP + regex boundary detection
3. **Embed** — Generate vector embeddings via sentence-transformers; store in ChromaDB
4. **Classify** — Assign CUAD clause types via embedding similarity + k-NN
5. **Extract** — Pull attributes (dates, amounts, durations) from clause text via pattern matching
6. **Risk** — Evaluate clauses against configurable risk rules (severity, conditions, weighted points)
7. **Benchmark** — Compare extracted attributes against market norms (percentile, z-score)
8. **Outlier** — Flag statistically unusual clauses (semantic + attribute-based outlier detection)
9. **Report** — Generate structured report with risk summary, findings, outliers, recommendations

## Key Components

### Backend (`backend/app/`)

| Module | Path | Responsibility |
|--------|------|----------------|
| Contracts | `routers/contracts.py` | Upload, list, download, delete contracts |
| Analysis | `routers/analysis.py` | Orchestrate full analysis pipeline |
| Reports | `routers/reports.py` | Retrieve analysis reports |
| Search | `routers/search.py` | Semantic search across clauses |
| Extraction | `services/extraction/` | PDF/DOCX text + clause extraction |
| Embeddings | `services/embeddings/` | Vector generation (sentence-transformers, hashing fallback) |
| Classification | `services/classification/` | CUAD clause type classifier |
| Risk | `services/risk_engine/` | Deterministic risk rule evaluation |
| Benchmarking | `benchmarking/` | Market norm comparison engine |
| Outlier | `outlier/` | Statistical outlier detection (percentile + z-score) |
| RAG | `services/rag/` | Retrieval-augmented generation context building |
| Reporting | `reporting/` | Structured report assembly |

### Frontend (`frontend/`)

- Next.js 14 App Router with TypeScript
- Tailwind CSS styling with responsive layout
- Recharts-based risk visualisation (severity breakdown, scores)
- Real-time progress tracking during analysis
- Clause-level drill-down with classification details

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- Python 3.11+ and Node.js 20+ (for local development)

### Docker (full stack)

```bash
docker compose up --build
```

| Service   | URL                  |
|-----------|----------------------|
| Frontend  | http://localhost:3000 |
| Backend   | http://localhost:8000 |
| ChromaDB  | http://localhost:8001 |
| PostgreSQL| localhost:5432        |

### Local Development

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
# Start PostgreSQL and ChromaDB separately, then:
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Configuration

Environment variables (`.env` at `backend/` root):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://normlens:normlens@localhost:5432/normlens` | PostgreSQL DSN |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_BACKEND` | `auto` | Backend: auto, sentence-transformers, or hashing |
| `UPLOAD_DIR` | `/tmp/normlens/uploads` | Upload storage path |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
cd backend
pytest
```

---

Built with the [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset) taxonomy for clause classification.
