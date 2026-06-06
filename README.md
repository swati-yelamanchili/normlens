# NormLens

Contract intelligence platform for clause classification, risk analysis, and statistical outlier detection using embeddings, semantic retrieval, and explainable rule-based reasoning.

## Features

- **Document Parsing** - PDF and DOCX contract parsing
- **Clause Segmentation** - Automatic clause splitting with heading detection
- **Clause Classification** - 30 CUAD-based clause types via embedding similarity
- **Attribute Extraction** - Deterministic rule-based extraction of key contract terms
- **Risk Detection** - Configurable rule engine with templated explanations
- **Benchmarking** - Peer-group comparison against market norms
- **Outlier Detection** - Statistical detection of unusual terms using percentile and z-score analysis
- **Semantic Search** - Embedding-based clause retrieval
- **Explainable Reports** - Fully traceable, reproducible risk reports

## Architecture

```
Upload → Parse → Segment → Embed → Classify → Extract → Detect Risks → Benchmark → Detect Outliers → Score → Report
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Frontend | Next.js (TypeScript, TailwindCSS) |
| ML | deterministic local embeddings by default, optional sentence-transformers |
| Database | PostgreSQL + ChromaDB |
| Document Parsing | PyMuPDF, pdfplumber, python-docx |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 16
- Docker (optional)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The base install uses lightweight deterministic hashing embeddings so local setup does
not download PyTorch. For production-quality semantic embeddings, install the optional
CPU-only ML stack and enable it:

```bash
pip install -r requirements-ml.txt
EMBEDDING_BACKEND=sentence-transformers uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/contracts/upload` | Upload a contract (PDF/DOCX) |
| GET | `/api/contracts/` | List all contracts |
| GET | `/api/contracts/{id}` | Get contract details |
| DELETE | `/api/contracts/{id}` | Delete a contract |
| POST | `/api/analysis/analyze/{id}` | Run full analysis pipeline |
| GET | `/api/analysis/status/{id}` | Check analysis status |
| GET | `/api/reports/{id}` | Get full analysis report |
| GET | `/api/reports/{id}/summary` | Get report summary |
| GET | `/api/search/?q=` | Semantic clause search |

## Supported Clause Types

Termination, Payment Terms, Liability, Limitation of Liability, Confidentiality, Non-Compete, Intellectual Property, Indemnification, Assignment, Governing Law, Arbitration, Insurance, Data Protection, and 17 more CUAD-based types.

## License

MIT
