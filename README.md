# NormLens

A contract intelligence platform that ingests contracts (PDF/DOCX), extracts clauses,
classifies them using a fine‑tuned Legal‑BERT clause classifier, detects risk patterns,
benchmarks terms against market norms, and flags statistical outliers — all exposed
through a REST API and a Next.js dashboard.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Frontend (Next.js 16 preview, React 18, Tailwind, Recharts) │
│  :3000                                                     │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP / JSON
┌──────────────────────▼─────────────────────────────────────┐
│  Backend (FastAPI, SQLAlchemy, uvicorn)                     │
│  :8000                                                      │
│  ┌───────────── ──────────── ──────────── ──────────┐      │
│  │  routers/    │  services/        │  models/       │      │
│  │  contracts   │  extraction       │  Contract      │      │
│  │  analysis    │  embeddings       │  Clause        │      │
│  │  reports     │  classification   │  RiskFinding   │      │
│  │  search      │  benchmarking     │  RiskRule      │      │
│  │              │  outlier          │  Benchmark     │      │
│  │              │  reporting        │  Report        │      │
│  └───────────── ──────────── ──────────── ──────────┘      │
│                                                             │
│  ┌────────────────── ML Inference ──────────────────────┐   │
│  │  Clause          │  Attribute     │  Risk      │ CT   │   │
│  │  Classifier      │  NER           │  Scorer   │ Type  │   │
│  │  (Legal‑BERT)    │  (spaCy)       │  (GBM)    │(TfIdf)│   │
│  └─────────────────────────────────────────────────────┘   │
└──────┬──────────────────────────┬──────────────────────────┘
       │                          │
┌──────▼──────┐          ┌───────▼────────┐
│  PostgreSQL │          │  ChromaDB      │
│  (metadata, │          │  (embeddings,  │
│   findings) │          │   similarity)  │
└─────────────┘          └────────────────┘
```

### Stack

| Layer         | Technology                                   |
|---------------|----------------------------------------------|
| Backend       | Python 3.11, FastAPI, SQLAlchemy             |
| Frontend      | Next.js 16 preview, React 18, Tailwind CSS   |
| Database      | PostgreSQL 16 (Alembic migrations)           |
| Vector DB     | ChromaDB 0.5.3                               |
| Embeddings    | sentence-transformers (all-MiniLM-L6-v2)     |
| Clause Class  | Legal‑BERT (nlpaueb/legal-bert-base-uncased) |
| NER           | spaCy (en_core_web_sm + custom entities)     |
| ML            | scikit-learn, transformers, torch, datasets  |
| PDF           | PyMuPDF, pdfplumber                          |

---

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

---

## API Endpoints

| Method | Path                         | Description                   |
|--------|------------------------------|-------------------------------|
| GET    | `/api/health`                | Health check                  |
| POST   | `/api/contracts/upload`      | Upload PDF/DOCX               |
| GET    | `/api/contracts/`            | List all contracts            |
| GET    | `/api/contracts/{id}`        | Get contract details          |
| GET    | `/api/contracts/{id}/download` | Download original file      |
| DELETE | `/api/contracts/{id}`        | Delete contract               |
| POST   | `/api/analysis/analyze/{id}` | Run full analysis pipeline    |
| GET    | `/api/analysis/status/{id}`  | Get analysis status           |
| GET    | `/api/reports/{id}`          | Get full report               |
| GET    | `/api/reports/{id}/summary`  | Get report summary            |
| GET    | `/api/search/`               | Semantic clause search        |

### ML Inference Server (optional, port 8001)

| Method | Path                         | Description                   |
|--------|------------------------------|-------------------------------|
| GET    | `/health`                    | Model health + availability   |
| POST   | `/predict/clause-type`       | Legal‑BERT clause prediction  |
| POST   | `/predict/entities`          | spaCy NER entity extraction   |
| POST   | `/predict/risk-score`        | Gradient‑boosted risk scoring |
| POST   | `/predict/contract-type`     | TF‑IDF + LR type prediction  |
| GET    | `/models`                    | List loaded model paths       |

---

## Analysis Pipeline

```
Upload ──► Parse ──► Segment ──► Embed ──► Classify ──► Extract ──► Risk ──► Benchmark ──► Outlier ──► Report
               │          │           │          │         │         │         │            │
          text/PDF   clause       vector    Legal‑BERT  regex/    rule     peer       z-score/
          extraction splitting   storage    + k‑NN     pattern  engine  comparison  percentile
                                            fusion
```

1. **Parse** — Extract raw text from PDF (PyMuPDF + pdfplumber) or DOCX (python-docx)
2. **Segment** — Split text into clauses using NLP + regex boundary detection
3. **Embed** — Generate vector embeddings via sentence-transformers; store in ChromaDB
4. **Classify** — Assign CUAD clause types (Legal‑BERT fine-tuned model + embedding k‑NN fusion)
5. **Extract** — Pull attributes (dates, amounts, durations) via regex + spaCy NER
6. **Risk** — Evaluate clauses against 22 configurable risk rules (severity, conditions, points)
7. **Benchmark** — Compare extracted attributes against market norms (percentile, z‑score)
8. **Outlier** — Flag statistically unusual clauses (semantic + attribute-based outlier detection)
9. **Report** — Generate structured report with risk summary, findings, outliers, recommendations

---

## Project Structure

```
normlens/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── benchmarking/       # Market norm comparison engine
│   │   ├── classification/     # Clause classifier (keyword + embedding fusion)
│   │   ├── embeddings/         # Vector embeddings + ChromaDB interface
│   │   ├── extraction/         # Attribute extraction (regex + spaCy NER)
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── outlier/            # Statistical outlier detection
│   │   ├── parsers/            # PDF/DOCX text extraction
│   │   ├── reporting/          # Report generation
│   │   ├── risk/               # Risk engine + rules + contract type detector
│   │   ├── routers/            # API endpoint definitions
│   │   ├── search/             # Semantic search
│   │   ├── segmentation/       # Clause segmentation
│   │   ├── services/           # Task manager, NLP service
│   │   ├── config.py
│   │   ├── database.py
│   │   └── main.py
│   │
│   ├── training/               # 🆕 ML model training scripts
│   │   ├── train_classifier.py      # Legal‑BERT fine-tuning on CUAD + LEDGAR
│   │   ├── train_ner.py             # spaCy NER from noisy regex labels
│   │   ├── train_risk_scorer.py     # Gradient‑boosted risk severity model
│   │   └── train_contract_type.py   # TF‑IDF + Logistic Regression
│   │
│   ├── evaluation/             # 🆕 Model evaluation & benchmarking
│   │   ├── eval_classifier.py       # Clause classifier vs baseline
│   │   ├── eval_ner.py              # NER model vs regex baseline
│   │   └── benchmark.py             # Full pipeline benchmark
│   │
│   ├── inference/              # 🆕 ML inference server
│   │   └── model_server.py          # FastAPI wrapper for all trained models
│   │
│   ├── models/                 # 🆕 Trained model artifacts
│   │   ├── clause_classifier/       # Legal‑BERT fine-tuned
│   │   ├── attribute_ner/           # spaCy NER with custom entities
│   │   ├── risk_scorer/             # GradientBoostingClassifier.pkl
│   │   └── contract_type/           # TF‑IDF vectorizer + LR model
│   │
│   ├── data/
│   │   ├── market_norms.json        # Market comparison distributions
│   │   ├── raw/                     # CUAD, LEDGAR, custom contracts
│   │   ├── processed/               # Tokenized, split datasets
│   │   └── labels/                  # Human annotations
│   │
│   ├── tests/                  # Test suite (68+ tests)
│   ├── requirements.txt
│   ├── requirements-ml.txt
│   └── Dockerfile
│
├── frontend/                   # Next.js dashboard
├── docker-compose.yml
└── README.md
```

---

## ML Component Overview

| Component               | Heuristic Baseline                          | ML Model                                      | Status         |
|-------------------------|---------------------------------------------|-----------------------------------------------|----------------|
| Clause Classification   | Embedding k‑NN + keyword fusion             | Legal‑BERT fine-tuned (CUAD + LEDGAR)         | Trained / v0   |
| Attribute Extraction    | 50+ regex patterns                          | spaCy NER with custom entity labels           | Trained / v0   |
| Risk Severity Scoring   | 22 hand‑coded rules (if‑then thresholds)    | Gradient‑boosted tree (sklearn)               | Trained / v0   |
| Contract Type Detection | Weighted keyword scoring                    | TF‑IDF + Logistic Regression                  | Trained / v0   |
| Missing Clause Detection| Boolean check (title regex)                 | Keep rules (no ML needed)                     | —              |
| Outlier Detection       | Percentile / z‑score                        | Keep statistical (sufficient)                 | —              |
| Report Generation       | Template‑based                              | Keep template (no ML needed)                  | —              |

### Model Performance

| Model                    | Accuracy     | Notes                                   |
|--------------------------|-------------|-----------------------------------------|
| Clause Classifier        | in progress  | 32 CUAD clause types, multi-class       |
| Contract Type Classifier | **92.3%**    | 9 contract types, 516 synthetic samples |
| Risk Scorer              | **95.5%**    | 4 severity levels, 2200 samples         |
| Attribute NER            | 100% eval   | 7 custom entities, 80 training examples |

---

## Training

Each model can be trained independently. All scripts accept `--output-dir` to
specify where the trained artifact is saved.

### Prerequisites

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt
```

### 1. Clause Classifier (Legal‑BERT)

Fine‑tunes `nlpaueb/legal-bert-base-uncased` on the CUAD dataset and LEDGAR
provision labels from HuggingFace Datasets, plus synthetic fallback data.

```bash
# Full training with CUAD + LEDGAR + synthetic (recommended with GPU)
python -m training.train_classifier --epochs 5 --batch_size 8

# Quick test with synthetic data only (CPU‑friendly)
python -m training.train_classifier --use_synthetic --epochs 3 --batch_size 4

# Custom output directory
python -m training.train_classifier --output-dir /app/models/clause_classifier
```

**Data sources:**
- [CUAD](https://huggingface.co/datasets/cuad) — 41 yes/no questions about contract clauses
- [LEDGAR](https://huggingface.co/datasets/lex_glue) — 60k+ labelled contract provisions
- Synthetic — 1700 clause samples generated from domain templates (fallback)

**Label map:** Saved as `label_map.json` alongside the model.

### 2. Attribute NER (spaCy)

Bootstraps training data from the existing regex patterns in the attribute
extractor to train a spaCy NER pipeline with custom entity types.

```bash
python -m training.train_ner --epochs 30 --output-dir models/attribute_ner
```

**Custom entity types:** `NOTICE_DAYS`, `LIABILITY_AMOUNT`, `PAYMENT_DEADLINE`,
`DURATION`, `MONEY`, `PERCENT`, `LAW`, `ORG`

### 3. Risk Severity Scorer (Gradient‑Boosted)

Generates 2200 synthetic samples from market norm distributions with realistic
z‑score severity labels, then trains a `GradientBoostingClassifier`.

```bash
python -m training.train_risk_scorer --epochs 200 --output-dir models/risk_scorer
```

**Feature vector (29 dimensions):**
- 25 market attribute z‑scores (padded across all clause types)
- Missing attribute count
- Total attribute count
- Clause type index
- Random noise feature

### 4. Contract Type Classifier (TF‑IDF + LR)

Generates 516 synthetic contract texts across 9 types, vectorises with
TF‑IDF (unigrams + bigrams + trigrams), and trains a multinomial logistic
regression.

```bash
python -m training.train_contract_type --output-dir models/contract_type
```

**Contract types:** SaaS Agreement, NDA, MSA, Consulting Agreement, Professional
Services Agreement, Employment Agreement, Vendor Agreement, License Agreement,
Government Contract.

---

## Evaluation

### Clause Classifier Evaluation

Compares the fine‑tuned Legal‑BERT model against the heuristic keyword +
embedding fusion baseline on a held‑out test set.

```bash
python -m evaluation.eval_classifier
python -m evaluation.eval_classifier --model-path models/clause_classifier/final
```

### NER Evaluation

Compares the trained spaCy model against the regex baseline on 8 test
sentences with gold entity spans.

```bash
python -m evaluation.eval_ner
python -m evaluation.eval_ner --model-path models/attribute_ner
```

### Full Pipeline Benchmark

Runs every analysis component (segmentation, classification, extraction,
outlier detection, risk scoring, contract type detection) against a test
contract and reports timing.

```bash
python -m evaluation.benchmark
python -m evaluation.benchmark --quick   # skip embedding benchmark
```

Results saved to `models/benchmark_results.json`.

---

## Inference Server

A standalone FastAPI server wraps all trained models for production inference.

```bash
# Start on port 8001
python -m inference.model_server
# or
uvicorn inference.model_server:app --host 0.0.0.0 --port 8001
```

**Endpoints:**

```bash
# Clause classification
curl -X POST http://localhost:8001/predict/clause-type \
  -H "Content-Type: application/json" \
  -d '{"text": "Either party may terminate upon 90 days notice.", "top_k": 3}'

# Entity extraction
curl -X POST http://localhost:8001/predict/entities \
  -H "Content-Type: application/json" \
  -d '{"text": "The liability cap is $5,000,000."}'

# Risk scoring
curl -X POST http://localhost:8001/predict/risk-score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, 0.0, 2]}'

# Contract type classification
curl -X POST http://localhost:8001/predict/contract-type \
  -H "Content-Type: application/json" \
  -d '{"text": "This Software as a Service Agreement is entered into..."}'
```

---

## Key Components

### Backend (`backend/app/`)

| Module              | Path                            | Responsibility                          |
|---------------------|---------------------------------|-----------------------------------------|
| Contracts           | `routers/contracts.py`          | Upload, list, download, delete contracts |
| Analysis            | `routers/analysis.py`           | Orchestrate full analysis pipeline      |
| Reports             | `routers/reports.py`            | Retrieve analysis reports               |
| Search              | `routers/search.py`             | Semantic search across clauses          |
| Extraction          | `extraction/attribute_extractor.py` | Regex + NER attribute extraction     |
| Embeddings          | `embeddings/embedding_service.py`   | Vector generation + ChromaDB          |
| Classification      | `classification/classifier.py`  | CUAD clause type classifier             |
| Risk                | `risk/risk_engine.py`           | Deterministic risk rule evaluation      |
| Benchmarking        | `benchmarking/benchmarking_engine.py` | Market norm comparison              |
| Outlier             | `outlier/outlier_detector.py`   | Statistical outlier detection           |
| Reporting           | `reporting/report_generator.py` | Structured report assembly              |

### Frontend (`frontend/`)

- Next.js 16 preview App Router with TypeScript
- Tailwind CSS styling with responsive layout
- Recharts-based risk visualisation (severity breakdown, scores)
- Real-time progress tracking during analysis
- Clause-level drill-down with classification details

---

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
# Start PostgreSQL and ChromaDB separately.
# For an existing database, apply migrations before starting the API:
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm ci
npm run dev
```

### Running the ML Inference Server

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt
python -m training.train_contract_type
python -m training.train_risk_scorer
python -m training.train_ner
python -m training.train_classifier --use_synthetic --epochs 3 --batch_size 4
python -m inference.model_server
```

---

## Configuration

Environment variables (`.env` at `backend/` root):

| Variable             | Default                                          | Description                          |
|----------------------|--------------------------------------------------|--------------------------------------|
| `DATABASE_URL`       | `postgresql://normlens:normlens@localhost:5432/normlens` | PostgreSQL DSN              |
| `CHROMA_HOST`        | `localhost`                                      | ChromaDB host                        |
| `CHROMA_PORT`        | `8001`                                           | ChromaDB port                        |
| `EMBEDDING_MODEL`    | `all-MiniLM-L6-v2`                               | Sentence-transformers model           |
| `EMBEDDING_BACKEND`  | `auto`                                           | auto, sentence-transformers, hashing  |
| `UPLOAD_DIR`         | `/tmp/normlens/uploads`                          | Upload storage path                   |
| `LOG_LEVEL`          | `INFO`                                           | Logging level                         |
| `ML_SERVER_PORT`     | `8001`                                           | ML inference server port              |

---

## Testing

```bash
cd backend
python -m pytest
```

The test suite covers:
- Clause classification (keyword + embedding)
- Attribute extraction (all 50+ attribute types)
- Benchmarking engine (market norms, percentiles)
- Outlier detection (statistical flags)
- Risk engine (rule evaluation, deduplication)
- Report generation (structure, severity breakdown)
- Search (semantic, exact match, intent filtering)
- Task manager (background analysis)
- Vector store (ChromaDB operations)
- NLP service (spaCy integration)

---

## References

- [CUAD — Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad)
- [LEDGAR — Labeled Contract Provision Dataset](https://zenodo.org/records/7930762)
- [Legal‑BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
- [LexGLUE benchmark](https://huggingface.co/datasets/lex_glue)
