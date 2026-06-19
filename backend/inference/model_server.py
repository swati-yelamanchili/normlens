"""
FastAPI server that wraps all trained ML models for inference.

Usage:
    uvicorn inference.model_server:app --host 0.0.0.0 --port 8001
    python -m inference.model_server
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.inference.server")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_models_loaded = False
_clause_classifier = None
_clause_tokenizer = None
_clause_label_map = None
_ner_model = None
_risk_scorer = None
_contract_type_vectorizer = None
_contract_type_model = None
_contract_type_map = None


def _load_models():
    global _models_loaded
    global _clause_classifier, _clause_tokenizer, _clause_label_map
    global _ner_model
    global _risk_scorer
    global _contract_type_vectorizer, _contract_type_model, _contract_type_map

    if _models_loaded:
        return

    models = {
        "clause_classifier": MODELS_DIR / "clause_classifier" / "final",
        "attribute_ner": MODELS_DIR / "attribute_ner",
        "risk_scorer": MODELS_DIR / "risk_scorer",
        "contract_type": MODELS_DIR / "contract_type",
    }

    cc_path = models["clause_classifier"]
    if cc_path.exists():
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            _clause_tokenizer = AutoTokenizer.from_pretrained(str(cc_path))
            _clause_classifier = AutoModelForSequenceClassification.from_pretrained(str(cc_path))
            _clause_classifier.eval()

            label_map_path = cc_path.parent / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path) as f:
                    _clause_label_map = json.load(f)
                _clause_label_map = {int(k): v for k, v in _clause_label_map.items()}
            logger.info(f"Clause classifier loaded from {cc_path}")
        except Exception as e:
            logger.warning(f"Cannot load clause classifier: {e}")

    ner_path = models["attribute_ner"]
    if ner_path.exists():
        try:
            import spacy
            _ner_model = spacy.load(str(ner_path))
            logger.info(f"NER model loaded from {ner_path}")
        except Exception as e:
            logger.warning(f"Cannot load NER model: {e}")

    rs_path = models["risk_scorer"]
    if rs_path.exists():
        try:
            import joblib
            model_file = rs_path / "risk_scorer.pkl"
            if model_file.exists():
                _risk_scorer = joblib.load(str(model_file))
                logger.info(f"Risk scorer loaded from {rs_path}")
        except Exception as e:
            logger.warning(f"Cannot load risk scorer: {e}")

    ct_path = models["contract_type"]
    if ct_path.exists():
        try:
            import joblib
            _contract_type_vectorizer = joblib.load(str(ct_path / "vectorizer.pkl"))
            _contract_type_model = joblib.load(str(ct_path / "contract_type_model.pkl"))

            type_map_path = ct_path / "contract_types.json"
            if type_map_path.exists():
                with open(type_map_path) as f:
                    _contract_type_map = json.load(f)
                _contract_type_map = {int(k): v for k, v in _contract_type_map.items()}
            logger.info(f"Contract type model loaded from {ct_path}")
        except Exception as e:
            logger.warning(f"Cannot load contract type model: {e}")

    _models_loaded = True
    logger.info("Model loading complete")


def predict_clause_type(text: str, top_k: int = 3) -> List[Dict]:
    if _clause_classifier is None:
        return [{"error": "Clause classifier not available"}]

    import torch

    inputs = _clause_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = _clause_classifier(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze(0)
    top_indices = torch.topk(probs, min(top_k, len(probs))).indices.tolist()

    results = []
    for idx in top_indices:
        results.append({
            "clause_type": _clause_label_map.get(idx, f"unknown_{idx}"),
            "confidence": round(float(probs[idx].item()), 4),
        })
    return results


def extract_entities(text: str) -> List[Dict]:
    if _ner_model is None:
        return [{"error": "NER model not available"}]

    doc = _ner_model(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def predict_risk_score(features: List[float]) -> Dict:
    if _risk_scorer is None:
        return {"error": "Risk scorer not available"}

    X = np.array(features).reshape(1, -1)
    pred = _risk_scorer.predict(X)[0]
    proba = _risk_scorer.predict_proba(X)[0]

    severity_levels = ["Low", "Moderate", "High", "Critical"]
    return {
        "severity": severity_levels[int(pred)] if int(pred) < len(severity_levels) else "Unknown",
        "severity_index": int(pred),
        "probabilities": {
            level: round(float(prob), 3)
            for level, prob in zip(severity_levels, proba)
        },
    }


def predict_contract_type(text: str) -> Dict:
    if _contract_type_vectorizer is None or _contract_type_model is None:
        return {"error": "Contract type model not available"}

    X = _contract_type_vectorizer.transform([text])
    pred = _contract_type_model.predict(X)[0]
    proba = _contract_type_model.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))

    return {
        "contract_type": _contract_type_map.get(top_idx, f"unknown_{top_idx}"),
        "confidence": round(float(proba[top_idx]), 4),
        "all_probabilities": {
            _contract_type_map.get(i, f"unknown_{i}"): round(float(p), 3)
            for i, p in enumerate(proba)
        },
    }


def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="NormLens ML Inference Server", version="1.0.0")

    class ClauseRequest(BaseModel):
        text: str
        top_k: int = 3

    class NERRequest(BaseModel):
        text: str

    class RiskRequest(BaseModel):
        features: List[float]

    class ContractTypeRequest(BaseModel):
        text: str

    @app.on_event("startup")
    async def startup():
        _load_models()

    @app.get("/health")
    async def health():
        status = {
            "clause_classifier": _clause_classifier is not None,
            "ner_model": _ner_model is not None,
            "risk_scorer": _risk_scorer is not None,
            "contract_type_model": _contract_type_vectorizer is not None,
        }
        return {
            "status": "ok" if any(status.values()) else "degraded",
            "models": status,
        }

    @app.post("/predict/clause-type")
    async def classify_clause(req: ClauseRequest):
        return {"predictions": predict_clause_type(req.text, req.top_k)}

    @app.post("/predict/entities")
    async def extract_entities_endpoint(req: NERRequest):
        return {"entities": extract_entities(req.text)}

    @app.post("/predict/risk-score")
    async def score_risk(req: RiskRequest):
        return predict_risk_score(req.features)

    @app.post("/predict/contract-type")
    async def classify_contract_type(req: ContractTypeRequest):
        return predict_contract_type(req.text)

    @app.get("/models")
    async def list_models():
        return {
            "clause_classifier": str(MODELS_DIR / "clause_classifier" / "final"),
            "attribute_ner": str(MODELS_DIR / "attribute_ner"),
            "risk_scorer": str(MODELS_DIR / "risk_scorer"),
            "contract_type": str(MODELS_DIR / "contract_type"),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_SERVER_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
