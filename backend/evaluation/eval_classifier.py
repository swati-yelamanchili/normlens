"""
Evaluate the trained clause classifier against the existing keyword+embedding baseline.

Usage:
    python -m evaluation.eval_classifier
    python -m evaluation.eval_classifier --model-path models/clause_classifier/final
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.evaluation.classifier")

LABELS = [
    "Termination", "Payment Terms", "Liability", "Limitation of Liability",
    "Confidentiality", "Non-Compete", "Intellectual Property", "Indemnification",
    "Assignment", "Governing Law", "Arbitration", "Insurance", "Data Protection",
    "Data Ownership", "Security Obligations", "Force Majeure", "Warranty",
    "Representations and Warranties", "Entire Agreement", "Amendments", "Notices",
    "Severability", "Waiver", "Survival", "Counterparts", "Definitions",
    "Expenses", "Publicity", "Subcontracting", "No Waiver", "Further Assurances",
    "No Third Party Beneficiaries",
]

TEST_CLAUSES = {
    "Termination": [
        "Either party may terminate this agreement upon 90 days written notice.",
        "This agreement may be terminated for cause immediately upon written notice.",
        "In the event of termination, all license rights shall cease.",
    ],
    "Payment Terms": [
        "All invoices shall be paid within 30 calendar days of receipt.",
        "A late payment charge of 1.5% per month shall apply to overdue amounts.",
        "The total purchase price shall be paid in three equal installments.",
    ],
    "Confidentiality": [
        "The receiving party shall hold all confidential information in strict confidence.",
        "This non-disclosure agreement shall survive termination for a period of 5 years.",
        "Confidential information shall not be disclosed to any third party without written consent.",
    ],
    "Limitation of Liability": [
        "In no event shall either party's aggregate liability exceed $1,000,000.",
        "Neither party shall be liable for any indirect, incidental, or consequential damages.",
        "The limitation of liability set forth herein shall not apply to indemnification obligations.",
    ],
    "Non-Compete": [
        "The employee shall not engage in any competing business for 12 months post-termination.",
        "During the term and for 6 months thereafter, Consultant shall not solicit any client.",
        "The non-competition covenant shall apply within a 50-mile radius.",
    ],
    "Boilerplate": [
        "IN WITNESS WHEREOF, the parties have executed this agreement as of the date first written above.",
        "This agreement may be executed in counterparts, each of which shall be deemed an original.",
        "The parties acknowledge receipt of the exhibit attached hereto and incorporated by reference.",
    ],
}


def load_ml_model(model_path):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        label_map_path = os.path.join(os.path.dirname(model_path), "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}
        else:
            label_map = {i: l for i, l in enumerate(LABELS)}

        def predict(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            return label_map[pred_idx], float(probs[pred_idx].item())

        return predict
    except Exception as e:
        logger.warning(f"Cannot load ML model: {e}")
        return None


def evaluate_baseline():
    from app.classification import ClauseClassifier
    from app.embeddings import EmbeddingService

    classifier = ClauseClassifier()
    results = {}

    for expected_type, texts in TEST_CLAUSES.items():
        type_results = []
        for text in texts:
            start = time.time()
            pred_type, confidence = classifier.classify_best(text)
            elapsed = time.time() - start
            type_results.append({
                "text": text,
                "expected": expected_type,
                "predicted": pred_type,
                "confidence": confidence,
                "correct": pred_type == expected_type if expected_type != "Boilerplate" else pred_type is None,
                "time_ms": round(elapsed * 1000, 1),
            })
        results[expected_type] = type_results

    return results


def evaluate_ml(predict_fn):
    results = {}

    for expected_type, texts in TEST_CLAUSES.items():
        type_results = []
        for text in texts:
            start = time.time()
            pred_type, confidence = predict_fn(text)
            elapsed = time.time() - start
            type_results.append({
                "text": text,
                "expected": expected_type,
                "predicted": pred_type,
                "confidence": confidence,
                "correct": pred_type == expected_type if expected_type != "Boilerplate" else pred_type is None,
                "time_ms": round(elapsed * 1000, 1),
            })
        results[expected_type] = type_results

    return results


def print_results(name, results):
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"{'='*60}")

    total = 0
    correct = 0
    total_time = 0

    for type_name, type_results in results.items():
        type_correct = sum(1 for r in type_results if r["correct"])
        type_total = len(type_results)
        total += type_total
        correct += type_correct
        for r in type_results:
            total_time += r["time_ms"]
            status = "✓" if r["correct"] else "✗"
            logger.info(f"  {status} Expected={r['expected']:<25} Pred={r['predicted']:<25} Conf={r['confidence']:.3f} ({r['time_ms']}ms)")
            if not r["correct"]:
                logger.info(f"     Text: {r['text'][:80]}...")

    acc = correct / total * 100 if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    logger.info(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")
    logger.info(f"  Avg time: {avg_time:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Evaluate clause classifier")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    baseline_results = evaluate_baseline()
    print_results("BASELINE (keyword + embedding fusion)", baseline_results)

    if args.model_path:
        predict_fn = load_ml_model(args.model_path)
        if predict_fn:
            ml_results = evaluate_ml(predict_fn)
            print_results("ML MODEL (fine-tuned Legal-BERT)", ml_results)
        else:
            logger.warning("ML model not available, skipping ML evaluation")


if __name__ == "__main__":
    main()
