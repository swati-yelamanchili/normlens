"""
Evaluate the trained NER model for attribute extraction against the regex baseline.

Usage:
    python -m evaluation.eval_ner
    python -m evaluation.eval_ner --model-path models/attribute_ner
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.evaluation.ner")

TEST_EXAMPLES = [
    {
        "text": "This agreement may be terminated upon 90 days written notice.",
        "entities": [("90 days", "DATE"), ("90", "NOTICE_DAYS")],
    },
    {
        "text": "The total liability of either party shall not exceed $5,000,000.",
        "entities": [("$5,000,000", "MONEY"), ("5,000,000", "LIABILITY_AMOUNT")],
    },
    {
        "text": "All invoices shall be paid within 30 calendar days of receipt.",
        "entities": [("30", "PAYMENT_DEADLINE"), ("30", "NOTICE_DAYS")],
    },
    {
        "text": "Confidentiality obligations shall continue for 5 years.",
        "entities": [("5 years", "DATE"), ("5", "DURATION")],
    },
    {
        "text": "Late payments shall accrue interest at 1.5% per annum.",
        "entities": [("1.5%", "PERCENT")],
    },
    {
        "text": "The non-compete obligations shall survive for 12 months.",
        "entities": [("12 months", "DURATION"), ("12", "DATE")],
    },
    {
        "text": "This Agreement is governed by the laws of the State of New York.",
        "entities": [("laws of the State of New York", "LAW")],
    },
    {
        "text": "Provider shall maintain CGL insurance of $2,000,000 per occurrence.",
        "entities": [("$2,000,000", "MONEY")],
    },
]


def load_spacy_model(model_path):
    try:
        import spacy
        nlp = spacy.load(model_path)
        return nlp
    except Exception as e:
        logger.warning(f"Cannot load spaCy model: {e}")
        return None


def evaluate_regex():
    ent_patterns = {
        "MONEY": re.compile(r"\$\s*[\d,]+(?:\.\d+)?"),
        "PERCENT": re.compile(r"\d+(?:\.\d+)?\s*[%]"),
        "NOTICE_DAYS": re.compile(r"(\d+)\s*(?:day|days)(?=.*\bnotice\b)", re.IGNORECASE),
        "LIABILITY_AMOUNT": re.compile(r"(?:liability|limit|cap)\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
        "PAYMENT_DEADLINE": re.compile(r"(?:due|paid|payable|within)\s*(\d+)\s*(?:day|days)\b", re.IGNORECASE),
        "DURATION": re.compile(r"(\d+)\s*(?:month|months|year|years)\b", re.IGNORECASE),
        "LAW": re.compile(r"(?:laws?\s+of|governed\s+by)\s+[A-Za-z\s]+(?:law|regulation|act)", re.IGNORECASE),
    }

    results = []
    for example in TEST_EXAMPLES:
        text = example["text"]
        gold = set(example["entities"])
        pred = set()

        for ent_type, pattern in ent_patterns.items():
            for match in pattern.finditer(text):
                pred.add((match.group().strip(), ent_type))

        correct = pred & gold
        precision = len(correct) / len(pred) if pred else 1.0
        recall = len(correct) / len(gold) if gold else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "text": text[:60],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "gold": len(gold),
            "pred": len(pred),
            "correct": len(correct),
        })

    return results


def evaluate_spacy(nlp):
    results = []
    for example in TEST_EXAMPLES:
        text = example["text"]
        gold = set(example["entities"])

        start = time.time()
        doc = nlp(text)
        elapsed = time.time() - start

        pred = set()
        for ent in doc.ents:
            pred.add((ent.text, ent.label_))

        correct = pred & gold
        precision = len(correct) / len(pred) if pred else 1.0
        recall = len(correct) / len(gold) if gold else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "text": text[:60],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "gold": len(gold),
            "pred": len(pred),
            "correct": len(correct),
            "time_ms": round(elapsed * 1000, 1),
        })

    return results


def print_results(name, results):
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"{'='*60}")

    total_gold = 0
    total_pred = 0
    total_correct = 0
    total_time = 0

    for r in results:
        total_gold += r["gold"]
        total_pred += r["pred"]
        total_correct += r["correct"]
        if "time_ms" in r:
            total_time += r["time_ms"]
        logger.info(f"  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} | gold={r['gold']} pred={r['pred']} ok={r['correct']}")
        logger.info(f"    {r['text']}")

    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"\n  Macro Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    if total_time:
        avg_time = total_time / len(results)
        logger.info(f"  Avg inference time: {avg_time:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    regex_results = evaluate_regex()
    print_results("REGEX BASELINE", regex_results)

    if args.model_path:
        nlp = load_spacy_model(args.model_path)
        if nlp:
            spacy_results = evaluate_spacy(nlp)
            print_results("SPACY NER MODEL", spacy_results)


if __name__ == "__main__":
    main()
