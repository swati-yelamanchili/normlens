"""
Train a spaCy NER model for attribute extraction using noisy labels from regex.
Bootstraps training data from existing regex patterns in the attribute extractor.

Usage:
    python -m training.train_ner --output-dir models/attribute_ner --epochs 30
    python -m training.train_ner --output-dir models/attribute_ner --force-cpu
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.training.ner")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "attribute_ner"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ENTITY_TYPES = {
    "MONEY": {"patterns": [r"\$\s*[\d,]+(?:\.\d+)?", r"[\d,]+(?:\.\d+)?\s*(?:USD|EUR|GBP)"]},
    "DATE": {"patterns": [r"\b\d+\s*(?:day|days|month|months|year|years)\b"]},
    "PERCENT": {"patterns": [r"\d+(?:\.\d+)?\s*%", r"\d+(?:\.\d+)?\s*percent"]},
    "NOTICE_DAYS": {"patterns": [r"(?:\bnotice\b).{0,50}?\b(\d+)\s*(?:day|days)\b"]},
    "LIABILITY_AMOUNT": {"patterns": [r"(?:liability|limit|cap|capped)\s*[:\s]*\$?\s*([\d,]+(?:\.\d+)?)"]},
    "PAYMENT_DEADLINE": {"patterns": [r"(?:due|paid|payable|within)\s*(\d+)\s*(?:day|days)\b"]},
    "DURATION": {"patterns": [r"(\d+)\s*(?:month|months|year|years)\b"]},
    "ORG": {"patterns": []},
    "LAW": {"patterns": [
        r"(?:laws?\s+of|governed\s+by|under)\s+[A-Z][A-Za-z\s]+(?:law|regulation|act)",
        r"GDPR|CCPA|HIPAA|SOX|PCI\s*DSS",
    ]},
}

BOILERPLATE = [
    "This agreement is entered into as of the effective date",
    "IN WITNESS WHEREOF, the parties have executed this agreement",
    "This agreement may be executed in counterparts",
    "The parties acknowledge receipt of the exhibit",
    "This is a standard miscellaneous clause",
]


def _extract_entities_with_regex(text, entity_type):
    config = ENTITY_TYPES.get(entity_type)
    if not config:
        return []
    found = []
    for pattern in config["patterns"]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "type": entity_type,
            })
    return found


def _generate_training_sentences():
    sentences = []

    templates_notice = [
        "This agreement may be terminated upon {days} days written notice.",
        "Either party may terminate with {days} days prior notice.",
        "A notice period of {days} days shall be required for termination.",
        "The terminating party shall provide {days} calendar days notice.",
    ]
    for t in templates_notice:
        for d in [14, 30, 45, 60, 90, 120]:
            sentences.append(("Termination", t.format(days=d)))

    templates_liability = [
        "The total liability of either party shall not exceed ${amount}.",
        "Each party's aggregate liability is capped at ${amount}.",
        "Liability shall be limited to ${amount}.",
        "In no event shall liability exceed ${amount}.",
    ]
    for t in templates_liability:
        for a in [100000, 500000, 1000000, 5000000, 10000000]:
            sentences.append(("Liability", t.format(amount=f"{a:,}")))

    templates_payment = [
        "All invoices shall be paid within {days} days of receipt.",
        "Payment shall be due within {days} calendar days.",
        "The buyer shall pay within {days} days of invoice date.",
    ]
    for t in templates_payment:
        for d in [15, 30, 45, 60, 90]:
            sentences.append(("Payment Terms", t.format(days=d)))

    templates_duration = [
        "This agreement shall remain in effect for {n} months.",
        "The initial term shall be {n} months from the effective date.",
        "The non-compete obligations shall survive for {n} months.",
        "Confidentiality obligations shall continue for {n} years.",
    ]
    for t in templates_duration:
        for n in [6, 12, 24, 36, 60]:
            sentences.append(("Duration", t.format(n=n)))

    templates_penalty = [
        "A late fee of {pct}% per month shall apply to overdue amounts.",
        "Late payments shall accrue interest at {pct}% per annum.",
        "The penalty for late payment is {pct} percent.",
    ]
    for t in templates_penalty:
        for p in [0.5, 1.0, 1.5, 2.0]:
            sentences.append(("Payment Terms", t.format(pct=p)))

    for b in BOILERPLATE:
        sentences.append(("Other", b))

    logger.info(f"Generated {len(sentences)} training sentences")
    return sentences


def train_spacy_ner(args):
    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example
    import random

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")

    for entity_type in ENTITY_TYPES:
        if entity_type not in ner.labels:
            ner.add_label(entity_type)
            logger.info(f"Added label: {entity_type}")

    sentences = _generate_training_sentences()

    examples = []
    for clause_type, text in sentences:
        doc = nlp.make_doc(text)
        entities = []
        for ent_type in ENTITY_TYPES:
            for match in re.finditer(ENTITY_TYPES[ent_type]["patterns"][0], text, re.IGNORECASE) if ENTITY_TYPES[ent_type]["patterns"] else []:
                _ = 0
            found = _extract_entities_with_regex(text, ent_type)
            for f in found:
                span = doc.char_span(f["start"], f["end"], label=f["type"])
                if span is not None:
                    entities.append(span)

        orgs = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Ltd|Inc|Corp|LLC|LLP|LP|PLC|GmbH|AG)\b', text)
        for org in orgs:
            for m in re.finditer(re.escape(org), text):
                span = doc.char_span(m.start(), m.end(), label="ORG")
                if span is not None:
                    entities.append(span)

        laws = re.findall(r'(?:GDPR|CCPA|HIPAA|SOX|PCI\s*DSS)', text, re.IGNORECASE)
        for law in laws:
            for m in re.finditer(re.escape(law), text, re.IGNORECASE):
                span = doc.char_span(m.start(), m.end(), label="LAW")
                if span is not None:
                    entities.append(span)

        if entities:
            filtered = []
            entities.sort(key=lambda s: (s.start, -s.end))
            for span in entities:
                if not any(
                    s.start <= span.start and s.end >= span.end
                    for s in filtered
                ):
                    filtered.append(span)
            if filtered:
                doc.set_ents(filtered)
                example = Example(doc, doc)
                examples.append(example)

    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    eval_examples = examples[split:]

    logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)} examples")

    other_pipe_names = [name for name, pipe in nlp.pipeline if name != "ner"]
    with nlp.select_pipes(enable=["ner"]):
        optimizer = nlp.resume_training()
        for epoch in range(args.epochs):
            random.shuffle(train_examples)
            losses = {}
            for batch in spacy.util.minibatch(train_examples, size=8):
                nlp.update(batch, sgd=optimizer, losses=losses, drop=0.3)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {losses.get('ner', 0):.4f}")

            if (epoch + 1) % 10 == 0:
                correct = 0
                total = 0
                for ex in eval_examples:
                    pred_doc = nlp(ex.text)
                    pred_ents = {(e.text, e.label_) for e in pred_doc.ents}
                    gold_ents = {(e.text, e.label_) for e in ex.reference.ents}
                    correct += len(pred_ents & gold_ents)
                    total += len(gold_ents)
                if total > 0:
                    logger.info(f"  Eval precision: {correct}/{total} = {correct/total:.3f}")

    output_path = args.output_dir or str(OUTPUT_DIR)
    nlp.to_disk(output_path)
    logger.info(f"Model saved to {output_path}")

    ner_labels = list(ner.labels)
    with open(os.path.join(output_path, "entity_types.json"), "w") as f:
        json.dump(ner_labels, f, indent=2)
    logger.info(f"Entity types: {ner_labels}")

    return nlp


def main():
    parser = argparse.ArgumentParser(description="Train spaCy NER for attribute extraction")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.force_cpu:
        import spacy
        spacy.require_gpu()

    os.makedirs(args.output_dir or str(OUTPUT_DIR), exist_ok=True)
    train_spacy_ner(args)


if __name__ == "__main__":
    main()
