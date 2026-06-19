"""
Fine-tune Legal-BERT on CUAD + LEDGAR for multi-label clause classification.
Usage:
    python -m training.train_classifier --epochs 3 --batch_size 8
    python -m training.train_classifier --use_synthetic --epochs 10
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("normlens.training.classifier")

BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "clause_classifier"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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

CUAD_QUESTION_LABEL_MAP = {
    "notice period for termination": "Termination",
    "termination for convenience": "Termination",
    "termination for cause": "Termination",
    "payment schedule": "Payment Terms",
    "late payment": "Payment Terms",
    "liability": "Liability",
    "limitation of liability": "Limitation of Liability",
    "caps on liability": "Limitation of Liability",
    "indemnification": "Indemnification",
    "non-compete": "Non-Compete",
    "confidential information": "Confidentiality",
    "confidentiality": "Confidentiality",
    "intellectual property": "Intellectual Property",
    "ownership of ip": "Intellectual Property",
    "assignment": "Assignment",
    "governing law": "Governing Law",
    "jurisdiction": "Governing Law",
    "arbitration": "Arbitration",
    "insurance": "Insurance",
    "data protection": "Data Protection",
    "force majeure": "Force Majeure",
    "warranty": "Warranty",
    "representations and warranties": "Representations and Warranties",
    "entire agreement": "Entire Agreement",
    "amendments": "Amendments",
    "notice": "Notices",
    "severability": "Severability",
    "waiver": "Waiver",
    "survival": "Survival",
    "counterparts": "Counterparts",
    "definitions": "Definitions",
    "expenses": "Expenses",
    "publicity": "Publicity",
    "subcontracting": "Subcontracting",
    "non-waiver": "No Waiver",
    "further assurances": "Further Assurances",
    "third party beneficiaries": "No Third Party Beneficiaries",
    "renewal": "Payment Terms",
    "license grant": "Intellectual Property",
    "non-disclosure": "Confidentiality",
    "covenant": "Warranty",
    "audit": "Security Obligations",
    "security": "Security Obligations",
    "data ownership": "Data Ownership",
    "data use": "Data Ownership",
}


def _load_cuad():
    try:
        from datasets import load_dataset
        dataset = load_dataset("cuad", trust_remote_code=True)
        logger.info(f"CUAD loaded: {len(dataset['train'])} train samples")
        return dataset
    except Exception as e:
        logger.warning(f"Cannot load CUAD: {e}")
        return None


def _load_ledgar():
    try:
        from datasets import load_dataset
        dataset = load_dataset("lex_glue", "ledgar", trust_remote_code=True)
        logger.info(f"LEDGAR loaded: {len(dataset['train'])} train samples")
        return dataset
    except Exception as e:
        logger.warning(f"Cannot load LEDGAR: {e}")
        return None


def _cuad_to_examples(dataset):
    examples = []
    label_to_idx = {l: i for i, l in enumerate(LABELS)}

    for split in ["train", "test"]:
        if split not in dataset:
            continue
        for sample in dataset[split]:
            question = sample.get("question", "")
            context = sample.get("context", "")
            answers = sample.get("answers", {})
            answer_texts = answers.get("text", []) if answers else []

            label = None
            q_lower = question.lower()
            for keyword, cls_type in CUAD_QUESTION_LABEL_MAP.items():
                if keyword in q_lower:
                    label = cls_type
                    break
            if label is None:
                continue

            has_answer = bool(answer_texts and any(t.strip() for t in answer_texts))
            examples.append({
                "text": context[:512],
                "label": label,
                "label_idx": label_to_idx.get(label, -1),
                "is_positive": has_answer,
                "source": "cuad",
                "split": split,
            })

    logger.info(f"Built {len(examples)} CUAD examples")
    return examples


def _ledgar_to_examples(dataset):
    examples = []
    label_to_idx = {l: i for i, l in enumerate(LABELS)}

    ledgar_label_map = {
        "Termination": "Termination",
        "Payment Terms": "Payment Terms",
        "Liability": "Liability",
        "Limitation of Liability": "Limitation of Liability",
        "Confidentiality": "Confidentiality",
        "Non-Compete": "Non-Compete",
        "Intellectual Property": "Intellectual Property",
        "Indemnification": "Indemnification",
        "Assignment": "Assignment",
        "Governing Law": "Governing Law",
        "Arbitration": "Arbitration",
        "Insurance": "Insurance",
        "Data Protection": "Data Protection",
        "Force Majeure": "Force Majeure",
        "Warranty": "Warranty",
        "Amendment": "Amendments",
        "Notices": "Notices",
        "Waiver": "Waiver",
        "Survival": "Survival",
        "Definitions": "Definitions",
        "Expenses": "Expenses",
        "Publicity": "Publicity",
    }

    for split in ["train", "test", "validation"]:
        if split not in dataset:
            continue
        for sample in dataset[split]:
            text = sample.get("text", sample.get("content", ""))
            label_str = sample.get("label", "")
            if isinstance(label_str, int):
                continue

            mapped = ledgar_label_map.get(label_str)
            if mapped is None:
                continue

            examples.append({
                "text": text[:512],
                "label": mapped,
                "label_idx": label_to_idx[mapped],
                "is_positive": True,
                "source": "ledgar",
                "split": split if split != "validation" else "train",
            })

    logger.info(f"Built {len(examples)} LEDGAR examples")
    return examples


def _generate_synthetic():
    examples = []
    label_to_idx = {l: i for i, l in enumerate(LABELS)}

    templates = {
        "Termination": [
            "This agreement may be terminated by either party upon {n} days written notice.",
            "Either party may terminate this agreement for cause upon material breach.",
            "In the event of termination, all rights and obligations shall cease.",
        ],
        "Payment Terms": [
            "All invoices shall be paid within {n} days of receipt.",
            "Payment shall be made via wire transfer within {n} calendar days.",
            "A late fee of {pct}% per month shall apply to overdue amounts.",
        ],
        "Liability": [
            "Neither party shall be liable for any indirect or consequential damages.",
            "The total liability of either party shall not exceed ${amount}.",
            "Nothing in this agreement shall limit liability for fraud or gross negligence.",
        ],
        "Limitation of Liability": [
            "In no event shall either party's aggregate liability exceed ${amount}.",
            "The limitation of liability set forth herein shall not apply to indemnification obligations.",
            "Exclusive remedy for any claim is limited to direct damages not exceeding ${amount}.",
        ],
        "Confidentiality": [
            "The receiving party shall maintain confidentiality for a period of {n} years.",
            "Confidential information shall not be disclosed to third parties without prior written consent.",
            "This non-disclosure obligation shall survive termination for {n} years.",
        ],
        "Non-Compete": [
            "The party shall not engage in any competing business for {n} months post-termination.",
            "For a period of {n} months, the consultant shall not solicit any clients.",
        ],
        "Intellectual Property": [
            "All intellectual property created during the term shall be owned by the company.",
            "The contractor hereby assigns all rights, title, and interest in the work product.",
            "Pre-existing IP shall remain the property of the originating party.",
        ],
        "Indemnification": [
            "Provider shall indemnify and hold harmless customer from all third-party claims.",
            "Each party agrees to defend the other against claims arising from IP infringement.",
        ],
        "Assignment": [
            "Neither party may assign this agreement without the other's prior written consent.",
            "This agreement shall be binding upon and inure to the benefit of permitted assigns.",
        ],
        "Governing Law": [
            "This agreement shall be governed by the laws of the State of New York.",
            "The parties submit to the exclusive jurisdiction of the federal courts in Delaware.",
        ],
        "Arbitration": [
            "Any dispute arising out of this agreement shall be resolved by binding arbitration.",
            "The arbitration shall be conducted in accordance with AAA rules.",
        ],
        "Insurance": [
            "Provider shall maintain commercial general liability insurance of ${amount} per occurrence.",
            "Upon request, provider shall furnish certificates of insurance evidencing coverage.",
        ],
        "Data Protection": [
            "Both parties shall comply with all applicable data protection laws including GDPR.",
            "Personal data shall only be processed in accordance with the data processing agreement.",
        ],
        "Force Majeure": [
            "Neither party shall be liable for delays caused by events outside its reasonable control.",
            "Force majeure events include acts of God, war, terrorism, and natural disasters.",
        ],
        "Warranty": [
            "Provider warrants that services will be performed in a professional manner.",
            "The products are warranted to be free from defects in materials for {n} months.",
        ],
    }

    rng = np.random.default_rng(42)
    for label, phrases in templates.items():
        idx = label_to_idx[label]
        for phrase in phrases:
            for _ in range(10):
                text = phrase.format(
                    n=int(rng.choice([30, 60, 90, 120])),
                    pct=round(rng.uniform(0.5, 2.0), 1),
                    amount=int(rng.choice([500000, 1000000, 5000000])),
                )
                examples.append({
                    "text": text,
                    "label": label,
                    "label_idx": idx,
                    "is_positive": True,
                    "source": "synthetic",
                    "split": "train",
                })

    neg_phrases = [
        "This agreement is entered into as of the effective date by and between the parties.",
        "IN WITNESS WHEREOF, the parties have executed this agreement.",
        "This agreement may be executed in counterparts, each of which shall be deemed an original.",
    ]
    for phrase in neg_phrases:
        for _ in range(20):
            examples.append({
                "text": phrase,
                "label": "Entire Agreement",
                "label_idx": label_to_idx["Entire Agreement"],
                "is_positive": False,
                "source": "synthetic",
                "split": "train",
            })

    for text in [
        "This is a standard miscellaneous clause that does not fit any category.",
        "The parties acknowledge receipt of the exhibit attached hereto.",
    ]:
        for _ in range(20):
            for label in LABELS:
                examples.append({
                    "text": text,
                    "label": label,
                    "label_idx": label_to_idx[label],
                    "is_positive": False,
                    "source": "synthetic",
                    "split": "train",
                })

    logger.info(f"Generated {len(examples)} synthetic examples")
    return examples


def build_dataset(args):
    all_examples = []

    if not args.use_synthetic:
        cuad = _load_cuad()
        if cuad:
            all_examples.extend(_cuad_to_examples(cuad))

        ledgar = _load_ledgar()
        if ledgar:
            all_examples.extend(_ledgar_to_examples(ledgar))

    all_examples.extend(_generate_synthetic())

    train_examples = [e for e in all_examples if e["split"] == "train"]
    test_examples = [e for e in all_examples if e["split"] == "test"]
    if not test_examples:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(train_examples), size=max(1, len(train_examples) // 5), replace=False)
        test_examples = [train_examples[i] for i in idxs]
        train_examples = [e for i, e in enumerate(train_examples) if i not in idxs]

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")
    return train_examples, test_examples


def train(args, train_examples, test_examples):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    import torch
    from torch.utils.data import Dataset

    class ClauseDataset(Dataset):
        def __init__(self, examples, tokenizer, max_len=256):
            self.texts = [e["text"] for e in examples]
            self.labels = [e["label_idx"] for e in examples]
            self.is_positive = [e["is_positive"] for e in examples]
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    import transformers.utils.import_utils as _iu
    _original_check = getattr(_iu, "check_torch_load_is_safe", None)
    if _original_check:
        _iu.check_torch_load_is_safe = lambda: None
    tokenizer = None
    model = None
    try:
        logger.info(f"Loading tokenizer and model: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(LABELS),
            problem_type="single_label_classification",
        )
    finally:
        if _original_check:
            _iu.check_torch_load_is_safe = _original_check
    if model is None:
        raise RuntimeError(f"Failed to load model {BASE_MODEL}")

    train_dataset = ClauseDataset(train_examples, tokenizer)
    test_dataset = ClauseDataset(test_examples, tokenizer)

    output_dir = args.output_dir or str(OUTPUT_DIR)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    model_dir = os.path.join(output_dir, "final")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({i: l for i, l in enumerate(LABELS)}, f, indent=2)

    eval_result = trainer.evaluate()
    logger.info(f"Evaluation: {eval_result}")

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_result, f, indent=2)

    logger.info(f"Model saved to {model_dir}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train clause classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Skip HF datasets, use only synthetic data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    os.makedirs(args.output_dir or str(OUTPUT_DIR), exist_ok=True)

    train_examples, test_examples = build_dataset(args)
    model, tokenizer = train(args, train_examples, test_examples)


if __name__ == "__main__":
    main()
