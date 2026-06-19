"""
Train a TF-IDF + Logistic Regression model for contract type classification.
Bootstraps training data from existing regex-based contract type detector.

Usage:
    python -m training.train_contract_type --output-dir models/contract_type
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.training.contract_type")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "contract_type"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONTRACT_TYPES = [
    "SaaS Agreement",
    "NDA",
    "MSA",
    "Consulting Agreement",
    "Professional Services Agreement",
    "Employment Agreement",
    "Vendor Agreement",
    "License Agreement",
    "Government Contract",
]

TYPE_KEYWORDS = {
    "SaaS Agreement": [
        r"\bsaas\b", r"\bsoftware\s+as\s+a\s+service\b", r"\bsubscription\s+(?:fee|service|agreement)\b",
        r"\bcloud\s+(?:service|computing)\b", r"\bplatform\s+as\s+a\s+service\b",
        r"\bpaas\b", r"\bservice\s+level\s+agreement\b", r"\bsla\b",
        r"\buser\s+subscription\b", r"\bhosted\s+service\b",
    ],
    "NDA": [
        r"\bnon.?disclosure\b", r"\bconfidentiality\s+agreement\b",
        r"\bnda\b", r"\bconfidential\s+information\b",
        r"\bnon.?circumvention\b", r"\bmutual\s+non.?disclosure\b",
    ],
    "MSA": [
        r"\bmaster\s+services?\s+agreement\b", r"\bmsa\b",
        r"\bstatement\s+of\s+work\b", r"\bsow\b",
        r"\bwork\s+order\b", r"\bservice\s+order\b",
        r"\bgeneral\s+terms\s+and\s+conditions\b",
    ],
    "Consulting Agreement": [
        r"\bconsulting\s+(?:agreement|services)\b", r"\bconsultant\b",
        r"\badvisory\s+(?:services|agreement)\b", r"\bconsultancy\b",
    ],
    "Professional Services Agreement": [
        r"\bprofessional\s+services\b", r"\bpsa\b",
        r"\bstatement\s+of\s+work\b", r"\bengagement\s+letter\b",
        r"\bservices?\s+agreement\b",
    ],
    "Employment Agreement": [
        r"\bemployment\s+(?:agreement|contract)\b", r"\bemployee\b",
        r"\bemployer\b", r"\bcompensation\b", r"\bsalary\b",
        r"\btermination\s+of\s+employment\b", r"\bat.?will\s+employment\b",
    ],
    "Vendor Agreement": [
        r"\bvendor\b", r"\bsupplier\s+(?:agreement|contract)\b",
        r"\bprocurement\b", r"\bgoods\s+and\s+services\b",
        r"\bpurchase\s+order\b", r"\bsupply\s+(?:agreement|contract)\b",
    ],
    "License Agreement": [
        r"\blicense\s+(?:agreement|contract)\b", r"\blicensee\b",
        r"\blicensor\b", r"\blicensing\b", r"\broyalt\w+\b",
        r"\bintellectual\s+property\s+license\b", r"\bsublicense\b",
    ],
    "Government Contract": [
        r"\bgovernment\s+(?:contract|agreement)\b", r"\bfederal\s+(?:acquisition|contract)\b",
        r"\bfar\b", r"\bdefense\s+contract\b",
        r"\bpublic\s+procurement\b", r"\bstate\s+contract\b",
    ],
}


def _generate_training_data():
    rng = __import__("numpy").random.default_rng(42)
    examples = []

    templates = {
        "SaaS Agreement": [
            "This Software as a Service Agreement (the 'Agreement') is entered into as of {date} between {co1} and {co2}.",
            "The subscription fee shall be {amount} per month for {n} licensed users.",
            "Service Level Agreement: Provider shall maintain {pct}% uptime availability.",
            "The platform as a service solution includes hosting, maintenance, and support.",
            "Customer shall pay the annual subscription fee in advance.",
        ],
        "NDA": [
            "This Non-Disclosure Agreement is entered into as of {date} between {co1} and {co2}.",
            "Confidential information shall not be disclosed to any third party.",
            "This NDA shall survive for a period of {n} years from the date of disclosure.",
            "The receiving party agrees to hold all proprietary information in confidence.",
        ],
        "MSA": [
            "This Master Services Agreement is entered into as of {date} between {co1} and {co2}.",
            "Statements of Work shall be executed hereunder for specific projects.",
            "The General Terms and Conditions set forth herein govern all services provided.",
            "Each Work Order shall reference this MSA and specify the scope of work.",
        ],
        "Consulting Agreement": [
            "This Consulting Agreement is entered into as of {date} between {co1} (Client) and {co2} (Consultant).",
            "The Consultant shall provide advisory services in the field of {field}.",
            "Consultant shall invoice Client monthly for services rendered.",
            "The consulting services shall commence on {date} and continue until completion.",
        ],
        "Professional Services Agreement": [
            "This Professional Services Agreement is entered into as of {date}.",
            "Provider shall perform the services described in each Statement of Work.",
            "Engagement letter attached hereto specifies the scope and fees.",
        ],
        "Employment Agreement": [
            "This Employment Agreement is entered into as of {date} between {co1} (Employer) and {name} (Employee).",
            "Employee shall receive an annual salary of {amount}, payable in equal installments.",
            "Upon termination of employment, Employee shall return all company property.",
        ],
        "Vendor Agreement": [
            "This Vendor Agreement is entered into as of {date} between {co1} (Buyer) and {co2} (Supplier).",
            "Supplier shall provide the goods described in each Purchase Order.",
            "All goods shall be delivered FOB {location} within {n} days of order.",
        ],
        "License Agreement": [
            "This License Agreement is entered into as of {date} between {co1} (Licensor) and {co2} (Licensee).",
            "Licensor grants Licensee a non-exclusive, non-transferable license to use the software.",
            "Licensee shall pay royalties of {pct}% of net sales.",
        ],
        "Government Contract": [
            "This Government Contract is entered into as of {date} by and between the {agency} and {co1}.",
            "This contract is subject to FAR Part {n} regulations.",
            "The Contractor shall comply with all applicable federal acquisition regulations.",
        ],
    }

    for ctype, phrases in templates.items():
        type_idx = CONTRACT_TYPES.index(ctype)
        for phrase in phrases:
            for _ in range(15):
                text = phrase.format(
                    date="June 1, 2025",
                    co1="Acme Corporation",
                    co2="Beta Solutions Inc",
                    name="John Doe",
                    amount=f"${rng.choice([50000, 100000, 250000, 500000, 1000000]):,}",
                    n=rng.choice([30, 60, 90, 180]),
                    pct=rng.choice([95, 99, 99.5, 99.9]),
                    field=rng.choice(["technology", "management", "finance", "strategy"]),
                    location=rng.choice(["New York", "Chicago", "Atlanta", "Dallas"]),
                    agency=rng.choice(["Department of Defense", "State of California", "Federal Aviation Administration"]),
                )
                examples.append((text, type_idx, ctype))

    neg_texts = [
        "This document is a purchase order for office supplies.",
        "Memo regarding quarterly financial results.",
        "Meeting minutes for the board of directors.",
        "Internal company policy on vacation time.",
    ]
    for text in neg_texts:
        for ctype in CONTRACT_TYPES:
            examples.append((text, CONTRACT_TYPES.index(ctype), ctype))

    logger.info(f"Generated {len(examples)} training examples")
    return examples


def train_tfidf_lr(args, examples):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    texts = [e[0] for e in examples]
    labels = [e[1] for e in examples]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        C=1.0,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=CONTRACT_TYPES)}")

    output_dir = args.output_dir or str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(output_dir, "contract_type_model.pkl"))

    with open(os.path.join(output_dir, "contract_types.json"), "w") as f:
        json.dump({i: t for i, t in enumerate(CONTRACT_TYPES)}, f, indent=2)

    metrics = {
        "accuracy": float(acc),
        "vocab_size": len(vectorizer.get_feature_names_out()),
        "model_type": "TF-IDF + LogisticRegression",
        "classes": CONTRACT_TYPES,
    }
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {output_dir}/")
    return vectorizer, model


def main():
    parser = argparse.ArgumentParser(description="Train contract type classifier")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    os.makedirs(args.output_dir or str(OUTPUT_DIR), exist_ok=True)
    examples = _generate_training_data()
    train_tfidf_lr(args, examples)


if __name__ == "__main__":
    main()
