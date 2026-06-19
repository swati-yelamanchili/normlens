"""
Train a gradient-boosted risk severity model.
Generates synthetic training data from existing risk rules + market norms.

Usage:
    python -m training.train_risk_scorer --epochs 200 --output-dir models/risk_scorer
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("normlens.training.risk_scorer")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "risk_scorer"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEVERITY_LEVELS = ["Low", "Moderate", "High", "Critical"]


def _load_market_norms():
    path = DATA_DIR / "market_norms.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _generate_features_and_labels():
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    market_data = _load_market_norms()
    rng = np.random.default_rng(42)

    features = []
    labels = []
    clause_types = [
        "Termination", "Payment Terms", "Liability", "Limitation of Liability",
        "Confidentiality", "Non-Compete", "Intellectual Property",
        "Indemnification", "Insurance", "Data Ownership",
        "Security Obligations",
    ]

    all_attrs = []
    for ct in clause_types:
        norms = market_data.get(ct, {})
        for attr in norms:
            if attr not in all_attrs:
                all_attrs.append(attr)
    n_attrs = len(all_attrs)

    for clause_type in clause_types:
        norms = market_data.get(clause_type, {})
        attr_names = list(norms.keys())

        for _ in range(200):
            vec = [0.0] * n_attrs
            missing = 0

            for attr_name in attr_names:
                idx = all_attrs.index(attr_name)
                info = norms.get(attr_name, {})
                vals = info.get("values", [])
                if vals and isinstance(vals[0], (int, float)) and max(vals) > 1:
                    val = float(rng.choice(vals))
                    mean_v = float(np.mean(vals))
                    std_v = float(np.std(vals)) if len(vals) > 1 else mean_v * 0.3
                    z = (val - mean_v) / std_v if std_v > 0 else 0
                    vec[idx] = z
                elif vals and max(vals) <= 1:
                    val = float(rng.choice(vals))
                    vec[idx] = val

                if rng.random() < 0.1:
                    missing += 1

            vec.append(missing)
            vec.append(len(attr_names))
            vec.append(clause_types.index(clause_type))
            vec.append(rng.integers(0, 3))

            features.append(vec)

            z_scores = [abs(v) for v in vec[:n_attrs] if isinstance(v, float)]
            max_z = max(z_scores) if z_scores else 0

            if max_z > 2.5 or missing >= 2:
                sev = "Critical"
            elif max_z > 1.5 or missing >= 1:
                sev = "High"
            elif max_z > 0.5:
                sev = "Moderate"
            else:
                sev = "Low"

            labels.append(SEVERITY_LEVELS.index(sev))

    features = np.array(features, dtype=np.float64)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    logger.info(f"Generated {len(features)} samples with {features.shape[1]} features (n_attrs={n_attrs})")
    severity_dist = {SEVERITY_LEVELS[i]: labels.count(i) for i in range(4)}
    logger.info(f"Severity distribution: {severity_dist}")

    return features, labels, scaler, clause_types


def train_gb(args, X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = GradientBoostingClassifier(
        n_estimators=args.epochs,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=SEVERITY_LEVELS)}")

    importances = model.feature_importances_
    logger.info(f"Feature importances: {importances}")

    output_dir = args.output_dir or str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "risk_scorer.pkl"))
    logger.info(f"Model saved to {output_dir}/risk_scorer.pkl")

    metrics = {
        "accuracy": float(acc),
        "n_estimators": args.epochs,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_classes": 4,
        "class_names": SEVERITY_LEVELS,
    }
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train risk severity scorer")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    os.makedirs(args.output_dir or str(OUTPUT_DIR), exist_ok=True)
    X, y, scaler, clause_types = _generate_features_and_labels()
    train_gb(args, X, y)


if __name__ == "__main__":
    main()
