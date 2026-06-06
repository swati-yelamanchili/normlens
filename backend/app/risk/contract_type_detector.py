"""
Contract Type Detector
Detects the type of contract from full text using keyword scoring.
No ML dependency — uses regex patterns with weighted scoring.
"""

import re
from typing import Dict, Tuple


# Each entry: (pattern, weight)
CONTRACT_TYPE_SIGNALS: Dict[str, list] = {
    "SaaS": [
        (r"\bsoftware[\s\-]+as[\s\-]+a[\s\-]+service\b", 10),
        (r"\bsaas\b", 10),
        (r"\bsubscription[\s\-]+based\b", 5),
        (r"\bcloud[\s\-]+service[s]?\b", 5),
        (r"\bhosted[\s\-]+service[s]?\b", 4),
        (r"\bplatform[\s\-]+as[\s\-]+a[\s\-]+service\b", 8),
        (r"\bpaas\b", 8),
        (r"\buptime\b", 4),
        (r"\bservice[\s\-]+level[\s\-]+agreement\b", 4),
        (r"\bsla\b", 3),
        (r"\btenant\b", 3),
        (r"\bapi[\s\-]+access\b", 3),
        (r"\bdata[\s\-]+processing[\s\-]+agreement\b", 3),
    ],
    "NDA": [
        (r"\bnon[\s\-]?disclosure[\s\-]+agreement\b", 10),
        (r"\bnda\b", 8),
        (r"\bmutual[\s\-]+confidentiality\b", 7),
        (r"\bconfidentiality[\s\-]+agreement\b", 9),
        (r"\bproprietary[\s\-]+information[\s\-]+agreement\b", 7),
        (r"\bsecrecy[\s\-]+agreement\b", 6),
        (r"\bsole[\s\-]+purpose\b.*\bevaluating\b", 5),
        (r"\bpurpose\b.*\bexploring.*\bpotential.*\bbusiness\b", 4),
    ],
    "MSA": [
        (r"\bmaster[\s\-]+service[s]?[\s\-]+agreement\b", 10),
        (r"\bmaster[\s\-]+services[\s\-]+agreement\b", 10),
        (r"\bmsa\b", 6),
        (r"\bstatement[s]?[\s\-]+of[\s\-]+work\b", 5),
        (r"\bsow\b", 3),
        (r"\bwork[\s\-]+order[s]?\b", 4),
        (r"\bmaster[\s\-]+agreement\b", 6),
        (r"\bframework[\s\-]+agreement\b", 5),
    ],
    "Consulting": [
        (r"\bconsulting[\s\-]+agreement\b", 10),
        (r"\bconsultancy[\s\-]+agreement\b", 9),
        (r"\bconsultant\b", 6),
        (r"\badvisory[\s\-]+service[s]?\b", 5),
        (r"\bindependent[\s\-]+contractor\b", 5),
        (r"\bprofessional[\s\-]+advice\b", 4),
        (r"\btime[\s\-]+and[\s\-]+materials\b", 4),
    ],
    "Professional Services": [
        (r"\bprofessional[\s\-]+services[\s\-]+agreement\b", 10),
        (r"\bprofessional[\s\-]+services\b", 7),
        (r"\bservice[s]?[\s\-]+agreement\b", 5),
        (r"\bdeliverable[s]?\b", 5),
        (r"\bmilestone[s]?\b", 4),
        (r"\bproject[\s\-]+completion\b", 4),
        (r"\bscope[\s\-]+of[\s\-]+work\b", 5),
    ],
    "Employment": [
        (r"\bemployment[\s\-]+agreement\b", 10),
        (r"\bemployment[\s\-]+contract\b", 10),
        (r"\bemployee\b", 6),
        (r"\bemployer\b", 6),
        (r"\bwages?\b", 5),
        (r"\bsalary\b", 5),
        (r"\bbenefits\b", 4),
        (r"\bat[\s\-]+will[\s\-]+employment\b", 7),
        (r"\bprobationary[\s\-]+period\b", 5),
        (r"\btermination[\s\-]+for[\s\-]+cause\b", 3),
    ],
    "Vendor": [
        (r"\bvendor[\s\-]+agreement\b", 10),
        (r"\bsupplier[\s\-]+agreement\b", 10),
        (r"\bpurchase[\s\-]+order[s]?\b", 7),
        (r"\bsupply[\s\-]+agreement\b", 8),
        (r"\bprocurement\b", 5),
        (r"\bgoods?\b.*\bservices?\b", 4),
        (r"\bwarranty\b.*\bproduct[s]?\b", 4),
        (r"\bdelivery[\s\-]+terms\b", 4),
        (r"\bincoterms\b", 5),
    ],
    "License Agreement": [
        (r"\blicense[\s\-]+agreement\b", 10),
        (r"\blicensing[\s\-]+agreement\b", 9),
        (r"\bsoftware[\s\-]+licen[cs]e\b", 8),
        (r"\bperpetual[\s\-]+licen[cs]e\b", 7),
        (r"\blicensor\b", 6),
        (r"\blicensee\b", 6),
        (r"\broyalt(?:y|ies)\b", 5),
        (r"\bopen[\s\-]+source\b", 4),
        (r"\bend[\s\-]+user[\s\-]+licen[cs]e[\s\-]+agreement\b", 8),
        (r"\beula\b", 7),
    ],
    "Government Contract": [
        (r"\bgovernment[\s\-]+contract\b", 10),
        (r"\bfederal[\s\-]+acquisition[\s\-]+regulation\b", 10),
        (r"\bfar\b.*\bclause\b", 8),
        (r"\bdfars?\b", 8),
        (r"\bpublic[\s\-]+sector\b", 6),
        (r"\bgovernment[\s\-]+contractor\b", 8),
        (r"\bfederal[\s\-]+contract\b", 9),
        (r"\bsmall[\s\-]+business\b.*\badministration\b", 5),
        (r"\bgsa\b.*\bschedule\b", 7),
        (r"\bnaics\b", 6),
    ],
}

# Required clauses per contract type (for context-aware missing clause analysis)
CONTRACT_TYPE_REQUIRED_CLAUSES: Dict[str, Dict[str, Tuple[str, int]]] = {
    "SaaS": {
        "Confidentiality":      ("High", 25),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 15),
        "Intellectual Property": ("High", 25),
        "Data Ownership":       ("High", 25),
        "Security Obligations": ("High", 25),
    },
    "NDA": {
        "Confidentiality":      ("Critical", 30),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 10),
        "Intellectual Property": ("Low", 5),
        "Data Ownership":       ("Low", 5),
        "Security Obligations": ("Low", 5),
    },
    "MSA": {
        "Confidentiality":      ("High", 25),
        "Governing Law":        ("Medium", 10),
        "Termination":          ("High", 25),
        "Intellectual Property": ("High", 25),
        "Data Ownership":       ("Medium", 15),
        "Security Obligations": ("Medium", 15),
    },
    "Consulting": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 15),
        "Intellectual Property": ("High", 25),
        "Data Ownership":       ("Low", 5),
        "Security Obligations": ("Low", 5),
    },
    "Professional Services": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 15),
        "Intellectual Property": ("High", 25),
        "Data Ownership":       ("Low", 10),
        "Security Obligations": ("Low", 5),
    },
    "Employment": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Low", 5),
        "Termination":          ("High", 25),
        "Intellectual Property": ("Medium", 15),
        "Data Ownership":       ("Low", 5),
        "Security Obligations": ("Low", 5),
    },
    "Vendor": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Medium", 10),
        "Termination":          ("Medium", 15),
        "Intellectual Property": ("Medium", 15),
        "Data Ownership":       ("Medium", 15),
        "Security Obligations": ("Low", 5),
    },
    "License Agreement": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 15),
        "Intellectual Property": ("Critical", 30),
        "Data Ownership":       ("Low", 5),
        "Security Obligations": ("Low", 5),
    },
    "Government Contract": {
        "Confidentiality":      ("High", 25),
        "Governing Law":        ("High", 25),
        "Termination":          ("High", 25),
        "Intellectual Property": ("High", 25),
        "Data Ownership":       ("High", 25),
        "Security Obligations": ("High", 25),
    },
    "General Commercial Agreement": {
        "Confidentiality":      ("Medium", 15),
        "Governing Law":        ("Low", 5),
        "Termination":          ("Medium", 10),
        "Intellectual Property": ("Medium", 15),
        "Data Ownership":       ("Low", 10),
        "Security Obligations": ("Medium", 15),
    },
}


def detect_contract_type(full_text: str) -> dict:
    """
    Detect the contract type from full text using keyword scoring.

    Returns:
        {
            "contract_type": str,
            "confidence": float,   # 0.0–1.0
            "scores": dict,        # raw scores per type
            "rationale": str,
        }
    """
    text_lower = full_text.lower()
    scores: Dict[str, float] = {}

    for contract_type, signals in CONTRACT_TYPE_SIGNALS.items():
        total = 0.0
        for pattern, weight in signals:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            # Cap each signal at 3 hits to avoid single-word domination
            total += min(len(matches), 3) * weight
        scores[contract_type] = total

    if not scores or max(scores.values()) == 0:
        return {
            "contract_type": "General Commercial Agreement",
            "confidence": 0.0,
            "scores": scores,
            "rationale": "No contract-type signals detected.",
        }

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    total_score = sum(scores.values())

    # Confidence = winner's share of total signal weight
    confidence = round(best_score / total_score, 3) if total_score > 0 else 0.0

    # If best score is too low or close to second best, fall back to general
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0

    if best_score < 10:
        return {
            "contract_type": "General Commercial Agreement",
            "confidence": confidence,
            "scores": scores,
            "rationale": f"Weak signals detected (best: {best_type} = {best_score:.0f} pts). Defaulting to General Commercial Agreement.",
        }

    rationale = f"Detected as {best_type} (score: {best_score:.0f}, confidence: {confidence:.0%})"
    if second_score > 0:
        rationale += f". Runner-up: {sorted_types[1][0]} (score: {second_score:.0f})."

    return {
        "contract_type": best_type,
        "confidence": confidence,
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "rationale": rationale,
    }


def get_required_clauses_for_type(contract_type: str) -> Dict[str, Tuple[str, int]]:
    """
    Return the required clauses and their context-adjusted severity/points
    for the given contract type.
    """
    return CONTRACT_TYPE_REQUIRED_CLAUSES.get(
        contract_type,
        CONTRACT_TYPE_REQUIRED_CLAUSES["General Commercial Agreement"],
    )
