"""
predict.py
==========
Inference module for the Payment Query Intent Classifier.

Loads the saved model once at startup and exposes a single
predict() function that the Streamlit app calls.

Can also be run directly from the terminal to test a query:
    python src/predict.py "why was I charged twice for the same transaction"

Returns:
    - Predicted intent label
    - Confidence score (probability of top class)
    - Top 3 intents with probabilities
    - A low-confidence flag if confidence < threshold
"""

import os
import re
import sys
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS + CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_PATH           = os.path.join("outputs", "model.pkl")
CONFIDENCE_THRESHOLD = 0.50   # below this the query gets flagged for human review

INTENT_LABELS = {
    0: "Card Fee Charge",
    1: "Fraud / Disputed Payment",
    2: "Deposit Delay",
    3: "ATM Partial Withdrawal",
    4: "Cash Withdrawal Charge",
    5: "Duplicate Transaction",
    6: "Declined Withdrawal",
    7: "Transfer Fee",
    8: "Transfer Delay",
    9: "Missing Money",
}

# Reverse map — intent name back to numeric label
INTENT_TO_LABEL = {v: k for k, v in INTENT_LABELS.items()}

# Business routing descriptions shown in the Streamlit app
INTENT_DESCRIPTIONS = {
    "Card Fee Charge"          : "Customer is querying an unexpected fee on a card transaction. Route to Card Services.",
    "Fraud / Disputed Payment" : "Customer believes a payment was made without authorisation. Route to Fraud & Disputes — high priority.",
    "Deposit Delay"            : "Customer is reporting a delayed deposit or cheque clearing. Route to Deposit Operations.",
    "ATM Partial Withdrawal"   : "Customer received less cash than requested from an ATM. Route to ATM & Cash Services.",
    "Cash Withdrawal Charge"   : "Customer was charged a fee for a cash withdrawal. Route to Card Services.",
    "Duplicate Transaction"    : "Customer has been charged twice for the same transaction. Route to Transaction Disputes.",
    "Declined Withdrawal"      : "Customer's ATM or cash withdrawal was declined. Route to Card Operations.",
    "Transfer Fee"             : "Customer is querying a fee applied to a transfer. Route to Payments & Transfers.",
    "Transfer Delay"           : "Customer's transfer has not arrived within the expected timeframe. Route to Payments & Transfers.",
    "Missing Money"            : "Customer reports funds are missing from their account. Route to Account Operations — review urgently.",
}

# ─────────────────────────────────────────────────────────────
# TEXT CLEANING
# Mirrors the logic in preprocess.py exactly
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Applies the same cleaning pipeline used during training.
    Ensures inference uses identical preprocessing to training.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# Loaded once at module import — not on every prediction call
# ─────────────────────────────────────────────────────────────

def _load_model(model_path: str = MODEL_PATH):
    """
    Loads the saved sklearn pipeline from disk.
    Called once when the module is first imported.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run src/train.py first to generate the model."
        )
    return joblib.load(model_path)


# Load model at import time
_model = _load_model()


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def _validate_query(query: str) -> tuple[bool, str]:
    """
    Basic validation before passing to the model.
    Returns (is_valid, error_message).
    """
    if not query or not query.strip():
        return False, "Query is empty. Please enter a payment-related question."

    if len(query.strip()) < 5:
        return False, "Query is too short to classify reliably. Please provide more detail."

    if len(query.strip()) > 1000:
        return False, "Query exceeds maximum length. Please shorten your message."

    return True, ""


# ─────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────

def predict(query: str) -> dict:
    """
    Classifies a payment query and returns a structured result.

    Parameters
    ----------
    query : str
        Raw customer query text.

    Returns
    -------
    dict with keys:
        success         : bool
        error           : str or None
        query_original  : str
        query_cleaned   : str
        intent          : str
        confidence      : float
        low_confidence  : bool
        routing         : str
        top_3           : list of {intent, probability}
    """
    # Validate
    is_valid, error_msg = _validate_query(query)
    if not is_valid:
        return {
            "success"       : False,
            "error"         : error_msg,
            "query_original": query,
            "query_cleaned" : "",
            "intent"        : None,
            "confidence"    : None,
            "low_confidence": None,
            "routing"       : None,
            "top_3"         : [],
        }

    # Clean
    cleaned = clean_text(query)

    # Predict
    # LinearSVC doesn't support predict_proba natively
    # so we use decision_function and convert to pseudo-probabilities
    clf = _model.named_steps["clf"]
    clf_name = type(clf).__name__

    if hasattr(clf, "predict_proba"):
        # Logistic Regression, MLP, Naive Bayes
        proba = _model.predict_proba([cleaned])[0]
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
        top_3_idx  = np.argsort(proba)[::-1][:3]
        top_3 = [
            {
                "intent"     : INTENT_LABELS[i],
                "probability": round(float(proba[i]), 4),
            }
            for i in top_3_idx
        ]

    else:
        # LinearSVC — use decision function scores
        decision = _model.decision_function([cleaned])[0]
        # Softmax to convert raw scores to pseudo-probabilities
        exp_d      = np.exp(decision - np.max(decision))
        proba_soft = exp_d / exp_d.sum()
        pred_label = int(np.argmax(proba_soft))
        confidence = float(proba_soft[pred_label])
        top_3_idx  = np.argsort(proba_soft)[::-1][:3]
        top_3 = [
            {
                "intent"     : INTENT_LABELS[i],
                "probability": round(float(proba_soft[i]), 4),
            }
            for i in top_3_idx
        ]

    intent         = INTENT_LABELS[pred_label]
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    routing        = INTENT_DESCRIPTIONS.get(intent, "Route to general customer support.")

    return {
        "success"       : True,
        "error"         : None,
        "query_original": query,
        "query_cleaned" : cleaned,
        "intent"        : intent,
        "confidence"    : round(confidence, 4),
        "low_confidence": low_confidence,
        "routing"       : routing,
        "top_3"         : top_3,
    }


# ─────────────────────────────────────────────────────────────
# CLI — run directly to test a query
# ─────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    """Pretty prints a prediction result to the terminal."""
    print("\n" + "=" * 55)
    print("  PAYMENT INTENT CLASSIFIER — PREDICTION")
    print("=" * 55)

    if not result["success"]:
        print(f"\n  Error: {result['error']}\n")
        return

    confidence_pct = result["confidence"] * 100
    flag = " ⚠  LOW CONFIDENCE — flag for human review" if result["low_confidence"] else ""

    print(f"\n  Query    : {result['query_original']}")
    print(f"  Cleaned  : {result['query_cleaned']}")
    print(f"\n  Intent   : {result['intent']}")
    print(f"  Confidence: {confidence_pct:.1f}%{flag}")
    print(f"\n  Routing  : {result['routing']}")
    print(f"\n  Top 3 predictions:")
    for i, item in enumerate(result["top_3"], 1):
        bar_len = int(item["probability"] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {i}. {item['intent']:<30} {bar} {item['probability']*100:.1f}%")

    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a payment query: ")

    result = predict(query)
    _print_result(result)