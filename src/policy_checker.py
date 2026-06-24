"""
policy_checker.py

Loads the fine-tuned DistilBERT classifier from models/risk_classifier/
and scores any response text.  No hardcoded keyword patterns — the model
learns what's unsafe from your dataset.

Returns:
    prediction  (int)  0 = safe, 1 = unsafe
    details     (dict) raw probabilities + model confidence
"""

from pathlib import Path
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_dir":         "models/risk_classifier",
    "max_length":        128,
    "unsafe_threshold":  0.5,   # tune up to reduce false positives
}
# ─────────────────────────────────────────────────────────────────────────────

_tokenizer = None
_model     = None
_device    = None


def _load():
    global _tokenizer, _model, _device

    model_path = Path(CONFIG["model_dir"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run train_risk_model.py first."
        )

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    _model     = DistilBertForSequenceClassification.from_pretrained(model_path).to(_device)
    _model.eval()


def check_policy(response: str, threshold: float | None = None):
    """
    Parameters
    ----------
    response : str
        The LLM-generated text to evaluate.

    Returns
    -------
    prediction : int
        0 = safe, 1 = unsafe
    details : dict
        {
            "safe_prob":   float,
            "unsafe_prob": float,
            "threshold":   float,
            "model_dir":   str,
        }
    """
    global _tokenizer, _model, _device

    if _model is None:
        _load()

    inputs = _tokenizer(
        response,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"],
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0]

    safe_prob   = float(probs[0])
    unsafe_prob = float(probs[1])

    effective_threshold = CONFIG["unsafe_threshold"] if threshold is None else threshold
    prediction = 1 if unsafe_prob >= effective_threshold else 0

    details = {
        "safe_prob":   round(safe_prob,   4),
        "unsafe_prob": round(unsafe_prob, 4),
        "threshold":   effective_threshold,
        "model_dir":   CONFIG["model_dir"],
    }

    return prediction, details
