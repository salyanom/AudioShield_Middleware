"""Append-only JSONL security audit logging."""

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from config import settings


_write_lock = Lock()


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def log_security_event(event: dict, path: str | None = None) -> None:
    destination = Path(path or settings.log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    line = json.dumps(record, default=_json_default, ensure_ascii=False)
    with _write_lock:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def log_event(audio_file, transcript, response, similarity, risk_score,
              unsafe_probability, decision):
    """Compatibility wrapper for older callers."""
    log_security_event({
        "audio_file": str(audio_file),
        "transcript": str(transcript),
        "raw_response": str(response),
        "similarity": float(similarity),
        "output_policy": {
            "prediction": int(risk_score),
            "unsafe_probability": float(unsafe_probability),
        },
        "decision": str(decision),
    })
