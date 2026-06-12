import json
from datetime import datetime
import numpy as np

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return str(obj)

def log_event(audio_file,
              transcript,
              response,
              similarity,
              risk_score,
              unsafe_probability,
              decision):

    log = {
    "timestamp": str(datetime.now()),
    "audio_file": str(audio_file),
    "transcript": str(transcript),
    "response": str(response),
    "similarity": float(similarity),
    "risk_score": int(risk_score),
    "unsafe_probability": float(
        unsafe_probability
    ),
    "decision": str(decision)
    }

    with open("logs.jsonl","a") as f:
        f.write(json.dumps(log, default=convert))
        f.write("\n")