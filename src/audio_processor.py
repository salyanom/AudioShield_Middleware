import os

import whisper

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(os.getenv("AUDIOSHIELD_WHISPER_MODEL", "base"))
    return _model

def transcribe_audio(audio_path):
    result = _get_model().transcribe(
        audio_path,
        language="en",
        fp16=False
    )

    return result["text"]
