import os
from config import settings

_model = None
_engine = None


def _get_model():
    global _model, _engine
    if _model is not None:
        return _model, _engine

    engine = settings.whisper_engine.lower()
    model_name = os.getenv("AUDIOSHIELD_WHISPER_MODEL", "base")

    if engine == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use float16 for CUDA, int8 for CPU to minimize latency
            compute_type = "float16" if device == "cuda" else "int8"
            _model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _engine = "faster-whisper"
            print(f"[audio_processor] Loaded faster-whisper on {device} ({compute_type})")
            return _model, _engine
        except ImportError:
            print("[audio_processor] faster-whisper package not found. Falling back to openai-whisper.")
            engine = "openai-whisper"

    if engine == "openai-whisper":
        import whisper
        _model = whisper.load_model(model_name)
        _engine = "openai-whisper"
        print(f"[audio_processor] Loaded openai-whisper base model")
        return _model, _engine

    raise ValueError(f"Unknown whisper engine config: {engine}")


def get_whisper_model(engine_name: str | None = None):
    model, engine = _get_model()
    return model, engine


def transcribe_audio(audio_path):
    model, engine = _get_model()
    if engine == "faster-whisper":
        segments, info = model.transcribe(
            audio_path,
            beam_size=settings.whisper_beam_size,
            language="en"
        )
        return "".join(segment.text for segment in segments)
    else:
        # Optimize decoding settings for openai-whisper to improve latency
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            beam_size=settings.whisper_beam_size,
            best_of=settings.whisper_best_of,
            temperature=0.0
        )
        return result["text"]
