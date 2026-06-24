"""
audio_embedder.py

Generates audio embeddings using CLAP (Contrastive Language-Audio Pretraining).
Produces a 512-dimensional vector per audio file that captures the semantic
content of the audio signal directly — not the transcript.

This is the key difference from transcript-level analysis:
two audio files that transcribe identically can have very different
CLAP embeddings if their acoustic content differs (pitch shift, echo,
adversarial perturbation, hidden injection mixed in at low amplitude).

Model: laion/clap-htsat-unfused
  - 512-dim audio embeddings
  - Trained on 630K audio-text pairs
  - Supports WAV and MP3 via librosa resampling to 48kHz

Install:
    pip install transformers librosa soundfile

Usage:
    from audio_embedder import get_audio_embedding, audio_text_similarity
    emb = get_audio_embedding("data/benign/test.wav")       # (512,) numpy array
    sim = audio_text_similarity("data/benign/test.wav", "tell me about networking")
"""

import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name":   "laion/clap-htsat-unfused",
    "sample_rate":  48000,   # CLAP requires 48kHz
    "max_seconds":  30,      # truncate long audio to keep memory reasonable
}
# ─────────────────────────────────────────────────────────────────────────────

_model     = None
_processor = None
_device    = None


def _load():
    global _model, _processor, _device
    import torch
    from transformers import ClapModel, ClapProcessor

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = ClapProcessor.from_pretrained(CONFIG["model_name"])
    _model     = ClapModel.from_pretrained(CONFIG["model_name"]).to(_device)
    _model.eval()
    print(f"[audio_embedder] CLAP loaded on {_device}")


def _load_audio(path: str) -> np.ndarray:
    """Load any audio file and resample to CLAP's required 48kHz."""
    import librosa
    max_samples = CONFIG["sample_rate"] * CONFIG["max_seconds"]
    audio, _ = librosa.load(
        path,
        sr=CONFIG["sample_rate"],
        mono=True,
        duration=CONFIG["max_seconds"],
    )
    return audio.astype(np.float32)


def get_audio_embedding(audio_path: str) -> np.ndarray:
    """
    Produce a normalised 512-dim CLAP embedding for an audio file.

    Parameters
    ----------
    audio_path : str   Path to WAV, MP3, FLAC, or any librosa-readable file.

    Returns
    -------
    np.ndarray   Shape (512,), L2-normalised.
    """
    import torch

    if _model is None:
        _load()

    audio = _load_audio(audio_path)

    inputs = _processor(
        audios=audio,
        return_tensors="pt",
        sampling_rate=CONFIG["sample_rate"],
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = _model.get_audio_features(**inputs)   # (1, 512)

    emb = emb[0].cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8              # L2 normalise
    return emb


def get_text_embedding(text: str) -> np.ndarray:
    """
    Produce a normalised 512-dim CLAP embedding for a text string.
    Uses CLAP's text encoder — same space as audio embeddings.

    Parameters
    ----------
    text : str   Any text string (transcript, response, description).

    Returns
    -------
    np.ndarray   Shape (512,), L2-normalised.
    """
    import torch

    if _model is None:
        _load()

    inputs = _processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = _model.get_text_features(**inputs)    # (1, 512)

    emb = emb[0].cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


def audio_text_similarity(audio_path: str, text: str) -> float:
    """
    Compute cosine similarity between an audio file and a text string
    in CLAP's shared embedding space.

    This measures whether the audio content is semantically consistent
    with the text — directly, without going through transcription.

    Returns float in [-1, 1].  Higher = more similar.
    """
    audio_emb = get_audio_embedding(audio_path)
    text_emb  = get_text_embedding(text)
    return float(np.dot(audio_emb, text_emb))


def audio_audio_similarity(path_a: str, path_b: str) -> float:
    """
    Compute cosine similarity between two audio files in CLAP space.
    Useful for comparing benign vs adversarial versions of the same utterance.

    Returns float in [-1, 1].
    """
    emb_a = get_audio_embedding(path_a)
    emb_b = get_audio_embedding(path_b)
    return float(np.dot(emb_a, emb_b))