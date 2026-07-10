"""
context_verifier.py

Dual-channel semantic consistency verification:

Channel 1 — Transcript-level (MiniLM):
    Compares the STT transcript against the LLM response using
    SentenceTransformer cosine similarity.  Fast, always available.

Channel 2 — Audio-level (CLAP):
    Compares the raw audio embedding against the LLM response text
    embedding in CLAP's shared audio-text space.  This catches
    adversarial perturbations that survive transcription unchanged
    but alter the audio's semantic fingerprint.

verify_context() returns the transcript-level similarity (float)
for backwards compatibility with middleware.py.

verify_context_full() returns a dict with both scores and the audio
embedding itself, for use in the hybrid decision engine.
"""

import numpy as np

from config import settings

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "minilm_model": "all-MiniLM-L6-v2",
}
# ─────────────────────────────────────────────────────────────────────────────

_minilm = None


def _load_minilm():
    global _minilm
    from sentence_transformers import SentenceTransformer
    _minilm = SentenceTransformer(CONFIG["minilm_model"])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def verify_context(transcript: str, response: str) -> float:
    """
    Compute transcript-level semantic similarity (MiniLM).
    Backwards-compatible interface used by middleware.py.

    Returns float in [-1, 1].  Higher = more similar.
    """
    if _minilm is None:
        _load_minilm()

    embs = _minilm.encode([transcript, response])
    return _cosine(embs[0], embs[1])


def verify_context_full(
    transcript: str,
    response: str,
    audio_path: str | None = None,
) -> dict:
    """
    Full dual-channel context verification.

    Parameters
    ----------
    transcript  : str   STT transcript of the audio.
    response    : str   LLM-generated response.
    audio_path  : str | None
        Path to original audio file.  If provided and settings.use_clap
        is True, CLAP audio-text similarity is computed.

    Returns
    -------
    dict with keys:
        transcript_similarity   float   MiniLM cosine similarity
        audio_similarity        float | None   CLAP cosine similarity
        audio_embedding         np.ndarray | None   (512,) CLAP audio embedding
        clap_available          bool
    """
    transcript_sim = verify_context(transcript, response)

    audio_sim = None
    audio_emb = None
    clap_ok   = False

    if audio_path and settings.use_clap:
        try:
            from audio_embedder import get_audio_embedding, get_text_embedding
            audio_emb = get_audio_embedding(audio_path)
            text_emb  = get_text_embedding(response)
            audio_sim = float(np.dot(audio_emb, text_emb))
            clap_ok   = True
        except Exception:
            import traceback
            traceback.print_exc()
            print("[context_verifier] CLAP unavailable (see traceback above) — using transcript only")

    return {
        "transcript_similarity": transcript_sim,
        "audio_similarity":      audio_sim,
        "audio_embedding":       audio_emb,
        "clap_available":        clap_ok,
    }