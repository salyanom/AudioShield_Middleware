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

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "minilm_model": "all-MiniLM-L6-v2",
    "use_clap":     True,    # set False to disable CLAP (faster, transcript only)
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
        Path to original audio file.  If provided and CONFIG["use_clap"]
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

    if audio_path and CONFIG["use_clap"]:
        try:
            from audio_embedder import audio_text_similarity, get_audio_embedding
            audio_sim = audio_text_similarity(audio_path, response)
            audio_emb = get_audio_embedding(audio_path)
            clap_ok   = True
        except Exception as e:
            print(f"[context_verifier] CLAP unavailable: {e} — using transcript only")

    return {
        "transcript_similarity": transcript_sim,
        "audio_similarity":      audio_sim,
        "audio_embedding":       audio_emb,
        "clap_available":        clap_ok,
    }