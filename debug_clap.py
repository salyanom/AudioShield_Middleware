"""
debug_clap.py — fast, standalone CLAP shape diagnostic.

audio_embedder.py assumes ClapModel.get_audio_features()/get_text_features()
return pooled (1, 512) embeddings. On this machine they're coming back as
(1, 768, 2, 32) and (1, 45, 768) instead -- unpooled feature maps, not
embeddings -- which is why np.dot() in context_verifier.py blows up.

This script bypasses Whisper and the PGD attack entirely (which take
minutes to re-run) and directly probes what your installed transformers
version's ClapModel actually returns, under a few call variants, so we
fix this from real data instead of another guess.

Usage:
    python debug_clap.py [path/to/audio.wav]
"""

import sys

import librosa
import torch
import transformers
from transformers import ClapModel, ClapProcessor

print(f"transformers version: {transformers.__version__}")
print(f"torch version:        {torch.__version__}")

audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/benign/benign_00.wav"
audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=30)
print(f"loaded audio: {audio.shape} samples @ {sr} Hz ({len(audio) / sr:.2f}s)")

processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
model.eval()

print(f"model.config.projection_dim = {getattr(model.config, 'projection_dim', 'N/A')}")
print(f"model class: {type(model).__name__}")
print(f"get_audio_features doc: {model.get_audio_features.__doc__!r:.200}")

audio_variants = [
    ("bare array, no extra kwargs",         dict(audio=audio)),
    ("list-wrapped, no extra kwargs",       dict(audio=[audio])),
    ("list-wrapped + repeatpad/rand_trunc", dict(audio=[audio], padding="repeatpad", truncation="rand_trunc")),
]

def _describe(obj, indent="    "):
    """Print every field of a HF ModelOutput (dict-like) object with its shape/type."""
    if hasattr(obj, "shape"):
        print(f"{indent}<tensor> shape={tuple(obj.shape)}")
        return
    if hasattr(obj, "items"):
        for k, v in obj.items():
            if v is None:
                print(f"{indent}{k}: None")
            elif hasattr(v, "shape"):
                print(f"{indent}{k}: shape={tuple(v.shape)}")
            else:
                print(f"{indent}{k}: {type(v).__name__}")
        return
    print(f"{indent}{type(obj).__name__} (no .items/.shape): {obj!r:.200}")


for label, kw in audio_variants:
    print(f"\n--- audio variant: {label} ---")
    try:
        inputs = processor(**kw, sampling_rate=48000, return_tensors="pt")
        print("  processor output shapes:", {k: tuple(v.shape) for k, v in inputs.items()})
        with torch.no_grad():
            af = model.get_audio_features(**inputs)
        print(f"  get_audio_features() -> {type(af).__name__}")
        _describe(af)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- text ---")
text = "Here is information about the topic you mentioned."
try:
    tinputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    print("  processor output shapes:", {k: tuple(v.shape) for k, v in tinputs.items()})
    with torch.no_grad():
        tf = model.get_text_features(**tinputs)
    print(f"  get_text_features() -> {type(tf).__name__}")
    _describe(tf)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- manual low-level path: submodule pooler_output -> projection layer ---")
print("  model submodule attrs:", [a for a in dir(model) if "proj" in a.lower() or a in ("audio_model", "text_model")])
try:
    inputs = processor(audio=[audio], sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        raw_audio_out = model.audio_model(**{k: v for k, v in inputs.items() if k in ("input_features", "is_longer")})
        audio_pooled = raw_audio_out.pooler_output
        audio_projected = model.audio_projection(audio_pooled)
    print("  audio_model pooler_output shape:  ", tuple(audio_pooled.shape))
    print("  audio_projection(pooler_output) ->", tuple(audio_projected.shape))
except Exception as e:
    print(f"  FAILED (audio): {type(e).__name__}: {e}")

try:
    tinputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        raw_text_out = model.text_model(**tinputs)
        text_pooled = raw_text_out.pooler_output
        text_projected = model.text_projection(text_pooled)
    print("  text_model pooler_output shape:   ", tuple(text_pooled.shape))
    print("  text_projection(pooler_output) -> ", tuple(text_projected.shape))
except Exception as e:
    print(f"  FAILED (text): {type(e).__name__}: {e}")
