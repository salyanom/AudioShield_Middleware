"""
generate_adversarial.py

Generates adversarial audio samples from a source file using configurable
attack parameters.  Each attack is a pure function — no global state.

Attacks implemented:
    speed       - time-stretch without pitch change (pydub)
    volume      - gain shift in dB (pydub)
    noise       - additive white gaussian noise (numpy)
    pitch       - pitch shift via resampling (scipy)
    echo        - single delayed echo (numpy)
    reverb      - multi-tap reverb via convolution (scipy)

Usage:
    python generate_adversarial.py
    python generate_adversarial.py --input data/benign/test.mp3 --attacks noise pitch
    python generate_adversarial.py --list-attacks
"""

import argparse
import os
import struct
import wave
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import resample_poly
from math import gcd


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "input_path":  "data/benign/test.mp3",
    "output_dir":  "data/adversarial",
    "output_fmt":  "mp3",           # mp3 | wav  (mp3 requires pydub+ffmpeg)

    "attacks": {

        "speed": {
            "enabled":        True,
            "playback_speed": 1.1,          # >1 = faster, <1 = slower
        },

        "volume": {
            "enabled":        True,
            "gain_db":        6.0,          # positive = louder, negative = quieter
        },

        "noise": {
            "enabled":        True,
            "snr_db":         20.0,         # signal-to-noise ratio in dB
                                            # lower = noisier (try 10–30)
        },

        "pitch": {
            "enabled":        True,
            "semitones":      2.0,          # positive = higher, negative = lower
                                            # 12 semitones = 1 octave
        },

        "echo": {
            "enabled":        True,
            "delay_ms":       300,          # echo delay in milliseconds
            "decay":          0.5,          # echo amplitude (0.0–1.0)
        },

        "reverb": {
            "enabled":        True,
            "room_scale":     0.6,          # 0.0 (dry) → 1.0 (large room)
            "wet_mix":        0.35,         # blend: 0.0 = dry, 1.0 = full wet
        },
    },
}
# ─────────────────────────────────────────────────────────────────────────────


# ── WAV helpers ───────────────────────────────────────────────────────────────

def _wav_to_array(path: str):
    """Read a WAV file → (float32 array in [-1, 1], sample_rate, n_channels)."""
    with wave.open(path, "rb") as wf:
        sr         = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    arr   = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    max_val = float(2 ** (8 * sampwidth - 1))
    arr    /= max_val                        # normalise to [-1, 1]

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels)

    return arr, sr, n_channels


def _array_to_wav(arr: np.ndarray, sr: int, n_channels: int, path: str):
    """Write a float32 array ([-1, 1]) back to a WAV file (16-bit PCM)."""
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _load_audio(path: str):
    """
    Load any audio file to a float32 mono array + sample rate.
    WAV: stdlib wave.  MP3/other: pydub (requires ffmpeg).
    Falls back gracefully with a clear error if pydub is missing.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        arr, sr, n_channels = _wav_to_array(path)
        # Mix down to mono if stereo
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr, sr

    # Non-WAV → pydub
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_sample_width(2)
        sr  = seg.frame_rate
        raw = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        return raw, sr
    except ImportError:
        raise ImportError(
            f"pydub is required to load '{ext}' files.\n"
            "Install with: pip install pydub\n"
            "Then ensure ffmpeg is on your PATH."
        )


def _save_audio(arr: np.ndarray, sr: int, path: str, fmt: str):
    """
    Save float32 mono array to disk.
    WAV: stdlib wave.  MP3/other: pydub.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt == "wav":
        _array_to_wav(arr, sr, 1, path)
        return

    try:
        from pydub import AudioSegment
        pcm  = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        seg  = AudioSegment(
            pcm.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        seg.export(path, format=fmt)
    except ImportError:
        # Fallback: save as wav with .mp3 extension warning
        wav_path = os.path.splitext(path)[0] + ".wav"
        print(f"  [warn] pydub not found — saving as WAV instead: {wav_path}")
        _array_to_wav(arr, sr, 1, wav_path)


# ── Attack functions ──────────────────────────────────────────────────────────

def attack_speed(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Time-stretch: resample to change playback speed without pitch shift.
    playback_speed > 1 → shorter (faster); < 1 → longer (slower).
    """
    speed = cfg["playback_speed"]
    # Rational approximation for resample_poly (max denominator = 500)
    from fractions import Fraction
    frac = Fraction(speed).limit_denominator(500)
    up, down = frac.denominator, frac.numerator  # swap: faster = fewer samples
    return resample_poly(arr, up, down).astype(np.float32)


def attack_volume(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Gain shift in dB.  Clips to prevent overflow."""
    gain_linear = 10 ** (cfg["gain_db"] / 20.0)
    return np.clip(arr * gain_linear, -1.0, 1.0).astype(np.float32)


def attack_noise(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Additive White Gaussian Noise at a target SNR.
    snr_db = 10 * log10(signal_power / noise_power)
    Lower snr_db = more noise.
    """
    snr_db      = cfg["snr_db"]
    signal_power = np.mean(arr ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.random.normal(0, np.sqrt(noise_power), arr.shape).astype(np.float32)
    return np.clip(arr + noise, -1.0, 1.0).astype(np.float32)


def attack_pitch(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Pitch shift via resampling trick:
      1. Resample to change pitch (alters duration).
      2. Resample back to original length (restores duration, keeps pitch shift).

    semitones > 0 → higher pitch; < 0 → lower pitch.
    """
    semitones   = cfg["semitones"]
    ratio       = 2 ** (semitones / 12.0)      # pitch ratio
    n_orig      = len(arr)

    # Step 1: shift pitch (resamples to different length)
    from fractions import Fraction
    frac         = Fraction(ratio).limit_denominator(500)
    up1, down1   = frac.numerator, frac.denominator
    pitched      = resample_poly(arr, up1, down1)

    # Step 2: resample back to original length to restore duration
    n_pitched    = len(pitched)
    frac2        = Fraction(n_orig / n_pitched).limit_denominator(500)
    up2, down2   = frac2.numerator, frac2.denominator
    restored     = resample_poly(pitched, up2, down2)

    # Trim or pad to exactly original length
    if len(restored) >= n_orig:
        return restored[:n_orig].astype(np.float32)
    else:
        return np.pad(restored, (0, n_orig - len(restored))).astype(np.float32)


def attack_echo(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Single-tap echo: mix original with a delayed, attenuated copy.
    output[t] = input[t] + decay * input[t - delay_samples]
    """
    delay_ms      = cfg["delay_ms"]
    decay         = cfg["decay"]
    delay_samples = int(sr * delay_ms / 1000)

    out           = arr.copy().astype(np.float32)
    if delay_samples < len(arr):
        out[delay_samples:] += decay * arr[:-delay_samples]

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def attack_reverb(arr: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Multi-tap reverb via convolution with a synthetic impulse response (IR).

    IR is built from exponentially decaying random noise — a standard
    approximation of a room impulse response.  room_scale controls the
    IR length (larger = longer decay = bigger room feel).
    wet_mix blends the reverb signal with the dry original.
    """
    room_scale = cfg["room_scale"]
    wet_mix    = cfg["wet_mix"]

    # IR length: 0.1s (small room) → 2.0s (large hall)
    ir_duration  = 0.1 + room_scale * 1.9
    ir_samples   = int(sr * ir_duration)

    # Exponential decay envelope
    decay_rate   = 6.0 / ir_duration          # reaches ~0.0025 at end
    t            = np.linspace(0, ir_duration, ir_samples)
    envelope     = np.exp(-decay_rate * t).astype(np.float32)

    np.random.seed(42)                        # reproducible IR
    ir           = (np.random.randn(ir_samples) * envelope).astype(np.float32)
    ir           /= np.abs(ir).max() + 1e-8  # normalise IR

    # Convolve
    wet          = fftconvolve(arr, ir, mode="full")[: len(arr)].astype(np.float32)
    wet          /= np.abs(wet).max() + 1e-8  # prevent clipping

    mixed        = (1.0 - wet_mix) * arr + wet_mix * wet
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


# ── Registry ──────────────────────────────────────────────────────────────────

ATTACK_REGISTRY = {
    "speed":  attack_speed,
    "volume": attack_volume,
    "noise":  attack_noise,
    "pitch":  attack_pitch,
    "echo":   attack_echo,
    "reverb": attack_reverb,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_attacks(input_path: str, output_dir: str, fmt: str, attack_cfgs: dict, attacks: list = None):
    """
    Apply each enabled attack to input_path and save results to output_dir.

    Parameters
    ----------
    attacks : list[str] | None
        If provided, only run these attacks (must still be enabled in cfg).
        If None, run all enabled attacks.
    """
    print(f"Loading: {input_path}")
    arr, sr = _load_audio(input_path)
    print(f"  {len(arr)} samples @ {sr} Hz  ({len(arr)/sr:.2f}s)")

    stem    = os.path.splitext(os.path.basename(input_path))[0]
    results = {}

    for name, fn in ATTACK_REGISTRY.items():
        cfg = attack_cfgs.get(name, {})
        if not cfg.get("enabled", False):
            continue
        if attacks and name not in attacks:
            continue

        print(f"\n  [{name}] applying attack...")
        try:
            perturbed    = fn(arr, sr, cfg)
            out_filename = f"{stem}_{name}.{fmt}"
            out_path     = os.path.join(output_dir, out_filename)
            _save_audio(perturbed, sr, out_path, fmt)
            print(f"  [{name}] saved → {out_path}")
            results[name] = {"path": out_path, "status": "ok"}
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            results[name] = {"path": None, "status": f"error: {e}"}

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate adversarial audio samples")
    p.add_argument("--input",    default=CONFIG["input_path"],
                   help="Path to source audio file")
    p.add_argument("--output",   default=CONFIG["output_dir"],
                   help="Output directory")
    p.add_argument("--fmt",      default=CONFIG["output_fmt"],
                   help="Output format: mp3 or wav")
    p.add_argument("--attacks",  nargs="+",
                   choices=list(ATTACK_REGISTRY.keys()),
                   help="Run only these attacks (default: all enabled)")
    p.add_argument("--list-attacks", action="store_true",
                   help="Print available attacks and exit")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list_attacks:
        print("Available attacks:")
        for name in ATTACK_REGISTRY:
            cfg     = CONFIG["attacks"].get(name, {})
            enabled = "enabled" if cfg.get("enabled") else "disabled"
            print(f"  {name:<10} [{enabled}]")
        return

    results = run_attacks(
        input_path  = args.input,
        output_dir  = args.output,
        fmt         = args.fmt,
        attack_cfgs = CONFIG["attacks"],
        attacks     = args.attacks,
    )

    print("\n── Summary ──────────────────────────────")
    for name, r in results.items():
        status = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status} {name:<10} {r['path'] or r['status']}")


if __name__ == "__main__":
    main()