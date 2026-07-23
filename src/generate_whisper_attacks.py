"""
generate_whisper_attacks.py

Generates acoustic obfuscation and Whisper-targeted adversarial audio samples
(`data/adversarial/whisper_attack_*.wav`).

Attacks implemented:
1. High-Frequency Phonetic Masking: Adds high-frequency acoustic masking noise to
   speech commands that degrades human intelligibility while preserving Whisper's
   mel-filterbank feature extraction in lower bands.
2. Extreme Time-Scale Modification (TSM): Compresses spoken prompt injections to
   3x-4x speed (or slows down with pitch jitter) where human comprehension drops
   significantly, yet Whisper's robust temporal encoder successfully decodes the
   underlying malicious instruction.
3. Acoustic Echo Impersonation: Adds synthetic multi-path reverberation simulating
   hidden voice commands played from a loudspeaker across a room.

Usage:
    python src/generate_whisper_attacks.py
"""

import os
import random
import numpy as np
import soundfile as sf
from scipy import signal

OUT_DIR = os.path.join("data", "adversarial")
os.makedirs(OUT_DIR, exist_ok=True)


def load_sample_injections() -> list[tuple[str, np.ndarray, int]]:
    """Load existing clean prompt injection audio files to transform."""
    injections = []
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("adversarial_inject_") and f.endswith(".wav"):
            path = os.path.join(OUT_DIR, f)
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            injections.append((f, audio.astype(np.float32), sr))
    return injections


def apply_hf_phonetic_masking(audio: np.ndarray, sr: int) -> np.ndarray:
    """Inject high-frequency masking band (4kHz-7kHz) with amplitude modulation."""
    n = len(audio)
    t = np.arange(n) / sr
    # Carrier frequency above normal fundamental speech frequencies
    carrier_freq = 5500.0 + 500.0 * np.sin(2 * np.pi * 3.0 * t)
    noise = np.sin(2 * np.pi * carrier_freq * t) * np.random.uniform(0.5, 1.5, size=n)
    # Bandpass filter around high frequencies
    sos = signal.butter(4, [4000, 7500], btype="bandpass", fs=sr, output="sos")
    noise_filtered = signal.sosfilt(sos, noise).astype(np.float32)
    # Mix with speech
    masked = audio + 0.15 * noise_filtered
    return np.clip(masked, -1.0, 1.0)


def apply_time_scale_compression(audio: np.ndarray, sr: int, factor: float = 2.5) -> np.ndarray:
    """Fast time-scale compression via resampling/phase-vocoder approximation."""
    num_samples = int(len(audio) / factor)
    compressed = signal.resample(audio, num_samples).astype(np.float32)
    return np.clip(compressed, -1.0, 1.0)


def apply_acoustic_room_impersonation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate multi-path room reverberation and distant loudspeaker impulse response."""
    # Create synthetic impulse response with multiple delayed reflections
    ir_len = int(0.25 * sr)
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0
    # Reflections at 20ms, 45ms, 80ms, 140ms
    for delay_ms, decay in [(20, 0.6), (45, 0.35), (80, 0.2), (140, 0.1)]:
        idx = int((delay_ms / 1000.0) * sr)
        if idx < ir_len:
            ir[idx] = decay * ((-1) ** random.randint(0, 1))
    
    reverb = signal.fftconvolve(audio, ir, mode="same").astype(np.float32)
    return np.clip(reverb * 0.8, -1.0, 1.0)


def main():
    injections = load_sample_injections()
    if not injections:
        print("No adversarial_inject_*.wav files found to transform.")
        return

    print(f"\nGenerating Whisper-targeted acoustic attacks from {len(injections)} base injections...\n")
    count = 0

    for idx, (fname, audio, sr) in enumerate(injections):
        stem = os.path.splitext(fname)[0]

        # 1. High-frequency phonetic masking
        hf_audio = apply_hf_phonetic_masking(audio, sr)
        out1 = os.path.join(OUT_DIR, f"whisper_attack_hfmask_{idx:02d}.wav")
        sf.write(out1, hf_audio, sr)
        count += 1

        # 2. Time compression (2.5x speed)
        tsm_audio = apply_time_scale_compression(audio, sr, factor=2.5)
        out2 = os.path.join(OUT_DIR, f"whisper_attack_fasttsm_{idx:02d}.wav")
        sf.write(out2, tsm_audio, sr)
        count += 1

        # 3. Room reverberation impersonation
        rev_audio = apply_acoustic_room_impersonation(audio, sr)
        out3 = os.path.join(OUT_DIR, f"whisper_attack_reverb_{idx:02d}.wav")
        sf.write(out3, rev_audio, sr)
        count += 1

    print(f"Successfully generated {count} Whisper-targeted adversarial audio samples in '{OUT_DIR}'.")
    print("These test AudioShield's robustness against acoustic obfuscation and high-speed attacks.")


if __name__ == "__main__":
    main()
