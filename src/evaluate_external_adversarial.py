"""Evaluate published DeepSpeech adversarial WAVs against Whisper.

This measures cross-model transfer: whether examples that attacked the source
DeepSpeech recognizer also alter AudioShield's Whisper transcript.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_processor import transcribe_audio
from context_verifier import verify_context


DATA_DIR = Path("data/external/zhenghuatan")
OUTPUT_PATH = Path("evaluation_external_adversarial.csv")

PAIRS = [
    {
        "dataset": "Common Voice",
        "attack": "white-box",
        "original": "000621_original.wav",
        "adversarial": "000621_adv-medium2medium.wav",
    },
    {
        "dataset": "Common Voice",
        "attack": "white-box",
        "original": "000639_original.wav",
        "adversarial": "000639_adv-medium2medium.wav",
    },
    {
        "dataset": "Speech Commands",
        "attack": "black-box targeted yes-to-right",
        "original": "yes_original.wav",
        "adversarial": "yes2right-black.wav",
    },
    {
        "dataset": "Speech Commands",
        "attack": "white-box targeted yes-to-right",
        "original": "yes_original.wav",
        "adversarial": "yes2right-white.wav",
    },
]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return float(bool(hyp))
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1] / len(ref)


def signal_metrics(original_path: Path, adversarial_path: Path) -> tuple[float, float]:
    original, original_sr = sf.read(original_path, dtype="float32", always_2d=False)
    adversarial, adversarial_sr = sf.read(adversarial_path, dtype="float32", always_2d=False)
    if original_sr != adversarial_sr:
        raise ValueError(f"Sample-rate mismatch: {original_sr} vs {adversarial_sr}")
    length = min(len(original), len(adversarial))
    original = np.asarray(original[:length])
    adversarial = np.asarray(adversarial[:length])
    noise = adversarial - original
    noise_power = float(np.mean(noise ** 2))
    signal_power = float(np.mean(original ** 2))
    snr_db = float("inf") if noise_power == 0 else 10 * np.log10(signal_power / noise_power)
    max_delta = float(np.max(np.abs(noise)))
    return float(snr_db), max_delta


def evaluate() -> list[dict]:
    missing = [
        str(DATA_DIR / pair[key])
        for pair in PAIRS
        for key in ("original", "adversarial")
        if not (DATA_DIR / pair[key]).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(sorted(set(missing))))

    transcript_cache: dict[str, str] = {}
    rows = []
    for pair in PAIRS:
        original_path = DATA_DIR / pair["original"]
        adversarial_path = DATA_DIR / pair["adversarial"]
        for path in (original_path, adversarial_path):
            if path.name not in transcript_cache:
                print(f"Transcribing {path.name}...")
                transcript_cache[path.name] = transcribe_audio(str(path)).strip()

        original_text = transcript_cache[original_path.name]
        adversarial_text = transcript_cache[adversarial_path.name]
        snr_db, max_delta = signal_metrics(original_path, adversarial_path)
        rows.append({
            **pair,
            "original_transcript": original_text,
            "adversarial_transcript": adversarial_text,
            "word_error_rate": round(word_error_rate(original_text, adversarial_text), 4),
            "semantic_similarity": round(float(verify_context(original_text, adversarial_text)), 4),
            "snr_db": round(snr_db, 3),
            "max_absolute_delta": round(max_delta, 6),
            "transferred_to_whisper": original_text.lower() != adversarial_text.lower(),
        })

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    results = evaluate()
    for result in results:
        print(
            f"{result['adversarial']}: WER={result['word_error_rate']:.4f}, "
            f"similarity={result['semantic_similarity']:.4f}, "
            f"SNR={result['snr_db']:.2f} dB, "
            f"Whisper changed={result['transferred_to_whisper']}"
        )
    print(f"Saved {OUTPUT_PATH}")
