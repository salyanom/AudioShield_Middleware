"""
benchmark_faster_whisper.py

Benchmarks latency and Real-Time Factor (RTF) comparing standard `openai-whisper`
against `faster-whisper` (CTranslate2 int8/float16 optimized engine) across
the benign and adversarial evaluation datasets.

Usage:
    python src/benchmark_faster_whisper.py
"""

import os
import time
import soundfile as sf
import numpy as np
import pandas as pd

BENIGN_DIR = os.path.join("data", "benign")
ADV_DIR = os.path.join("data", "adversarial")


def get_sample_files(n_each: int = 5) -> list[str]:
    files = []
    if os.path.exists(BENIGN_DIR):
        bf = [os.path.join(BENIGN_DIR, f) for f in sorted(os.listdir(BENIGN_DIR)) if f.endswith(".wav")]
        files.extend(bf[:n_each])
    if os.path.exists(ADV_DIR):
        af = [os.path.join(ADV_DIR, f) for f in sorted(os.listdir(ADV_DIR)) if f.endswith(".wav")]
        files.extend(af[:n_each])
    return files


def benchmark_engine(engine_name: str, sample_files: list[str]) -> pd.DataFrame:
    print(f"\nBenchmarking [{engine_name}] across {len(sample_files)} audio files...")
    rows = []

    # Initialize model
    start_load = time.perf_counter()
    if engine_name == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel("base", device=device, compute_type=compute_type)
        except ImportError:
            print("  faster-whisper package not found, skipping benchmark.")
            return pd.DataFrame()
    elif engine_name == "openai-whisper":
        import whisper
        model = whisper.load_model("base")
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    load_time = (time.perf_counter() - start_load) * 1000.0
    print(f"  Model load time: {load_time:.1f} ms")

    for path in sample_files:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration_sec = len(audio) / sr

        start_trans = time.perf_counter()
        if engine_name == "faster-whisper":
            segments, _ = model.transcribe(audio, beam_size=1)
            text = " ".join(s.text for s in segments).strip()
        else:
            result = model.transcribe(path, fp16=False, beam_size=1, temperature=0.0)
            text = result["text"].strip()
        latency_ms = (time.perf_counter() - start_trans) * 1000.0
        rtf = (latency_ms / 1000.0) / duration_sec if duration_sec > 0 else 0.0

        rows.append({
            "Engine": engine_name,
            "File": os.path.basename(path),
            "Duration (s)": round(duration_sec, 2),
            "Latency (ms)": round(latency_ms, 1),
            "RTF": round(rtf, 3),
            "Transcript Preview": text[:40] + "..." if len(text) > 40 else text,
        })
        print(f"  [{os.path.basename(path):<22s} | {duration_sec:4.1f}s audio] -> {latency_ms:6.1f} ms (RTF: {rtf:.3f})")

    return pd.DataFrame(rows)


def main():
    sample_files = get_sample_files(n_each=5)
    if not sample_files:
        print("No sample files found to benchmark.")
        return

    df_openai = benchmark_engine("openai-whisper", sample_files)
    df_faster = benchmark_engine("faster-whisper", sample_files)

    if not df_openai.empty and not df_faster.empty:
        df_all = pd.concat([df_openai, df_faster], ignore_index=True)
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "whisper_benchmark_comparison.csv")
        df_all.to_csv(csv_path, index=False)

        avg_openai_lat = df_openai["Latency (ms)"].mean()
        avg_faster_lat = df_faster["Latency (ms)"].mean()
        avg_openai_rtf = df_openai["RTF"].mean()
        avg_faster_rtf = df_faster["RTF"].mean()
        speedup = avg_openai_lat / avg_faster_lat if avg_faster_lat > 0 else 1.0

        print("\n" + "=" * 65)
        print("  WHISPER STT BENCHMARK COMPARISON SUMMARY")
        print("=" * 65)
        print(f"  {'Metric':<22s} | {'openai-whisper':<16s} | {'faster-whisper':<16s}")
        print("-" * 65)
        print(f"  {'Avg Latency (ms)':<22s} | {avg_openai_lat:<16.1f} | {avg_faster_lat:<16.1f} ({speedup:.2f}x speedup)")
        print(f"  {'Avg Real-Time Factor':<22s} | {avg_openai_rtf:<16.3f} | {avg_faster_rtf:<16.3f}")
        print("=" * 65)
        print(f"Saved detailed benchmark comparison to '{csv_path}'.\n")


if __name__ == "__main__":
    main()
