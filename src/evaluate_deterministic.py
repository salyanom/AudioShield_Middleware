"""
evaluate_deterministic.py

Runs the AudioShield pipeline with a fixed, deterministic LLM response
so evaluation results are reproducible across runs.

The stub backend already uses the transcript directly, but similarity
scores vary because Whisper itself has slight non-determinism on short audio.
This script pins the LLM response to the transcript (echo mode) so the
context verifier always gets a fair, stable similarity score.

Usage:
    python evaluate_deterministic.py
    python evaluate_deterministic.py --benign data/external/benign --adversarial data/external/adversarial --out results/external/
"""

import argparse
import os
import sys
sys.path.insert(0, "src")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "benign_dir":      "data/benign",
    "adversarial_dir": "data/adversarial",
    "output_dir":      "results",
}
# ─────────────────────────────────────────────────────────────────────────────


def main(benign_dir, adversarial_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Import after sys.path set
    from middleware import process_audio
    import evaluate

    # Monkey-patch LLM to echo transcript deterministically
    # This removes LLM variance from evaluation so similarity scores are stable
    import llm_engine
    _original_generate = llm_engine.generate_response

    def _deterministic_response(transcript: str, **kwargs) -> str:
        """Return a clean, topically consistent response for any transcript."""
        if not transcript or not transcript.strip():
            return "I did not catch that. Could you please repeat?"
        return f"Here is information about the topic you mentioned: {transcript.strip()}"

    llm_engine.generate_response = _deterministic_response

    # Also patch provider.generate if middleware uses build_provider directly
    try:
        from config import settings
        from llm_engine import build_provider
        provider = build_provider(settings)
        provider.generate = _deterministic_response
    except Exception:
        pass

    print("=" * 56)
    print("  AUDIOSHIELD DETERMINISTIC EVALUATION")
    print("  (LLM pinned to echo mode for reproducibility)")
    print("=" * 56)

    import pandas as pd
    import numpy as np

    def _files(directory):
        ext = {".wav", ".mp3", ".flac"}
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in ext
        )

    def _attack_type(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        return parts[-1] if len(parts) > 1 else "unknown"

    rows = []
    benign_files = _files(benign_dir)
    adv_files    = _files(adversarial_dir)

    print(f"Benign files     : {len(benign_files)}")
    print(f"Adversarial files: {len(adv_files)}")

    for path in benign_files:
        print(f"  [benign]      {os.path.basename(path)}", end=" ", flush=True)
        try:
            r = process_audio(path)
            row = evaluate._to_row(r, 0, "benign")
            rows.append(row)
            print(f"→ {r.decision}  sim={row['similarity']:.3f}  unsafe={row['unsafe_prob']:.3f}")
        except Exception as e:
            print(f"→ ERROR: {e}")

    for path in adv_files:
        attack = _attack_type(path)
        print(f"  [adversarial] {os.path.basename(path)}", end=" ", flush=True)
        try:
            r = process_audio(path)
            row = evaluate._to_row(r, 1, attack)
            rows.append(row)
            print(f"→ {r.decision}  sim={row['similarity']:.3f}  unsafe={row['unsafe_prob']:.3f}")
        except Exception as e:
            print(f"→ ERROR: {e}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "evaluation_results_deterministic.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    cols = ["file", "attack_type", "similarity", "unsafe_prob", "decision", "latency_ms"]
    print("\n" + df[cols].to_string(index=False))

    print("\n" + "=" * 56)
    print("  DETECTION METRICS")
    print("=" * 56)
    metrics = evaluate.compute_metrics(df)
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}" if isinstance(v, float) else f"  {k:<14} {v}")

    print("\n" + "=" * 56)
    print("  GRAPHS")
    print("=" * 56)
    graph_dir = os.path.join(out_dir, "deterministic")
    os.makedirs(graph_dir, exist_ok=True)
    evaluate.plot_similarity(df, graph_dir)
    evaluate.plot_decisions(df, graph_dir)
    evaluate.plot_unsafe_hist(df, graph_dir)
    evaluate.plot_roc(df, graph_dir)
    evaluate.plot_confusion(df, graph_dir)
    evaluate.plot_latency(df, graph_dir)

    print(f"\nAll outputs → {graph_dir}/")

    # Restore original
    llm_engine.generate_response = _original_generate


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--benign",      default=CONFIG["benign_dir"])
    p.add_argument("--adversarial", default=CONFIG["adversarial_dir"])
    p.add_argument("--out",         default=CONFIG["output_dir"])
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(a.benign, a.adversarial, a.out)