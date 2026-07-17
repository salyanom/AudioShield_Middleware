"""
evaluate_external_benign.py

Runs the AudioShield pipeline on external benign datasets to validate
that optimized thresholds generalize to unseen data (FPR should be 0%).

Compares results under original (manual) vs optimized thresholds.

Usage:
    python src/evaluate_external_benign.py
    python src/evaluate_external_benign.py --out results/external_benign/
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Settings
from middleware import process_transcript
from audio_processor import transcribe_audio


# ── Threshold configs to compare ─────────────────────────────────────────────

ORIGINAL_CONFIG = Settings(
    llm_provider="stub",
    weight_policy=0.40,
    weight_context=0.35,
    weight_audio=0.25,
    mitigate_threshold=0.40,
    block_threshold=0.60,
)

OPTIMIZED_CONFIG = Settings(
    llm_provider="stub",
    weight_policy=0.65,
    weight_context=0.10,
    weight_audio=0.25,
    mitigate_threshold=0.42,
    block_threshold=0.43,
)

CONFIGS = {
    "Original (Manual)": ORIGINAL_CONFIG,
    "Optimized (Grid)": OPTIMIZED_CONFIG,
}


def _audio_files(directory):
    """List all audio files in a directory."""
    exts = {".wav", ".flac", ".mp3"}
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


def evaluate_dataset(dataset_name, audio_dir, cfg, out_dir=None):
    """Run pipeline on all files in audio_dir and return results DataFrame."""
    files = _audio_files(audio_dir)
    if not files:
        print(f"  No audio files found in {audio_dir}")
        return pd.DataFrame()

    rows = []
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            transcript = transcribe_audio(fpath)
            if not transcript or not transcript.strip():
                print(f"    [{i+1}/{len(files)}] {fname}: empty transcript, skipping")
                continue

            result = process_transcript(
                transcript,
                audio_path=fpath,
                cfg=cfg,
            )
            rows.append({
                "dataset": dataset_name,
                "file": fname,
                "transcript": transcript[:80],
                "decision": result.decision,
                "risk_score": result.risk_score,
                "unsafe_prob": result.output_unsafe_probability,
                "similarity": result.context_similarity,
                "audio_similarity": result.audio_similarity,
            })
            status = "OK" if result.decision == "ALLOW" else f"** {result.decision} **"
            print(f"    [{i+1}/{len(files)}] {fname}: {status} (risk={result.risk_score:.3f})")

        except Exception as e:
            print(f"    [{i+1}/{len(files)}] {fname}: ERROR - {e}")
            rows.append({
                "dataset": dataset_name,
                "file": fname,
                "transcript": f"ERROR: {e}",
                "decision": "ERROR",
                "risk_score": None,
                "unsafe_prob": None,
                "similarity": None,
                "audio_similarity": None,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate external benign datasets")
    parser.add_argument("--data", default="data/external_benign",
                        help="Root directory of external benign datasets")
    parser.add_argument("--out", default="results/external_benign",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Discover datasets
    if not os.path.isdir(args.data):
        print(f"Error: {args.data} not found. Run download_datasets.py first.")
        return

    datasets = sorted(
        d for d in os.listdir(args.data)
        if os.path.isdir(os.path.join(args.data, d)) and not d.startswith("_")
    )

    if not datasets:
        print(f"No dataset directories found in {args.data}")
        return

    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")
    print()

    # Evaluate each dataset under each config
    all_results = {}
    for config_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  CONFIG: {config_name}")
        print(f"{'='*60}")

        config_dfs = []
        for ds_name in datasets:
            ds_path = os.path.join(args.data, ds_name)
            print(f"\n  Dataset: {ds_name}")
            df = evaluate_dataset(ds_name, ds_path, cfg, args.out)
            if not df.empty:
                config_dfs.append(df)

        if config_dfs:
            all_results[config_name] = pd.concat(config_dfs, ignore_index=True)

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  EXTERNAL BENIGN VALIDATION: THRESHOLD COMPARISON")
    print("=" * 72)

    for config_name, df in all_results.items():
        valid = df[df["decision"] != "ERROR"]
        total = len(valid)
        allow = len(valid[valid["decision"] == "ALLOW"])
        mitigate = len(valid[valid["decision"] == "MITIGATE"])
        block = len(valid[valid["decision"] == "BLOCK"])
        fpr = (mitigate + block) / total if total > 0 else 0.0

        print(f"\n  {config_name}:")
        print(f"    Total files:  {total}")
        print(f"    ALLOW:        {allow}")
        print(f"    MITIGATE:     {mitigate}  (false positive)")
        print(f"    BLOCK:        {block}  (false positive)")
        print(f"    FPR:          {fpr:.4f}")

        # Per-dataset breakdown
        for ds_name in datasets:
            ds_df = valid[valid["dataset"] == ds_name]
            if ds_df.empty:
                continue
            ds_total = len(ds_df)
            ds_allow = len(ds_df[ds_df["decision"] == "ALLOW"])
            ds_fp = ds_total - ds_allow
            ds_fpr = ds_fp / ds_total if ds_total > 0 else 0.0
            print(f"      {ds_name:>20s}: {ds_allow}/{ds_total} ALLOW, FPR={ds_fpr:.4f}")

    print("\n" + "=" * 72)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    for config_name, df in all_results.items():
        safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        csv_path = os.path.join(args.out, f"external_benign_{safe_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Results saved to {csv_path}")

    # ── Generate comparison bar chart ─────────────────────────────────────────
    if len(all_results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        config_names = list(all_results.keys())
        colors = ["#4C72B0", "#DD8452"]

        x_labels = datasets + ["TOTAL"]
        x = np.arange(len(x_labels))
        width = 0.35

        for ci, config_name in enumerate(config_names):
            df = all_results[config_name]
            valid = df[df["decision"] != "ERROR"]
            fprs = []
            for ds_name in datasets:
                ds_df = valid[valid["dataset"] == ds_name]
                if ds_df.empty:
                    fprs.append(0.0)
                else:
                    ds_fp = len(ds_df[ds_df["decision"] != "ALLOW"])
                    fprs.append(ds_fp / len(ds_df))
            # Total FPR
            total_fp = len(valid[valid["decision"] != "ALLOW"])
            total_fpr = total_fp / len(valid) if len(valid) > 0 else 0.0
            fprs.append(total_fpr)

            offset = -width/2 + ci * width
            bars = ax.bar(x + offset, fprs, width, label=config_name,
                          color=colors[ci], edgecolor="white", linewidth=0.8)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                        f"{height:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("False Positive Rate", fontsize=12)
        ax.set_title("External Benign Validation: FPR by Dataset", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, max(0.15, ax.get_ylim()[1] * 1.2))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        chart_path = os.path.join(args.out, "external_benign_fpr_comparison.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"  FPR comparison chart saved to {chart_path}")

    print("\nExternal benign evaluation complete.")


if __name__ == "__main__":
    main()
