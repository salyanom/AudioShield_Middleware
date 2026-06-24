"""
evaluate.py

Runs the full AudioShield pipeline against every benign and adversarial file,
then computes detection metrics and saves 6 publication-ready graphs.

Usage:
    python src/evaluate.py
    python src/evaluate.py --benign data/benign --adversarial data/adversarial --out results/
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
import matplotlib.patches as mpatches

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "benign_dir":      "data/benign",
    "adversarial_dir": "data/adversarial",
    "output_dir":      "results",
    "csv_name":        "evaluation_results.csv",
    "audio_extensions": {".wav", ".mp3", ".flac"},
    "detected_decisions": {"BLOCK", "MITIGATE"},  # what counts as "detected"
    "fig_dpi": 150,
    "colors": {
        "benign":     "#4C72B0",
        "adversarial":"#DD8452",
        "allow":      "#2ca02c",
        "mitigate":   "#ff7f0e",
        "block":      "#d62728",
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def _files(directory: str) -> list:
    ext = CONFIG["audio_extensions"]
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in ext
    )


def _attack_type(path: str) -> str:
    stem   = os.path.splitext(os.path.basename(path))[0]
    parts  = stem.split("_")
    return parts[-1] if len(parts) > 1 else "unknown"


def run_evaluation(benign_dir: str, adv_dir: str) -> pd.DataFrame:
    # Lazy import — Whisper/SentenceTransformer only loaded when evaluation runs
    from middleware import process_audio

    rows = []
    benign_files = _files(benign_dir)
    adv_files    = _files(adv_dir)

    print(f"Benign files     : {len(benign_files)}")
    print(f"Adversarial files: {len(adv_files)}")

    for path in benign_files:
        print(f"  [benign]      {os.path.basename(path)}", end=" ", flush=True)
        try:
            r = process_audio(path)
            rows.append(_to_row(r, label=0, attack_type="benign"))
            print(f"→ {r.decision}")
        except Exception as e:
            print(f"→ ERROR: {e}")

    for path in adv_files:
        attack = _attack_type(path)
        print(f"  [adversarial] {os.path.basename(path)} ({attack})", end=" ", flush=True)
        try:
            r = process_audio(path)
            rows.append(_to_row(r, label=1, attack_type=attack))
            print(f"→ {r.decision}")
        except Exception as e:
            print(f"→ ERROR: {e}")

    return pd.DataFrame(rows)


def _to_row(r, label: int, attack_type: str) -> dict:
    """Works with PipelineResult dataclass from the real middleware."""
    latency_ms = r.latency_ms
    total_latency = sum(latency_ms.values()) if isinstance(latency_ms, dict) else float(latency_ms)

    # unsafe_prob: use output if available, else input
    unsafe_prob = (
        r.output_unsafe_probability
        if r.output_unsafe_probability is not None
        else r.input_unsafe_probability
    )
    similarity = r.context_similarity if r.context_similarity is not None else 0.0

    return {
        "file":        os.path.basename(r.source.get("path", "") if isinstance(r.source, dict) else ""),
        "label":       label,
        "attack_type": attack_type,
        "transcript":  r.transcript,
        "similarity":  float(similarity),
        "unsafe_prob": float(unsafe_prob),
        "decision":    r.decision,
        "blocked_at":  r.reason,
        "detected":    1 if r.decision in CONFIG["detected_decisions"] else 0,
        "latency_ms":  total_latency,
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"].values
    y_pred = df["detected"].values
    y_prob = df["unsafe_prob"].values

    if len(np.unique(y_true)) < 2:
        print("[warn] Only one class present — classification metrics skipped.")
        return {}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "accuracy":  (tp + tn) / len(y_true),
        "fpr":       fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr":       fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


# ── Graphs ────────────────────────────────────────────────────────────────────

def _style():
    available = plt.style.available
    for s in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        if s in available:
            plt.style.use(s)
            return


def _save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_similarity(df: pd.DataFrame, out_dir: str):
    _style()
    grp   = df.groupby("attack_type")["similarity"]
    means = grp.mean().sort_values(ascending=False)
    stds  = grp.std().fillna(0).reindex(means.index)
    c     = CONFIG["colors"]
    colors = [c["benign"] if t == "benign" else c["adversarial"] for t in means.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(means.index, means.values, yerr=stds, color=colors,
                  capsize=4, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.8, color=c["allow"],   linestyle="--", linewidth=1, label="High similarity (0.8)")
    ax.axhline(0.5, color=c["block"],   linestyle="--", linewidth=1, label="Low similarity (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Attack Type", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Transcript Semantic Similarity per Attack Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "similarity_by_attack.png")


def plot_decisions(df: pd.DataFrame, out_dir: str):
    _style()
    decisions = ["ALLOW", "MITIGATE", "BLOCK"]
    c = CONFIG["colors"]
    clr = {"ALLOW": c["allow"], "MITIGATE": c["mitigate"], "BLOCK": c["block"]}

    grp = df.groupby(["attack_type", "decision"]).size().unstack(fill_value=0)
    for d in decisions:
        if d not in grp.columns:
            grp[d] = 0

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(grp))
    for d in decisions:
        vals = grp[d].values
        ax.bar(grp.index, vals, bottom=bottom, label=d,
               color=clr[d], edgecolor="white", linewidth=0.8)
        bottom += vals
    ax.set_xlabel("Attack Type", fontsize=11)
    ax.set_ylabel("Number of Files", fontsize=11)
    ax.set_title("Decision Distribution per Attack Type", fontsize=13, fontweight="bold")
    ax.legend(title="Decision", fontsize=9)
    _save(fig, out_dir, "decision_distribution.png")


def plot_unsafe_hist(df: pd.DataFrame, out_dir: str):
    _style()
    c    = CONFIG["colors"]
    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df["label"] == 0]["unsafe_prob"], bins=bins,
            alpha=0.7, label="Benign",      color=c["benign"],     edgecolor="white")
    ax.hist(df[df["label"] == 1]["unsafe_prob"], bins=bins,
            alpha=0.7, label="Adversarial", color=c["adversarial"], edgecolor="white")
    ax.axvline(0.50, color=c["mitigate"], linestyle="--", linewidth=1.5, label="Mitigate (0.50)")
    ax.axvline(0.85, color=c["block"],    linestyle="--", linewidth=1.5, label="Block (0.85)")
    ax.set_xlabel("Unsafe Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Unsafe Probability: Benign vs Adversarial", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "unsafe_prob_distribution.png")


def plot_roc(df: pd.DataFrame, out_dir: str):
    if len(np.unique(df["label"])) < 2:
        print("  [skip] ROC needs both classes.")
        return
    _style()
    fpr_arr, tpr_arr, _ = roc_curve(df["label"], df["unsafe_prob"])
    auc = roc_auc_score(df["label"], df["unsafe_prob"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_arr, tpr_arr, color=CONFIG["colors"]["benign"],
            lw=2, label=f"DistilBERT (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve — Adversarial Audio Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out_dir, "roc_curve.png")


def plot_confusion(df: pd.DataFrame, out_dir: str):
    if len(np.unique(df["label"])) < 2:
        print("  [skip] Confusion matrix needs both classes.")
        return
    _style()
    cm = confusion_matrix(df["label"], df["detected"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Adversarial"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    _save(fig, out_dir, "confusion_matrix.png")


def plot_latency(df: pd.DataFrame, out_dir: str):
    _style()
    c     = CONFIG["colors"]
    types = sorted(df["attack_type"].unique())
    data  = [df[df["attack_type"] == t]["latency_ms"].values for t in types]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=types, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, t in zip(bp["boxes"], types):
        patch.set_facecolor(c["benign"] if t == "benign" else c["adversarial"])
        patch.set_alpha(0.7)

    ax.set_xlabel("Attack Type",  fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Pipeline Latency per Attack Type", fontsize=13, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=c["benign"],      alpha=0.7, label="Benign"),
        mpatches.Patch(color=c["adversarial"], alpha=0.7, label="Adversarial"),
    ], fontsize=9)
    _save(fig, out_dir, "latency_boxplot.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(benign_dir: str, adv_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 54)
    print("  AUDIOSHIELD EVALUATION")
    print("=" * 54)

    df = run_evaluation(benign_dir, adv_dir)

    csv_path = os.path.join(out_dir, CONFIG["csv_name"])
    df.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    # ── Per-file table ────────────────────────────────────────────────────────
    cols = ["file", "attack_type", "similarity", "unsafe_prob", "decision", "latency_ms"]
    print("\n" + df[cols].to_string(index=False))

    # ── Detection metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  DETECTION METRICS")
    print("=" * 54)
    metrics = compute_metrics(df)
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}" if isinstance(v, float) else f"  {k:<14} {v}")

    # ── Similarity summary ────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  SIMILARITY SUMMARY")
    print("=" * 54)
    print(df.groupby("attack_type")["similarity"]
            .agg(["mean", "std", "min", "max"]).round(4).to_string())

    # ── Decision counts ───────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  DECISION COUNTS")
    print("=" * 54)
    print(df.groupby(["attack_type", "decision"]).size().to_string())

    # ── Graphs ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  GRAPHS")
    print("=" * 54)
    plot_similarity(df, out_dir)
    plot_decisions(df, out_dir)
    plot_unsafe_hist(df, out_dir)
    plot_roc(df, out_dir)
    plot_confusion(df, out_dir)
    plot_latency(df, out_dir)

    print(f"\nAll outputs → {out_dir}/")


def _args():
    p = argparse.ArgumentParser(description="AudioShield evaluation")
    p.add_argument("--benign",      default=CONFIG["benign_dir"])
    p.add_argument("--adversarial", default=CONFIG["adversarial_dir"])
    p.add_argument("--out",         default=CONFIG["output_dir"])
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(a.benign, a.adversarial, a.out)