"""
validate_thresholds.py

Compares classification metrics under the original (manually chosen)
thresholds vs. the grid-search-optimized thresholds on the same
evaluation dataset. Generates side-by-side comparison charts.

Usage:
    python src/validate_thresholds.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ── Configurations to compare ────────────────────────────────────────────────

OLD_CONFIG = {
    "label": "Original (Manual)",
    "wp": 0.40, "wc": 0.35, "wa": 0.25,
    "mitigate_threshold": 0.40,
    "block_threshold": 0.60,
}

OPTIMIZED_CONFIG = {
    "label": "Optimized (Grid Search)",
    "wp": 0.65, "wc": 0.10, "wa": 0.25,
    "mitigate_threshold": 0.42,
    "block_threshold": 0.43,
}


def _compute_risk_scores(df, wp, wc, wa):
    """Compute hybrid risk scores for all rows."""
    scores = []
    for _, row in df.iterrows():
        up = row["unsafe_prob"]
        st = row["similarity"]
        sa = row["audio_similarity"]

        if pd.isna(sa):
            total = wp + wc
            if total > 0:
                wp_r, wc_r = wp / total, wc / total
            else:
                wp_r, wc_r = 0.5, 0.5
            score = wp_r * up + wc_r * (1.0 - st)
        else:
            score = wp * up + wc * (1.0 - st) + wa * (1.0 - sa)
        scores.append(score)
    return np.array(scores)


def _compute_metrics(y_true, y_pred):
    """Compute classification metrics from binary arrays."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = float((tp + tn) / len(y_true))
    fpr       = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr       = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "FPR": fpr,
        "FNR": fnr,
    }


def validate():
    csv_path = "results/evaluation_results.csv"
    output_dir = "results"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run evaluate.py first.")
        return

    y_true = df["label"].values
    configs = [OLD_CONFIG, OPTIMIZED_CONFIG]
    all_metrics = {}
    all_scores = {}

    for cfg in configs:
        label = cfg["label"]
        scores = _compute_risk_scores(df, cfg["wp"], cfg["wc"], cfg["wa"])
        y_pred = (scores >= cfg["mitigate_threshold"]).astype(int)
        metrics = _compute_metrics(y_true, y_pred)
        all_metrics[label] = metrics
        all_scores[label] = scores

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  THRESHOLD VALIDATION: OLD vs OPTIMIZED")
    print("=" * 72)
    print(f"{'Metric':<20} {'Original (Manual)':>20} {'Optimized (Grid)':>20}")
    print("-" * 72)
    for metric in ["Precision", "Recall", "F1-Score", "Accuracy", "FPR", "FNR"]:
        old_val = all_metrics["Original (Manual)"][metric]
        new_val = all_metrics["Optimized (Grid Search)"][metric]
        delta = new_val - old_val
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {metric:<18} {old_val:>18.4f} {new_val:>18.4f}  {arrow} {abs(delta):.4f}")
    print("=" * 72)

    # ── Chart 1: ROC Comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#4C72B0", "#DD8452"]

    for i, (label, scores) in enumerate(all_scores.items()):
        fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr_curve, tpr_curve)
        ax.plot(fpr_curve, tpr_curve, color=colors[i], lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Comparison: Original vs Optimized Thresholds", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = os.path.join(output_dir, "threshold_comparison_roc.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"  ROC comparison saved to {roc_path}")

    # ── Chart 2: F1 / Precision / Recall Bar Comparison ───────────────────────
    metrics_to_plot = ["Precision", "Recall", "F1-Score", "FPR"]
    old_vals = [all_metrics["Original (Manual)"][m] for m in metrics_to_plot]
    new_vals = [all_metrics["Optimized (Grid Search)"][m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, old_vals, width, label="Original (Manual)",
                   color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, new_vals, width, label="Optimized (Grid Search)",
                   color="#DD8452", edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Metric Comparison: Original vs Optimized Thresholds", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f1_path = os.path.join(output_dir, "threshold_comparison_f1.png")
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)
    print(f"  Metric comparison saved to {f1_path}")

    print("\nThreshold validation complete.")


if __name__ == "__main__":
    validate()
