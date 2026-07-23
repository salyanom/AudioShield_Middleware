#!/usr/bin/env bash
# ============================================================================
# run_all.sh — Reproduce the complete AudioShield evaluation pipeline
#
# Prerequisites:
#   1. Python 3.10+ virtual environment with requirements.txt installed
#   2. Audio dataset in data/benign/ and data/adversarial/
#   3. Ollama running locally (or set AUDIOSHIELD_LLM_PROVIDER=stub for offline)
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo "  AudioShield — Full Evaluation Pipeline"
echo "============================================"
echo ""

# ─── Step 0: Environment check ──────────────────────────────────────────────
echo "[0/6] Checking environment..."
python --version
if [ ! -d "data/adversarial" ] || [ ! -d "data/benign" ]; then
    echo "ERROR: Dataset not found. Ensure data/adversarial/ and data/benign/ exist."
    exit 1
fi

ADV_COUNT=$(find data/adversarial -name "*.wav" | wc -l)
BEN_COUNT=$(find data/benign -name "*.wav" | wc -l)
echo "       Dataset: ${ADV_COUNT} adversarial + ${BEN_COUNT} benign = $(( ADV_COUNT + BEN_COUNT )) total"
echo ""

# ─── Step 1: Train the DistilBERT risk classifier ───────────────────────────
echo "[1/6] Training DistilBERT risk classifier..."
python src/train_risk_model.py
echo "       Model saved to models/risk_classifier/"
echo ""

# ─── Step 2: Extract features ───────────────────────────────────────────────
echo "[2/6] Extracting audio features..."
python src/extract_features.py
echo "       Features saved to features/"
echo ""

# ─── Step 3: Run systematic threshold evaluation ────────────────────────────
echo "[3/6] Running threshold evaluation across full dataset..."
export AUDIOSHIELD_LLM_PROVIDER="${AUDIOSHIELD_LLM_PROVIDER:-stub}"
python src/evaluate_thresholds_systematic.py
echo "       Raw results:  results/threshold_eval_raw.csv"
echo "       Metrics:      results/threshold_metrics.csv"
echo ""

# ─── Step 4: Generate evaluation plots ──────────────────────────────────────
echo "[4/6] Generating evaluation plots..."
python src/analyze_eval_results.py
echo "       Plots saved to results/plots/"
echo ""

# ─── Step 5: Run automated tests ────────────────────────────────────────────
echo "[5/6] Running pytest test suite..."
pytest -v tests/ || echo "       WARNING: Some tests failed (see output above)"
echo ""

# ─── Step 6: Summary ────────────────────────────────────────────────────────
echo "[6/6] Pipeline complete!"
echo ""
echo "============================================"
echo "  Generated Outputs"
echo "============================================"
echo "  results/threshold_eval_raw.csv     Per-sample evaluation scores"
echo "  results/threshold_metrics.csv      Aggregated metrics at each threshold"
echo "  results/plots/confusion_matrix_new.png"
echo "  results/plots/decision_dist_new.png"
echo "  results/plots/latency_chart.png"
echo "  results/plots/risk_score_hist.png"
echo "  results/plots/roc_curve.png"
echo "  results/plots/threshold_comparison.png"
echo "============================================"
