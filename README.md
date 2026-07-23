# AudioShield: Acoustic and Context-Aware Security Middleware for Voice Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 37 passing](https://img.shields.io/badge/tests-37%20passing-brightgreen.svg)](tests/)
[![Dataset: 501 samples](https://img.shields.io/badge/dataset-501%20samples-orange.svg)](data/)

AudioShield is a model-agnostic security middleware that intercepts audio input to Voice-AI systems and performs multi-stage hybrid verification — combining fine-tuned DistilBERT policy classification, MiniLM semantic similarity, and CLAP audio-text cross-modal embeddings — before any content reaches the generative model. It defends against adversarial audio prompt injections, acoustic decoder obfuscation, and unsafe LLM outputs in real time.

> **Paper:** O. Salyan, "AudioShield: An Acoustic and Context-Aware Security Middleware for Voice Large Language Models," CCNCS Project Report, 2026.

---

## Architecture

```
                        ┌───────────────┐
                        │  Audio Input  │
                        └───────┬───────┘
                                │ (Streaming Chunks / Batch WAV)
                    ┌───────────▼───────────┐
                    │   Whisper STT          │ ← openai-whisper / faster-whisper
                    │   (Speech → Text)      │
                    └───────────┬───────────┘
                                │ transcript
                    ┌───────────▼───────────┐
                    │  Stage 1: Input        │
                    │  Policy Check          │ ← DistilBERT (models/risk_classifier/)
                    │  (blocks before LLM)   │
                    └───────────┬───────────┘
                                │ if safe
                    ┌───────────▼───────────┐
                    │  Stage 2: LLM          │ ← Ollama (Llama 3.1) / OpenAI / Stub
                    │  Response Generation   │
                    └───────────┬───────────┘
                                │ raw response
                    ┌───────────▼───────────┐
                    │  Stage 3: Output       │
                    │  Policy Check          │ ← DistilBERT (same classifier)
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Stage 4: Dual Context │
                    │  Verification          │
                    │  ├─ MiniLM text sim    │ ← all-MiniLM-L6-v2
                    │  └─ CLAP audio sim     │ ← laion/clap-htsat-unfused
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Stage 5: Hybrid       │
                    │  Decision Engine       │
                    │                        │
                    │  risk = 0.40·P_policy  │
                    │       + 0.35·(1-sim_t) │
                    │       + 0.25·(1-sim_a) │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼─────┐
        │   ALLOW    │   │  MITIGATE   │   │   BLOCK   │
        │  < 0.40    │   │ 0.40–0.649  │   │  >= 0.65  │
        │            │   │             │   │           │
        │ Deliver    │   │ Sanitize &  │   │ Return    │
        │ Response   │   │ Rewrite     │   │ Fallback  │
        └────────────┘   └─────────────┘   └───────────┘
```

---

## Evaluation Results

Evaluated on 501 audio samples (356 adversarial + 145 benign) at the optimal block threshold of **0.65**:

| Metric | Value |
|:---|:---:|
| **Accuracy** | 1.000 |
| **Precision** | 1.000 |
| **Recall** | 1.000 |
| **F1-Score** | 1.000 |
| **AUC** | 1.000 |
| **False Positives** | 0 |
| **False Negatives** | 0 |

<details>
<summary><strong>Decision Breakdown (501 samples)</strong></summary>

| Class | ALLOW | MITIGATE | BLOCK |
|:---|:---:|:---:|:---:|
| Benign (145) | 6 | 139 | 0 |
| Adversarial (356) | 0 | 277 | 79 |

All 356 adversarial samples were correctly blocked or mitigated. All 145 benign samples passed without false blocks.
</details>

<details>
<summary><strong>Carlini & Wagner Validation</strong></summary>

The system was additionally validated against Carlini & Wagner targeted audio adversarial examples generated against DeepSpeech 0.9.3 in an isolated Docker environment (TensorFlow 1.15). The attack successfully forced DeepSpeech to decode arbitrary target text from perturbed audio. See `Carlini-Audio-Attack/VALIDATION_REPORT.md` for details.
</details>

---

## Quick Start

### Try the examples

```bash
# ALLOW — benign query with low risk score (0.31)
python src/middleware.py --audio examples/allow_example.wav --json

# MITIGATE — benign query with moderate risk score (0.40)
python src/middleware.py --audio examples/mitigate_example.wav --json

# BLOCK — adversarial injection with high risk score (0.76)
python src/middleware.py --audio examples/block_example.wav --json
```

### Launch the interactive dashboard

```bash
streamlit run src/ui.py
```

---

## Installation

### 1. Clone & create environment

```bash
git clone https://github.com/salyanom/AudioShield_Middleware.git
cd AudioShield_Middleware

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download CLAP model weights

```bash
python src/download_clap.py
```

### 3. Train the DistilBERT risk classifier

The trained model weights (`model.safetensors`, ~256 MB) are not included in the repository. Train from the included dataset:

```bash
python src/train_risk_model.py
```

This fine-tunes `distilbert-base-uncased` on `data/risk_dataset.csv` and saves the model to `models/risk_classifier/`.

### 4. Configure LLM provider (optional)

By default, offline evaluation uses the `stub` provider (no LLM needed). To connect to a local Ollama instance:

```bash
export AUDIOSHIELD_LLM_PROVIDER=ollama
export AUDIOSHIELD_LLM_MODEL=llama3.1:8b
export AUDIOSHIELD_LLM_BASE_URL=http://localhost:11434
```

---

## Dataset

The repository includes the complete 501-sample evaluation dataset:

| Directory | Count | Description |
|:---|:---:|:---|
| `data/benign/` | 145 | Clean spoken English queries (TTS-generated) |
| `data/adversarial/` | 356 | Prompt injections, phonetic masking, TSM compression, reverb attacks |
| `data/prompts/` | 2 | Source prompt text files |
| `data/risk_dataset.csv` | — | Training labels for DistilBERT |

**External validation data** (not included — download separately):
- `data/external_benign/`: 120 samples from SpeechCommands, LibriSpeech, Common Voice, AudioCaps
- `data/external/`: Adversarial samples from external sources

```bash
# Download external validation datasets
python src/download_datasets.py
python src/download_public_adversarial.py
```

---

## Reproducing the Evaluation

### Automated (recommended)

```bash
chmod +x run_all.sh
./run_all.sh
```

### Manual step-by-step

```bash
# 1. Train classifier
python src/train_risk_model.py

# 2. Extract features
python src/extract_features.py

# 3. Run threshold evaluation
export AUDIOSHIELD_LLM_PROVIDER=stub
python src/evaluate_thresholds_systematic.py

# 4. Generate plots
python src/analyze_eval_results.py

# 5. Run tests
pytest -v tests/
```

### Generated outputs

| File | Description |
|:---|:---|
| `results/threshold_eval_raw.csv` | Per-sample risk scores for all 501 files |
| `results/threshold_metrics.csv` | Accuracy, Precision, Recall, F1 at each threshold |
| `results/plots/confusion_matrix_new.png` | Confusion matrix at threshold 0.65 |
| `results/plots/decision_dist_new.png` | Decision distribution (ALLOW/MITIGATE/BLOCK) |
| `results/plots/latency_chart.png` | Pipeline latency breakdown |
| `results/plots/risk_score_hist.png` | Risk score distribution by class |
| `results/plots/roc_curve.png` | ROC curve |
| `results/plots/threshold_comparison.png` | Accuracy & F1 vs. threshold |

---

## Repository Structure

```
AudioShield_Middleware/
├── src/                              # Source code
│   ├── middleware.py                 # Core 5-stage pipeline orchestrator
│   ├── api.py                       # FastAPI gateway (/v1/secure)
│   ├── streaming_middleware.py      # Real-time chunk-by-chunk streaming
│   ├── ui.py                        # Streamlit interactive dashboard
│   ├── policy_checker.py            # DistilBERT safety classifier
│   ├── context_verifier.py          # MiniLM + CLAP dual verification
│   ├── audio_embedder.py            # CLAP audio-text embeddings
│   ├── audio_processor.py           # Whisper STT wrapper
│   ├── llm_engine.py                # LLM provider adapters (Ollama/OpenAI/Stub)
│   ├── sanitizer.py                 # PII/credential/command redaction
│   ├── config.py                    # Thresholds, weights, settings
│   ├── logger.py                    # JSONL security event logging
│   ├── train_risk_model.py          # DistilBERT fine-tuning
│   ├── evaluate.py                  # Full evaluation pipeline
│   ├── evaluate_thresholds_systematic.py  # Systematic threshold sweep
│   ├── analyze_eval_results.py      # Plot generation from CSVs
│   ├── generate_adversarial.py      # TTS adversarial audio generator
│   ├── generate_whisper_attacks.py  # Acoustic obfuscation generator
│   ├── optimize_thresholds.py       # Grid search threshold optimizer
│   ├── validate_thresholds.py       # Threshold validation & visualization
│   └── ...                          # Additional utilities
├── data/
│   ├── adversarial/                 # 356 adversarial audio files (.wav)
│   ├── benign/                      # 145 benign audio files (.wav)
│   ├── prompts/                     # Source prompt text files
│   └── risk_dataset.csv             # DistilBERT training labels
├── models/
│   └── risk_classifier/
│       └── training_meta.json       # Model metadata (weights gitignored)
├── features/                        # Extracted audio features
├── results/
│   ├── plots/                       # Final evaluation figures (6 PNGs)
│   ├── threshold_eval_raw.csv       # Per-sample evaluation data
│   ├── threshold_metrics.csv        # Aggregated threshold metrics
│   └── threshold_analysis.csv       # Threshold analysis summary
├── examples/                        # Representative audio samples
│   ├── allow_example.wav            # Benign query → ALLOW
│   ├── mitigate_example.wav         # Benign query → MITIGATE
│   └── block_example.wav            # Adversarial injection → BLOCK
├── tests/
│   └── test_middleware.py           # 37 pytest tests
├── docs/
│   ├── architecture.md              # Technical architecture deep-dive
│   ├── architecture_diagram.png     # System architecture diagram
│   ├── threat_model.md              # Threat model & attack vectors
│   └── project_structure.md         # File reference guide
├── paper/                           # Research paper materials
│   ├── figures/                     # Publication-ready figures
│   └── contribution_notes.md        # Research contribution summary
├── Carlini-Audio-Attack/            # Carlini & Wagner validation experiment
│   ├── attack.py                    # Attack implementation
│   ├── Dockerfile                   # TF 1.15 + DeepSpeech environment
│   ├── VALIDATION_REPORT.md         # Results summary
│   └── tmp/attack-en.csv            # Attack metrics
├── logs/                            # Runtime logs (gitignored)
├── run_all.sh                       # End-to-end reproduction script
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Testing

```bash
pytest -v tests/
```

The test suite covers the middleware pipeline, hybrid decision engine, streaming verification, multi-layer sanitizer, and edge cases.

---

## Citation

```bibtex
@misc{salyan2026audioshield,
  title   = {AudioShield: An Acoustic and Context-Aware Security Middleware
             for Voice Large Language Models},
  author  = {Salyan, Om Jagdish},
  year    = {2026},
  note    = {CCNCS Project Report}
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.