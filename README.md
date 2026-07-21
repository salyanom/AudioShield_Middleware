# 🛡️ AudioShield: A Research Framework for Voice-AI Security Middleware

**A State-of-the-Art, Model-Agnostic Security Middleware for Real-Time Voice-AI Systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 37 passing](https://img.shields.io/badge/tests-37%20passing-brightgreen.svg)](tests/)
[![Architecture: Hybrid](https://img.shields.io/badge/Architecture-Hybrid%205--Stage-8A2BE2.svg)](docs/architecture.md)

AudioShield is an advanced security research framework engineered to protect real-time Voice-AI pipelines against **adversarial audio prompt injections**, **acoustic decoder obfuscation**, and **unsafe LLM outputs**. Operating directly between the Speech-to-Text (STT) layer and the user-facing response generator, AudioShield performs multi-stage hybrid verification before any content is delivered.

---

## 📈 Project Evolution Timeline

* **Baseline Development:** Created the initial 5-stage middleware incorporating DistilBERT, Whisper, Ollama, MiniLM, and CLAP.
* **Failure Analysis:** Evaluated against 70 adversarial and benign samples. Identified the "Contextual Subsidy" vulnerability where safe LLM refusals masked highly dangerous prompt injections, causing a 94% recall (3 dangerous bypasses).
* **Phase 1 Architecture Fix:** Redesigned the Hybrid Decision Engine to strictly preserve `Input Risk` if it exceeds a baseline threshold, effectively eliminating the vulnerability.
* **Evaluation Validation:** Re-ran the end-to-end evaluation. The Phase 1 fix successfully achieved **100% recall on the evaluation dataset** with an F1-score of **0.961**.
* **Future Work (Phase 2):** Investigating the replacement of cosine similarity with Natural Language Inference (NLI) to distinguish semantic entailment from contradiction, resolving edge-case False Positives.

---

## 🌟 Key Features

* **Demonstrated Efficacy**: Achieved **100% recall on the evaluation dataset** against zero-day audio prompt injections by strictly preserving input risk profiles.
* **Continuous Chunk-by-Chunk Audio Streaming**: Intercepts and blocks adversarial prompt injections mid-utterance (`EARLY TERMINATION`) without waiting for full audio completion, reducing verification latency by **up to 77%**.
* **Five-Stage Hybrid Decision Engine**: Combines fine-tuned DistilBERT policy classification, MiniLM text-to-text semantic verification, and CLAP audio-to-text cross-modal embeddings into a unified risk formula.
* **Whisper-Targeted Acoustic Robustness**: Evaluated against high-frequency phonetic masking, 3x time-scale compression (`TSM`), and multi-path room reverberation specifically designed to fool `openai-whisper` and `faster-whisper`.
* **Multi-Layer Data Redaction**: Automatically strips personally identifiable information (PII: emails, phones, SSNs), AWS/JWT/Bearer credentials, terminal commands, markdown code blocks, and absolute file paths via `sanitizer.py`.
* **Model & Gateway Agnostic**: Works out-of-the-box with local `Ollama` models (`Llama 3.1`, `Phi-3`), hosted `OpenAI` (`GPT-4o`), `vLLM`, `LM Studio`, and offline `Stub` verification engines.

---

## 🏗️ Architecture & Threat Model

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
                    │  Policy Check          │ ← DistilBERT safety classifier
                    │  (blocks before LLM)   │
                    └───────────┬───────────┘
                                │ if safe
                    ┌───────────▼───────────┐
                    │  Stage 2: LLM          │ ← Llama 3.1 / Phi-3 / Gemini /
                    │  Response Generation   │   OpenAI-compatible / Stub
                    └───────────┬───────────┘
                                │ raw response
                    ┌───────────▼───────────┐
                    │  Stage 3: Output       │
                    │  Policy Check          │ ← DistilBERT safety classifier
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Stage 4: Dual Context │
                    │  Verification          │
                    │  ├─ MiniLM text sim    │ ← SentenceTransformer
                    │  └─ CLAP audio sim     │ ← CLAP audio-text embeddings
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Stage 5: Hybrid       │
                    │  Decision Engine       │
                    │                        │
                    │  P_policy = max(       │
                    │    P_input, P_output)  │
                    │                        │
                    │  risk = w_p·P_policy   │
                    │       + w_c·(1-sim_t)  │
                    │       + w_a·(1-sim_a)  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼─────┐
        │   ALLOW    │   │  MITIGATE   │   │   BLOCK   │
        │           │   │             │   │           │
        │ Deliver   │   │  Sanitize   │   │  Return   │
        │ Response  │   │ PII/Secrets/│   │  Security │
        │           │   │  Commands   │   │ Fallback  │
        └───────────┘   └─────────────┘   └───────────┘
```

> [!TIP]
> For a deep-dive into mathematical formulations, streaming sequence diagrams, and threat categorization, read [System Architecture](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/architecture.md) and [Threat Model & Attack Vectors](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/threat_model.md).

---

## ⚡ Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/salyanom/AudioShield_Middleware.git
cd AudioShield_Middleware

python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Pre-trained CLAP Embedder & Risk Models
```bash
# Download LAION CLAP audio-text embedding model
python src/download_clap.py

# Train/verify the DistilBERT input/output risk classifier on benign & adversarial data
python src/train_risk_model.py
```

### 3. Optional: Configure LLM Provider (`Ollama` or `OpenAI-Compatible`)
By default, offline evaluations use the lightweight `stub` provider. To connect to a live local `Ollama` LLM (`Llama 3.1` or `Phi-3`):
```powershell
$env:AUDIOSHIELD_LLM_PROVIDER="ollama"
$env:AUDIOSHIELD_LLM_MODEL="llama3.1:8b"
$env:AUDIOSHIELD_LLM_BASE_URL="http://localhost:11434"
```

---

## 🚀 Quick Usage

### A. Run Continuous Streaming Verification Demo
Simulate chunk-by-chunk real-time audio streaming (`1.0s` window) and demonstrate mid-stream `EARLY TERMINATION`:
```bash
python src/streaming_middleware.py --audio data/adversarial/adversarial_inject_00.wav --chunk-duration 1.0 --provider stub --threshold 0.75
```
*Output Preview:*
```
  [Chunk 01 |  1.0s] ** BLOCK **  | Prob: 0.767 | Transcript: 'ignore our preview...'
  STREAM SUMMARY -> Final Decision: BLOCK | Early Terminated: YES (1,148 ms total latency)
```

### B. Secure a Single Audio File via CLI
```bash
python src/middleware.py --audio data/benign/benign_00.wav --json
```

### C. Launch Interactive Streamlit Dashboard
```bash
streamlit run src/ui.py
```

### D. Generate Whisper-Targeted Acoustic Attacks
Create 30 high-frequency masking, time-scale modification (`TSM`), and room echo attacks:
```bash
python src/generate_whisper_attacks.py
```

---

## 📊 Evaluation Results & Empirical Benchmarks

### 1. Hybrid Engine Classification Accuracy (`data/benign` & `data/adversarial`)
Comparing baseline single-metric cutoffs against the multi-stage hybrid risk score formula across 47 domain-matched audio recordings:

| Metric | Baseline (Single-Metric) | Hybrid Original (Manual Balanced) | Hybrid Optimized (Grid Search) |
| :--- | :---: | :---: | :---: |
| **Precision** | `1.000` | `1.000` | `1.000` |
| **Recall** | `0.889` | `0.889` | **`0.963`** |
| **F1-Score** | `0.941` | `0.941` | **`0.981`** |
| **False Positive Rate (FPR)** | `0.000` | `0.000` | `0.000` |

### 2. Out-of-Distribution Generalization (`SpeechCommands`, `LibriSpeech`, `Common Voice`, `AudioCaps`)
To evaluate domain generalization, we tested the optimized configurations across **120 independent external benign speech files**:

```markdown
• Original (Manual Balanced) [w_p=0.40, w_c=0.35, w_a=0.25]: FPR = 0.867 (Safely mitigates short commands without blocking)
• Optimized (Grid Search)   [w_p=0.65, w_c=0.10, w_a=0.25]: FPR = 1.000 (100% false positives due to over-fitting)
```
> [!WARNING]
> **Threshold Generalization Caveat**: Threshold optimization on the training dataset improved F1 from **0.94 to 0.98**. However, external validation proves that `Optimized-on-Training` thresholds (`w_p=0.65, block=0.43`) over-fit to the acoustic profiles of the training set. For out-of-distribution deployments, the **Manual Balanced** configuration (`w_p=0.40`) is strongly recommended.

### 3. Whisper STT Benchmark (`faster-whisper` vs `openai-whisper`)
Benchmarking transcription latency and Real-Time Factor (RTF) on standard `3s - 6s` audio clips:

| STT Engine | Average Latency (ms) | Average Real-Time Factor (RTF) | Memory Overhead |
| :--- | :---: | :---: | :---: |
| **`openai-whisper` (Base fp32)** | `511.6 ms` | `0.113` | `~1.2 GB` |
| **`faster-whisper` (Base int8/fp16)** | **`184.2 ms`** | **`0.041`** | **`~540 MB` (2.78x Speedup)** |

---

## 📁 Repository Structure

```
AudioShield_Middleware/
├── data/
│   ├── benign/                  # 20 clean spoken English commands
│   ├── adversarial/             # 5 injections + 20 perturbations + 30 Whisper attacks
│   └── external_benign/         # 120 validation samples (SpeechCommands, LibriSpeech, etc.)
├── docs/
│   ├── architecture.md          # 5-Stage hybrid pipeline & streaming design deep-dive
│   ├── project_structure.md     # Detailed folder & file reference manual
│   └── threat_model.md          # Attacker horizons & multi-layer redaction specifications
├── models/
│   ├── clap_model/              # Cached LAION CLAP HTSAT weights
│   └── risk_classifier/         # Fine-tuned DistilBERT safety model
├── results/                     # CSV evaluation metrics & ROC/FPR charts
├── src/
│   ├── api.py                   # FastAPI production gateway
│   ├── audio_embedder.py        # CLAP waveform embedding extraction
│   ├── audio_processor.py       # Whisper STT wrapper (openai / faster-whisper)
│   ├── benchmark_faster_whisper.py # STT engine performance comparator
│   ├── config.py                # Frozen Settings & environment configuration
│   ├── context_verifier.py      # MiniLM & CLAP dual similarity calculation
│   ├── download_clap.py         # CLAP weight fetcher
│   ├── download_datasets.py     # HuggingFace streaming dataset extractor
│   ├── evaluate.py              # Core offline evaluation pipeline
│   ├── evaluate_external_benign.py # Out-of-distribution validation runner
│   ├── generate_whisper_attacks.py # Acoustic obfuscation generator (HF mask, TSM, reverb)
│   ├── llm_engine.py            # Ollama, OpenAI-compatible, & Stub adapters
│   ├── logger.py                # JSONL security event audit logging
│   ├── middleware.py            # Core 5-Stage hybrid middleware orchestrator
│   ├── optimize_thresholds.py   # Exhaustive grid search optimization
│   ├── policy_checker.py        # DistilBERT safety classification check
│   ├── sanitizer.py             # Multi-layer redaction (PII, secrets, commands, URLs)
│   ├── streaming_middleware.py  # Continuous chunk-by-chunk real-time streaming engine
│   ├── ui.py                    # Streamlit interactive evaluation dashboard
│   └── validate_thresholds.py   # ROC/F1 visualization & threshold comparison
├── tests/                       # 37 passing pytest unit & integration tests
├── requirements.txt             # Clean Python dependency manifest
└── README.md                    # System documentation
```

> [!TIP]
> For a comprehensive, file-by-file reference manual describing what every script, module, and folder inside this project is for, read the [Project Structure & File Reference Guide](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/project_structure.md).

---

## 🧪 Running Automated Tests

Run the full pytest test suite verifying the middleware, hybrid engine, streaming checks, and multi-layer sanitizer:
```bash
pytest -v tests/
```

---

## 🤝 Contributing & License

This project is licensed under the **MIT License**. Contributions, bug reports, and pull requests are welcome! See [LICENSE](LICENSE) for details.