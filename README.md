# AudioShield

**A Model-Agnostic Security Middleware for Voice-AI Systems**

AudioShield is a lightweight security middleware that protects Voice-AI pipelines against adversarial audio prompt injection attacks and unsafe LLM outputs. It operates between the Speech-to-Text (STT) layer and the user-facing response, performing multi-stage verification before any content is delivered.

---

## Architecture

```
Audio Input
     ↓
Whisper STT
     ↓
Transcript
     ↓
┌─────────────────────────┐
│  Input Policy Check     │  ← DistilBERT safety classifier
│  (blocks before LLM)   │
└─────────────────────────┘
     ↓ (if safe)
LLM Response Generation
(Llama 3.1 / Phi-3 / Gemini / Stub)
     ↓
┌─────────────────────────┐
│  Context Verification   │  ← SentenceTransformer cosine similarity
│  (semantic consistency) │
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│  Output Policy Check    │  ← DistilBERT safety classifier
│  (response safety)      │
└─────────────────────────┘
     ↓
Decision Engine
     ↓
ALLOW / MITIGATE / BLOCK
     ↓
JSONL Audit Log
```

---

## Key Results

| Metric | Value |
|---|---|
| Precision | **1.000** |
| Recall | **0.444** |
| F1 Score | **0.615** |
| ROC-AUC | **0.906** |
| False Positive Rate | **0.000** |

Evaluated on 20 benign + 27 adversarial samples (20 prompt injection attacks + 7 signal perturbations).

---

## Project Structure

```
CCNCSP1/
├── data/
│   ├── benign/                  # 20 TTS-generated benign WAV files
│   ├── adversarial/             # 20 prompt injection + 7 signal perturbation WAVs
│   ├── external/
│   │   ├── benign/              # Original WAVs from external adversarial dataset
│   │   └── adversarial/        # adv-medium2medium + yes2right variants
│   └── risk_dataset.csv        # 253-row safe/unsafe training dataset
├── features/
│   ├── benign_features.csv
│   └── dataset.csv
├── logs/
│   └── security_events.jsonl   # JSONL audit log (git-ignored)
├── models/
│   └── risk_classifier/        # Fine-tuned DistilBERT weights
├── results/
│   ├── evaluation_results.csv
│   ├── similarity_by_attack.png
│   ├── decision_distribution.png
│   ├── unsafe_prob_distribution.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── latency_boxplot.png
├── src/
│   ├── api.py                  # FastAPI gateway
│   ├── audio_processor.py      # Whisper STT
│   ├── config.py               # Environment-variable-driven settings
│   ├── context_verifier.py     # SentenceTransformer cosine similarity
│   ├── evaluate.py             # Full pipeline evaluation + graphs
│   ├── evaluate_external_adversarial.py
│   ├── generate_adversarial.py # 6-attack audio perturbation generator
│   ├── generate_dataset.py     # TTS dataset generator (pyttsx3)
│   ├── llm_engine.py           # Multi-backend LLM (Ollama/Gemini/OpenAI/stub)
│   ├── logger.py               # JSONL audit logger
│   ├── middleware.py           # Main pipeline orchestrator
│   ├── policy_checker.py       # DistilBERT inference
│   ├── train_risk_model.py     # DistilBERT fine-tuning pipeline
│   └── ui.py                   # Streamlit operator dashboard
├── tests/
│   └── test_middleware.py
└── requirements.txt
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA GPU recommended for Whisper + DistilBERT inference
- [FFmpeg](https://ffmpeg.org/) on PATH (for audio processing)

### Install

```bash
pip install -r requirements.txt
```

### Train the safety classifier

```bash
python src/train_risk_model.py
```

Trains DistilBERT on `data/risk_dataset.csv` (253 examples, balanced).
Saves model to `models/risk_classifier/`.

Training results: val_accuracy = 100% at epoch 2 (from 97.56% at epoch 1).

---

## Usage

### Run the middleware on a single audio file

```bash
python src/middleware.py --audio data/benign/benign_00.wav
```

### Run full evaluation

```bash
python src/evaluate.py \
  --benign data/benign \
  --adversarial data/adversarial \
  --out results/
```

Generates 6 graphs and prints precision/recall/F1/ROC-AUC/FPR/FNR.

### Generate adversarial audio samples

```bash
python src/generate_adversarial.py --input data/benign/benign_00.wav --fmt wav
```

Attacks: `speed`, `volume`, `noise`, `pitch`, `echo`, `reverb`.

### Generate TTS evaluation dataset

```bash
pip install pyttsx3
python src/generate_dataset.py --benign 20 --adversarial 20
```

### Start the API gateway

```bash
uvicorn src.api:app --port 8000
```

### Start the Streamlit UI

```bash
python -m streamlit run src/ui.py
```

---

## Configuration

All settings are environment-variable driven via `src/config.py`. No hardcoded values.

| Variable | Default | Description |
|---|---|---|
| `AUDIOSHIELD_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `gemini`, `stub` |
| `AUDIOSHIELD_LLM_MODEL` | `llama3.1:8b` | Model name |
| `AUDIOSHIELD_LLM_BASE_URL` | `http://localhost:11434` | API base URL |
| `AUDIOSHIELD_LLM_API_KEY` | — | API key (Gemini/OpenAI) |
| `AUDIOSHIELD_INPUT_RISK_THRESHOLD` | `0.80` | Block transcript above this unsafe_prob |
| `AUDIOSHIELD_OUTPUT_RISK_THRESHOLD` | `0.50` | Mitigate/block response above this |
| `AUDIOSHIELD_CONTEXT_THRESHOLD` | `0.25` | Block if similarity below this |
| `AUDIOSHIELD_LOG_PATH` | `logs/security_events.jsonl` | Audit log path |

### Using with Ollama (local Llama 3.1)

```bash
ollama pull llama3.1:8b
python src/middleware.py --audio data/benign/benign_00.wav
```

### Using with Gemini API

```bash
export AUDIOSHIELD_LLM_PROVIDER=gemini
export AUDIOSHIELD_LLM_API_KEY=your_key_here
python src/middleware.py --audio data/benign/benign_00.wav
```

### Offline testing (stub backend)

```bash
export LLM_BACKEND=stub
python src/evaluate.py --benign data/benign --adversarial data/adversarial --out results/
```

---

## Attack Types

| Attack | Method | Detected by |
|---|---|---|
| Prompt injection | Spoken override instructions | Policy checker (unsafe_prob) |
| Noise injection | AWGN at configurable SNR | Context verifier (similarity) |
| Echo | Single-tap delayed reflection | Context verifier |
| Reverb | Synthetic room impulse response | Context verifier |
| Pitch shift | Resampling-based pitch change | Context verifier |
| Speed change | Time-stretch without pitch shift | Context verifier |
| Volume increase | Gain amplification | Context verifier |

---

## Carlini & Wagner Investigation

The C&W gradient-based attack (targeted at DeepSpeech 0.9.3) was investigated as a potential evaluation source. The original Docker image (`nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04`) has been removed from Docker Hub, making the build impossible without manual environment reconstruction.

An external adversarial dataset (`data/external/`) was used instead to evaluate attack transferability. **Key finding:** C&W-style attacks targeting DeepSpeech do not transfer to Whisper — WER = 0.0 and cosine similarity ≈ 1.0 across all original/adversarial pairs, confirming STT model specificity of gradient-based attacks.

---

## Technology Stack

| Component | Technology |
|---|---|
| Speech-to-Text | OpenAI Whisper |
| Safety Classifier | DistilBERT (fine-tuned) |
| Context Verification | SentenceTransformer (all-MiniLM-L6-v2) |
| LLM Backend | Llama 3.1 / Phi-3 / Gemini / OpenAI-compatible |
| API Gateway | FastAPI |
| UI | Streamlit |
| Audio Processing | Librosa, SciPy, pydub |
| Logging | JSONL |

---

## Repository

[github.com/salyanom/AudioShield_Middleware](https://github.com/salyanom/AudioShield_Middleware)