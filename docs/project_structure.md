# 📁 AudioShield Repository & Project Structure Reference

This document provides a comprehensive, directory-by-directory and file-by-file reference manual explaining the purpose, architecture role, and exact functionality of every folder and script within the **AudioShield** repository.

---

## 🗺️ High-Level Directory Tree

```
AudioShield_Middleware/
├── 📁 data/                      # Spoken audio datasets (`.wav`) for training, evaluation, & robustness testing
│   ├── 📁 benign/                # 20 clean, domain-matched spoken English queries
│   ├── 📁 adversarial/           # 55 prompt injections, signal perturbations, & Whisper-targeted attacks
│   └── 📁 external_benign/       # 120 out-of-distribution validation samples (SpeechCommands, LibriSpeech, etc.)
├── 📁 docs/                      # System documentation, threat models, & engineering deep-dives
├── 📁 models/                    # Local cache for pre-trained weights & fine-tuned safety classifiers
│   ├── 📁 clap_model/            # LAION CLAP cross-modal audio-text embedding model
│   └── 📁 risk_classifier/       # Fine-tuned DistilBERT sequence safety classifier
├── 📁 results/                   # Evaluation CSV tables, JSONL security logs, & benchmark charts (`.png`)
│   └── 📁 external_benign/       # Out-of-distribution threshold comparison logs (`Original vs Grid`)
├── 📁 src/                       # Core system source code (26 Python scripts & modules)
└── 📁 tests/                     # Automated unit and integration test suite (37 passing tests)
```

---

## 📄 Root Directory Files

| File Name | Description & Purpose |
| :--- | :--- |
| **[`README.md`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/README.md)** | **Primary Project Documentation**: Entry point featuring badges, ASCII system diagrams, quick-start installation commands, usage examples, and empirical evaluation results. |
| **[`RESULTS.md`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/RESULTS.md)** | **Exhaustive Evaluation Report**: Documents baseline single-metric vs hybrid multi-stage accuracy (`98.1% F1`), grid search threshold optimization, out-of-distribution False Positive Rates (`Original vs Grid`), latency breakdowns (`511ms vs 184ms`), and ROC curves. |
| **[`requirements.txt`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/requirements.txt)** | **Python Dependency Manifest**: Lists all required third-party packages including `torch`, `transformers`, `openai-whisper`, `faster-whisper`, `sentence-transformers`, `streamlit`, `fastapi`, `soundfile`, `scipy`, and `pydantic-settings`. |
| **[`LICENSE`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/LICENSE)** | **Open-Source License**: Standard MIT License governing the repository and code usage. |
| **[`.gitignore`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/.gitignore)** | **Git Exclusion Rules**: Prevents tracking of heavy model weights (`models/`), virtual environments (`venv/`), `__pycache__`, large dataset downloads (`data/external_benign/`), and temporary scratch logs. |

---

## 💻 `src/` — Source Code & Core Modules

The `src/` directory contains 26 Python scripts that power the 5-Stage Hybrid Pipeline, continuous audio streaming, multi-provider LLM integration, and dataset generation.

### 1. Core Pipeline & Middleware Orchestration
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/middleware.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/middleware.py)** | **Master Pipeline Orchestrator**: Implements `AudioShieldMiddleware` and the 5-Stage Hybrid verification pipeline (`process_audio` and `process_transcript`). Coordinates Whisper STT transcription, Stage 1 input check, Stage 2 LLM generation, Stage 3 output check, Stage 4 dual similarity verification, and Stage 5 hybrid risk scoring (`ALLOW`, `MITIGATE`, `BLOCK`). |
| **[`src/streaming_middleware.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/streaming_middleware.py)** | **Continuous Audio Streaming Engine**: Implements `StreamingAudioShield` to process live audio streams chunk-by-chunk (`1.0s` rolling window). Runs partial Whisper STT and real-time DistilBERT policy checks to trigger mid-stream **`EARLY TERMINATION (BLOCK)`** before audio finishes (`77% latency reduction`). Includes `--threshold` and `--provider` CLI flags. |
| **[`src/config.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/config.py)** | **System Configuration**: Uses `pydantic-settings` (`Settings`) to define global environment settings, default risk thresholds (`input_risk_threshold=0.80`, `mitigate_threshold=0.40`, `block_threshold=0.60`), hybrid weights (`w_p=0.40, w_c=0.35, w_a=0.25`), and fallback mitigation messages. |
| **[`src/utils.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/utils.py)** | **Timing & Helper Utilities**: Provides utility wrappers, specifically `_timed(call, *args)` which precisely records millisecond execution latency for each stage of the security pipeline. |

### 2. Verification Engines & Policy Classifiers (Stages 1, 3, & 4)
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/policy_checker.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/policy_checker.py)** | **DistilBERT Safety Classifier (Stage 1 & Stage 3)**: Loads the fine-tuned sequence classifier from `models/risk_classifier/` to evaluate incoming transcripts and generated LLM responses. Returns unsafe probability ($P_{\text{unsafe}}$) and binary classification (`0=safe, 1=unsafe`). |
| **[`src/context_verifier.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/context_verifier.py)** | **Dual Context Verifier (Stage 4)**: Computes text-to-text semantic similarity (`sim_t` via `all-MiniLM-L6-v2`) between the prompt and response, and cross-modal audio-text similarity (`sim_a` via `CLAP`) between the raw `.wav` audio and response text. |
| **[`src/audio_embedder.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/audio_embedder.py)** | **CLAP Audio Feature Extractor**: Loads the LAION CLAP (`laion/clap-htsat-unfused`) model on CUDA or CPU, extracts 512-dimensional L2-normalized audio embeddings from `.wav` waveforms, and computes cross-modal cosine alignment with response embeddings. |
| **[`src/sanitizer.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/sanitizer.py)** | **Multi-Layer Deterministic Redaction**: Applied when Stage 5 outputs `MITIGATE`. Redacts PII (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_SSN]`), credentials (`AWS keys`, `JWT tokens`, `API keys`), shell commands (`sudo`, `rm -rf`, `curl`), code blocks, URLs, and absolute file paths. |

### 3. Speech-to-Text (STT) & LLM Engines
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/audio_processor.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/audio_processor.py)** | **Unified Whisper STT Wrapper**: Dynamically loads either `openai-whisper` (`fp32/fp16`) or `faster-whisper` (`CTranslate2 int8/fp16`) to transcribe audio into text with minimal latency. Exposes `get_whisper_model()` and `transcribe_audio()`. |
| **[`src/llm_engine.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/llm_engine.py)** | **Multi-Backend LLM Engine**: Implements `LLMEngine` capable of interfacing with local `Ollama` servers (`Llama 3.1:8b`, `Phi-3`), hosted `OpenAI` (`GPT-4o`), `vLLM` / `LM Studio` endpoints (`http://localhost:1234/v1`), or offline `Stub` generation (`[Stub response for...]`). |

### 4. Evaluation, Optimization, & Benchmarking Scripts
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/evaluate.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/evaluate.py)** | **Core Batch Evaluation Runner**: Runs AudioShield across all files in `data/benign/` and `data/adversarial/`. Logs precision, recall, F1-score, latency metrics, and similarity distributions to `results/evaluation_results.csv`. |
| **[`src/evaluate_deterministic.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/evaluate_deterministic.py)** | **Deterministic Baseline Evaluator**: Evaluates the pipeline using fixed, pre-defined response outputs (`StubProvider`) to ensure reproducible baseline metric calculations without requiring live LLM network calls. |
| **[`src/evaluate_external_benign.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/evaluate_external_benign.py)** | **Out-of-Distribution Threshold Validator**: Evaluates `Original (Manual Balanced)` vs `Optimized (Grid Search)` thresholds across 120 external speech files (`SpeechCommands`, `LibriSpeech`, etc.) to empirically prove generalizability and over-fitting behavior. |
| **[`src/evaluate_external_adversarial.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/evaluate_external_adversarial.py)** | **External Adversarial Evaluator**: Tests AudioShield against simulated or external adversarial prompt injections and audio jailbreak datasets. |
| **[`src/optimize_thresholds.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/optimize_thresholds.py)** | **Exhaustive Grid Search Optimizer**: Iterates through `414,000 combinations` of policy weights ($w_p, w_c, w_a$) and decision cutoffs ($\tau_{\text{mit}}, \tau_{\text{blk}}$) to find the configuration maximizing domain F1-score (`0.981 F1`). |
| **[`src/validate_thresholds.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/validate_thresholds.py)** | **Threshold Validation & Plot Generator**: Validates optimal cutoffs and generates visualization charts including `roc_curve.png`, `threshold_comparison_roc.png`, and `threshold_comparison_f1.png`. |
| **[`src/benchmark_faster_whisper.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/benchmark_faster_whisper.py)** | **STT Latency & RTF Benchmark**: Measures transcription latency (`ms`) and Real-Time Factor (`RTF`) comparing `openai-whisper` vs `faster-whisper` across evaluation recordings (`2.78x speedup`). |

### 5. Dataset Generation & Attack Synthesis
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/generate_adversarial.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/generate_adversarial.py)** | **Signal Perturbation Generator**: Applies 20 acoustic distortions (`Gaussian noise SNR 10-20dB`, `echo 50-200ms`, `pitch shifting ±2-4 semitones`, `time stretching 0.7x-1.5x`) to prompt injections to test Stage 4 CLAP stability. |
| **[`src/generate_whisper_attacks.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/generate_whisper_attacks.py)** | **Whisper Acoustic Attack Generator**: Synthesizes 30 specialized acoustic obfuscation attacks (`data/adversarial/whisper_attack_*`) including high-frequency phonetic masking (`hfmask`), `2.5x` fast time compression (`fasttsm`), and room reverberation (`reverb`). |
| **[`src/download_datasets.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/download_datasets.py)** | **HuggingFace Instant Dataset Streamer**: Streams 30 evaluation clips each from `SpeechCommands v2`, `LibriSpeech test-clean`, `Common Voice / VoxPopuli`, and `AudioCaps` (`streaming=True`) without downloading multi-gigabyte `.tar.gz` archives. |
| **[`src/generate_dataset.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/generate_dataset.py)** | **Base Audio Synthesizer**: Uses `pyttsx3` text-to-speech to generate synthetic `.wav` files for `data/benign/` and `data/adversarial/` when real audio recordings are unavailable. |
| **[`src/extract_features.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/extract_features.py)** | **Acoustic Feature Extractor**: Extracts Mel-Frequency Cepstral Coefficients (`MFCCs`) and spectrogram features from `.wav` files for diagnostic audio analysis. |
| **[`src/build_dataset.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/build_dataset.py)** | **Dataset Setup Utility**: Quick helper utility that creates and initializes directory hierarchies for audio datasets. |

### 6. Production Interfaces, Training, & Logging
| File Name | Description & Purpose |
| :--- | :--- |
| **[`src/api.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/api.py)** | **FastAPI Production Gateway**: Exposes REST API endpoints (`POST /process`, `POST /verify`, and `GET /health`) for deploying AudioShield as a microservice in enterprise cloud pipelines. |
| **[`src/ui.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/ui.py)** | **Streamlit Interactive Dashboard**: Web UI allowing users to upload audio files (`.wav/.mp3`) or record live microphone speech, visualize STT transcription, view similarity score dials, and inspect hybrid decision reasoning. |
| **[`src/logger.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/logger.py)** | **Audit Event Logger**: Appends tamper-evident JSONL records (`request_id`, `probabilities`, `similarities`, `decision`, `latency`) to `results/security_audit.log` for compliance and post-incident forensics. |
| **[`src/train_risk_model.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/train_risk_model.py)** | **DistilBERT Fine-Tuning Script**: Fine-tunes `distilbert-base-uncased` on safe queries vs prompt injections/jailbreaks, saving weights (`model.safetensors`) to `models/risk_classifier/`. |
| **[`src/download_clap.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/download_clap.py)** | **CLAP Weight Downloader**: Fetches and caches the `laion/clap-htsat-unfused` cross-modal audio-text embedding model from HuggingFace directly to `models/clap_model/`. |
| **[`src/train_detector.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/train_detector.py)** | **Auxiliary Detector Trainer**: Small helper utility script for training secondary linear or baseline anomaly classifiers. |
| **[`src/test_policy.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/src/test_policy.py)** | **CLI Policy Testing Utility**: Quick command-line script to test random text strings against the `DistilBERT` risk classifier. |

---

## 🎧 `data/` — Spoken Audio Datasets

```
data/
├── benign/                      # 20 clean spoken queries (benign_00.wav - benign_19.wav)
├── adversarial/                 # 55 adversarial audio files across 3 attack classes
│   ├── adversarial_inject_*.wav # Direct spoken prompt injections (00 - 04)
│   ├── adversarial_perturb_*.wav# Signal distortions: noise, echo, pitch, time stretch (00 - 19)
│   └── whisper_attack_*.wav     # Whisper acoustic attacks: hfmask, fasttsm, reverb (30 files)
└── external_benign/             # 120 independent out-of-distribution evaluation files
    ├── speechcommands/          # 30 random clips from Google SpeechCommands v2
    ├── librispeech/             # 30 random clips from LibriSpeech test-clean
    ├── commonvoice/             # 30 random clips from Common Voice / VoxPopuli
    └── audiocaps/               # 30 random clips from Clotho / AudioCaps
```

---

## 📚 `docs/` — System Documentation

| File Name | Description & Purpose |
| :--- | :--- |
| **[`docs/architecture.md`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/architecture.md)** | **Engineering Architecture Deep-Dive**: Explains the 5-Stage Hybrid verification flow with sequence diagrams (`Mermaid`), mathematical formulations (`Risk Score`), continuous streaming timing charts, and STT benchmark data. |
| **[`docs/threat_model.md`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/threat_model.md)** | **Threat Model & Redaction Specifications**: Details attacker horizons, capabilities (`Black-box audio vs Acoustic obfuscation`), attack family breakdowns, and the 6 deterministic redaction layers in `sanitizer.py`. |
| **[`docs/project_structure.md`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/docs/project_structure.md)** | **Directory & File Reference Guide**: This exact reference manual. |

---

## 🧠 `models/` — Cached Model Weights

| Folder Path | Contents & Purpose |
| :--- | :--- |
| **`models/clap_model/`** | Contains the cached weights, configuration (`config.json`), and tokenizer files for `laion/clap-htsat-unfused`. Used during Stage 4 to extract cross-modal audio-text similarity. |
| **`models/risk_classifier/`** | Contains the fine-tuned `DistilBertForSequenceClassification` model weights (`model.safetensors`, `config.json`, `vocab.txt`) used during Stage 1 and Stage 3 to assign unsafe probabilities (`0.0 - 1.0`). |

---

## 📈 `results/` — Metrics, Logs, & Charts

| File Name / Folder | Description & Purpose |
| :--- | :--- |
| **[`results/evaluation_results.csv`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/results/evaluation_results.csv)** | Master evaluation metrics table containing `transcript`, `sim_t`, `sim_a`, `risk_score`, `unsafe_prob`, `decision`, and `latency_ms` across the 47 baseline audio recordings. |
| **`results/external_benign/`** | Contains `external_benign_original_manual.csv` (`0.867 FPR`) and `external_benign_optimized_grid.csv` (`1.000 FPR`), proving how threshold optimization over-fits to training data vs out-of-distribution generalizability. |
| **[`results/whisper_benchmark_comparison.csv`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/results/whisper_benchmark_comparison.csv)** | Raw benchmark table recording file duration, latency (`ms`), and Real-Time Factor (`RTF`) across `openai-whisper` and `faster-whisper`. |
| **[`results/security_audit.log`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/results/security_audit.log)** | JSON Lines (`.jsonl`) audit log recording every security decision processed by the middleware or streaming engine. |
| **`results/*.png` Charts** | Generated visualization charts: `roc_curve.png`, `similarity_by_attack.png`, `latency_boxplot.png`, `unsafe_prob_distribution.png`, `decision_distribution.png`, `threshold_comparison_roc.png`, and `threshold_comparison_f1.png`. |

---

## 🧪 `tests/` — Automated Pytest Suite

| File Name | Description & Purpose |
| :--- | :--- |
| **[`tests/test_middleware.py`](file:///c:/Users/Om%20Jagdish%20Salyan/Downloads/CCNCSP1/tests/test_middleware.py)** | **Comprehensive Test Suite (37 Passing Tests)**: Exercises every core module and edge case across AudioShield using `pytest`: <br>• **Risk Computation (`TestComputeRiskScore`)**: Verifies weight redistribution when `audio_sim` (CLAP) is `None` or available.<br>• **Decision Routing (`TestAllowPath`, `TestBlockAtInput`, `TestBlockAtOutput`, `TestMitigatePath`)**: Verifies correct `ALLOW`, `MITIGATE`, and `BLOCK` decisions.<br>• **Sanitizer Redaction (`TestSanitizerEngine`)**: Tests exact regex matching and stripping for AWS keys, emails, SSNs, phone numbers, JWT tokens, markdown code blocks, URLs, and shell commands (`sudo`, `rm -rf`, `curl`).<br>• **Smart Mitigation (`TestSmartMitigationFeatures`)**: Verifies LLM prompt rewriting and safety re-checks.<br>• **Structure & Edge Cases (`TestPipelineResultStructure`, `TestEdgeCases`)**: Validates UUID request IDs, millisecond timing keys, and exception handling for empty/whitespace transcripts. |
