# AudioShield Middleware - Project Structure

The AudioShield Middleware repository is organized to separate core production logic from lifecycle and evaluation scripts. 

## Directory Structure

```text
AudioShield_Middleware/
├── data/                      # Evaluation datasets (benign, adversarial, and external)
├── docs/                      # Architectural documentation, diagrams, and threat models
├── features/                  # Extracted feature CSVs for model training
├── logs/                      # System and security event logs
├── models/                    # DistilBERT safety classifier weights (git-ignored if large)
├── results/                   # Evaluation results, CSVs, and generated plots
├── src/                       # Core Middleware Logic & Active Pipeline Scripts
├── tests/                     # Unit and integration tests
├── .gitignore                 
├── PROJECT_STRUCTURE.md       # Directory structure documentation
├── README.md                  
├── RESULTS.md                 
└── requirements.txt           
```

## `src/` Architecture Overview

The `src/` directory contains all Python scripts. Because of a strict **Repository Freeze Policy** leading up to submission, active scripts, obsolete scripts, and training scripts currently reside together. 

### 1. Core Middleware (Production Runtime)
These files implement the live streaming defense gateway.
* `api.py`: FastAPI server for production deployment.
* `middleware.py`: The core pipeline orchestration (STT → DistilBERT → Llama → MiniLM/CLAP).
* `streaming_middleware.py`: Continuous stream-processing variant of the middleware.
* `policy_checker.py`: Wraps DistilBERT for input/output policy evaluation.
* `context_verifier.py`: Wraps MiniLM and CLAP for semantic and acoustic verification.
* `audio_processor.py`: Wraps Whisper for Speech-to-Text.
* `llm_engine.py`: Wraps Ollama for LLM generation.
* `audio_embedder.py`: Utility for extracting CLAP audio embeddings.
* `sanitizer.py`: Fallback safety response generator.
* `logger.py`: Standardized logging module.
* `utils.py`: Helper functions.
* `config.py`: Thresholds and model configurations.
* `ui.py`: Streamlit dashboard for operator monitoring.

### 2. Active Evaluation Pipeline
These scripts represent the final systematic evaluation methodology used for Phase 1 Validation.
* `evaluate_thresholds_systematic.py`: Runs the complete pipeline against all 70 samples.
* `analyze_eval_results.py`: Computes metrics and generates publication-ready plots.

### 3. Lifecycle & Training Scripts (Deferred Refactoring)
These scripts are part of the project lifecycle but are not executed during runtime inference.
* `build_dataset.py`, `generate_dataset.py`: Utilities to build custom testing datasets.
* `download_datasets.py`, `download_clap.py`: Stream and cache HuggingFace datasets/models.
* `extract_features.py`: Feature extraction for custom risk models.
* `generate_adversarial.py`, `generate_whisper_attacks.py`: Automated adversarial audio generation.
* `train_detector.py`, `train_risk_model.py`: Training routines for custom classifiers.

### 4. Legacy, Diagnostic, & Benchmarking Scripts (Deferred Refactoring)
These scripts have been superseded or are intended for one-off tasks.
* `evaluate.py`, `evaluate_deterministic.py`, `evaluate_external_*.py`: Obsolete evaluation scripts replaced by `evaluate_thresholds_systematic.py`.
* `optimize_thresholds.py`, `validate_thresholds.py`: Early hyperparameter sweeps.
* `benchmark_faster_whisper.py`: One-off STT latency benchmark.
* `test_policy.py`: Quick verification script.
* `recompute_scores.py`: Experimental script used during Phase 1 for fast-track math evaluation.
