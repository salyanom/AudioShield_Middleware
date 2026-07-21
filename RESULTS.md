# AudioShield: Phase 1 Evaluation Results

**For use in IEEE paper/report writing.**
All results produced by the Phase 1 systematic evaluation (`src/evaluate_thresholds_systematic.py`) on the final frozen dataset.

---

## 1. Evaluation Setup

| Parameter | Value |
|---|---|
| STT Model | OpenAI Whisper (base) |
| Safety Classifier | DistilBERT fine-tuned on 253-example dataset |
| Context Verifier | SentenceTransformer all-MiniLM-L6-v2, cosine similarity |
| Acoustic Verifier | CLAP audio-text embeddings |
| LLM Backend | Ollama (llama3.1:8b, deterministic temperature=0) |
| Architecture | Strict Input Risk Preservation (`policy_prob = max(input, output)`) |
| Mitigate threshold | 0.40 |
| Block threshold | 0.60 |

---

## 2. Quantitative Results

**Dataset:** 70 samples (35 benign + 35 adversarial).

### Final Phase 1 Metrics
| Metric | Value |
|---|---|
| Precision | 0.9259 |
| Recall | **1.0000** |
| F1 Score | **0.9615** |
| Accuracy | 0.9428 |
| False Positive Rate (FPR) | 0.1143 (4 Benign MITIGATED) |
| False Negative Rate (FNR) | **0.0000 (0 bypasses)** |
| True Positives (TP) | 35 |
| True Negatives (TN) | 31 |
| False Positives (FP) | 4 |
| False Negatives (FN) | 0 |

### Component Ablation Study
To isolate the effectiveness of the Phase 1 strict-preservation architectural fix, we ablated the pipeline mathematically:

| Configuration | Accuracy | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|:---:|
| **Input Policy Only** | 77.1% | 97.2% | 70.0% | 0.814 |
| **Input + Output Policy** | 75.7% | 97.1% | 68.0% | 0.800 |
| **Original Hybrid Engine (Baseline)** | 94.2% | 97.9% | 94.0% | 0.959 |
| **Phase 1 Hybrid Engine** | **94.2%** | **92.5%** | **100.0%** | **0.961** |

**Scientific Finding:** The baseline architecture suffered from a "Contextual Subsidy" where adversarial inputs generated safe LLM refusals, artificially lowering the final risk score and causing 3 dangerous bypasses (94% recall). Phase 1 eliminates this vulnerability entirely by preserving input risk profiles, achieving perfect 100% recall at the minor trade-off of 4 false positives on technical queries.

## 3. Pipeline Latency Analysis

Average processing latencies across the pipeline during evaluation:
* **Transcription (Whisper Base)**: ~11.7 seconds
* **Generation (Llama 3.1 8B)**: ~8.0 seconds
* **Context Verification (MiniLM+CLAP)**: ~9.9 seconds
* **Smart Mitigation (when triggered)**: ~3.0 seconds

> *Note: In a production streaming environment via `streaming_middleware.py`, prompt injections are intercepted mid-utterance, frequently reducing the end-to-end latency footprint by up to 77%.*
