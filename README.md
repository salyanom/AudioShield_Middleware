# AudioShield

## Defending Voice-AI Pipelines: A Context-Aware Middleware for Real-Time Detection of Adversarial Audio Prompt Injection

AudioShield is a security middleware designed to protect Voice-AI systems against adversarial audio prompt injection attacks. The middleware analyzes audio inputs, verifies response relevance, detects unsafe model outputs, and decides whether the generated response should be allowed or blocked.

---

## Features

### Speech-to-Text Transcription

* OpenAI Whisper
* Converts audio input into text transcripts

### Context Verification

* Sentence Transformers
* Semantic similarity analysis between transcript and generated response
* Detects context deviation and prompt injection attempts

### Policy Checking

* Fine-tuned DistilBERT classifier
* Classifies responses as Safe or Unsafe
* Produces risk probabilities for decision making

### Decision Engine

* Combines:

  * Context similarity score
  * Policy classification result
* Produces:

  * ALLOW
  * BLOCK

### Logging

* Stores middleware decisions and metadata
* JSONL-based logging format

### Adversarial Audio Evaluation

* Speed perturbation attacks
* Volume perturbation attacks
* Robustness evaluation against modified audio samples

---

## Project Structure

```text
CCNCSP1/
│
├── data/
│   ├── benign/
│   ├── adversarial/
│   └── risk_dataset.csv
│
├── features/
│
├── models/
│   └── risk_classifier/
│
├── src/
│   ├── audio_processor.py
│   ├── context_verifier.py
│   ├── policy_checker.py
│   ├── middleware.py
│   ├── logger.py
│   ├── evaluate.py
│   ├── generate_adversarial.py
│   ├── train_risk_model.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Architecture

```text
Audio Input
      │
      ▼
Whisper STT
      │
      ▼
Transcript
      │
      ▼
LLM Response
      │
      ▼
Middleware
 ├── Context Verification
 ├── Policy Checking
 ├── Decision Engine
 └── Logger
      │
      ▼
ALLOW / BLOCK
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd CCNCSP1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install FFmpeg

Whisper requires FFmpeg.

Windows:

```powershell
winget install Gyan.FFmpeg
```

Verify:

```powershell
ffmpeg -version
```

---

## Train Policy Classifier

The DistilBERT classifier must be trained before running the middleware.

```bash
python src/train_risk_model.py
```

This generates:

```text
models/risk_classifier/model.safetensors
```

Note:
The trained model file is excluded from Git because it exceeds GitHub's file size limit.

---

## Running the Middleware

```bash
python src/middleware.py
```

Example input:

```text
Audio File Path:
data/benign/test.mp3
```

Output:

```text
Transcript:
...

Generated Response:
...

========== REPORT ==========

Similarity Score  : 0.81
Unsafe Probability: 0.23
Policy Score      : 0
Decision          : ALLOW

============================
```

---

## Generating Adversarial Audio

```bash
python src/generate_adversarial.py
```

Current attacks:

* Speed perturbation
* Volume perturbation

---

## Evaluating Robustness

```bash
python src/evaluate.py
```

Produces:

```text
evaluation_results.csv
```

containing similarity scores between benign and adversarial transcripts.

---

## Technologies Used

* Python
* OpenAI Whisper
* Sentence Transformers
* DistilBERT
* Hugging Face Transformers
* PyTorch
* Scikit-learn
* Pandas
* NumPy
* Pydub

---

## Current Status

### Implemented

* Whisper transcription
* Context verification
* DistilBERT policy classifier
* Decision engine
* Logging system
* Adversarial audio generation
* Evaluation pipeline

### Planned

* Noise attacks
* Pitch-shift attacks
* Echo/Reverb attacks
* Response mitigation layer
* Expanded attack dataset

---

## Authors

* Om Jagdish Salyan
* Siddhartha Aakash Rao

### Guide

Dr. Vinodha K.

Summer Internship 2026
Centre for Cognitive Neural Computing Systems (CCNCS)
PES University
