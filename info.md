# AudioShield: Context-Aware Middleware for Voice-AI Security

## Project Title

**Defending Voice-AI Pipelines: A Context-Aware Middleware for Real-Time Detection and Mitigation of Adversarial Audio Prompt Injection**

---

# Project Overview

Voice-enabled AI systems are increasingly being integrated into assistants, smart devices, customer support systems, and enterprise workflows. However, these systems are vulnerable to adversarial audio attacks and prompt injection attempts that can manipulate downstream Large Language Models (LLMs) into generating unsafe or malicious responses.

This project aims to design and implement a context-aware middleware layer that sits between the Speech-to-Text (STT) system and the final user-facing response. The middleware continuously analyzes transcripts and generated responses to detect, block, and mitigate adversarial behavior before harmful content reaches the user.

---

# Project Objectives

The primary objectives are:

* Detect adversarial audio prompt injection attacks.
* Verify contextual consistency between audio transcripts and generated responses.
* Identify unsafe, malicious, or policy-violating outputs.
* Block or mitigate suspicious responses.
* Log security events for auditing and analysis.
* Improve the robustness of Voice-AI pipelines.

---

# System Architecture

Audio Input
↓
Whisper Speech-to-Text
↓
Transcript
↓
Response Generation Layer (LLM)
↓
Middleware Layer
├── Context Verification Module
├── Policy Checking Module
├── Decision Engine
└── Logging Module
↓
ALLOW / BLOCK / MITIGATE
↓
Final User Response

---

# Current Implementation Status

## Completed

### Speech-to-Text Pipeline

* Integrated OpenAI Whisper.
* Successfully transcribes WAV and MP3 files.
* FFmpeg configured for audio processing.

### Audio Feature Extraction

Implemented extraction of:

* MFCC Mean
* MFCC Standard Deviation
* Zero Crossing Rate (ZCR)
* Spectral Centroid
* Spectral Rolloff
* RMS Energy

### Context Verification

Implemented semantic consistency verification using:

* Sentence Transformers
* all-MiniLM-L6-v2
* Cosine Similarity

Used to compare:

* Transcript
* Generated Response

### Policy Checking

Implemented response risk analysis using:

* TF-IDF Vectorization
* Logistic Regression Classifier

Current output:

* Safe
* Unsafe

### Decision Engine

Combines:

* Context Verification Score
* Policy Risk Score

Produces:

* ALLOW
* BLOCK

### Logging

Implemented JSONL logging for:

* Timestamp
* Audio file path
* Transcript
* Generated response
* Similarity score
* Risk score
* Final decision

### Adversarial Audio Generation

Implemented:

* Speed Perturbation
* Volume Perturbation

Generated samples:

* test_speed.mp3
* test_louder.mp3

### Evaluation Framework

Implemented robustness evaluation using:

* Transcript generation
* Embedding comparison
* Similarity computation

Results exported to:

evaluation_results.csv

---

# Technologies Used

## Speech Recognition

### OpenAI Whisper

Purpose:

* Audio transcription
* Speech-to-text conversion

Dependencies:

* openai-whisper
* ffmpeg

---

## Audio Processing

### Librosa

Purpose:

* Feature extraction
* Audio analysis

Features:

* MFCC
* Spectral Features
* Energy Features

---

## Semantic Analysis

### Sentence Transformers

Model:

all-MiniLM-L6-v2

Purpose:

* Context verification
* Semantic similarity analysis

---

## Machine Learning

### Scikit-Learn

Algorithms:

* Logistic Regression
* TF-IDF Vectorizer

Purpose:

* Risk classification
* Policy violation detection

---

## Data Handling

Libraries:

* Pandas
* NumPy

Purpose:

* Dataset management
* Feature processing

---

# Folder Structure

CCNCSP1

data/
├── benign/
├── adversarial/
└── risk_dataset.csv

features/
├── benign_features.csv
└── dataset.csv

models/
├── risk_model.pkl
└── risk_vectorizer.pkl

src/
├── audio_processor.py
├── build_dataset.py
├── context_verifier.py
├── evaluate.py
├── extract_features.py
├── generate_adversarial.py
├── llm_engine.py
├── logger.py
├── middleware.py
├── policy_checker.py
├── train_detector.py
├── train_risk_model.py
└── utils.py

logs.jsonl

evaluation_results.csv

archidia-v1.png

---

# File Descriptions

## audio_processor.py

Purpose:

* Audio loading
* Speech transcription

Main Function:

transcribe_audio(audio_path)

Output:

Transcript text

---

## extract_features.py

Purpose:

Extract acoustic features from audio files.

Output:

features/benign_features.csv

---

## build_dataset.py

Purpose:

Prepare ML-ready datasets from extracted audio features.

Output:

features/dataset.csv

---

## train_detector.py

Purpose:

Train audio attack detection model using extracted features.

Current Result:

Accuracy = 1.0

Note:

Current dataset is small and requires expansion.

---

## context_verifier.py

Purpose:

Verify semantic consistency between transcript and response.

Method:

* Embedding generation
* Cosine similarity

Output:

Similarity score

---

## policy_checker.py

Purpose:

Detect unsafe responses.

Current Method:

* TF-IDF
* Logistic Regression

Output:

Risk score

Future Goal:

Replace with transformer-based safety classifier.

---

## train_risk_model.py

Purpose:

Train response safety classifier.

Input:

data/risk_dataset.csv

Output:

models/risk_model.pkl
models/risk_vectorizer.pkl

---

## middleware.py

Purpose:

Main integration pipeline.

Workflow:

Audio
↓
Transcript
↓
Generated Response
↓
Context Verification
↓
Policy Checking
↓
Decision
↓
Logging

Output:

ALLOW or BLOCK

---

## logger.py

Purpose:

Store middleware events.

Output:

logs.jsonl

---

## llm_engine.py

Purpose:

Generate responses from transcripts.

Current Version:

Simple summarization logic.

Future Version:

* Phi-3 Mini
* Gemini API
* Other lightweight LLMs

---

## generate_adversarial.py

Purpose:

Generate perturbed audio samples.

Current Attacks:

* Speed change
* Volume increase

Future Attacks:

* Noise injection
* Echo
* Reverb
* Pitch shift

---

## evaluate.py

Purpose:

Measure robustness against adversarial audio.

Process:

1. Transcribe benign audio.
2. Transcribe adversarial audio.
3. Generate embeddings.
4. Compute similarity.

Output:

evaluation_results.csv

---

# Mitigation Strategy

The middleware performs both detection and mitigation.

## Current Mitigation

When unsafe content is detected:

* Response is blocked.
* Event is logged.
* Unsafe output is prevented from reaching the user.

Example:

Generated Response:

Download malware from https://evil.com

Result:

BLOCK

## Future Mitigation

Planned enhancements:

* Response sanitization
* Safe response regeneration
* Human-in-the-loop verification
* Adaptive risk scoring

Example:

Unsafe Response:

Download malware from https://evil.com

Sanitized Output:

The generated response contained potentially unsafe instructions and was removed.

---

# Experimental Results

## Benign Audio

Similarity Score:

~0.81

Policy Score:

0

Decision:

ALLOW

---

## Malicious Response Injection

Response:

Download malware from https://evil.com

Detected Indicators:

* URL
* Download instruction
* Malware-related content

Decision:

BLOCK

---

## Adversarial Audio Evaluation

### test_speed.mp3

Similarity:

0.961

### test_louder.mp3

Similarity:

0.975

These results indicate that moderate speed and volume perturbations do not significantly alter transcript semantics.

---

# Current Limitations

## Small Risk Dataset

Current dataset contains only a limited number of examples.

Target:

100+ safe responses
100+ unsafe responses

---

## Rule-Based Components

Current implementation partially relies on manually defined security patterns.

Future work will replace these with learned safety models.

---

## Limited Adversarial Attacks

Currently implemented:

* Speed perturbation
* Volume perturbation

Future attacks:

* Noise
* Echo
* Reverb
* Pitch shifting
* Carlini-style adversarial perturbations

---

## Simplified Response Generator

Current implementation uses a basic response generation approach.

Future work will integrate a real LLM.

---

# Future Work

## Short-Term

* Expand risk classification dataset.
* Generate more adversarial audio samples.
* Improve evaluation metrics.
* Produce graphs and result visualizations.

## Medium-Term

Replace current policy checker with:

* DistilBERT
* RoBERTa
* DeBERTa

for semantic safety classification.

## Long-Term

Integrate:

* Phi-3 Mini
* Gemini API
* Llama Guard
* ShieldGemma

to create a fully context-aware secure Voice-AI pipeline.

---

# Weekly Progress Summary

## Week 1

Completed:

* Literature Survey
* Threat Analysis
* Existing Defense Study
* Architecture Design
* Project Scope Definition

---

## Week 2

Completed:

* Whisper Integration
* Audio Feature Extraction
* Context Verification Module
* Policy Checking Module
* Decision Engine
* Logging System
* Adversarial Audio Generation
* Evaluation Pipeline
* End-to-End Middleware Integration

---

# Current Project Status

Architecture: Completed

Implementation: Completed (Prototype Version)

Middleware Integration: Completed

Detection Pipeline: Completed

Mitigation Pipeline: Partially Implemented

Evaluation: Ongoing

Dataset Expansion: Ongoing

Project Phase:

**Experimentation, Evaluation, and Security Enhancement**
