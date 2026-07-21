# AudioShield: IEEE Paper Contribution Outline

**Working Title:** AudioShield: A Model-Agnostic, Hybrid Security Middleware for Defending Real-Time Voice-AI Systems Against Contextual Prompt Injections

---

## 1. Research Problem
Modern Voice-AI pipelines are increasingly vulnerable to adversarial audio attacks, including zero-day prompt injections, high-frequency phonetic masking, and acoustic decoder obfuscation. While significant research exists on securing text-based Large Language Models (LLMs), Voice-AI systems introduce an uncontrolled acoustic vector. When an adversarial prompt bypasses the Speech-to-Text (STT) layer, standard text-based LLM safeguards often fail to identify the malicious context of the spoken instruction. Existing defenses typically rely on monolithic, model-specific fine-tuning or rigid single-metric thresholds that fail to generalize across diverse operational environments.

## 2. Identified Limitation: The Contextual Subsidy Vulnerability
During baseline evaluation of our proposed 5-Stage Hybrid Decision Engine, we identified a critical architectural vulnerability termed the **"Contextual Subsidy."** The baseline engine computed the final risk score as a weighted sum of the maximum predicted policy risk (Input vs Output) and the dual-context similarity metrics (Semantic Text + Acoustic Audio).

However, highly adversarial prompt injections often bypassed the DistilBERT input classifier with moderately high risk scores (e.g., $P_{\text{input}} = 0.70$) while provoking the LLM to generate a safe, polite refusal (e.g., $P_{\text{output}} = 0.20$). Because the LLM response was technically safe and its embedding matched the refusal context, the output risk effectively "subsidized" the input threat. The hybrid engine discarded the severe input warning, yielding a final risk score below the mitigation threshold. This structural flaw resulted in a 94% recall rate and 3 successful bypasses on our evaluation dataset.

## 3. Proposed Architectural Improvement
To eliminate the Contextual Subsidy vulnerability, we redesigned the Hybrid Decision Engine (Phase 1) to enforce a **Strict Input Risk Preservation** policy. The core modification ensures that if the input risk exceeds a baseline threat threshold, it is strictly preserved as the foundational policy threat, regardless of the LLM's output behavior:

$$P_{\text{policy}} = \begin{cases} P_{\text{input}} & \text{if } P_{\text{input}} \ge 0.60 \\ P_{\text{output}} & \text{otherwise} \end{cases}$$

$$\mathcal{R} = w_p \cdot P_{\text{policy}} + w_c \cdot (1 - \text{sim}_t) + w_a \cdot (1 - \text{sim}_a)$$

This non-linear preservation guarantees that any audio prompt flagged as suspicious directly translates its threat weight into the final decision formula, eliminating the LLM's ability to artificially mask an injection attempt.

## 4. Experimental Validation
We evaluated the AudioShield Phase 1 architecture against a robust dataset of 70 samples (35 benign technical queries, 35 adversarial prompt injections and signal perturbations) passing through `openai-whisper` and `llama3.1:8b`.

**Key Results:**
* **Recall:** Achieved 100% recall (0 False Negatives) against all zero-day prompt injections.
* **Precision:** Maintained high precision (92.5%), proving the preservation logic does not catastrophically collapse benign utility.
* **F1-Score:** 0.961 overall performance metric.
* **Latency:** In streaming deployment (`streaming_middleware.py`), the mid-utterance early termination capabilities reduced verification latency by up to 77% compared to post-generation verification.

## 5. Limitations
The strict input risk preservation inherently increases sensitivity. On our evaluation dataset, this resulted in 4 False Positives—highly technical benign queries (e.g., coding instructions) that triggered baseline suspicion in DistilBERT and were mitigated. The current contextual verification module (MiniLM Cosine Similarity) calculates geometric distance in vector space but cannot inherently distinguish between semantic entailment (a valid answer) and contradiction (a polite refusal) if they share similar technical vocabulary.

## 6. Future Work (Phase 2)
Future research will address the remaining False Positives by replacing the geometric cosine similarity function with a Natural Language Inference (NLI) model (e.g., DeBERTa/RoBERTa MNLI). An NLI model can explicitly classify the relationship between the user's transcript and the LLM's response as Entailment, Contradiction, or Neutral. This will allow the Hybrid Engine to mathematically distinguish a safe refusal from a hallucinated or adversarial compliance, bridging the final gap in context-aware Voice-AI security.
