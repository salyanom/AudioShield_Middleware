"""Runtime configuration for AudioShield."""

from dataclasses import dataclass
import os


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, default))

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # LLM backend
    llm_provider:    str        = os.getenv("AUDIOSHIELD_LLM_PROVIDER",  "ollama")
    llm_model:       str        = os.getenv("AUDIOSHIELD_LLM_MODEL",     "phi3")
    llm_base_url:    str        = os.getenv("AUDIOSHIELD_LLM_BASE_URL",  "http://localhost:11434")
    llm_api_key:     str | None = os.getenv("AUDIOSHIELD_LLM_API_KEY")
    request_timeout: float      = _float_env("AUDIOSHIELD_REQUEST_TIMEOUT", 120.0)

    # Legacy single thresholds (kept for backwards compatibility)
    input_risk_threshold:  float = _float_env("AUDIOSHIELD_INPUT_RISK_THRESHOLD",  0.80)
    output_risk_threshold: float = _float_env("AUDIOSHIELD_OUTPUT_RISK_THRESHOLD", 0.50)
    context_threshold:     float = _float_env("AUDIOSHIELD_CONTEXT_THRESHOLD",     0.25)

    # Hybrid decision engine weights (must sum to 1.0).
    # Re-tuned via ablation on the phi3 backend: the original 0.40/0.35/0.25
    # split gave CLAP too much influence relative to how well it actually
    # discriminates benign vs. adversarial audio here (80% false-mitigate
    # rate on clean audio). This split cut that to 25% while only costing
    # 1/10 recall on the whisper_attack.py batch (F1 0.556 -> 0.750,
    # accuracy 0.467 -> 0.800). See RESULTS.md, Finding 7, for the full
    # comparison and the acknowledgment that this is a better-found point,
    # not an exhaustively-tuned optimum.
    weight_policy:   float = _float_env("AUDIOSHIELD_WEIGHT_POLICY",  0.45)
    weight_context:  float = _float_env("AUDIOSHIELD_WEIGHT_CONTEXT", 0.45)
    weight_audio:    float = _float_env("AUDIOSHIELD_WEIGHT_AUDIO",   0.10)

    # Hybrid risk score thresholds
    block_threshold:    float = _float_env("AUDIOSHIELD_BLOCK_THRESHOLD",    0.60)
    mitigate_threshold: float = _float_env("AUDIOSHIELD_MITIGATE_THRESHOLD", 0.40)

    # CLAP audio embeddings
    use_clap:   bool = _bool_env("AUDIOSHIELD_USE_CLAP", True)
    clap_model: str  = os.getenv("AUDIOSHIELD_CLAP_MODEL", "laion/clap-htsat-unfused")

    # Logging
    log_path:           str = os.getenv("AUDIOSHIELD_LOG_PATH",
                                        "logs/security_events.jsonl")
    mitigation_message: str = os.getenv(
        "AUDIOSHIELD_MITIGATION_MESSAGE",
        "I couldn't safely return the generated answer. Please rephrase your request.",
    )


settings = Settings()