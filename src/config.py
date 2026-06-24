"""Runtime configuration for AudioShield.

Values can be overridden with environment variables so the same middleware can
sit in front of a local Ollama model or an OpenAI-compatible production API.
"""

from dataclasses import dataclass
import os


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("AUDIOSHIELD_LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("AUDIOSHIELD_LLM_MODEL", "llama3.1:8b")
    llm_base_url: str = os.getenv("AUDIOSHIELD_LLM_BASE_URL", "http://localhost:11434")
    llm_api_key: str | None = os.getenv("AUDIOSHIELD_LLM_API_KEY")
    request_timeout: float = _float_env("AUDIOSHIELD_REQUEST_TIMEOUT", 120.0)

    input_risk_threshold: float = _float_env("AUDIOSHIELD_INPUT_RISK_THRESHOLD", 0.80)
    output_risk_threshold: float = _float_env("AUDIOSHIELD_OUTPUT_RISK_THRESHOLD", 0.50)
    context_threshold: float = _float_env("AUDIOSHIELD_CONTEXT_THRESHOLD", 0.25)

    log_path: str = os.getenv("AUDIOSHIELD_LOG_PATH", "logs/security_events.jsonl")
    mitigation_message: str = os.getenv(
        "AUDIOSHIELD_MITIGATION_MESSAGE",
        "I couldn't safely return the generated answer. Please rephrase your request.",
    )


settings = Settings()
