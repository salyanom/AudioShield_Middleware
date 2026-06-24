"""AudioShield's connected, end-to-end security pipeline."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from config import Settings, settings
from llm_engine import LLMProvider, build_provider
from logger import log_security_event


@dataclass
class PipelineResult:
    request_id: str
    decision: str
    transcript: str
    response: str
    raw_response: str | None
    input_unsafe_probability: float
    output_unsafe_probability: float | None
    context_similarity: float | None
    reason: str
    provider: str | None
    model: str | None
    source: dict
    latency_ms: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _timed(call: Callable, *args, **kwargs):
    started = perf_counter()
    value = call(*args, **kwargs)
    return value, round((perf_counter() - started) * 1000, 2)


def process_transcript(
    transcript: str,
    *,
    provider: LLMProvider | None = None,
    supplied_response: str | None = None,
    cfg: Settings = settings,
    policy_check: Callable | None = None,
    context_check: Callable | None = None,
    log_path: str | None = None,
    source: dict | None = None,
    initial_latency: dict[str, float] | None = None,
) -> PipelineResult:
    """Secure a transcript before and after response generation.

    A dangerous input is BLOCKed before the LLM is called. A dangerous or
    contextually detached output is MITIGATEd by replacing it with a safe
    fallback. This two-stage design is the core real-world middleware boundary.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    if policy_check is None:
        from policy_checker import check_policy
        policy_check = check_policy
    if context_check is None:
        from context_verifier import verify_context
        context_check = verify_context

    request_id = str(uuid4())
    latency: dict[str, float] = dict(initial_latency or {})
    active_provider = provider

    (input_prediction, input_details), latency["input_policy"] = _timed(
        policy_check, transcript, cfg.input_risk_threshold
    )
    input_probability = float(input_details["unsafe_prob"])

    if input_prediction:
        result = PipelineResult(
            request_id=request_id,
            decision="BLOCK",
            transcript=transcript,
            response=cfg.mitigation_message,
            raw_response=None,
            input_unsafe_probability=input_probability,
            output_unsafe_probability=None,
            context_similarity=None,
            reason="Input policy check rejected the transcript before model generation.",
            provider=None,
            model=None,
            source=source or {"type": "text"},
            latency_ms=latency,
        )
        log_security_event(result.to_dict(), log_path)
        return result

    if supplied_response is None:
        active_provider = active_provider or build_provider(cfg)
        raw_response, latency["generation"] = _timed(
            active_provider.generate, transcript
        )
    else:
        raw_response = supplied_response
        latency["generation"] = 0.0

    (output_prediction, output_details), latency["output_policy"] = _timed(
        policy_check, raw_response, cfg.output_risk_threshold
    )
    similarity, latency["context"] = _timed(
        context_check, transcript, raw_response
    )
    output_probability = float(output_details["unsafe_prob"])
    similarity = float(similarity)

    if output_prediction:
        decision = "MITIGATE"
        reason = "Unsafe model output was replaced before reaching the user."
    elif similarity < cfg.context_threshold:
        decision = "MITIGATE"
        reason = "Contextually inconsistent model output was replaced."
    else:
        decision = "ALLOW"
        reason = "Input and output passed policy and context checks."

    final_response = raw_response if decision == "ALLOW" else cfg.mitigation_message
    result = PipelineResult(
        request_id=request_id,
        decision=decision,
        transcript=transcript,
        response=final_response,
        raw_response=raw_response,
        input_unsafe_probability=input_probability,
        output_unsafe_probability=output_probability,
        context_similarity=similarity,
        reason=reason,
        provider=getattr(active_provider, "name", None),
        model=getattr(active_provider, "model", None),
        source=source or {"type": "text"},
        latency_ms=latency,
    )
    log_security_event(result.to_dict(), log_path)
    return result


def process_audio(audio_path: str, **kwargs) -> PipelineResult:
    from audio_processor import transcribe_audio

    transcript, transcription_ms = _timed(transcribe_audio, audio_path)
    return process_transcript(
        transcript,
        source={"type": "audio", "path": str(Path(audio_path))},
        initial_latency={"transcription": transcription_ms},
        **kwargs,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the AudioShield middleware")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Audio file to transcribe and secure")
    source.add_argument("--text", help="Existing transcript to secure")
    parser.add_argument(
        "--response",
        help="Optional existing model response; skips generation for gateway integration/testing",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main():
    args = _parse_args()
    result = (
        process_audio(args.audio, supplied_response=args.response)
        if args.audio
        else process_transcript(args.text, supplied_response=args.response)
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"\nDecision: {result.decision}")
    print(f"Reason:   {result.reason}")
    print(f"Response: {result.response}")
    print(f"Request:  {result.request_id}")


if __name__ == "__main__":
    main()
