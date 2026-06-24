"""
middleware.py

AudioShield full pipeline with hybrid decision engine.

Decision engine:
    risk_score = w_policy  * unsafe_prob
               + w_context * (1 - transcript_similarity)
               + w_audio   * (1 - audio_similarity)   [if CLAP available]

    BLOCK    if risk_score >= block_threshold
    MITIGATE if risk_score >= mitigate_threshold
    ALLOW    otherwise
"""

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
    request_id:                str
    decision:                  str
    transcript:                str
    response:                  str
    raw_response:              str | None
    input_unsafe_probability:  float
    output_unsafe_probability: float | None
    context_similarity:        float | None
    audio_similarity:          float | None
    risk_score:                float | None
    reason:                    str
    provider:                  str | None
    model:                     str | None
    source:                    dict
    latency_ms:                dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            try:
                if v is not None and hasattr(v, "item"):
                    d[k] = v.item()
            except Exception:
                pass
        return d


def _timed(call: Callable, *args, **kwargs):
    started = perf_counter()
    value   = call(*args, **kwargs)
    return value, round((perf_counter() - started) * 1000, 2)


def _compute_risk_score(
    unsafe_prob:    float,
    transcript_sim: float,
    audio_sim:      float | None,
    cfg:            Settings,
) -> float:
    w_policy  = cfg.weight_policy
    w_context = cfg.weight_context
    w_audio   = cfg.weight_audio

    if audio_sim is None:
        total    = w_policy + w_context
        w_policy  = w_policy  / total
        w_context = w_context / total
        w_audio   = 0.0

    score = (
        w_policy  * unsafe_prob +
        w_context * (1.0 - transcript_sim) +
        (w_audio  * (1.0 - audio_sim) if audio_sim is not None else 0.0)
    )
    return round(float(score), 4)


def process_transcript(
    transcript:        str,
    *,
    audio_path:        str | None = None,
    provider:          LLMProvider | None = None,
    supplied_response: str | None = None,
    cfg:               Settings = settings,
    policy_check:      Callable | None = None,
    context_check:     Callable | None = None,
    log_path:          str | None = None,
    source:            dict | None = None,
    initial_latency:   dict[str, float] | None = None,
) -> PipelineResult:
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    if policy_check is None:
        from policy_checker import check_policy
        policy_check = check_policy
    if context_check is None:
        from context_verifier import verify_context_full
        context_check = verify_context_full

    request_id      = str(uuid4())
    latency: dict   = dict(initial_latency or {})
    active_provider = provider

    # Stage 1: Input policy
    (input_pred, input_details), latency["input_policy"] = _timed(
        policy_check, transcript, cfg.input_risk_threshold
    )
    input_prob = float(input_details["unsafe_prob"])

    if input_pred:
        result = PipelineResult(
            request_id=request_id,
            decision="BLOCK",
            transcript=transcript,
            response=cfg.mitigation_message,
            raw_response=None,
            input_unsafe_probability=input_prob,
            output_unsafe_probability=None,
            context_similarity=None,
            audio_similarity=None,
            risk_score=None,
            reason="Input policy check rejected the transcript before model generation.",
            provider=None, model=None,
            source=source or {"type": "text"},
            latency_ms=latency,
        )
        log_security_event(result.to_dict(), log_path)
        return result

    # Stage 2: LLM generation
    if supplied_response is None:
        active_provider = active_provider or build_provider(cfg)
        raw_response, latency["generation"] = _timed(
            active_provider.generate, transcript
        )
    else:
        raw_response          = supplied_response
        latency["generation"] = 0.0

    # Stage 3: Output policy
    (output_pred, output_details), latency["output_policy"] = _timed(
        policy_check, raw_response, cfg.output_risk_threshold
    )
    output_prob = float(output_details["unsafe_prob"])

    # Stage 4: Dual context verification (MiniLM + CLAP)
    ctx_result, latency["context"] = _timed(
        context_check, transcript, raw_response, audio_path
    )
    transcript_sim = float(ctx_result["transcript_similarity"])
    audio_sim      = ctx_result["audio_similarity"]
    if audio_sim is not None:
        audio_sim = float(audio_sim)

    # Stage 5: Hybrid decision engine
    risk_score = _compute_risk_score(
        unsafe_prob    = output_prob,
        transcript_sim = transcript_sim,
        audio_sim      = audio_sim,
        cfg            = cfg,
    )

    if risk_score >= cfg.block_threshold:
        decision = "BLOCK"
        reason   = (f"Hybrid risk score {risk_score:.3f} >= block threshold "
                    f"{cfg.block_threshold:.2f}.")
    elif risk_score >= cfg.mitigate_threshold:
        decision = "MITIGATE"
        reason   = (f"Hybrid risk score {risk_score:.3f} >= mitigate threshold "
                    f"{cfg.mitigate_threshold:.2f}.")
    else:
        decision = "ALLOW"
        reason   = (f"Hybrid risk score {risk_score:.3f} within safe bounds.")

    final_response = raw_response if decision == "ALLOW" else cfg.mitigation_message

    result = PipelineResult(
        request_id=request_id,
        decision=decision,
        transcript=transcript,
        response=final_response,
        raw_response=raw_response,
        input_unsafe_probability=input_prob,
        output_unsafe_probability=output_prob,
        context_similarity=transcript_sim,
        audio_similarity=audio_sim,
        risk_score=risk_score,
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
        audio_path     = audio_path,
        source         = {"type": "audio", "path": str(Path(audio_path))},
        initial_latency= {"transcription": transcription_ms},
        **kwargs,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the AudioShield middleware")
    src    = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="Audio file to transcribe and secure")
    src.add_argument("--text",  help="Existing transcript to secure directly")
    parser.add_argument("--response", help="Existing model response (skips generation)")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def main():
    args   = _parse_args()
    result = (
        process_audio(args.audio, supplied_response=args.response)
        if args.audio
        else process_transcript(args.text, supplied_response=args.response)
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"\nDecision:    {result.decision}")
    print(f"Risk score:  {result.risk_score}")
    print(f"Sim (text):  {result.context_similarity:.4f}" if result.context_similarity else "")
    print(f"Sim (audio): {result.audio_similarity:.4f}" if result.audio_similarity else "  Sim (audio): N/A")
    print(f"Reason:      {result.reason}")
    print(f"Response:    {result.response}")
    print(f"Request:     {result.request_id}")


if __name__ == "__main__":
    main()