"""
streaming_middleware.py

Continuous Audio Stream Middleware for real-time voice AI pipelines.
Simulates or handles chunk-by-chunk audio streaming:
  Chunk 1 -> Chunk 2 -> Chunk 3 ...
  
At each incoming chunk:
  1. Appends audio samples to a rolling buffer.
  2. Runs incremental partial STT (Speech-To-Text).
  3. Evaluates the partial transcript against the Input Security Policy (DistilBERT).
  4. If a prompt injection, jailbreak, or policy violation is detected mid-stream,
     it triggers EARLY TERMINATION (BLOCK) instantly without waiting for the
     remainder of the audio stream to finish.

Usage:
    python src/streaming_middleware.py --audio data/adversarial/adversarial_inject_00.wav --chunk-duration 1.0
"""

import argparse
import time
from dataclasses import dataclass, field
import soundfile as sf
import numpy as np

from config import Settings, settings
from logger import log_security_event
from policy_checker import check_policy as policy_check
from audio_processor import get_whisper_model


@dataclass
class StreamChunkResult:
    chunk_index: int
    duration_processed: float
    partial_transcript: str
    input_unsafe_prob: float
    is_violation: bool
    decision: str
    reason: str
    latency_ms: float


@dataclass
class StreamingPipelineResult:
    request_id: str
    audio_path: str
    total_duration: float
    chunk_duration: float
    chunks_processed: int
    early_terminated: bool
    final_decision: str
    final_transcript: str
    final_response: str | None
    chunk_history: list[StreamChunkResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "audio_path": self.audio_path,
            "total_duration": self.total_duration,
            "chunk_duration": self.chunk_duration,
            "chunks_processed": self.chunks_processed,
            "early_terminated": self.early_terminated,
            "final_decision": self.final_decision,
            "final_transcript": self.final_transcript,
            "final_response": self.final_response,
            "total_latency_ms": self.total_latency_ms,
            "history": [
                {
                    "chunk": c.chunk_index,
                    "duration": c.duration_processed,
                    "transcript": c.partial_transcript,
                    "unsafe_prob": c.input_unsafe_prob,
                    "decision": c.decision,
                    "latency_ms": c.latency_ms,
                }
                for c in self.chunk_history
            ]
        }


class StreamingAudioShield:
    """Continuously monitors incoming audio chunks and blocks threats mid-stream."""

    def __init__(self, cfg: Settings = settings, chunk_duration_sec: float = 1.0):
        self.cfg = cfg
        self.chunk_duration = chunk_duration_sec
        self.whisper_model, self.whisper_engine = get_whisper_model(cfg.whisper_engine)

    def process_audio_stream(self, audio_path: str, request_id: str | None = None) -> StreamingPipelineResult:
        import uuid
        if not request_id:
            request_id = str(uuid.uuid4())

        start_total = time.perf_counter()
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_samples = len(audio)
        total_duration = total_samples / sr
        chunk_samples = int(self.chunk_duration * sr)

        chunk_history = []
        accumulated_audio = np.array([], dtype=np.float32)
        early_terminated = False
        final_decision = "ALLOW"
        final_transcript = ""
        final_response = None
        chunks_processed = 0

        # Process stream chunk by chunk
        for start_idx in range(0, total_samples, chunk_samples):
            chunks_processed += 1
            chunk_start = time.perf_counter()
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = audio[start_idx:end_idx].astype(np.float32)
            accumulated_audio = np.concatenate((accumulated_audio, chunk))
            duration_so_far = len(accumulated_audio) / sr

            # 1. Incremental transcription on accumulated buffer
            if self.whisper_engine == "faster-whisper":
                segments, _ = self.whisper_model.transcribe(accumulated_audio, beam_size=self.cfg.whisper_beam_size)
                partial_transcript = " ".join(s.text for s in segments).strip()
            else:
                result = self.whisper_model.transcribe(
                    accumulated_audio,
                    beam_size=self.cfg.whisper_beam_size,
                    best_of=self.cfg.whisper_best_of,
                )
                partial_transcript = result.get("text", "").strip()

            final_transcript = partial_transcript

            # 2. Real-time input policy check
            start_policy = time.perf_counter()
            is_violation, details = policy_check(
                partial_transcript, self.cfg.input_risk_threshold
            )
            unsafe_prob = float(details.get("unsafe_prob", 0.0))
            chunk_latency = (time.perf_counter() - chunk_start) * 1000.0

            if is_violation:
                decision = "BLOCK"
                reason = f"Mid-stream policy check detected threat (unsafe_prob={unsafe_prob:.3f} >= {self.cfg.input_risk_threshold}) at {duration_so_far:.1f}s."
                early_terminated = True
                final_decision = "BLOCK"
                final_response = self.cfg.mitigation_message
                chunk_history.append(StreamChunkResult(
                    chunk_index=chunks_processed,
                    duration_processed=duration_so_far,
                    partial_transcript=partial_transcript,
                    input_unsafe_prob=unsafe_prob,
                    is_violation=True,
                    decision=decision,
                    reason=reason,
                    latency_ms=chunk_latency,
                ))
                break
            else:
                decision = "CONTINUE"
                reason = f"Partial transcript safe at {duration_so_far:.1f}s."
                chunk_history.append(StreamChunkResult(
                    chunk_index=chunks_processed,
                    duration_processed=duration_so_far,
                    partial_transcript=partial_transcript,
                    input_unsafe_prob=unsafe_prob,
                    is_violation=False,
                    decision=decision,
                    reason=reason,
                    latency_ms=chunk_latency,
                ))

        total_latency = (time.perf_counter() - start_total) * 1000.0

        # If stream finished without early termination, run full output/hybrid stage
        if not early_terminated:
            from middleware import process_transcript
            res = process_transcript(
                final_transcript,
                audio_path=audio_path,
                cfg=self.cfg,
                source={"type": "streaming_audio", "path": audio_path},
            )
            final_decision = res.decision
            final_response = res.response

        result = StreamingPipelineResult(
            request_id=request_id,
            audio_path=audio_path,
            total_duration=total_duration,
            chunk_duration=self.chunk_duration,
            chunks_processed=chunks_processed,
            early_terminated=early_terminated,
            final_decision=final_decision,
            final_transcript=final_transcript,
            final_response=final_response,
            chunk_history=chunk_history,
            total_latency_ms=total_latency,
        )
        log_security_event(result.to_dict(), self.cfg.log_path)
        return result


def main():
    parser = argparse.ArgumentParser(description="Run AudioShield continuous streaming middleware simulation")
    parser.add_argument("--audio", required=True, help="Path to audio file to stream")
    parser.add_argument("--chunk-duration", type=float, default=1.0, help="Duration of each audio chunk in seconds (default: 1.0s)")
    parser.add_argument("--provider", type=str, default="stub", help="LLM backend provider (default: stub)")
    parser.add_argument("--threshold", type=float, default=None, help="Input risk cutoff threshold (default: settings.input_risk_threshold)")
    args = parser.parse_args()

    cfg_kwargs = {"llm_provider": args.provider}
    if args.threshold is not None:
        cfg_kwargs["input_risk_threshold"] = args.threshold
    cfg = Settings(**cfg_kwargs)

    print(f"\n========================================================================")
    print(f"  AUDIOSHIELD CONTINUOUS STREAMING MIDDLEWARE DEMO")
    print(f"========================================================================")
    print(f"  Stream Input : {args.audio}")
    print(f"  Chunk Window : {args.chunk_duration} seconds")
    print(f"  LLM Provider : {args.provider}")
    print(f"  Risk Cutoff  : {cfg.input_risk_threshold}")
    print(f"========================================================================\n")

    streamer = StreamingAudioShield(cfg=cfg, chunk_duration_sec=args.chunk_duration)
    res = streamer.process_audio_stream(args.audio)

    for c in res.chunk_history:
        status_color = "** BLOCK **" if c.decision == "BLOCK" else "CONTINUE"
        print(f"  [Chunk {c.chunk_index:02d} | {c.duration_processed:4.1f}s] {status_color:<12s} | Prob: {c.input_unsafe_prob:.3f} | Transcript: '{c.partial_transcript[:40]}...'")

    print(f"\n========================================================================")
    print(f"  STREAM SUMMARY")
    print(f"========================================================================")
    print(f"  Final Decision    : {res.final_decision}")
    print(f"  Early Terminated  : {'YES (Threat intercepted mid-stream!)' if res.early_terminated else 'NO (Processed full stream)'}")
    print(f"  Chunks Processed  : {res.chunks_processed} / {int(np.ceil(res.total_duration / res.chunk_duration))}")
    print(f"  Total Latency     : {res.total_latency_ms:.1f} ms")
    print(f"========================================================================\n")


if __name__ == "__main__":
    main()
