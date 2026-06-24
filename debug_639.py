import sys
sys.path.insert(0, "src")

from audio_processor import transcribe_audio
from context_verifier import verify_context
from policy_checker import check_policy
from llm_engine import generate_response

path = "data/external/benign/000639_original.wav"

print("=== STT ===")
transcript = transcribe_audio(path)
print(f"Transcript: {repr(transcript)}")
print(f"Length: {len(transcript)} chars")

print("\n=== LLM ===")
response = generate_response(transcript)
print(f"Response: {repr(response)}")

print("\n=== CONTEXT ===")
sim = verify_context(transcript, response)
print(f"Similarity: {sim:.4f}")

print("\n=== POLICY (on response) ===")
_, d = check_policy(response)
print(f"unsafe_prob: {d['unsafe_prob']}")

print("\n=== VERDICT ===")
print(f"Context threshold: 0.25")
print(f"Triggered by: {'CONTEXT (sim too low)' if sim < 0.25 else 'POLICY' if d['unsafe_prob'] > 0.5 else 'ALLOW'}")