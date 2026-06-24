import sys
sys.path.insert(0, "src")

from audio_processor import transcribe_audio
from policy_checker import check_policy

t = transcribe_audio("data/benign/test.mp3")
print("Transcript:", t)
print()

_, d = check_policy(t)
print("Input unsafe_prob:", d["unsafe_prob"])

stub_response = "Here is a response to your request: " + t[:200]
_, d2 = check_policy(stub_response)
print("Output unsafe_prob:", d2["unsafe_prob"])