import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import Settings
from middleware import process_transcript


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, response="A relevant, safe answer."):
        self.response = response
        self.calls = 0

    def generate(self, transcript):
        self.calls += 1
        return self.response


def policy_by_marker(text, threshold):
    probability = 0.99 if "UNSAFE" in text else 0.01
    return int(probability >= threshold), {"unsafe_prob": probability}


def relevant_context(transcript, response):
    return 0.9


class MiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.temp = TemporaryDirectory()
        self.log_path = str(Path(self.temp.name) / "events.jsonl")

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, text, provider, context_check=relevant_context):
        return process_transcript(
            text,
            provider=provider,
            cfg=self.cfg,
            policy_check=policy_by_marker,
            context_check=context_check,
            log_path=self.log_path,
        )

    def test_blocks_unsafe_input_before_generation(self):
        provider = FakeProvider()
        result = self.run_pipeline("UNSAFE input", provider)
        self.assertEqual(result.decision, "BLOCK")
        self.assertEqual(provider.calls, 0)
        self.assertIsNone(result.raw_response)

    def test_allows_safe_relevant_output(self):
        provider = FakeProvider()
        result = self.run_pipeline("safe request", provider)
        self.assertEqual(result.decision, "ALLOW")
        self.assertEqual(result.response, provider.response)

    def test_mitigates_unsafe_output(self):
        provider = FakeProvider("UNSAFE generated answer")
        result = self.run_pipeline("safe request", provider)
        self.assertEqual(result.decision, "MITIGATE")
        self.assertNotEqual(result.response, result.raw_response)

    def test_mitigates_context_drift(self):
        provider = FakeProvider()
        result = self.run_pipeline("safe request", provider, lambda _a, _b: 0.1)
        self.assertEqual(result.decision, "MITIGATE")

    def test_every_request_is_logged(self):
        self.run_pipeline("safe request", FakeProvider())
        lines = Path(self.log_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('"decision": "ALLOW"', lines[0])


if __name__ == "__main__":
    unittest.main()
