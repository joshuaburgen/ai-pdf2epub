import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bookocr.storage import write_json
from scripts.run_vision_batch import run


class VisionBatchTests(unittest.TestCase):
    def fixture(self, directory):
        out = Path(directory)
        write_json(out/"pilot.json", {"jobs": [{}]})
        return out

    def test_received_invalid_response_waits_for_review_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.fixture(directory)
            with patch("scripts.run_vision_batch.submit", return_value={
                    "missing_responses": 0, "valid_proposals": 0}) as send:
                result = run(out, {})
            self.assertEqual(send.call_count, 1)
            self.assertEqual(result["status"], "awaiting_unified_review")

    def test_transient_retry_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.fixture(directory)
            write_json(out/"results/job.json", {"error": "API HTTP 503; response body omitted"})
            with patch("scripts.run_vision_batch.submit", return_value={"missing_responses": 1}) as send, \
                    patch("scripts.run_vision_batch.time.sleep"):
                result = run(out, {})
            self.assertEqual(send.call_count, 3)
            self.assertFalse(send.call_args_list[0].kwargs["retry_failed"])
            self.assertTrue(send.call_args_list[1].kwargs["retry_failed"])
            self.assertEqual(result["status"], "incomplete")

    def test_permanent_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.fixture(directory)
            write_json(out/"results/job.json", {"error": "API HTTP 401; response body omitted"})
            with patch("scripts.run_vision_batch.submit", return_value={"missing_responses": 1}) as send:
                run(out, {})
            self.assertEqual(send.call_count, 1)
