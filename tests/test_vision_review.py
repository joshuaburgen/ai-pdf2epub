import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bookocr.storage import atomic_text, digest, read_json, signature, write_json
from bookocr.vision_review import (
    APIError, api_settings, parse_proposal, reuse_successful_results, select_jobs, submit,
)


def response(job, verdict="unchanged", edits=None):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
        "block_id": job["block_id"], "verdict": verdict, "edits": edits or [], "notes": "",})}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class VisionReviewTests(unittest.TestCase):
    def test_generic_provider_prefix_and_environment_override(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'JUDGE_API_KEY': 'environment-key'}, clear=True):
            path = Path(d)/'env'
            atomic_text(path, 'JUDGE_BASE_URL=https://judge.example/v1\nJUDGE_MODEL=independent\nJUDGE_API_KEY=file-key\n')
            settings = api_settings(path, 'JUDGE')
            self.assertEqual(settings['model'], 'independent')
            self.assertEqual(settings['key'], 'environment-key')
            with self.assertRaises(ValueError):
                api_settings(path, 'invalid-prefix')
    def setUp(self):
        self.job = {"block_id": "p0001-b0000", "page": 1, "original_text": "這是思難的例子。", "cohort": "control"}
        self.job["original_sha256"] = signature(self.job["original_text"])

    def test_minimal_edit_remains_pending(self):
        edits = [{"original": "思難", "replacement": "思維", "reason": "visible in image"}]
        proposal = parse_proposal(response(self.job, "correction", edits), self.job)
        self.assertEqual(proposal["candidate_text"], "這是思維的例子。")
        self.assertEqual(proposal["review_status"], "pending")

    def test_unknown_id_or_source_span_fails(self):
        for changed in ["unknown", self.job["block_id"]]:
            value = response(self.job, "correction", [{"original": "不存在", "replacement": "字"}])
            payload = json.loads(value["choices"][0]["message"]["content"])
            payload["block_id"] = changed
            value["choices"][0]["message"]["content"] = json.dumps(payload)
            with self.assertRaises(ValueError):
                parse_proposal(value, self.job)

    def test_ambiguous_or_overlapping_edits_fail(self):
        job = {**self.job, "original_text": "原文原文"}
        with self.assertRaises(ValueError):
            parse_proposal(response(job, "correction", [{"original": "原文", "replacement": "文字"}]), job)
        edits = [{"original": "這是思難", "replacement": "這是思維"}, {"original": "思難", "replacement": "思維"}]
        with self.assertRaises(ValueError):
            parse_proposal(response(self.job, "correction", edits), self.job)

    def test_uncertain_cannot_carry_changes(self):
        with self.assertRaises(ValueError):
            parse_proposal(response(self.job, "uncertain", [{"original": "思難", "replacement": "思維"}]), self.job)

    def test_truncated_or_empty_answer_fails(self):
        value = response(self.job)
        value["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            parse_proposal(value, self.job)

    def test_pilot_is_stratified_and_excludes_resolved_records(self):
        blocks = [{"id": f"p{n:04}-b0000", "page": n, "role": "text", "bbox": [10, 10, 100, 200], "text": "原書文字用來進行圖像逐字核驗。"*3} for n in range(1, 31)]
        flags = {b["id"]: {"reasons": "low_confidence"} for b in blocks[:20]}
        jobs = select_jobs(blocks, flags, {blocks[0]["id"]}, 10, 5)
        self.assertEqual(len(jobs), 15)
        self.assertEqual(sum(j["cohort"] == "flagged" for j in jobs), 10)
        self.assertEqual(sum(j["cohort"] == "control" for j in jobs), 5)
        self.assertNotIn(blocks[0]["id"], {j["block_id"] for j in jobs})
        self.assertEqual(len({j["block_id"] for j in jobs}), 15)

    def test_all_flagged_selection_supports_full_book_batches(self):
        blocks = [{"id": f"p{n:04}-b0000", "page": n, "role": "text", "bbox": [10, 10, 100, 200], "text": "原書文字用來進行圖像逐字核驗。"*3} for n in range(1, 151)]
        flags = {b["id"]: {"reasons": "low_confidence"} for b in blocks}
        jobs = select_jobs(blocks, flags, set(), None, 0)
        self.assertEqual(len(jobs), len(blocks))
        self.assertTrue(all(job["cohort"] == "flagged" for job in jobs))

    def test_all_includes_short_empty_and_resolved_flags(self):
        blocks = [{"id": f"p0001-b{n:04}", "page": 1, "role": role,
                   "bbox": [10, 10, 100, 200], "text": text}
                  for n, (role, text) in enumerate([
                      ("text", ""), ("number", "12"), ("running_header", "header"),
                      ("text", "previously reviewed text"), ("text", "x"*1801)])]
        flags = {b["id"]: {"reasons": "low_confidence"} for b in blocks}
        jobs = select_jobs(blocks, flags, {blocks[3]["id"]}, None, 0)
        self.assertEqual({j["block_id"] for j in jobs}, set(flags))

    def test_all_refuses_missing_source_blocks(self):
        with self.assertRaisesRegex(ValueError, "missing from source"):
            select_jobs([], {"missing": {"reasons": "low_confidence"}}, set(), None, 0)

    def test_settings_read_key_without_exposing_it_and_require_https(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            path = Path(d)/"env"
            atomic_text(path, 'DEEPSEEK_BASE_URL=https://example.test/v1\nDEEPSEEK_MODEL=model\nDEEPSEEK_API_KEY="test-secret"\n')
            cfg = api_settings(path)
            self.assertEqual(cfg["key"], "test-secret")
            atomic_text(path, 'DEEPSEEK_BASE_URL=http://example.test\nDEEPSEEK_MODEL=model\n')
            with self.assertRaises(ValueError):
                api_settings(path)

    def pilot(self, directory):
        out = Path(directory)
        image = out/"image.png"
        image.write_bytes(b"test image bytes; transport is mocked")
        settings = {"base_url": "https://example.test/v1", "model": "test-model", "key": "test-secret"}
        write_json(out/"pilot.json", {"provider": {k: settings[k] for k in ("base_url", "model")},
                                    "jobs": [self.job], "images": {self.job["block_id"]: {"path": image.name, "sha256": digest(image)}}})
        return out, settings

    def test_resume_does_not_repeat_paid_request_or_approve_changes(self):
        with tempfile.TemporaryDirectory() as d:
            out, settings = self.pilot(d)
            calls = []
            def fake(url, key, body):
                calls.append(body)
                return response(self.job)
            submit(out, settings, 1, transport=fake)
            summary = submit(out, settings, 1, transport=fake)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary["automatic_approvals"], 0)
            self.assertEqual(summary["usage"]["total_tokens"], 15)
            self.assertNotIn("test-secret", (out/"model_suggestions.csv").read_text(encoding="utf-8"))

    def test_reuses_only_matching_successful_results(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"source"
            source.mkdir()
            source, settings = self.pilot(source)
            source_manifest = read_json(source/"pilot.json")
            source_manifest.update({"source_run": "run", "pdf_sha256": "pdf", "prompt_version": 1})
            write_json(source/"pilot.json", source_manifest)
            calls = []

            def fake(*args):
                calls.append(True)
                return response(self.job)

            submit(source, settings, 1, transport=fake)
            target = Path(d)/"target"
            target.mkdir()
            image = target/"image.png"
            image.write_bytes((source/"image.png").read_bytes())
            write_json(target/"pilot.json", {
                **source_manifest,
                "images": {self.job["block_id"]: {"path": image.name, "sha256": digest(image)}},
            })
            self.assertEqual(reuse_successful_results(source, target), 1)
            submit(target, settings, 1, transport=lambda *args: self.fail("duplicate request sent"))
            self.assertEqual(len(calls), 1)
            # A paid but truncated response is also reusable for manual review.
            result = source/"results"/(self.job["block_id"]+".json")
            record = read_json(result)
            record.update(status="error", error="Model response did not finish normally")
            record.pop("proposal")
            record["response"]["choices"][0]["finish_reason"] = "length"
            write_json(result, record)
            destination = target/"results"/result.name
            destination.unlink()
            self.assertEqual(reuse_successful_results(source, target), 0)
            self.assertEqual(reuse_successful_results(source, target, include_received=True), 1)
            submit(target, settings, 1, retry_failed=True,
                   transport=lambda *args: self.fail("paid response resubmitted"))

    def test_parallel_submission_is_bounded_and_saves_all_responses(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            settings = {"base_url": "https://example.test/v1", "model": "test-model", "key": "test-secret"}
            jobs, images = [], {}
            for index in range(4):
                job = {**self.job, "block_id": f"p{index+1:04}-b0000", "page": index+1,
                       "original_text": f"原書文字 {index}"}
                job["original_sha256"] = signature(job["original_text"])
                image = out/f"{index}.png"
                image.write_bytes(f"image {index}".encode())
                jobs.append(job)
                images[job["block_id"]] = {"path": image.name, "sha256": digest(image)}
            write_json(out/"pilot.json", {
                "provider": {"base_url": settings["base_url"], "model": settings["model"]},
                "jobs": jobs, "images": images,
            })
            lock = threading.Lock()
            active = maximum = 0

            def fake(url, key, body):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(.02)
                payload = json.loads(body["messages"][1]["content"][0]["text"])
                job = next(item for item in jobs if item["block_id"] == payload["block_id"])
                with lock:
                    active -= 1
                return response(job)

            summary = submit(out, settings, 4, transport=fake, workers=3)
            self.assertEqual(summary["valid_proposals"], 4)
            self.assertEqual(len(list((out/"results").glob("*.json"))), 4)
            self.assertGreater(maximum, 1)
            self.assertLessEqual(maximum, 3)

            # An isolated transport failure must not prevent later jobs running.
            for path in (out/"results").glob("*.json"):
                path.unlink()

            def transient(url, key, body):
                payload = json.loads(body["messages"][1]["content"][0]["text"])
                if payload["block_id"] == jobs[0]["block_id"]:
                    raise APIError("API network/TLS/timeout failure; no automatic retry")
                return response(next(j for j in jobs if j["block_id"] == payload["block_id"]))

            summary = submit(out, settings, 4, transport=transient, workers=1)
            self.assertEqual(summary["saved_responses"], 4)
            self.assertEqual(summary["received_responses"], 3)
            self.assertEqual(summary["missing_responses"], 1)

    def test_network_failure_is_saved_redacted_and_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as d:
            out, settings = self.pilot(d)
            calls = []
            def failed(*args):
                calls.append(True)
                raise APIError("failure test-secret")
            submit(out, settings, 1, transport=failed)
            submit(out, settings, 1, transport=failed)
            self.assertEqual(len(calls), 1)
            submit(out, settings, 1, retry_failed=True, transport=failed)
            self.assertEqual(len(calls), 2)
            saved = (out/"results"/(self.job["block_id"]+".json")).read_text(encoding="utf-8")
            self.assertNotIn("test-secret", saved)
            self.assertIn("[REDACTED]", saved)

    def test_retry_does_not_repeat_received_but_invalid_model_response(self):
        with tempfile.TemporaryDirectory() as d:
            out, settings = self.pilot(d)
            calls = []

            def truncated(*args):
                calls.append(True)
                value = response(self.job)
                value["choices"][0]["finish_reason"] = "length"
                return value

            submit(out, settings, 1, transport=truncated)
            submit(out, settings, 1, retry_failed=True, transport=truncated)
            self.assertEqual(len(calls), 1)

    def test_changed_image_or_provider_fails_before_request(self):
        with tempfile.TemporaryDirectory() as d:
            out, settings = self.pilot(d)
            (out/"image.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                submit(out, settings, 1, transport=lambda *args: self.fail("request sent"))
            settings["model"] = "other-model"
            with self.assertRaises(ValueError):
                submit(out, settings, 1, transport=lambda *args: self.fail("request sent"))


if __name__ == "__main__":
    unittest.main()
