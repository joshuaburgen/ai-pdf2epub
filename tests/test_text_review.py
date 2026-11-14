import csv
import json
import tempfile
import unittest
from pathlib import Path

from bookocr.storage import atomic_text, digest, read_json
from bookocr.review import csv_text
from bookocr.text_review import export, load_review, parse_response, prepare, submission_lock, submit
from bookocr.vision_review import APIError


class TextReviewTests(unittest.TestCase):
    settings = {'base_url': 'https://example.test/v1', 'model': 'review-model', 'key': 'secret-test-key'}

    def fixture(self, directory):
        root = Path(directory)
        source, out = root/'book.md', root/'review'
        text = '## 第一章\n\n<!-- PDF page 5 -->\n\n这是唯一一处需要讨论的原文句子。\n\n'+'内容正常，继续阅读。'*40
        atomic_text(source, text)
        manifest = prepare(source, out, self.settings, size=160, context=40)
        return source, out, manifest, text

    def response(self, chunk, issues=None, more=False):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'chunk_id': chunk['chunk_id'], 'issues': issues or [], 'more_issues': more,
            'notes': ''}, ensure_ascii=False)}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}

    def issue(self, quote='这是唯一一处需要讨论的原文句子。'):
        return {'quote': quote, 'category': 'broken_sentence', 'severity': 'medium',
                'reason': '示例审查意见，等待独立评测。', 'suggestion': '', 'needs_source_image': True,
                'minimal_original': '', 'minimal_replacement': ''}

    def test_chunks_cover_source_exactly_with_boundary_context(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, m, text = self.fixture(d)
            self.assertEqual(''.join(text[c['start']:c['end']] for c in m['chunks']), text)
            self.assertTrue(all(c['context_start'] <= c['start'] < c['end'] <= c['context_end'] for c in m['chunks']))
            self.assertEqual(load_review(out)[1], text)

    def test_exact_anchor_page_line_and_evaluator_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            source, out, manifest, text = self.fixture(d)
            before = digest(source)
            calls = []
            def fake(url, key, body):
                payload = json.loads(body['messages'][1]['content'])
                calls.append(payload)
                chunk = next(c for c in manifest['chunks'] if c['chunk_id'] == payload['chunk_id'])
                return self.response(chunk, [self.issue()] if chunk['start'] == 0 else [])
            summary = submit(out, self.settings, transport=fake, workers=2)
            self.assertEqual(summary['status'], 'awaiting_independent_evaluation')
            self.assertEqual(summary['reviewed_core_characters'], len(text))
            issue = json.loads((out/'issues.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(text[issue['start']:issue['end']], issue['quote'])
            self.assertEqual(issue['pdf_page'], 5)
            self.assertEqual(issue['section'], '第一章')
            task = json.loads((out/'evaluation_tasks.jsonl').read_text(encoding='utf-8'))
            self.assertIn(issue['quote'], task['context'])
            self.assertEqual(task['claim']['issue_id'], issue['issue_id'])
            self.assertEqual(digest(source), before)
            # Exporting again must retain an evaluator's completed verdict.
            atomic_text(out/'evaluation_template.csv', csv_text([
                {'issue_id': issue['issue_id'], 'verdict': 'disagree', 'reason': '人工或第二模型的判断',
                 'suggestion': ''}], ['issue_id', 'verdict', 'reason', 'suggestion']))
            export(out)
            with (out/'evaluation_template.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['verdict'], 'disagree')
            submit(out, self.settings, transport=lambda *a: self.fail('duplicate paid request'))
            self.assertEqual(len(calls), len(manifest['chunks']))

    def test_invalid_quotes_are_saved_for_diagnosis_not_exported_as_findings(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            parsed = parse_response(self.response(m['chunks'][0], [self.issue('原文里不存在的虚构句子')]), m, m['chunks'][0], text)
            self.assertEqual(parsed['issues'], [])
            self.assertEqual(len(parsed['rejected']), 1)

    def test_wrong_chunk_and_truncated_response_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            response = self.response(m['chunks'][-1])
            with self.assertRaises(ValueError):
                parse_response(response, m, m['chunks'][0], text)
            response['choices'][0]['finish_reason'] = 'length'
            with self.assertRaises(ValueError):
                parse_response(response, m, m['chunks'][-1], text)

    def test_short_unique_quote_remains_locatable(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            parsed = parse_response(self.response(m['chunks'][0], [self.issue('唯一一处')]), m, m['chunks'][0], text)
            self.assertEqual(len(parsed['issues']), 1)
            self.assertEqual(parsed['rejected'], [])

    def test_overlap_context_issue_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            # Use the unique phrase from the first chunk as context of another.
            start = text.index('这是唯一')
            chunk = {'chunk_id': 'c9999', 'start': start+25, 'end': start+60,
                     'context_start': 0, 'context_end': 120}
            parsed = parse_response(self.response(chunk, [self.issue()]), m, chunk, text)
            self.assertEqual(parsed['issues'], [])
            self.assertEqual(parsed['outside_focus'], 1)

    def test_changed_snapshot_and_provider_fail_before_request(self):
        with tempfile.TemporaryDirectory() as d:
            source, out, _, _ = self.fixture(d)
            changed = {**self.settings, 'model': 'other'}
            with self.assertRaises(ValueError):
                submit(out, changed, transport=lambda *a: self.fail('request sent'))
            atomic_text(out/'source.md', 'changed')
            with self.assertRaises(ValueError):
                export(out)
            with self.assertRaises(ValueError):
                prepare(source, out, self.settings, 160, 40)

    def test_invalid_received_response_is_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, m, _ = self.fixture(d)
            def fake(*args):
                response = self.response(m['chunks'][0])
                response['choices'][0]['finish_reason'] = 'length'
                return response
            submit(out, self.settings, limit=1, transport=fake)
            summary = submit(out, self.settings, limit=1, retry_failed=True,
                             transport=lambda *a: self.fail('paid response repeated'))
            self.assertEqual(summary['received_chunks'], 1)
            self.assertEqual(summary['parsed_chunks'], 0)

    def test_network_retry_is_explicit_redacted_and_archived(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, m, _ = self.fixture(d)
            def failure(*args):
                raise APIError('API network/TLS/timeout secret-test-key')
            submit(out, self.settings, limit=1, transport=failure)
            submit(out, self.settings, limit=1, transport=lambda *a: self.fail('implicit retry'))
            summary = submit(out, self.settings, limit=1, retry_failed=True,
                             transport=lambda *a: self.response(m['chunks'][0]))
            self.assertEqual(summary['archived_attempts'], 1)
            self.assertNotIn(self.settings['key'], (out/'attempts/c0001_001.json').read_text())

    def test_saturation_does_not_claim_review_complete(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, m, _ = self.fixture(d)
            def fake(url, key, body):
                chunk = json.loads(body['messages'][1]['content'])
                return self.response(chunk, more=True)
            summary = submit(out, self.settings, transport=fake)
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(summary['chunks_with_more_issues'], len(m['chunks']))

    def test_duplicate_submission_is_locked(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, _, _ = self.fixture(d)
            with submission_lock(out):
                with self.assertRaises(ValueError):
                    with submission_lock(out):
                        self.fail('second process lock succeeded')

    def test_terminology_and_low_impact_edits_are_out_of_scope(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            item = self.issue()
            item['category'] = 'terminology'
            parsed = parse_response(self.response(m['chunks'][0], [item]), m, m['chunks'][0], text)
            self.assertEqual(parsed['issues'], [])
            item = {**self.issue(), 'severity': 'low'}
            parsed = parse_response(self.response(m['chunks'][0], [item]), m, m['chunks'][0], text)
            self.assertEqual(parsed['issues'], [])

    def test_large_rewrite_is_not_a_residual_repair(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, m, text = self.fixture(d)
            item = {**self.issue(), 'minimal_original': '原文句子', 'minimal_replacement': '新'*17}
            parsed = parse_response(self.response(m['chunks'][0], [item]), m, m['chunks'][0], text)
            self.assertEqual(parsed['issues'], [])
            item = {**self.issue(), 'minimal_original': '原文句子', 'minimal_replacement': '原句'}
            parsed = parse_response(self.response(m['chunks'][0], [item]), m, m['chunks'][0], text)
            self.assertEqual(len(parsed['issues']), 1)

    def test_stop_saves_current_batch_and_prevents_next_request(self):
        with tempfile.TemporaryDirectory() as d:
            _, out, m, _ = self.fixture(d)
            calls = []
            def fake(url, key, body):
                calls.append(body)
                atomic_text(out/'STOP', 'stop')
                return self.response(m['chunks'][0])
            summary = submit(out, self.settings, workers=1, transport=fake)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary['received_chunks'], 1)
            self.assertEqual(summary['status'], 'stopped')
