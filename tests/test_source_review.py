import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from bookocr import source_review as review, text_review
from bookocr.storage import atomic_text, read_json, write_json


SETTINGS = {'base_url': 'https://model.invalid/v1', 'model': 'vision', 'key': 'test-secret'}


def response(value, status='completed'):
    return {'status': status, 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps(value, ensure_ascii=False)}]}]}


class SourceReviewTests(unittest.TestCase):
    def fixture(self, directory, overlap=False):
        root = Path(directory)
        source, primary, out, pdf = root/'book.md', root/'primary', root/'visual', root/'source.pdf'
        atomic_text(source, '<!-- PDF page 1 -->\n\nPlease read teh book.\n')
        with pymupdf.open() as doc:
            doc.new_page().insert_text((40, 60), 'Please read the book.')
            doc.save(pdf)
        manifest = text_review.prepare(source, primary, SETTINGS)
        item = {'quote': 'Please read teh book.', 'category': 'ocr_typo', 'severity': 'medium',
                'reason': 'Mechanical OCR typo', 'suggestion': 'Read source', 'needs_source_image': True,
                'minimal_original': '', 'minimal_replacement': ''}
        items = [item, {**item, 'category': 'broken_sentence'}] if overlap else [item]
        def first(*args):
            value = {'chunk_id': manifest['chunks'][0]['chunk_id'], 'issues': items, 'more_issues': False}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        text_review.submit(primary, SETTINGS, transport=first)
        manifest = review.prepare(primary, pdf, out, SETTINGS)
        return source, primary, out, manifest

    def decision(self, task, **changes):
        return {'task_id': task['task_id'], 'verdict': 'correct', 'certainty': 'certain',
                'source_readable': True, 'source_page': 1, 'source_reading': 'the',
                'original': 'teh', 'replacement': 'the', 'evidence': 'The image prints the.',
                'reason': 'Visible source differs from OCR.', **changes}

    def fake(self, values=None):
        def call(url, key, body):
            self.assertTrue(url.endswith('/responses'))
            self.assertEqual(body['reasoning'], {'effort': 'none'})
            content = body['input'][0]['content']
            self.assertEqual(sum(p['type'] == 'input_image' for p in content), 3)
            task = json.loads(content[0]['text'])
            return response(self.decision(task, **(values or {})))
        return call

    def test_source_images_cache_and_missing_first_pass_patch_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            source, primary, out, manifest = self.fixture(d)
            self.assertEqual(review.prepare(primary, Path(d)/'source.pdf', out, SETTINGS), manifest)
            self.assertEqual(review.submit(out, SETTINGS, transport=self.fake())['auto_apply_candidates'], 1)
            review.submit(out, SETTINGS, transport=lambda *a: self.fail('Repeated paid request'))
            result = review.apply(out, Path(d)/'final')
            self.assertEqual(result['applied_patches'], 1)
            self.assertIn('read the book', Path(result['markdown']).read_text(encoding='utf-8'))
            self.assertIn('read teh book', source.read_text(encoding='utf-8'))
            self.assertTrue(review.apply(out, Path(d)/'final')['cached'])

    def test_uncertain_or_unreadable_source_is_queued_not_applied(self):
        for changes in ({'certainty': 'uncertain'}, {'source_readable': False}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as d:
                source, _, out, _ = self.fixture(d)
                review.submit(out, SETTINGS, transport=self.fake(changes))
                result = review.apply(out, Path(d)/'final')
                self.assertEqual(result['applied_patches'], 0)
                self.assertEqual(result['next_stage'], 1)
                self.assertEqual(Path(result['markdown']).read_text(encoding='utf-8'), source.read_text(encoding='utf-8'))

    def test_invalid_quote_evidence_scope_and_truncation_never_apply(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, _, manifest = self.fixture(d)
            task = manifest['tasks'][0]
            invalid = [{'source_page': 2}, {'source_reading': 'other'}, {'original': 'absent'},
                       {'original': 'Please read teh book.', 'replacement': 'rewritten'},
                       {'replacement': '[the]'}, {'evidence': ''}, {'verdict': 'unchanged'}]
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    review.parse(response(self.decision(task, **changes)), task)
            with self.assertRaises(ValueError):
                review.parse(response(self.decision(task), 'incomplete'), task)
            value = self.decision(task, source_reading='書', replacement='书')
            self.assertTrue(review.parse(response(value), task)['auto_apply'])
            # Claimed certainty cannot turn an original printing into a semantic correction.
            historic = {**task, 'issue': {**task['issue'], 'quote': '原书作大缪。'}}
            value = self.decision(task, original='大缪', replacement='大谬', source_reading='大繆')
            with self.assertRaises(ValueError):
                review.parse(response(value), historic)

    def test_conflicting_repairs_remain_for_next_stage(self):
        with tempfile.TemporaryDirectory() as d:
            source, _, out, manifest = self.fixture(d, overlap=True)
            for n, task in enumerate(manifest['tasks']):
                value = self.decision(task)
                if n:
                    value.update(original='teh book', replacement='the book', source_reading='the book')
                write_json(out/'results'/f"{task['task_id']}.json", {
                    'request_key': review.task_key(manifest, task), 'response': response(value)})
            result = review.apply(out, Path(d)/'final')
            self.assertEqual(result['conflicts'], 2)
            self.assertEqual(result['next_stage'], 2)
            self.assertEqual(Path(result['markdown']).read_text(encoding='utf-8'), source.read_text(encoding='utf-8'))

    def test_missing_invalid_and_unchanged_decisions_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, out, manifest = self.fixture(d)
            self.assertEqual(review.export(out)['next_stage'], 1)
            task = manifest['tasks'][0]
            # A response-bearing failure is cached, never silently recharged.
            write_json(out/'results'/f"{task['task_id']}.json", {
                'request_key': review.task_key(manifest, task), 'response': {'status': 'incomplete'}})
            summary = review.submit(out, SETTINGS, retry_failed=True,
                                    transport=lambda *a: self.fail('Repeated paid failure'))
            self.assertEqual(summary['missing_or_invalid'], 1)
            write_json(out/'results'/f"{task['task_id']}.json", {
                'request_key': review.task_key(manifest, task), 'response': response(self.decision(
                    task, verdict='unchanged', original='', replacement=''))})
            result = review.apply(out, Path(d)/'final')
            self.assertEqual(result['unchanged'], 1)
            self.assertEqual(result['next_stage'], 0)

    def test_changed_image_source_pdf_or_page_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, out, manifest = self.fixture(d)
            info = manifest['images']['1'][0]
            (out/info['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                review.load(out)
        with tempfile.TemporaryDirectory() as d:
            source, _, out, manifest = self.fixture(d)
            source.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                review.apply(out, Path(d)/'final')
        with tempfile.TemporaryDirectory() as d:
            _, _, out, manifest = self.fixture(d)
            Path(manifest['pdf']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                review.load(out)
        with self.assertRaises(ValueError):
            review.issue_pages({'start': 23, 'end': 24, 'pdf_page': 2, 'category': 'ocr_typo'},
                               '<!-- PDF page 1 -->text', 2)


if __name__ == '__main__':
    unittest.main()
