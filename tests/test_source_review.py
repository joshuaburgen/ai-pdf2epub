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
    def test_variant_width_proofs_and_overlapping_context_anchors(self):
        def task(quote):
            return {'task_id': 'r', 'pages': [1], 'retry': {},
                    'issue': {'quote': quote, 'start': 0, 'category': 'ocr_typo'}}
        def value(edits):
            return {'task_id': 'r', 'verdict': 'correct', 'certainty': 'certain', 'source_readable': True,
                    'source_page': 1, 'evidence': 'Visible print', 'reason': 'OCR', 'edits': edits}
        for quote, after, printed in [('战门', '战斗', '戰鬪'),
                                      ('(注41下面', '(注41）下面', '（註41）下面')]:
            with self.subTest(quote=quote):
                parsed = review.parse(response(value([{'original': quote, 'replacement': after,
                                                        'source_reading': printed}])), task(quote))
                self.assertTrue(parsed['auto_apply'])
        quote = '富有有影响力好像发病了'
        edits = [{'original': '富有有影响力', 'replacement': '富有、有影响力', 'source_reading': '富有、有影響力'},
                 {'original': '有影响力好像发病了', 'replacement': '有影响力）好像发病了', 'source_reading': '有影響力）好像發病了'}]
        parsed = review.parse(response(value(edits)), task(quote))
        self.assertEqual(len(parsed['patches']), 2)
        self.assertLessEqual(parsed['patches'][0]['end'], parsed['patches'][1]['start'])

    def test_retry_accepts_traditional_anchor_and_splits_only_image_supported_changes(self):
        task = {'task_id': 'retry1', 'pages': [1], 'retry': {},
                'issue': {'quote': '这是演示文本，内容正常，这是什么·，感受和开使。', 'start': 20, 'category': 'ocr_typo'}}
        value = {'task_id': 'retry1', 'verdict': 'correct', 'certainty': 'certain',
                 'source_readable': True, 'source_page': 1, 'evidence': 'Visible scan', 'reason': 'OCR differences',
                 'edits': [{'original': '這是演示文本，內容正常，這是什麽·，感受和開使。',
                            'replacement': '这是演示文本，内容正常，这是什么，感受和开始。',
                            'source_reading': '這是演示文本，內容正常，這是什麼，感受和開始。'}]}
        parsed = review.parse(response(value), task)
        nested = {'assessment': {k: v for k, v in value.items() if k != 'edits'}, 'edits': value['edits']}
        self.assertEqual(review.parse(response(nested), task), parsed)
        self.assertTrue(parsed['auto_apply'])
        self.assertGreater(len(parsed['patches']), 1)
        for patch in parsed['patches']:
            self.assertEqual(task['issue']['quote'][patch['start']-20:patch['end']-20], patch['original'])
            self.assertLessEqual(len(patch['original']), 16)
        bad = {**value, 'edits': [{'original': '开使', 'replacement': '开始', 'source_reading': '開使'}]}
        retained = review.parse(response(bad), task)
        self.assertEqual(retained['verdict'], 'unchanged')
        self.assertFalse(retained['auto_apply'])
        self.assertEqual(retained['model_verdict'], 'correct')

    def test_retry_reuses_closed_results_and_only_calls_unresolved_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, parent, first = self.fixture(d, overlap=True)
            review.submit(parent, SETTINGS, transport=self.fake())
            task = first['tasks'][1]
            write_json(parent/'results'/f"{task['task_id']}.json", {
                'request_key': review.task_key(first, task), 'response': response(self.decision(
                    task, verdict='uncertain', certainty='uncertain', original='', replacement=''))})
            out = Path(d)/'retry'
            child = review.prepare_retry(parent, out, SETTINGS)
            self.assertEqual(len(child['refinement']['retry_ids']), 1)
            self.assertEqual(len(child['refinement']['reused_ids']), 1)
            closed = child['tasks'][0]
            self.assertEqual(review.task_key(child, closed), review.task_key(first, closed))
            calls = []
            def fake(url, key, body):
                payload = json.loads(body['input'][0]['content'][0]['text'])
                calls.append(payload['task_id'])
                self.assertEqual(body['text']['format']['type'], 'json_schema')
                self.assertEqual(body['text']['format']['schema']['required'], ['assessment', 'edits'])
                self.assertEqual(sum(p['type'] == 'input_image' for p in body['input'][0]['content']), 5)
                value = self.decision(payload)
                value.pop('original'); value.pop('replacement'); value.pop('source_reading')
                value['edits'] = [{'original': 'teh', 'replacement': 'the', 'source_reading': 'the'}]
                edits = value.pop('edits')
                return response({'assessment': value, 'edits': edits})
            review.submit(out, SETTINGS, transport=fake)
            self.assertEqual(calls, [task['task_id']])
            review.submit(out, SETTINGS, transport=lambda *a: self.fail('Paid retry repeated'))
            result = review.apply(out, Path(d)/'final')
            self.assertEqual(result['applied_patches'], 1)
            self.assertEqual(result['applied_findings'], 2)
            self.assertEqual(result['next_stage'], 0)
            saved = read_json(parent/'tasks.json')
            write_json(parent/'tasks.json', {**saved, 'dpi': 121})
            with self.assertRaises(ValueError):
                review.load(out)

    def test_multi_edit_conflict_keeps_entire_decision_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, parent, _ = self.fixture(d, overlap=True)
            out = Path(d)/'retry'
            child = review.prepare_retry(parent, out, SETTINGS)
            for index, task in enumerate(child['tasks']):
                value = self.decision(task)
                for field in ('original', 'replacement', 'source_reading'):
                    value.pop(field)
                value['edits'] = [{'original': 'teh', 'replacement': 'the' if not index else 'THE',
                                   'source_reading': 'the' if not index else 'THE'}]
                if index:
                    value['edits'].append({'original': 'book', 'replacement': 'books', 'source_reading': 'books'})
                write_json(out/'results'/f"{task['task_id']}.json", {
                    'request_key': review.task_key(child, task), 'response': response(value)})
            result = review.apply(out, Path(d)/'final')
            self.assertEqual(result['applied_patches'], 0)
            self.assertEqual(result['next_stage'], 2)


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
                    value.update(original='teh book', replacement='new book', source_reading='new book')
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
