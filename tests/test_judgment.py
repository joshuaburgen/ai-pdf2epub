import json
import tempfile
import unittest
from pathlib import Path

from bookocr import judgment, text_review
from bookocr.storage import atomic_text, digest, read_json, signature, write_json


PRIMARY = {'base_url': 'https://primary.invalid/v1', 'model': 'primary', 'key': 'test-primary-secret'}
JUDGE = {'base_url': 'https://judge.invalid/v1', 'model': 'independent', 'key': 'test-judge-secret'}


def response(value):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value, ensure_ascii=False)}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}


class JudgmentTests(unittest.TestCase):
    def text_fixture(self, directory, issues=None):
        root = Path(directory)
        source = root/'book.md'
        text = '## 第一章\n\n<!-- PDF page 1 -->\n\n孩子把书放在桌上，然后开使阅读。\n'
        atomic_text(source, text)
        first = root/'primary'
        manifest = text_review.prepare(source, first, PRIMARY)
        issue = {'quote': '孩子把书放在桌上，然后开使阅读。', 'category': 'ocr_typo', 'severity': 'medium',
                 'reason': '明确的演示错字', 'suggestion': '开使改为开始', 'needs_source_image': False,
                 'minimal_original': '开使', 'minimal_replacement': '开始'}
        text_review.submit(first, PRIMARY, transport=lambda *a: response({
            'chunk_id': manifest['chunks'][0]['chunk_id'], 'issues': issues or [issue], 'more_issues': False}))
        out = root/'judge'
        judgment.prepare_text(first, out, JUDGE)
        return source, first, out, text

    def approve(self, out, verdict='approve'):
        calls = []
        def fake(url, key, body):
            data = json.loads(body['messages'][1]['content'])
            calls.append(data)
            return response({'task_id': data['task_id'], 'verdict': verdict, 'edits': [], 'reason': '独立判断'})
        judgment.submit(out, JUDGE, transport=fake)
        return calls

    def test_orthographic_equivalence_does_not_change_numbers_or_math(self):
        self.assertTrue(judgment.equivalent_text('潛意識：例子', '潜意识:例子'))
        self.assertFalse(judgment.equivalent_text('90年10日', '90年10月'))
        self.assertFalse(judgment.equivalent_text('x²', 'x2'))
        self.assertFalse(judgment.equivalent_text('a man', 'aman'))

    def test_vision_approval_cannot_silently_change_candidate(self):
        task = {'task_id': 'p0001-b0000', 'original_text': '这是一段故車的文字。', 'proposal': {
            'verdict': 'correction', 'candidate_text': '这是一段故事的文字。',
            'edits': [{'original': '故車', 'replacement': '故事', 'reason': '图像'}]}}
        good = response({'task_id': task['task_id'], 'verdict': 'approve', 'edits': [], 'reason': '确认'})
        self.assertEqual(judgment.parse_decision(good, task, 'visual_judgment_v1')['candidate_text'], '这是一段故事的文字。')
        bad = response({'task_id': task['task_id'], 'verdict': 'approve', 'reason': '其他',
                        'edits': [{'original': '故車', 'replacement': '故乡', 'reason': '改写'}]})
        with self.assertRaises(ValueError):
            judgment.parse_decision(bad, task, 'visual_judgment_v1')

    def test_text_approval_applies_minimal_patch_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as d:
            source, _, out, original = self.text_fixture(d)
            calls = self.approve(out)
            summary = judgment.apply_text(out, Path(d)/'final')
            self.assertEqual(summary['applied_patches'], 1)
            self.assertIn('开始阅读', (Path(d)/'final/book_simplified.md').read_text(encoding='utf-8'))
            self.assertEqual(source.read_text(encoding='utf-8'), original)
            judgment.submit(out, JUDGE, transport=lambda *a: self.fail('duplicate paid call'))
            self.assertEqual(len(calls), 1)
            self.assertTrue(judgment.apply_text(out, Path(d)/'final')['cached'])

    def test_uncertain_or_missing_judgment_cannot_change_text(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, out, original = self.text_fixture(d)
            with self.assertRaises(ValueError):
                judgment.apply_text(out, Path(d)/'not-created')
            self.approve(out, 'uncertain')
            result = judgment.apply_text(out, Path(d)/'final')
            self.assertEqual(result['applied_patches'], 0)
            self.assertEqual((Path(d)/'final/book_simplified.md').read_text(encoding='utf-8'), original)

    def test_input_change_after_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, _, out, _ = self.text_fixture(d)
            self.approve(out)
            atomic_text(source, 'changed')
            with self.assertRaises(ValueError):
                judgment.apply_text(out, Path(d)/'final')

    def test_same_provider_model_is_not_independent(self):
        with tempfile.TemporaryDirectory() as d:
            _, primary, _, _ = self.text_fixture(d)
            with self.assertRaises(ValueError):
                judgment.prepare_text(primary, Path(d)/'same-model', PRIMARY)

    def test_text_judge_cannot_add_new_edits(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, out, _ = self.text_fixture(d)
            task = read_json(out/'tasks.json')['tasks'][0]
            bad = response({'task_id': task['task_id'], 'verdict': 'approve', 'reason': '扩展修改',
                            'edits': [{'original': '孩子', 'replacement': '儿童'}]})
            with self.assertRaises(ValueError):
                judgment.parse_decision(bad, task, 'text_judgment_v1')

    def test_overlap_is_retained_and_duplicate_repairs_only_apply_once(self):
        with tempfile.TemporaryDirectory() as d:
            _, primary, out, _ = self.text_fixture(d)
            manifest = read_json(out/'tasks.json')
            task = manifest['tasks'][0]
            duplicate = {**task, 'task_id': 'duplicate'}
            manifest['tasks'].append(duplicate)
            write_json(out/'tasks.json', manifest)
            self.approve(out)
            self.assertEqual(judgment.apply_text(out, Path(d)/'final')['applied_patches'], 1)
        with tempfile.TemporaryDirectory() as d:
            _, primary, out, original = self.text_fixture(d)
            manifest = read_json(out/'tasks.json')
            task = manifest['tasks'][0]
            overlapping = {**task, 'task_id': 'overlap', 'repair_original': '使阅读',
                           'repair_replacement': '始阅读', 'repair_start': task['repair_start']+1,
                           'repair_end': task['repair_start']+4}
            manifest['tasks'].append(overlapping)
            write_json(out/'tasks.json', manifest)
            self.approve(out)
            result = judgment.apply_text(out, Path(d)/'final')
            self.assertEqual(result['applied_patches'], 0)
            self.assertEqual(result['conflicting_patches_retained'], 2)
            self.assertEqual((Path(d)/'final/book_simplified.md').read_text(encoding='utf-8'), original)

    def test_empty_visual_flags_are_valid_and_need_no_api(self):
        from bookocr.vision_review import select_jobs
        self.assertEqual(select_jobs([], {}, set(), None, 0), [])
