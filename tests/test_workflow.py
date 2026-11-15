import os
import shutil
import tempfile
import unittest
from pathlib import Path

from bookocr.demo import run_demo, make_fixture, fake_transport
from bookocr.storage import digest, load_config, read_json, write_json
from bookocr.workflow import execute


PANDOC = os.environ.get('PANDOC') or shutil.which('pandoc')


class WorkflowTests(unittest.TestCase):
    @unittest.skipUnless(PANDOC, 'Pandoc is required for source-image workflow integration')
    def test_source_image_stage_applies_and_publishes_without_independent_text_model(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'source-route'
            config = make_fixture(root, PANDOC)
            settings = {name: {'base_url': 'https://offline.invalid/v1', 'model': name, 'key': 'offline'}
                        for name in ('AI', 'JUDGE')}
            for stage in range(1, 5):
                execute(config, root/'run', stage, settings=settings, transport=fake_transport,
                        submit=stage in {2, 3, 4}, apply=stage == 3)
            def visual(url, key, body):
                self.assertTrue(url.endswith('/responses'))
                task = json.loads(body['input'][0]['content'][0]['text'])
                value = {'task_id': task['task_id'], 'verdict': 'correct', 'certainty': 'certain',
                         'source_readable': True, 'source_page': 1, 'source_reading': '開始',
                         'original': '开使', 'replacement': '开始', 'evidence': 'Synthetic image prints 開始',
                         'reason': 'Confirmed minimal OCR error'}
                return {'status': 'completed', 'output': [{'type': 'message', 'content': [
                    {'type': 'output_text', 'text': json.dumps(value)}]}]}
            result = execute(config, root/'run', 5, settings={'AI': settings['AI']},
                             source_images=True, submit=True, apply=True, transport=visual)
            self.assertEqual(result['applied_patches'], 1)
            self.assertIn('开始阅读', Path(result['markdown']).read_text(encoding='utf-8'))
            self.assertTrue((Path(result['markdown']).parent/'images').is_dir())
            publication = execute(config, root/'run', 6)
            self.assertFalse(publication['errors'])
            self.assertTrue(Path(publication['epub']).exists())

    def test_misplaced_apply_or_submit_flags_fail_before_creating_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'not-created'
            with self.assertRaises(ValueError):
                execute('not-read.json', out, 2, apply=True)
            with self.assertRaises(ValueError):
                execute('not-read.json', out, 6, submit=True)
            with self.assertRaises(ValueError):
                execute('not-read.json', out, 4, source_images=True)
            self.assertFalse(out.exists())
    @unittest.skipUnless(PANDOC, 'Pandoc is required for partial-book integration')
    def test_sample_page_selection_stays_partial_through_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'partial'
            config = make_fixture(root, PANDOC)
            settings = {name: {'base_url': 'https://offline.invalid/v1', 'model': name, 'key': 'offline'}
                        for name in ('AI', 'JUDGE')}
            execute(config, root/'run', 1, pages='1', settings=settings)
            for stage in range(2, 7):
                result = execute(config, root/'run', stage, settings=settings, transport=fake_transport,
                                 submit=stage in {2, 3, 4, 5}, apply=stage in {3, 5})
            self.assertTrue(result['partial_book'])
            self.assertEqual(result['processed_pdf_pages'], 1)
            with self.assertRaises(ValueError):
                execute(config, root/'run', 1, pages='all', settings=settings)
    def test_pandoc_command_is_resolved_from_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)/'book.json'
            write_json(cfg, {'title': '示例', 'pdf': 'demo.pdf', 'ocr': 'ocr.jsonl', 'pandoc': 'pandoc'})
            loaded = load_config(cfg)
            self.assertEqual(loaded['pandoc'], shutil.which('pandoc') or 'pandoc')

    @unittest.skipUnless(PANDOC, 'Pandoc is required for the six-stage integration test')
    def test_all_six_stages_and_final_patches_reach_epub(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_demo(Path(d)/'demo', PANDOC)
            self.assertTrue(Path(result['epub']).is_file())
            self.assertEqual(result['network_calls'], 0)
            state = read_json(Path(d)/'demo/run/workflow.json')
            self.assertEqual(set(state['stages']), set('123456'))
            final = Path(result['markdown']).read_text(encoding='utf-8')
            self.assertIn('故事', final)
            self.assertIn('开始阅读', final)
            self.assertNotIn('故车', final)
            self.assertNotIn('开使', final)
            epub_hash = digest(result['epub'])
            replay = execute(Path(d)/'demo/book.json', Path(d)/'demo/run', 6)
            self.assertTrue(replay['cached'])
            self.assertEqual(epub_hash, digest(result['epub']))
            import zipfile
            with zipfile.ZipFile(result['epub']) as archive:
                xhtml = ''.join(archive.read(n).decode('utf-8') for n in archive.namelist() if n.endswith('.xhtml'))
            self.assertIn('开始阅读', xhtml)
            self.assertNotIn('开使', xhtml)
