import json
import unittest

from bookocr.storage import signature
from scripts.build_readable_md import assemble_review


class ReadableDraftTests(unittest.TestCase):
    def fixture(self, verdict='correction', replacement='wordZZ text'):
        key = 'p0001-b0000'
        pages = [{'page': 1, 'blocks': [{'id': key, 'text': 'wordXX text', 'role': 'text'}]}]
        rows = [{'block_id': key, 'original_text': 'wordXX text',
                 'corrected_text': 'wordYY text', 'status': 'accepted'}]
        job = {'block_id': key, 'page': 1, 'original_text': 'wordYY text',
               'original_sha256': signature('wordYY text')}
        manifest = {'provider': {}, 'prompt_version': 1, 'jobs': [job],
                    'images': {key: {'sha256': 'image'}}}
        edits = [{'original': 'wordYY text', 'replacement': replacement}] if verdict == 'correction' else []
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'block_id': key, 'verdict': verdict, 'edits': edits, 'notes': ''})}}]}
        records = {key: {'response': response, 'original_sha256': job['original_sha256'],
                        'request_key': signature([{}, 1, job, 'image'])}}
        return pages, rows, manifest, records

    def test_model_edit_composes_with_prior_correction_without_duplicate_row(self):
        rows, decisions = assemble_review(*self.fixture())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['original_text'], 'wordXX text')
        self.assertEqual(rows[0]['corrected_text'], 'wordZZ text')
        self.assertEqual(rows[0]['review_method'], 'model_readability_policy')

    def test_unchanged_preserves_prior_accepted_correction(self):
        rows, _ = assemble_review(*self.fixture(verdict='unchanged'))
        self.assertEqual(rows[0]['status'], 'accepted')
        self.assertEqual(rows[0]['corrected_text'], 'wordYY text')

    def test_truncated_response_keeps_prior_text(self):
        args = self.fixture()
        args[3]['p0001-b0000']['response']['choices'][0]['finish_reason'] = 'length'
        rows, decisions = assemble_review(*args)
        self.assertEqual(rows[0]['corrected_text'], 'wordYY text')
        self.assertEqual(decisions[0]['decision'], 'retain_original_unresolved')

    def test_empty_candidate_cannot_erase_block(self):
        rows, decisions = assemble_review(*self.fixture(replacement=''))
        self.assertEqual(rows[0]['corrected_text'], 'wordYY text')
        self.assertEqual(decisions[0]['decision'], 'retain_original_unresolved')

    def test_changed_request_identity_is_rejected(self):
        args = self.fixture()
        args[3]['p0001-b0000']['request_key'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            assemble_review(*args)
