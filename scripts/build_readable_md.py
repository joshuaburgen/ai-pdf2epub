"""Build a readable draft from saved model proposals; no OCR or API calls."""
import argparse
import copy
import csv
from collections import Counter
from pathlib import Path

from bookocr.pipeline import run
from bookocr.review import FIELDS, apply_corrections, csv_text
from bookocr.storage import atomic_text, digest, read_json, signature, write_json
from bookocr.structure import SKIP_ROLES
from bookocr.vision_review import parse_proposal


def assemble_review(pages, baseline_rows, manifest, records):
    blocks = {b['id']: b for p in pages for b in p['blocks']}
    # Model input may already contain an accepted correction. Keep the layout
    # original in the formal ledger while composing the next candidate on top.
    baseline = copy.deepcopy(blocks)
    rows = {}
    for row in baseline_rows:
        key = row['block_id']
        if key in rows or key not in blocks or row['original_text'] != blocks[key]['text']:
            raise ValueError('Baseline review IDs/text do not match layout')
        if row['status'] not in {'accepted', 'rejected', 'pending'}:
            raise ValueError('Invalid baseline review status')
        rows[key] = dict(row)
        if row['status'] == 'accepted':
            baseline[key]['text'] = row['corrected_text']
    decisions = []
    seen = set()
    for job in manifest['jobs']:
        key = job['block_id']
        if key in seen or key not in baseline:
            raise ValueError('Duplicate or unknown model job')
        seen.add(key)
        if (signature(job['original_text']) != job['original_sha256']
                or baseline[key]['text'] != job['original_text']):
            raise ValueError(f'Model source text mismatch: {key}')
        record = records.get(key, {})
        expected = signature([manifest['provider'], manifest['prompt_version'], job,
                              manifest['images'][key]['sha256']])
        if record and (record.get('request_key') != expected or
                       record.get('original_sha256') != job['original_sha256']):
            raise ValueError(f'Model request identity mismatch: {key}')
        decision, proposal = 'retain_original_unresolved', {}
        try:
            proposal = parse_proposal(record['response'], job)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        if blocks[key]['role'] in SKIP_ROLES:
            decision = 'excluded_from_body'
        elif proposal.get('verdict') == 'correction' and proposal['candidate_text'].strip():
            decision = 'model_accepted_for_readability'
        elif proposal.get('verdict') == 'unchanged':
            decision = 'model_unchanged'
        row = rows.setdefault(key, {'block_id': key, 'page': job['page'],
                                   'original_text': blocks[key]['text'],
                                   'status': 'pending', 'corrected_text': ''})
        if decision == 'model_accepted_for_readability':
            row.update(status='accepted', corrected_text=proposal['candidate_text'],
                       review_method='model_readability_policy')
        elif decision in {'model_unchanged', 'excluded_from_body'} and row['status'] != 'accepted':
            row.update(status='rejected', review_method='model_readability_policy')
        row['decision_reason'] = decision
        decisions.append({'block_id': key, 'page': job['page'], 'decision': decision,
                          'model_verdict': proposal.get('verdict', 'invalid_or_failed'),
                          'error': record.get('error', '')})
    return list(rows.values()), decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='output/book_full_md')
    parser.add_argument('--review', default='output/deepseek_full_864')
    parser.add_argument('--out', default='output/book_readable_md')
    args = parser.parse_args()
    source, review, out = map(Path, (args.source, args.review, args.out))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new output directory; existing runs are preserved')
    cfg = read_json(source/'run.json')['config']
    manifest = read_json(review/'pilot.json')
    if digest(cfg['pdf']) != manifest['pdf_sha256']:
        raise ValueError('PDF differs from visual review source')
    with (source/'review/reviewed_lines.csv').open(encoding='utf-8-sig', newline='') as stream:
        baseline_rows = list(csv.DictReader(stream))
    run(cfg, out, 'all', until='quality')
    pages = [read_json(p)['data'] for p in sorted((out/'layout_json').glob('page_*.json'))]
    records = {p.stem: read_json(p) for p in (review/'results').glob('*.json')}
    rows, decisions = assemble_review(pages, baseline_rows, manifest, records)
    fields = FIELDS + ['review_method', 'decision_reason']
    reviewed = out/'review/reviewed_lines.csv'
    atomic_text(reviewed, csv_text(rows, fields=fields))
    apply_corrections(pages, reviewed)
    atomic_text(out/'review/model_decisions.csv', csv_text(decisions, fields=list(decisions[0])))
    run(cfg, out, 'all', until='simplify')
    write_json(out/'readable_build.json', {
        'policy': 'readable_draft_model_accepted_not_human_verified',
        'model_decisions': dict(Counter(d['decision'] for d in decisions)),
        'formal_review_statuses': dict(Counter(r['status'] for r in rows)),
        'source_manifest_sha256': digest(review/'pilot.json'),
        'ocr_or_api_calls': 0,
        'traditional_markdown': str(out/'book/book_structured.md'),
        'simplified_markdown': str(out/'book/book_simplified.md')})


if __name__ == '__main__':
    main()
