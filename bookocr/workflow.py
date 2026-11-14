"""Six explicit stages; AI requests and application require explicit switches."""
import json
import time
from pathlib import Path

import pymupdf
from opencc import OpenCC

from . import judgment, text_review, vision_review
from .epub import audit_epub, build_epub, write_audit
from .pipeline import run
from .storage import digest, load_config, read_json, signature, write_json


STAGES = {1: 'OCR and layout', 2: 'Primary visual review', 3: 'Independent visual judgment and import',
          4: 'Conservative residual text review', 5: 'Independent text judgment and import', 6: 'EPUB publication'}


def files_key(path):
    return signature({p.name: digest(p) for p in sorted(Path(path).glob('*.json'))})


def execute(config, output, stage, **kwargs):
    if kwargs.get('apply') and stage not in {3, 5}:
        raise ValueError('--apply is only valid for judgment stages 3 and 5')
    if kwargs.get('submit') and stage not in {2, 3, 4, 5}:
        raise ValueError('--submit is only valid for AI stages 2 through 5')
    output = Path(output).resolve()
    cfg = load_config(config)
    if not Path(cfg['pdf']).is_file():
        raise ValueError('Input PDF is missing')
    if output == Path(cfg['pdf']).resolve() or output in Path(cfg['pdf']).resolve().parents:
        raise ValueError('Workflow output must not contain the input PDF')
    if output.exists() and not (output/'workflow.json').exists() and any(p.name != 'submit.lock' for p in output.iterdir()):
        raise ValueError('Workflow output directory is not owned by this project')
    output.mkdir(parents=True, exist_ok=True)
    with text_review.submission_lock(output):
        return _execute(config, output, stage, **kwargs)


def _execute(config, output, stage, *, submit=False, apply=False, allow_pending=False, retry_failed=False,
            env_file='.env.local', ai_prefix='AI', judge_prefix='JUDGE', workers=2, pages='all',
            device='gpu:0', settings=None, transport=None):
    cfg = load_config(config)
    output = Path(output).resolve()
    pdf = Path(cfg['pdf']).resolve()
    if output == pdf or output in pdf.parents:
        raise ValueError('Workflow output must not contain the input PDF')
    identity = {'config': cfg, 'pdf_sha256': digest(pdf)}
    marker = output/'workflow.json'
    if marker.exists():
        state = read_json(marker)
        if state['identity'] != identity:
            raise ValueError('Book configuration/PDF changed; choose a new workflow directory')
    else:
        if output.exists() and any(p.name != 'submit.lock' for p in output.iterdir()):
            raise ValueError('Workflow output is not owned by this project')
        state = {'identity': identity, 'artifacts': {}, 'stages': {}}
        write_json(marker, state)
    artifacts = state['artifacts']

    def bind(key, value, invalidate=()):
        value = str(Path(value).resolve())
        if artifacts.get(key) != value:
            for child in invalidate:
                artifacts.pop(child, None)
        artifacts[key] = value
        write_json(marker, state)

    def need(key):
        if key not in artifacts:
            raise ValueError(f'Missing {key}; execute/apply the preceding stage first')
        return Path(artifacts[key])

    def provider(prefix):
        return settings[prefix] if settings is not None else vision_review.api_settings(env_file, prefix)

    def review_directory(name, inputs):
        return output/name/signature(inputs)[:12]

    kwargs = {'transport': transport} if transport is not None else {}
    if stage == 1:
        if state.get('page_selection', pages) != pages:
            raise ValueError('Page selection changed; choose a new workflow directory')
        state['page_selection'] = pages
        target = output/'01-ocr'
        ocr = Path(cfg['ocr']).resolve()
        if target == ocr or target in ocr.parents or cfg.get('source_format') == 'structure' and ocr in target.parents:
            raise ValueError('OCR cache and processed stage directory must be separate')
        if cfg.get('source_format') == 'structure':
            from .backends import recognize
            recognize(cfg['pdf'], cfg['ocr'], pages, device, cfg.get('ocr_scale', 2))
        elif not Path(cfg['ocr']).is_file():
            raise ValueError('Legacy OCR file missing; use source_format=structure for recognition')
        result = run(cfg, target, pages, until='structure')
        bind('ocr_run', target)
    elif stage == 2:
        source = need('ocr_run')
        primary = provider(ai_prefix)
        target = review_directory('02-vision', [digest(source/'run.json'), files_key(source/'corrected'),
                                  {k: primary[k] for k in ('base_url', 'model')}, vision_review.PROMPT_VERSION])
        manifest = vision_review.prepare(source, target, primary, None, 0)
        bind('visual_review', target, ('visual_judgment', 'reviewed_run', 'text_review', 'text_judgment', 'final_markdown', 'epub'))
        result = {'prepared': len(manifest['jobs']), 'directory': str(target), 'api_calls': 0}
        if submit and manifest['jobs']:
            result = vision_review.submit(target, primary, len(manifest['jobs']), retry_failed, workers=workers, **kwargs)
    elif stage == 3:
        source = need('visual_review')
        judge = provider(judge_prefix)
        target = review_directory('03-judge', [digest(source/'pilot.json'), files_key(source/'results'),
                                  {k: judge[k] for k in ('base_url', 'model')}, judgment.VISION_PROMPT])
        manifest = judgment.prepare_visual(source, target, judge)
        bind('visual_judgment', target)
        result = {'prepared': len(manifest['tasks']), 'automatic': len(manifest['automatic']), 'directory': str(target)}
        if submit and manifest['tasks']:
            result = judgment.submit(target, judge, workers, retry_failed, **kwargs)
        if apply:
            _, decisions, _ = judgment.decisions(target, not allow_pending)
            destination = review_directory('03-reviewed', [digest(target/'tasks.json'), decisions])
            result = judgment.apply_visual(target, destination, allow_pending)
            bind('reviewed_run', destination, ('text_review', 'text_judgment', 'final_markdown', 'epub'))
    elif stage == 4:
        source = need('reviewed_run')/'book/book_simplified.md'
        primary = provider(ai_prefix)
        target = review_directory('04-text', [digest(source), {k: primary[k] for k in ('base_url', 'model')}, text_review.PROMPT])
        manifest = text_review.prepare(source, target, primary)
        bind('text_review', target, ('text_judgment', 'final_markdown', 'epub'))
        result = {'prepared': len(manifest['chunks']), 'directory': str(target), 'api_calls': 0}
        if submit:
            result = text_review.submit(target, primary, workers, retry_failed=retry_failed, **kwargs)
    elif stage == 5:
        source = need('text_review')
        report = text_review.export(source)
        if report['status'] != 'awaiting_independent_evaluation' and not allow_pending:
            raise ValueError('Text review incomplete; resume it or explicitly --allow-pending to retain unresolved source')
        judge = provider(judge_prefix)
        target = review_directory('05-judge', [digest(source/'manifest.json'), digest(source/'issues.jsonl'),
                                  {k: judge[k] for k in ('base_url', 'model')}, judgment.TEXT_PROMPT])
        manifest = judgment.prepare_text(source, target, judge)
        bind('text_judgment', target)
        result = {'prepared': len(manifest['tasks']), 'directory': str(target), 'api_calls': 0}
        if submit and manifest['tasks']:
            result = judgment.submit(target, judge, workers, retry_failed, **kwargs)
        if apply:
            _, decisions, _ = judgment.decisions(target, not allow_pending)
            destination = review_directory('05-final', [digest(target/'tasks.json'), decisions])
            result = judgment.apply_text(target, destination, allow_pending)
            bind('final_markdown', destination/'book_simplified.md', ('epub',))
    elif stage == 6:
        source = need('final_markdown')
        assets = {p.name: digest(p) for p in (source.parent/'images').glob('*') if p.is_file()}
        target = review_directory('06-epub', [digest(source), cfg, assets])
        epub = target/'book.epub'
        converter = OpenCC(cfg.get('opencc', 't2s'))
        cached = False
        if (target/'publication.json').exists() and epub.exists():
            prior = read_json(target/'publication.json')
            cached = prior['source_markdown_sha256'] == digest(source) and prior['epub_sha256'] == digest(epub)
        if not cached:
            with pymupdf.open(cfg['pdf']) as document:
                build_epub(source, epub, cfg, document, converter)
        sections = read_json(need('reviewed_run')/'book/source_map.json')['section_count']
        result = audit_epub(epub, converter.convert(cfg['title']), sections)
        selected = read_json(need('reviewed_run')/'book/source_map.json')['selected_pages']
        with pymupdf.open(cfg['pdf']) as document:
            result['partial_book'] = selected != list(range(1, len(document)+1))
        result['processed_pdf_pages'] = len(selected)
        write_audit(target, result)
        if result['errors']:
            raise ValueError('EPUB audit failed: '+'; '.join(result['errors']))
        write_json(target/'publication.json', {'source_markdown_sha256': digest(source), 'epub_sha256': digest(epub),
                                             'readability_policy': 'residual_ocr_only'})
        bind('epub', epub)
        result = {**result, 'epub': str(epub), 'cached': cached}
    else:
        raise ValueError('Workflow stage must be 1..6')
    state['stages'][str(stage)] = {'name': STAGES[stage], 'updated_unix': time.time(),
                                  'submit_requested': submit, 'apply_requested': apply, 'result': result}
    write_json(marker, state)
    return result


def configure_parser(parser):
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', type=int, choices=range(1, 7), required=True)
    parser.add_argument('--submit', action='store_true', help='Send AI requests; without this only prepare tasks')
    parser.add_argument('--apply', action='store_true', help='Import validated judgments in stages 3/5')
    parser.add_argument('--allow-pending', action='store_true', help='Retain originals for missing/invalid judgments')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--env-file', default='.env.local')
    parser.add_argument('--ai-prefix', default='AI')
    parser.add_argument('--judge-prefix', default='JUDGE')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pages', default='all')
    parser.add_argument('--device', default='gpu:0')


def run_cli(args):
    return execute(args.config, args.out, args.stage, **{key: getattr(args, key) for key in (
        'submit', 'apply', 'allow_pending', 'retry_failed', 'env_file', 'ai_prefix', 'judge_prefix', 'workers', 'pages', 'device')})
