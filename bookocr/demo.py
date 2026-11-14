"""Original two-page fixture; real EPUB export, simulated OCR and model answers."""
import json
import shutil
from pathlib import Path

import pymupdf

from .storage import atomic_text, read_json, write_json
from .workflow import execute


def make_fixture(root, pandoc='pandoc'):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose an empty demo directory')
    root.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    page = pdf.new_page(width=400, height=550)
    page.insert_text((45, 60), 'AI PDF to EPUB Demo', fontsize=20)
    page.insert_text((45, 110), '這本小書介紹一個簡單的故事。', fontname='china-s', fontsize=14)
    page.insert_text((45, 140), '孩子把書放在桌上，然後開始閱讀。', fontname='china-s', fontsize=14)
    page = pdf.new_page(width=400, height=550)
    page.insert_text((45, 70), '第二章：自己畫的圖', fontname='china-s', fontsize=18)
    page.insert_text((45, 110), '這是為示範流程而原創的測試文字。', fontname='china-s', fontsize=14)
    page.draw_rect(pymupdf.Rect(50, 200, 150, 300), color=(0, .3, .7), fill=(.9, .95, 1))
    page.draw_circle(pymupdf.Point(100, 250), 28, color=(0, .3, .7))
    pdf.save(root/'demo.pdf')
    pdf.close()
    data = [{'page': 1, 'items': [{'text': '這本小書介紹一個簡單的故車。孩子把書放在桌上，然後開使閱讀。',
                                 'box': [80, 170, 720, 320], 'score': .6}]},
            {'page': 2, 'items': [{'text': '這是為示範流程而原創的測試文字。',
                                  'box': [80, 160, 720, 260], 'score': .99}]}]
    atomic_text(root/'ocr.jsonl', ''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in data))
    cfg = {'title': 'AI PDF to EPUB 原创演示', 'author': 'ai-pdf2epub contributors', 'language': 'zh-CN',
           'pdf': 'demo.pdf', 'ocr': 'ocr.jsonl', 'source_format': 'legacy', 'ocr_scale': 2,
           'pandoc': shutil.which(str(pandoc)) or str(pandoc), 'sample_pages': 'all', 'direction': 'horizontal',
           'confidence_threshold': .85, 'flag_simplified_candidates': False, 'opencc': 't2s',
           'sections': [{'page': 1, 'title': '第一章 开始阅读', 'kind': 'chapter'},
                        {'page': 2, 'title': '第二章 自己的图画', 'kind': 'chapter'}],
           'chapter_count': 2, 'figures': [{'page': 2, 'bbox': [100, 400, 300, 600], 'caption': '原创几何图'}],
           'toc_depth': 2, 'split_level': 2}
    write_json(root/'book.json', cfg)
    return root/'book.json'


def fake_transport(url, key, body):
    """No network. Deliberately leave one OCR error for the residual check."""
    content = body['messages'][1]['content']
    payload = json.loads(content[0]['text'] if isinstance(content, list) else content)
    if 'block_id' in payload:
        result = {'block_id': payload['block_id'], 'verdict': 'correction', 'edits': [
            {'original': '故車', 'replacement': '故事', 'reason': 'Synthetic image review fixture'}], 'notes': 'Simulated response'}
    elif 'chunk_id' in payload:
        result = {'chunk_id': payload['chunk_id'], 'issues': [{
            'quote': '孩子把书放在桌上，然后开使阅读。', 'category': 'ocr_typo', 'severity': 'medium',
            'reason': '“开使阅读”是本演示刻意保留的OCR错误。', 'suggestion': '仅将开使改为开始',
            'minimal_original': '开使', 'minimal_replacement': '开始', 'needs_source_image': False}],
            'more_issues': False, 'notes': 'Simulated response'}
    else:
        result = {'task_id': payload['task_id'], 'verdict': 'approve', 'edits': [], 'reason': 'Synthetic independent judgment fixture'}
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result, ensure_ascii=False)}}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}


def run_demo(root, pandoc='pandoc'):
    root = Path(root).resolve()
    config = make_fixture(root, pandoc)
    settings = {name: {'base_url': 'https://offline.invalid/v1', 'model': model, 'key': 'offline-fixture'}
                for name, model in [('AI', 'simulated-primary'), ('JUDGE', 'simulated-independent')]}
    for stage in range(1, 7):
        execute(config, root/'run', stage, submit=stage in {2, 3, 4, 5}, apply=stage in {3, 5},
                settings=settings, transport=fake_transport)
    state = read_json(root/'run/workflow.json')
    final = Path(state['artifacts']['final_markdown']).read_text(encoding='utf-8')
    if '故车' in final or '开使' in final or '开始阅读' not in final:
        raise ValueError('Synthetic corrections did not reach the EPUB input')
    return {'epub': state['artifacts']['epub'], 'markdown': state['artifacts']['final_markdown'],
            'ai_requests': 'simulated_only', 'ocr': 'synthetic_import', 'network_calls': 0}
