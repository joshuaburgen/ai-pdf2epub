"""Run a prepared text review with up to three passes for transient failures."""
import argparse
import json
import os
import time
from pathlib import Path

from bookocr.storage import read_json, write_json
from bookocr.text_review import submit
from bookocr.vision_review import api_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/deepseek_text_repair')
    parser.add_argument('--env-file', default='.env.local')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--passes', type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    out, settings = Path(args.out), api_settings(args.env_file)
    for attempt in range(1, args.passes+1):
        write_json(out/'batch_state.json', {'status': 'running', 'pid': os.getpid(),
                   'pass': attempt, 'updated_unix': time.time()})
        try:
            summary = submit(out, settings, args.workers, retry_failed=attempt > 1)
        except Exception:
            write_json(out/'batch_state.json', {'status': 'runner_error', 'pid': os.getpid(),
                       'pass': attempt, 'updated_unix': time.time()})
            raise
        write_json(out/'batch_state.json', {**summary, 'pid': os.getpid(), 'pass': attempt,
                   'updated_unix': time.time()})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if summary.get('stop_requested') or summary['received_chunks'] == summary['chunks']:
            break
        transient = ('API network/TLS/timeout', 'API HTTP 429;', 'API HTTP 500;',
                     'API HTTP 502;', 'API HTTP 503;', 'API HTTP 504;')
        records = [read_json(p) for p in (out/'results').glob('*.json')]
        if any(not r.get('error', '').startswith(transient) for r in records if 'response' not in r):
            break
        if attempt < args.passes:
            time.sleep(15)


if __name__ == '__main__':
    main()
