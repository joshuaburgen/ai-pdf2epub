"""Resume a prepared book review with bounded transport retries and saved status."""
import argparse
import json
import os
import time
from pathlib import Path

from bookocr.storage import read_json, write_json
from bookocr.vision_review import api_settings, submit


def run(out, settings, workers=4, passes=3):
    out = Path(out)
    total = len(read_json(out/"pilot.json")["jobs"])
    for attempt in range(1, passes+1):
        write_json(out/"batch_state.json", {"status": "running", "pid": os.getpid(),
                   "pass": attempt, "prepared": total, "updated_unix": time.time()})
        summary = submit(out, settings, total, retry_failed=attempt > 1, workers=workers)
        state = {**summary, "pass": attempt, "pid": os.getpid(), "updated_unix": time.time(),
                 "status": "awaiting_unified_review" if not summary["missing_responses"] else "incomplete"}
        write_json(out/"batch_state.json", state)
        print(json.dumps(state), flush=True)
        if not summary["missing_responses"]:
            return state
        errors = [read_json(path) for path in (out/"results").glob("*.json")]
        errors = [r.get("error", "") for r in errors if "response" not in r]
        transient = ("API network/TLS/timeout", "API HTTP 429;", "API HTTP 500;",
                     "API HTTP 502;", "API HTTP 503;", "API HTTP 504;")
        if any(not error.startswith(transient) for error in errors):
            return state
        if attempt < passes:
            time.sleep(15)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--passes", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    try:
        result = run(args.out, api_settings(args.env_file), args.workers, args.passes)
    except Exception:
        # The traceback may contain provider data. Persist only a neutral state.
        write_json(Path(args.out)/"batch_state.json", {"status": "runner_error",
                   "pid": os.getpid(), "updated_unix": time.time()})
        raise
    raise SystemExit(0 if result["status"] == "awaiting_unified_review" else 1)
