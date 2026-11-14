"""Prepare/probe/submit visual review proposals without applying corrections."""
import argparse
import json
from bookocr.vision_review import api_settings, prepare, probe, reuse_successful_results, submit
from bookocr.storage import read_json


def selection_count(value):
    if value.lower() == "all":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'all'") from exc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["prepare", "probe", "submit"])
parser.add_argument("--run", default="output/book_full_md")
parser.add_argument("--out", default="output/deepseek_pilot")
parser.add_argument("--env-file", default=".env.local")
parser.add_argument("--flagged", type=selection_count, default=20)
parser.add_argument("--controls", type=int, default=10)
parser.add_argument("--limit", type=selection_count, default=30)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--reuse-from")
parser.add_argument("--reuse-received", action="store_true", help="also preserve received responses requiring manual inspection")
parser.add_argument("--retry-failed", action="store_true")
args = parser.parse_args()
settings = api_settings(args.env_file)
if args.action == "prepare":
    result = prepare(args.run, args.out, settings, args.flagged, args.controls)
    reused = reuse_successful_results(args.reuse_from, args.out, args.reuse_received) if args.reuse_from else 0
    print(json.dumps({"prepared": len(result["jobs"]), "reused": reused, "status": "no_api_calls"}))
elif args.action == "probe":
    print(json.dumps(probe(settings)))
else:
    limit = len(read_json(f"{args.out}/pilot.json")["jobs"]) if args.limit is None else args.limit
    print(json.dumps(submit(args.out, settings, limit, args.retry_failed, workers=args.workers)))
