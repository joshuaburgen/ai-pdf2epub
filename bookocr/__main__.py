"""Run with python -m bookocr; OCR dependencies are loaded only on explicit request."""
import argparse
import json
import sys

from .storage import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bookocr")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "layout", "quality", "structure", "simplify", "epub"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out", required=True)
        sub.add_argument("--pages", help="1-based PDF pages, e.g. 3-6,30,34; all must be explicit")
    sub = commands.add_parser("audit")
    sub.add_argument("epub")
    sub.add_argument("--out", required=True)
    sub = commands.add_parser("recognize")
    sub.add_argument("--pdf", required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--pages", required=True)
    sub.add_argument("--device", default="gpu:0")
    sub = commands.add_parser("recheck")
    sub.add_argument("--run", required=True)
    sub.add_argument("--ids", required=True, help="Comma-separated flagged block IDs")
    sub.add_argument("--model", default="chinese_cht_PP-OCRv3")
    from .text_review import configure_parser
    configure_parser(commands.add_parser("text-review", help="Review full Markdown and export evaluation tasks"))
    from .source_review import configure_parser as source_parser
    source_parser(commands.add_parser("source-review", help="Verify residual findings against PDF images and apply certain repairs"))
    from .workflow import configure_parser as workflow_parser
    workflow_parser(commands.add_parser("workflow", help="Six-stage AI-assisted PDF to EPUB workflow"))
    sub = commands.add_parser("demo", help="Offline six-stage fixture with real Pandoc publication")
    sub.add_argument("--out", default="output/demo")
    sub.add_argument("--pandoc", default="pandoc")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.out, args.pandoc)
        elif args.command == "workflow":
            from .workflow import run_cli
            result = run_cli(args)
        elif args.command == "source-review":
            from .source_review import run_cli
            result = run_cli(args)
        elif args.command == "text-review":
            from .text_review import run_cli
            result = run_cli(args)
        elif args.command == "recognize":
            from .backends import recognize
            result = recognize(args.pdf, args.out, args.pages, args.device)
        elif args.command == "recheck":
            from .backends import recheck
            result = recheck(args.run, args.ids.split(","), args.model)
        elif args.command == "audit":
            from .epub import audit_epub, write_audit
            result = audit_epub(args.epub)
            write_audit(args.out, result)
            print(json.dumps(result, ensure_ascii=False))
            return bool(result["errors"])
        else:
            from .pipeline import run
            result = run(load_config(args.config), args.out, args.pages,
                         until="audit" if args.command == "run" else args.command)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"bookocr: {type(exc).__name__}: {exc}", file=sys.stderr)
        cause = exc.__cause__
        while cause is not None:
            print(f"  caused by {type(cause).__name__}: {cause}", file=sys.stderr)
            cause = cause.__cause__
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
