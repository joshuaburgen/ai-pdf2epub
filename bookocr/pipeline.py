"""Stage runner; expensive per-page work is cached with content validation."""
import json
import csv
import time
from pathlib import Path

import pymupdf

from .adapters import legacy_pages, structure_pages
from .epub import audit_epub, build_epub, write_audit
from .layout import reconstruct, render_preview
from .review import apply_corrections, detect, export_review, second_results
from .storage import Workspace, atomic_text, digest, pages_arg, signature, write_json
from .structure import build_book, escape, simplify_markdown


def run(cfg, output, selection=None, until="audit"):
    started = time.perf_counter()
    work = Workspace(output, cfg)
    root = work.root
    stats = {"stages": [], "cache_hits": 0, "processed_pages": 0}
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    def checkpoint(stage, **details):
        event = {"stage": stage, "seconds_since_start": round(time.perf_counter()-started, 3), **details}
        stats["stages"].append(event)
        with (log_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        write_json(root / "run_summary.json", stats)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return until == stage

    with pymupdf.open(cfg["pdf"]) as pdf:
        if any(s["page"] > len(pdf) for s in cfg.get("sections", [])):
            raise ValueError("Configured section is outside PDF")
        selection = selection or cfg.get("sample_pages", "1-10")
        numbers = pages_arg(selection, len(pdf))
        source = legacy_pages(cfg["ocr"]) if cfg.get("source_format", "legacy") == "legacy" else structure_pages(cfg["ocr"], cfg.get("repair_structure_regions", False))
        pages, found = [], set()
        for p in source:
            n = p["page"]
            if n > len(pdf):
                raise ValueError(f"OCR page {n} is outside PDF")
            if n not in numbers:
                continue
            found.add(n)
            p["width"] = pdf[n-1].rect.width*cfg.get("ocr_scale", 2)
            p["height"] = pdf[n-1].rect.height*cfg.get("ocr_scale", 2)
            if p["source"] == "pp-structurev3":
                p["width"] = p["raw_page"].get("width", p["width"])
                p["height"] = p["raw_page"].get("height", p["height"])
            # Keep a normalized immutable snapshot alongside reconstructed output.
            original, _ = work.cached_json(f"raw_pages/page_{n:04}.json", ["import", p], lambda p=p: p)
            data, hit = work.cached_json(f"layout_json/page_{n:04}.json", ["layout", original],
                                         lambda p=original: reconstruct(p, cfg))
            pages.append(data)
            stats["cache_hits"] += hit
        if found != set(numbers):
            raise ValueError(f"Missing selected OCR pages: {sorted(set(numbers)-found)}")
        pages.sort(key=lambda p: p["page"])
        preview_dir = root / "layout_previews"
        preview_dir.mkdir(exist_ok=True)
        for p in pages[:cfg.get("preview_limit", 20)]:
            render_preview(pdf[p["page"]-1], p, preview_dir / f"page_{p['page']:04}.png")
        stats["processed_pages"] = len(pages)
        stats["selected_pages"] = numbers
        text = "\n\n".join(f"<!-- PDF page {p['page']} -->\n" + "\n".join(escape(b["text"]) for b in p["blocks"]) for p in pages)
        atomic_text(root / "layout_reconstructed.md", text + "\n")
        if checkpoint("layout", pages=len(pages), cached=stats["cache_hits"]):
            return stats
        second = second_results(root / "review/second_engine.jsonl")
        flags, quality = detect(pages, cfg, second)
        export_review(root / "review", flags, quality, cfg, pdf)
        if checkpoint("quality", flagged=len(flags)):
            return stats
        review_path = root / "review/reviewed_lines.csv"
        corrected, applied = apply_corrections(pages, review_path)
        review_hash = digest(review_path)
        for p in corrected:
            work.cached_json(f"corrected/page_{p['page']:04}.json", [p, review_hash], lambda p=p: p)
        book_dir = root / "book"
        text, provenance = build_book(corrected, cfg, book_dir, pdf)
        atomic_text(book_dir / "book_structured.md", text)
        write_json(book_dir / "source_map.json", provenance)
        write_json(root / "review/applied_corrections.json", applied)
        if checkpoint("structure", sections=provenance["section_count"], applied_corrections=len(applied)):
            return stats
        from opencc import OpenCC
        converter = OpenCC(cfg.get("opencc", "t2s"))
        simple = simplify_markdown(text, converter)
        atomic_text(book_dir / "book_simplified.md", simple)
        if checkpoint("simplify", characters=len(simple)):
            return stats
        publication = root / "epub/book.epub"
        # Publication is reusable only if all material inputs and the EPUB bytes match.
        pub_key = signature([simple, cfg, work.code, {p.name: digest(p) for p in (book_dir / "images").glob("*")}])
        manifest = root / "epub/build.json"
        from .storage import read_json
        try:
            previous = read_json(manifest) if manifest.exists() else {}
        except (OSError, ValueError):
            previous = {}
        if not (publication.exists() and previous.get("key") == pub_key and previous.get("sha256") == digest(publication)):
            result = build_epub(book_dir / "book_simplified.md", publication, cfg, pdf, converter)
            write_json(manifest, {"key": pub_key, "sha256": digest(publication), **result})
            pub_cached = False
        else:
            pub_cached = True
        if checkpoint("epub", cached=pub_cached):
            return stats
        report = audit_epub(publication, converter.convert(cfg["title"]), provenance["section_count"])
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            resolved = {r["block_id"] for r in csv.DictReader(stream) if r["status"] in {"accepted", "rejected"}}
        report["pending_review_blocks"] = len(flags) - len(resolved & {r["block_id"] for r in flags})
        report["partial_book"] = numbers != list(range(1, len(pdf)+1))
        write_audit(root / "reports", report)
        checkpoint("audit", status=report["status"], errors=len(report["errors"]))
        if report["errors"]:
            raise ValueError("EPUB package audit failed: " + "; ".join(report["errors"]))
    return stats
