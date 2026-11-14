"""Optional recognition, always restricted to an explicit page or review whitelist."""
import csv
import importlib.metadata
import json
import math
import time
from pathlib import Path

from .storage import atomic_text, digest, pages_arg, read_json, signature, write_json
from .compat import certificate_fallback


@certificate_fallback()
def recognize(pdf_path, out, selection, device="gpu:0", scale=2):
    import pymupdf
    out = Path(out).resolve()
    pdf_path = Path(pdf_path).resolve()
    if out == pdf_path or out in pdf_path.parents:
        raise ValueError("Recognition output must not contain the PDF input")
    with pymupdf.open(pdf_path) as pdf:
        numbers = pages_arg(selection, len(pdf))
        identity = {"pdf_sha256": digest(pdf_path), "scale": scale, "device": device,
                    "paddleocr": importlib.metadata.version("paddleocr"), "backend": "PPStructureV3",
                    "paddlex": importlib.metadata.version("paddlex"), "backend_code": digest(__file__),
                    "text_models": ["PP-OCRv5_server_det", "PP-OCRv5_server_rec"],
                    "options": {"use_doc_orientation_classify": False, "use_doc_unwarping": False,
                                "use_textline_orientation": False, "use_table_recognition": False,
                                "use_formula_recognition": False, "use_seal_recognition": False,
                                "use_chart_recognition": False, "use_region_detection": False}}
        marker = out / "recognition_run.json"
        if out.exists() and not marker.exists() and any(out.iterdir()):
            raise ValueError("Recognition output directory is not owned by bookocr")
        if marker.exists() and read_json(marker) != identity:
            raise ValueError("Recognition parameters changed; choose a new output directory")
        out.mkdir(parents=True, exist_ok=True)
        write_json(marker, identity)
        engine = None
        summary = []
        for n in numbers:
            target = out / f"page_{n:04}.json"
            if target.exists():
                try:
                    old = read_json(target)
                    if old["pdf_page"] == n and old["checksum"] == signature(old["result"]):
                        summary.append({"page": n, "cached": True})
                        continue
                except (ValueError, KeyError):
                    pass
            if engine is None:
                from paddleocr import PPStructureV3
                engine = PPStructureV3(device=device, **identity["options"],
                                       text_detection_model_name="PP-OCRv5_server_det",
                                       text_recognition_model_name="PP-OCRv5_server_rec")
            started = time.perf_counter()
            # In-memory rendering avoids creating or replacing an input PDF.
            import numpy as np
            pix = pdf[n-1].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            results = list(engine.predict(rgb[:, :, ::-1].copy()))
            if len(results) != 1:
                raise ValueError(f"Expected one result for PDF page {n}, got {len(results)}")
            result = results[0].json
            if isinstance(result, str):
                result = json.loads(result)
            write_json(target, {"pdf_page": n, "image_size": [pix.width, pix.height],
                                "result": result, "checksum": signature(result)})
            summary.append({"page": n, "cached": False, "seconds": round(time.perf_counter()-started, 3)})
            print(json.dumps(summary[-1]), flush=True)
        write_json(out / "recognition_summary.json", summary)
        return summary


@certificate_fallback()
def recheck(root, selected_ids, model_name="chinese_cht_PP-OCRv3"):
    root = Path(root)
    with (root / "review/flagged_lines.csv").open(encoding="utf-8-sig", newline="") as stream:
        flags = {r["block_id"]: r for r in csv.DictReader(stream)}
    if not selected_ids or not set(selected_ids) <= flags.keys():
        raise ValueError("Choose explicit IDs present in review/flagged_lines.csv")
    from .review import second_results
    output = root / "review/second_engine.jsonl"
    saved = second_results(output)
    engine = None
    for key in selected_ids:
        row = flags[key]
        old = saved.get(key, {})
        version = importlib.metadata.version("cnocr")
        old_text = old.get("text", "")
        old_score = old.get("confidence")
        valid_success = (old.get("status") == "ok" and bool(old_text.strip())
                         and type(old_score) in (int, float) and math.isfinite(old_score)
                         and 0 <= old_score <= 1)
        valid_empty = old.get("status") == "empty" and not old_text.strip() and old_score is None
        if (old.get("original_sha256") == signature(row["original_text"])
                and old.get("model") == model_name and old.get("version") == version
                and (valid_success or valid_empty)):
            continue
        if engine is None:
            from cnocr import CnOcr
            engine = CnOcr(rec_model_name=model_name, context="cpu")
        value = {"block_id": key, "original_sha256": signature(row["original_text"]),
                 "engine": "cnocr", "model": model_name, "version": version}
        try:
            result = engine.ocr(str(root / "review" / row["crop"]))
            text = "".join(r["text"] for r in result)
            scores = [float(r["score"]) for r in result if r["text"].strip()]
            value.update(text=text, confidence=None)
            if not text.strip():
                value.update(status="empty")
            elif not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                raise ValueError("CnOCR returned a non-finite/out-of-range recognition score")
            else:
                value.update(status="ok", confidence=sum(scores)/len(scores))
        except Exception as exc:
            value.update(status="error", error=f"{type(exc).__name__}: {exc}")
        saved[key] = value
        atomic_text(output, "".join(json.dumps(v, ensure_ascii=False, allow_nan=False)+"\n" for v in saved.values()))
    return {"selected": len(selected_ids), "saved": len(saved)}
