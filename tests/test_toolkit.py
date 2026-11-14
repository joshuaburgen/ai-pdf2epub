import copy
import csv
import json
import os
import shutil
import ssl
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
from opencc import OpenCC

from bookocr.adapters import block, legacy_pages, repair_structure_regions, structure_page, structure_pages
from bookocr.epub import audit_epub
from bookocr.layout import reconstruct
from bookocr.pipeline import run
from bookocr.review import apply_corrections, csv_text, detect
from bookocr.storage import Workspace, atomic_text, digest, load_config, pages_arg, signature, write_json
from bookocr.structure import build_book, escape, simplify_markdown
from bookocr.compat import certificate_fallback


ROOT = Path(__file__).resolve().parents[1]


def page(blocks, number=1):
    return {"page": number, "blocks": blocks, "source": "legacy-paddleocr", "width": 600, "height": 800}


def b(index, text, box, number=1, score=.95):
    return block(number, index, text, box, score, {"untouched": text})


class DataAndLayoutTests(unittest.TestCase):
    def test_engine_number_region_at_margin_is_not_body_text(self):
        number = b(0, "I。", [10, 20, 30, 80])
        number["role"] = "number"
        letter = b(1, "I", [250, 300, 270, 350])
        letter["role"] = "text"
        roles = {v["id"]: v["role"] for v in reconstruct(page([number, letter]), {})["blocks"]}
        self.assertEqual(roles[number["id"]], "page_number")
        self.assertEqual(roles[letter["id"]], "text")
    def test_page_selection_rejects_reversed_and_out_of_bounds(self):
        self.assertEqual(pages_arg("1-3,2,5", 5), [1, 2, 3, 5])
        for value in ("0", "4-2", "6", "-1", "", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pages_arg(value, 5)

    def test_legacy_validates_duplicates_and_preserves_extra_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"ocr.jsonl"
            raw = {"page": 1, "note": "extra", "items": [{"text": "原文", "box": [1, 2, 3, 4], "score": .9, "extra": 42}]}
            atomic_text(path, json.dumps(raw)+"\n")
            result = list(legacy_pages(path))[0]
            self.assertEqual(result["blocks"][0]["raw"], raw["items"][0])
            self.assertEqual(result["raw_page"]["note"], "extra")
            atomic_text(path, (json.dumps(raw)+"\n")*2)
            with self.assertRaises(ValueError):
                list(legacy_pages(path))

    def test_bad_coordinates_and_nonfinite_scores_fail(self):
        for box, score in [([0, 0, 0, 3], .9), ([0, 0, 4, 3], float("nan")), ([0, 0, float("inf"), 3], .9)]:
            with self.assertRaises(ValueError):
                b(0, "文字", box, score=score)

    def test_structure_adapter_keeps_raw_and_does_not_invent_confidence(self):
        raw = {"parsing_res_list": [{"block_content": "段落", "block_bbox": [1, 2, 3, 4],
                                     "block_label": "text", "block_order": 2}], "overall_ocr_res": {"rec_scores": [.93]}}
        p = structure_page({"res": raw}, 8)
        self.assertIsNone(p["blocks"][0]["confidence"])
        self.assertEqual(p["raw_page"], raw)
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/"page_0008.json", {"pdf_page": 8, "result": raw})
            write_json(Path(d)/"recognition_run.json", {"backend": "metadata"})
            self.assertEqual(len(list(structure_pages(d))), 1)

    def test_vertical_columns_cluster_fragments_and_keep_all_sources(self):
        p = page([b(0, "左列", [100, 100, 120, 700]),
                  b(1, "右列下", [300, 400, 320, 700]), b(2, "右列上", [300, 100, 320, 350])])
        before = copy.deepcopy(p)
        result = reconstruct(p, {"direction": "vertical"})
        self.assertEqual([x["text"] for x in result["blocks"]], ["右列上", "右列下", "左列"])
        self.assertEqual(p, before)
        self.assertEqual({x["id"] for x in result["blocks"]}, {x["id"] for x in p["blocks"]})

    def test_structure_confidence_comes_from_recognition_boxes(self):
        raw = {"parsing_res_list": [{"block_content": "段落", "block_bbox": [0, 0, 100, 200],
                                     "block_label": "text", "block_order": 0}],
               "overall_ocr_res": {"rec_boxes": [[10, 10, 40, 40], [10, 50, 40, 80]], "rec_scores": [.6, .8]}}
        item = structure_page(raw, 1)["blocks"][0]
        self.assertAlmostEqual(item["confidence"], .7)
        self.assertEqual(item["ocr_item_indices"], [0, 1])

    def test_structure_region_repair_preserves_engine_text_and_ocr_members(self):
        raw = {"parsing_res_list": [{"block_content": "左右", "block_bbox": [0, 0, 100, 400],
                                     "block_label": "text", "block_order": 1}],
               "overall_ocr_res": {"rec_texts": ["左", "右"],
                                   "rec_boxes": [[10, 20, 20, 350], [70, 20, 80, 350]], "rec_scores": [.9, .95]}}
        p = repair_structure_regions(structure_page(raw, 1))
        self.assertEqual(p["blocks"][0]["text"], "右左")
        self.assertEqual(p["blocks"][0]["engine_text"], "左右")
        self.assertEqual(p["blocks"][0]["ordered_ocr_item_indices"], [1, 0])
        self.assertEqual(p["raw_page"], raw)

    def test_horizontal_columns_are_not_interleaved_by_global_y(self):
        p = page([b(0, "左一", [20, 100, 220, 120]), b(1, "右一", [370, 100, 570, 120]),
                  b(2, "左二", [20, 140, 220, 160]), b(3, "右二", [370, 140, 570, 160])])
        result = reconstruct(p, {"direction": "horizontal"})
        self.assertEqual([v["text"] for v in result["blocks"]], ["左一", "左二", "右一", "右二"])

    def test_only_margin_numbers_are_page_number_candidates(self):
        result = reconstruct(page([b(0, "123", [250, 300, 280, 320]), b(1, "9", [10, 10, 20, 30])]), {})
        roles = {v["id"]: v["role"] for v in result["blocks"]}
        self.assertEqual(roles["p0001-b0000"], "text")
        self.assertEqual(roles["p0001-b0001"], "page_number")

    def test_engine_order_is_used_instead_of_resorting_layout(self):
        p = page([b(0, "第一", [100, 100, 300, 120]), b(1, "第二", [100, 50, 300, 70])])
        p["source"] = "pp-structurev3"
        for i,v in enumerate(p["blocks"]):
            v["engine_order"] = i
        self.assertEqual([v["text"] for v in reconstruct(p, {})["blocks"]], ["第一", "第二"])


class ReviewAndStorageTests(unittest.TestCase):
    def test_unbalanced_quotes_and_stray_separator_are_review_candidates(self):
        p = page([b(0, "「引用未闭合", [20, 20, 50, 300]),
                  b(1, "功能|——思想", [70, 20, 100, 300])])
        flags, _ = detect([p], {})
        reasons = {v["block_id"]: v["reasons"] for v in flags}
        self.assertIn("unbalanced_quote_candidate", reasons["p0001-b0000"])
        self.assertIn("stray_separator_candidate", reasons["p0001-b0001"])
    def test_an_image_with_no_text_is_not_an_ocr_failure(self):
        image = b(0, "", [10, 10, 300, 300], score=None)
        image["role"] = "image"
        flags, report = detect([page([image])], {})
        self.assertEqual(flags, [])
        self.assertFalse(any(v["reason"] == "block_confidence_unavailable" for v in report["page_candidates"]))

    def test_certificate_workaround_preserves_verification_and_restores_function(self):
        original = ssl.create_default_context
        def broken_store(*args, **kwargs):
            if not kwargs.get("cafile"):
                raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA]")
            return original(*args, **kwargs)
        with patch.object(ssl, "create_default_context", broken_store):
            with certificate_fallback():
                context = ssl.create_default_context()
                self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(context.check_hostname)
            self.assertIs(ssl.create_default_context, broken_store)

    def test_second_engine_states_and_stale_text(self):
        p = page([b(0, "原文", [1, 2, 20, 90], score=.5)])
        flags, _ = detect([p], {})
        self.assertEqual(flags[0]["second_status"], "not_run")
        other = {"p0001-b0000": {"status": "ok", "original_sha256": signature("原文"), "text": "異文"}}
        self.assertIn("engine_disagreement", detect([p], {}, other)[0][0]["reasons"])
        other["p0001-b0000"]["original_sha256"] = "stale"
        with self.assertRaises(ValueError):
            detect([p], {}, other)

    def test_human_correction_requires_acceptance_and_matching_original(self):
        p = page([b(0, "原文", [1, 2, 20, 90])])
        row = {"block_id": "p0001-b0000", "original_text": "原文", "corrected_text": "正字", "status": "pending"}
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/"review.csv"
            atomic_text(file, csv_text([row]))
            self.assertEqual(apply_corrections([p], file)[0][0]["blocks"][0]["text"], "原文")
            row["status"] = "accepted"
            atomic_text(file, csv_text([row]))
            self.assertEqual(apply_corrections([p], file)[0][0]["blocks"][0]["text"], "正字")
            self.assertEqual(p["blocks"][0]["text"], "原文")
            row["original_text"] = "错版本"
            atomic_text(file, csv_text([row]))
            with self.assertRaises(ValueError):
                apply_corrections([p], file)

    def test_duplicate_corrections_fail(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/"review.csv"
            row = {"block_id": "p0001-b0000", "original_text": "原文", "status": "accepted", "corrected_text": "字"}
            atomic_text(file, csv_text([row, row]))
            with self.assertRaises(ValueError):
                apply_corrections([page([b(0, "原文", [1, 2, 20, 90])])], file)

    def test_cache_reuse_corruption_and_input_changes(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            atomic_text(base/"source", "original")
            cfg = {"pdf": str(base/"source"), "ocr": str(base/"source")}
            ws = Workspace(base/"out", cfg)
            self.assertFalse(ws.cached_json("data/p1.json", "key", lambda: {"text": "A"})[1])
            self.assertTrue(ws.cached_json("data/p1.json", "key", lambda: self.fail("cache missed"))[1])
            saved = json.loads((base/"out/data/p1.json").read_text(encoding="utf-8"))
            saved["data"] = {"text": "tampered"}
            write_json(base/"out/data/p1.json", saved)
            self.assertEqual(ws.cached_json("data/p1.json", "key", lambda: {"text": "rebuilt"})[0]["text"], "rebuilt")
            atomic_text(base/"source", "changed")
            with self.assertRaises(ValueError):
                Workspace(base/"out", cfg)
            with self.assertRaises(ValueError):
                Workspace(base, cfg)

    def test_recognition_summary_changes_do_not_invalidate_page_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            atomic_text(base/"pdf", "input")
            source = base/"recognition"
            write_json(source/"page_0001.json", {"pdf_page": 1})
            write_json(source/"recognition_summary.json", {"cached": False})
            cfg = {"pdf": str(base/"pdf"), "ocr": str(source)}
            first = Workspace(base/"run", cfg)
            write_json(source/"recognition_summary.json", {"cached": True})
            self.assertEqual(first.identity, Workspace(base/"run", cfg).identity)

    def test_opencc_protects_asset_paths_anchors_and_comments(self):
        text = "## 繁體\n<!-- 繁體 -->\n![圖說](images/繁體.png)\n[連結](繁體.html#錨點)\n[^繁體]\n`繁體`"
        result = simplify_markdown(text, OpenCC("t2s"))
        self.assertIn("## 繁体", result)
        self.assertIn("![图说](images/繁體.png)", result)
        self.assertIn("[连结](繁體.html#錨點)", result)
        self.assertIn("<!-- 繁體 -->", result)
        self.assertIn("[^繁體]", result)
        self.assertIn("`繁體`", result)

    def test_source_markup_is_escaped(self):
        self.assertEqual(escape("<script>*x*"), "&lt;script&gt;\\*x\\*")


class EndToEndTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        pdf = root/"input.pdf"
        with pymupdf.open() as doc:
            for _ in range(2):
                p = doc.new_page(width=300, height=400)
                p.insert_text((30, 40), "Source scan")
            doc.save(pdf)
        records = [
            {"page": 1, "items": [{"text": "繁體正文。", "box": [60, 100, 500, 140], "score": .99},
                                    {"text": "註釋內容。", "box": [60, 600, 500, 640], "score": .5}]},
            {"page": 2, "items": [{"text": "第二章正文。", "box": [60, 100, 500, 140], "score": .99}]}
        ]
        ocr = root/"ocr.jsonl"
        atomic_text(ocr, "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records))
        cfg = {"pdf": str(pdf), "ocr": str(ocr), "title": "測試書籍", "author": "作者",
               "pandoc": os.environ.get("PANDOC") or shutil.which("pandoc") or "pandoc", "direction": "horizontal",
               "sections": [{"page": 1, "title": "第一章", "kind": "chapter"},
                            {"page": 2, "title": "第二章", "kind": "chapter"}],
               "footnotes": [{"id": "note1", "anchor_block_id": "p0001-b0000", "body_block_id": "p0001-b0001"}],
               "figures": [{"page": 2, "bbox": [30, 30, 250, 90], "caption": "原圖"}]}
        return cfg

    def test_real_pandoc_epub_footnotes_images_navigation_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            before = {key: digest(cfg[key]) for key in ("pdf", "ocr")}
            out = Path(d)/"run"
            first = run(cfg, out, "all")
            self.assertEqual(first["processed_pages"], 2)
            audit = json.loads((out/"reports/audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["errors"], [])
            self.assertEqual(audit["section_headings"], 2)
            self.assertGreater(audit["footnote_references"], 0)
            self.assertGreaterEqual(audit["images"], 2)
            mtime = (out/"layout_json/page_0001.json").stat().st_mtime_ns
            second = run(cfg, out, "all")
            self.assertEqual(second["cache_hits"], 2)
            self.assertEqual((out/"layout_json/page_0001.json").stat().st_mtime_ns, mtime)
            self.assertTrue(next(v for v in second["stages"] if v["stage"] == "epub")["cached"])
            self.assertEqual(before, {key: digest(cfg[key]) for key in before})

    def test_audit_detects_broken_anchor_in_generated_epub(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            run(cfg, Path(d)/"run", "all")
            source = Path(d)/"run/epub/book.epub"
            target = Path(d)/"broken.epub"
            with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as damaged:
                for info in original.infolist():
                    data = original.read(info.filename)
                    if info.filename.endswith(".xhtml") and b"noteref" in data:
                        data = data.replace(b'href="#fn1"', b'href="#missing-note"')
                        # Handle Pandoc's cross-document references too.
                        data = data.replace(b'#fn1"', b'#missing-note"')
                    damaged.writestr(info, data)
            self.assertTrue(any("Broken anchor" in s for s in audit_epub(target)["errors"]))

    def test_partial_missing_ocr_fails_and_does_not_touch_sources(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            atomic_text(cfg["ocr"], json.dumps({"page": 1, "items": []})+"\n")
            before = digest(cfg["ocr"])
            with self.assertRaises(ValueError):
                run(cfg, Path(d)/"run", "all")
            self.assertEqual(before, digest(cfg["ocr"]))

    def test_config_paths_are_relative_to_config_and_chapter_count_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"config.json"
            write_json(path, {"pdf": "input.pdf", "ocr": "ocr.jsonl", "title": "书",
                              "chapter_count": 1, "sections": []})
            with self.assertRaises(ValueError):
                load_config(path)
            write_json(path, {"pdf": "input.pdf", "ocr": "ocr.jsonl", "title": "书"})
            self.assertEqual(load_config(path)["pdf"], str((Path(d)/"input.pdf").resolve()))

    def test_multicolumn_regression_keeps_original_source_indices(self):
        blocks = [b(0, "4", [10, 750, 40, 790], 4), b(1, "示例页眉", [70, 10, 500, 30], 4)]
        for index in range(2, 16):
            x = 550 if index == 13 else 80+index*20
            blocks.append(b(index, "右側第一欄" if index == 13 else f"示例正文第{index}欄", [x, 100, x+12, 700], 4))
        source = page(blocks, 4)
        result = reconstruct(source, {"direction": "vertical", "running_headers": ["示例页眉"]})
        body = [v for v in result["blocks"] if v["role"] == "text"]
        self.assertEqual(body[0]["text"], "右側第一欄")
        self.assertEqual(body[0]["source_item_index"], 13)
        self.assertEqual(len(result["blocks"]), 16)

    def test_recognition_whitelist_and_resume_without_reloading_model(self):
        from bookocr.backends import recognize
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            called, loads = [], []
            class FakeEngine:
                def __init__(self, **kwargs):
                    loads.append(kwargs["device"])
                def predict(self, image):
                    called.append(image.shape)
                    return [types.SimpleNamespace(json={"res": {"parsing_res_list": []}})]
            with patch.dict("sys.modules", {"paddleocr": types.SimpleNamespace(PPStructureV3=FakeEngine)}), \
                    patch("bookocr.backends.importlib.metadata.version", return_value="fixture-version"):
                recognize(cfg["pdf"], Path(d)/"recognition", "2")
                again = recognize(cfg["pdf"], Path(d)/"recognition", "2")
            self.assertEqual(loads, ["gpu:0"])
            self.assertEqual(len(called), 1)
            self.assertTrue(again[0]["cached"])
            self.assertFalse((Path(d)/"recognition/page_0001.json").exists())
            saved = json.loads((Path(d)/"recognition/page_0002.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["pdf_page"], 2)

    def test_cnocr_wrapper_only_processes_whitelisted_flags_and_resumes(self):
        from bookocr.backends import recheck
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            out = Path(d)/"run"
            run(cfg, out, "all", until="quality")
            called = []
            class FakeCnOcr:
                def __init__(self, **kwargs):
                    called.append("loaded")
                def ocr(self, path):
                    self_path = Path(path)
                    if not self_path.exists():
                        raise FileNotFoundError(path)
                    called.append(self_path.name)
                    return [{"text": "候選文字", "score": .8}]
            with patch.dict("sys.modules", {"cnocr": types.SimpleNamespace(CnOcr=FakeCnOcr)}), patch("bookocr.backends.importlib.metadata.version", return_value="test"):
                recheck(out, ["p0001-b0001"])
                recheck(out, ["p0001-b0001"])
                with self.assertRaises(ValueError):
                    recheck(out, ["p0001-b0000"])
            self.assertEqual(called, ["loaded", "p0001-b0001.png"])
            saved = json.loads((out/"review/second_engine.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ok")
            self.assertEqual(saved["original_sha256"], signature("註釋內容。"))

    def test_cnocr_empty_nan_result_is_serializable_and_not_success(self):
        from bookocr.backends import recheck
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            out = Path(d)/"run"
            run(cfg, out, "all", until="quality")
            loads = []
            class EmptyCnOcr:
                def __init__(self, **kwargs):
                    loads.append("loaded")
                def ocr(self, path):
                    return [{"text": "", "score": float("nan")}]
            old = {"block_id": "p0001-b0001", "original_sha256": signature("註釋內容。"),
                   "engine": "cnocr", "model": "chinese_cht_PP-OCRv3", "version": "test",
                   "status": "ok", "text": "", "confidence": None}
            atomic_text(out/"review/second_engine.jsonl", json.dumps(old)+"\n")
            with patch.dict("sys.modules", {"cnocr": types.SimpleNamespace(CnOcr=EmptyCnOcr)}), patch("bookocr.backends.importlib.metadata.version", return_value="test"):
                recheck(out, ["p0001-b0001"])
                recheck(out, ["p0001-b0001"])
            saved = json.loads((out/"review/second_engine.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "empty")
            self.assertIsNone(saved["confidence"])
            self.assertEqual(loads, ["loaded"])

    def test_cross_page_join_requires_explicit_adjacent_source_ids(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.fixture(d)
            cfg["sections"] = [{"page": 1, "title": "同一章"}]
            cfg["footnotes"] = []
            cfg["figures"] = []
            pages = [page([b(0, "跨頁前半", [40, 100, 70, 500])], 1),
                     page([b(0, "後半。", [40, 100, 70, 500], number=2)], 2)]
            for p in pages:
                p["direction"] = "vertical"
            with pymupdf.open(cfg["pdf"]) as pdf:
                plain, _ = build_book(copy.deepcopy(pages), cfg, Path(d)/"plain", pdf)
                self.assertNotIn("跨頁前半後半。", plain)
                cfg["cross_page_joins"] = [{"previous": "p0001-b0000", "next": "p0002-b0000"}]
                joined, source = build_book(copy.deepcopy(pages), cfg, Path(d)/"joined", pdf)
            self.assertIn("跨頁前半後半。", joined)
            self.assertEqual(source["paragraphs"][0]["block_ids"], ["p0001-b0000", "p0002-b0000"])


if __name__ == "__main__":
    unittest.main()
