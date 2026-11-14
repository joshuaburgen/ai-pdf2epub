"""Explicit adapters; retain original records instead of guessing nested schemas."""
import json
import statistics
from pathlib import Path

from .storage import box_valid


def block(page, index, text, box, score, raw, **extra):
    if not isinstance(text, str) or not box_valid(box):
        raise ValueError(f"Invalid text/box on page {page}, item {index}")
    if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
        raise ValueError(f"Invalid score on page {page}, item {index}")
    return {"id": f"p{page:04d}-b{index:04d}", "source_item_index": index,
            "text": text, "bbox": list(box), "confidence": score,
            "role": "text", "raw": raw, **extra}


def legacy_pages(path):
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            data = json.loads(line)
            n = data["page"]
            if type(n) is not int or n < 1 or n in seen:
                raise ValueError(f"Invalid/duplicate page on JSONL line {line_number}")
            seen.add(n)
            yield {"schema_version": 1, "page": n, "source": "legacy-paddleocr",
                   "raw_page": {k: v for k, v in data.items() if k != "items"},
                   "blocks": [block(n, i, v["text"], v["box"], v["score"], v)
                              for i, v in enumerate(data["items"])]}


def structure_page(data, page):
    # JsonMixin may wrap the serialized result in {"res": ...}.
    data = data.get("res", data)
    if "parsing_res_list" not in data:
        raise ValueError("PP-StructureV3 result lacks parsing_res_list")
    blocks = []
    for index, value in enumerate(data["parsing_res_list"]):
        blocks.append(block(page, index, value["block_content"], value["block_bbox"],
                            None, value, role=value["block_label"],
                            engine_order=value.get("block_order")))
    ocr = data.get("overall_ocr_res", {})
    boxes, scores = ocr.get("rec_boxes", []), ocr.get("rec_scores", [])
    unassigned = []
    if boxes and len(boxes) == len(scores):
        assigned = {b["id"]: [] for b in blocks}
        for index, (box, score) in enumerate(zip(boxes, scores)):
            if not box_valid(box) or type(score) not in (int, float) or not 0 <= score <= 1:
                continue
            x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
            owners = [b for b in blocks if b["bbox"][0] <= x <= b["bbox"][2]
                      and b["bbox"][1] <= y <= b["bbox"][3] and b["role"] not in {"image", "figure"}]
            if owners:
                owner = min(owners, key=lambda b: (b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]))
                assigned[owner["id"]].append((index, score))
            elif index < len(ocr.get("rec_texts", [])) and ocr["rec_texts"][index].strip():
                unassigned.append(index)
        for b in blocks:
            members = assigned[b["id"]]
            if members:
                b["confidence"] = statistics.mean(v for _, v in members)
                b["confidence_method"] = "mean_of_geometrically_assigned_rec_scores"
                b["ocr_item_indices"] = [i for i, _ in members]
    return {"schema_version": 1, "page": page, "source": "pp-structurev3",
            "raw_page": data, "unassigned_ocr_item_indices": unassigned, "blocks": blocks}


def repair_structure_regions(page):
    """Use layout regions, but reconstruct vertical text from retained OCR lines."""
    from .layout import order_region, orientation
    ocr = page["raw_page"].get("overall_ocr_res", {})
    texts, boxes, scores = (ocr.get(k, []) for k in ("rec_texts", "rec_boxes", "rec_scores"))
    if not (len(texts) == len(boxes) == len(scores)):
        raise ValueError("Cannot repair regions: OCR array lengths differ")
    for b in page["blocks"]:
        if orientation(b) != "vertical" or not b.get("ocr_item_indices"):
            continue
        lines = [block(page["page"], i, texts[i], boxes[i], scores[i], {}) for i in b["ocr_item_indices"]]
        ordered = order_region(lines, "vertical")
        b["engine_text"] = b["text"]
        b["text"] = "".join(v["text"] for v in ordered)
        b["ordered_ocr_item_indices"] = [v["source_item_index"] for v in ordered]
        b["text_method"] = "layout_region_with_ordered_ocr_lines"
    return page


def structure_pages(path, repair=False):
    seen = set()
    for file in sorted(Path(path).glob("page_*.json")):
        wrapper = json.loads(file.read_text(encoding="utf-8"))
        # Our recognition backend explicitly records physical PDF page numbers.
        n = wrapper.get("pdf_page")
        if type(n) is not int or n < 1 or n in seen:
            raise ValueError(f"Missing/duplicate pdf_page in {file}; import with an explicit page map")
        seen.add(n)
        page = structure_page(wrapper["result"], n)
        yield repair_structure_regions(page) if repair else page
