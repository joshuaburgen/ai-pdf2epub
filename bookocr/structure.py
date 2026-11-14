"""Explicit chapter boundaries and conservative paragraphs, with source maps."""
import html
import re
from pathlib import Path

from .review import crop


SKIP_ROLES = {"page_number", "running_header", "header", "footer", "exclude", "footnote", "section_title", "image"}


def escape(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"([\\`*_\[\]#!|])", r"\\\1", text)
    text = re.sub(r"(?m)^([+-])(?=\s)", r"\\\1", text)
    return re.sub(r"(?m)^(\d+)([.)])(?=\s)", r"\1\\\2", text)


def section_for(page, sections):
    return next((s for s in reversed(sections) if s["page"] <= page), None)


def paragraphs(blocks, direction, merge=True):
    result = []
    for b in blocks:
        if b["role"] in SKIP_ROLES or not b["text"].strip():
            continue
        special = b["role"] in {"title", "doc_title", "paragraph_title", "figure_title", "caption"}
        region = b.get("region", b.get("region_path"))
        if (merge and result and not special and not result[-1]["special"]
                and not re.search(r"[。！？.!?][”’」』）)]?$", result[-1]["text"])
                and result[-1]["region"] == region):
            # Only merge within a region and page. Cross-page joins are opt-in below.
            separator = " " if direction == "horizontal" and re.search(r"[A-Za-z0-9]$", result[-1]["text"]) and re.match(r"[A-Za-z0-9]", b["text"]) else ""
            result[-1]["text"] += separator + b["text"]
            result[-1]["ids"].append(b["id"])
        else:
            result.append({"text": b["text"], "ids": [b["id"]], "special": special,
                           "region": region})
    return result


def build_book(pages, cfg, root, pdf):
    root = Path(root)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    blocks_by_id = {b["id"]: b for p in pages for b in p["blocks"]}
    selected = {p["page"] for p in pages}
    emitted, output, source_map, exclusions = set(), [], [], []
    configured_sections = cfg.get("sections", [])
    footnotes = cfg.get("footnotes", [])
    for fn in footnotes:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", fn["id"]):
            raise ValueError("Footnote IDs must use ASCII letters, digits, _ or -")
        if fn["anchor_block_id"] not in blocks_by_id or fn["body_block_id"] not in blocks_by_id:
            if any(key in blocks_by_id for key in (fn["anchor_block_id"], fn["body_block_id"])):
                raise ValueError("Selected pages contain only half of a configured footnote")
            continue
        blocks_by_id[fn["body_block_id"]]["role"] = "footnote"
    for page in pages:
        n = page["page"]
        if n in cfg.get("exclude_pages", []):
            exclusions.append({"page": n, "reason": "configured_exclude_page"})
            continue
        section = section_for(n, configured_sections)
        if section and section["page"] not in emitted:
            title = section["title"]
            if section["page"] not in selected:
                title += "（选页片段）"
            output += ["## " + escape(title), ""]
            emitted.add(section["page"])
        if not section and "unclassified" not in emitted:
            output += ["## 未分类前置页面", ""]
            emitted.add("unclassified")
        output += [f"<!-- PDF page {n} -->", ""]
        figures = [v for v in cfg.get("figures", []) if v["page"] == n]
        figures += [{"page": n, "bbox": b["bbox"], "caption": ""}
                    for b in page["blocks"] if b["role"] in {"image", "figure", "chart"}
                    and not any(v["bbox"] == b["bbox"] for v in figures)]
        hidden = {key for v in figures for key in v.get("caption_block_ids", [])}
        text_blocks = [b for b in page["blocks"] if b["id"] not in hidden]
        for para in paragraphs(text_blocks, page["direction"], page.get("paragraph_mode") != "lines"):
            value = escape(para["text"])
            for fn in footnotes:
                if fn["anchor_block_id"] in para["ids"] and fn["body_block_id"] in blocks_by_id:
                    value += f"[^{fn['id']}]"
            # Joining across pages is allowed only through explicit approved source IDs.
            joins = cfg.get("cross_page_joins", [])
            requested = next((j for j in joins if j["next"] == para["ids"][0]), None)
            if requested and source_map and source_map[-1]["block_ids"][-1] == requested["previous"]:
                if n != source_map[-1]["page"] + 1 or source_map[-1]["section"] != (section or {}).get("page"):
                    raise ValueError("Cross-page join must remain within consecutive pages of one section")
                # The preceding paragraph is tracked by its exact output position.
                output[source_map[-1]["output_index"]] += value
                source_map[-1]["block_ids"].extend(para["ids"])
                source_map[-1]["joined_page"] = n
                continue
            index = len(output)
            output += [value, ""]
            source_map.append({"page": n, "section": (section or {}).get("page"),
                               "block_ids": para["ids"], "output_index": index})
        for index, figure in enumerate(figures):
            file = image_dir / f"page_{n:04}_figure_{index:02}.png"
            crop(pdf[n-1], figure["bbox"], cfg.get("ocr_scale", 2), file)
            caption = figure.get("caption") or "".join(blocks_by_id[key]["text"] for key in figure.get("caption_block_ids", []))
            output += [f"![{escape(caption)}](images/{file.name})", ""]
    for fn in footnotes:
        if fn["anchor_block_id"] in blocks_by_id and fn["body_block_id"] in blocks_by_id:
            output += [f"[^{fn['id']}]: {escape(blocks_by_id[fn['body_block_id']]['text'])}", ""]
    return "\n".join(output), {"paragraphs": source_map, "excluded_pages": exclusions,
                                  "block_dispositions": [{"id": b["id"], "page": p["page"], "role": b["role"], "empty": not bool(b["text"].strip())} for p in pages for b in p["blocks"]],
                                  "section_count": len(emitted),
                                  "selected_pages": sorted(selected),
                                  "status": "draft_requires_image_review"}


def simplify_markdown(text, converter):
    # Protect structural syntax, link destinations, comments and code spans.
    protected = r"(<!--.*?-->|!?\[[^\]]*\]\([^)]*\)|\[\^[^\]]+\]|`[^`]*`)"
    def convert_piece(piece):
        match = re.fullmatch(r"(!?\[)([^\]]*)(\]\([^)]*\))", piece, re.S)
        if match:
            return match[1] + converter.convert(match[2]) + match[3]
        return piece if re.fullmatch(protected, piece, re.S) else converter.convert(piece)
    return "".join(convert_piece(piece) for piece in re.split(protected, text, flags=re.S) if piece)
