"""Region-first deterministic ordering with explicit per-book overrides."""
import copy
import re
import statistics


def center(box, axis):
    return (box[axis] + box[axis+2]) / 2


def orientation(block):
    x1, y1, x2, y2 = block["bbox"]
    return "vertical" if y2-y1 > 2*(x2-x1) else "horizontal"


def gap_split(blocks, axis, threshold):
    intervals = sorted((b["bbox"][axis], b["bbox"][axis+2]) for b in blocks)
    end, gaps = intervals[0][1], []
    for start, stop in intervals[1:]:
        if start-end > threshold:
            gaps.append((start-end, (start+end)/2))
        end = max(end, stop)
    if not gaps:
        return None
    cut = max(gaps)[1]
    return ([b for b in blocks if center(b["bbox"], axis) < cut],
            [b for b in blocks if center(b["bbox"], axis) >= cut])


def order_region(blocks, direction):
    """Split whitespace regions before clustering columns or rows within each region."""
    if not blocks:
        return []
    char_size = statistics.median(min(b["bbox"][2]-b["bbox"][0],
                                      b["bbox"][3]-b["bbox"][1]) for b in blocks)
    # XY cuts separate a title, multi-column region or caption island.
    for axis in (1, 0):
        split = gap_split(blocks, axis, max(12, char_size*1.8))
        if split and all(split):
            groups = list(split)
            if axis == 0 and direction == "vertical":
                groups.reverse()
            for index, group in enumerate(groups):
                for b in group:
                    b.setdefault("region_path", []).append([axis, index])
            return sum((order_region(g, direction) for g in groups), [])
    major_axis = 0 if direction == "vertical" else 1
    minor_axis = 1-major_axis
    groups = []
    for b in sorted(blocks, key=lambda b: center(b["bbox"], major_axis)):
        candidate = next((g for g in groups
                          if abs(center(b["bbox"], major_axis)
                                 - statistics.mean(center(v["bbox"], major_axis) for v in g))
                          <= char_size*.65), None)
        if candidate is None:
            groups.append([b])
        else:
            candidate.append(b)
    if direction == "vertical":
        groups.reverse()
    return sum((sorted(g, key=lambda b: (b["bbox"][minor_axis], b["source_item_index"]))
                for g in groups), [])


def reconstruct(page, cfg):
    result = copy.deepcopy(page)
    width, height = page["width"], page["height"]
    override = cfg.get("page_overrides", {}).get(str(page["page"]), {})
    regions = override.get("regions", [])
    buckets = {i: [] for i in range(len(regions))}
    unassigned, ignored = [], []
    for b in result["blocks"]:
        box = b["bbox"]
        margin = (box[0] < width*.14 or box[2] > width*.9
                  or box[1] < height*.10 or box[3] > height*.93)
        if margin and (re.fullmatch(r"\d{1,4}", b["text"].strip()) or b["role"] == "number"):
            b["role"] = "page_number"
        if b["text"].strip() in cfg.get("running_headers", []) and margin:
            b["role"] = "running_header"
        if b["id"] in cfg.get("block_roles", {}):
            b["role"] = cfg["block_roles"][b["id"]]
        assigned = False
        for index, region in enumerate(regions):
            x, y = center(box, 0)/width, center(box, 1)/height
            x1, y1, x2, y2 = region["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                b["role"] = region.get("role", b["role"])
                b["region"] = index
                buckets[index].append(b)
                assigned = True
                break
        if not assigned:
            if b["role"] in ("page_number", "running_header", "header", "footer"):
                ignored.append(b)
            else:
                unassigned.append(b)
    direction = override.get("direction", cfg.get("direction", "auto"))
    if direction == "auto":
        nonempty = [b for b in unassigned if b["text"].strip()]
        direction = "vertical" if sum(orientation(b) == "vertical" for b in nonempty) > len(nonempty)/2 else "horizontal"
    if direction not in ("horizontal", "vertical"):
        raise ValueError(f"Invalid direction: {direction}")
    if page["source"] == "pp-structurev3" and not regions:
        ordered = sorted(unassigned, key=lambda b: (b.get("engine_order") is None,
                                                    b.get("engine_order") or 0,
                                                    b["source_item_index"]))
    else:
        ordered = []
        for i, region in enumerate(regions):
            ordered.extend(order_region(buckets[i], region.get("direction", direction)))
        ordered.extend(order_region(unassigned, direction))
    ordered.extend(ignored)
    for index, b in enumerate(ordered):
        b["reading_order"] = index
    assert sorted(b["id"] for b in ordered) == sorted(b["id"] for b in page["blocks"])
    result.update(blocks=ordered, direction=direction,
                  paragraph_mode=override.get("paragraph_mode", "heuristic" if direction == "vertical" and page["source"] != "pp-structurev3" else "lines"),
                  layout_method="configured-regions" if regions else "engine-order" if page["source"] == "pp-structurev3" else "xy-cut-clusters")
    return result


def render_preview(pdf_page, page, path):
    """Render an inspection overlay; never save changes into the source PDF."""
    from PIL import Image, ImageDraw
    pix = pdf_page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(image)
    sx, sy = pix.width/page["width"], pix.height/page["height"]
    for b in page["blocks"]:
        box = b["bbox"]
        coordinates = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
        color = "red" if b["role"] in ("page_number", "running_header", "exclude") else "blue"
        draw.rectangle(coordinates, outline=color, width=1)
        draw.text((coordinates[0], coordinates[1]), str(b["reading_order"]), fill=color)
    image.save(path)
