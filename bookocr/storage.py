"""Validated configuration, content fingerprints and atomic derived-file writes."""
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pages_arg(value, total):
    if value == "all":
        return list(range(1, total + 1))
    result = set()
    for token in value.split(","):
        bounds = token.split("-")
        if len(bounds) not in (1, 2) or any(not b.isdecimal() for b in bounds):
            raise ValueError(f"Invalid page selection: {token}")
        lo, hi = int(bounds[0]), int(bounds[-1])
        if not 1 <= lo <= hi <= total:
            raise ValueError(f"Page range outside 1..{total}: {token}")
        result.update(range(lo, hi + 1))
    return sorted(result)


def box_valid(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    # Paths are relative to the configuration file, never to the caller's cwd.
    for key in ("pdf", "ocr", "pandoc"):
        if cfg.get(key):
            if key == "pandoc" and not any(c in cfg[key] for c in ("/", "\\")) and not Path(cfg[key]).is_absolute():
                cfg[key] = shutil.which(cfg[key]) or cfg[key]
                continue
            cfg[key] = str((path.parent / cfg[key]).resolve())
    for key in ("pdf", "ocr", "title"):
        if not cfg.get(key):
            raise ValueError(f"Missing configuration field: {key}")
    if cfg.get("source_format", "legacy") not in ("legacy", "structure"):
        raise ValueError("source_format must be legacy or structure")
    scale = cfg.get("ocr_scale", 2)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("ocr_scale must be positive and finite")
    for key in ("toc_depth", "split_level"):
        if key in cfg and (type(cfg[key]) is not int or not 1 <= cfg[key] <= 6):
            raise ValueError(f"{key} must be an integer in 1..6")
    for override in cfg.get("page_overrides", {}).values():
        for region in override.get("regions", []):
            if not box_valid(region.get("bbox")) or any(v > 1 for v in region["bbox"]):
                raise ValueError("Region bbox must use normalized coordinates in 0..1")
            if region.get("direction", "horizontal") not in {"horizontal", "vertical"}:
                raise ValueError("Region direction must be horizontal or vertical")
    for figure in cfg.get("figures", []):
        if not box_valid(figure.get("bbox")):
            raise ValueError("Figure bbox must use positive OCR pixel coordinates")
    starts = [s["page"] for s in cfg.get("sections", [])]
    if any(type(n) is not int or n < 1 for n in starts) or starts != sorted(set(starts)):
        raise ValueError("sections must have unique, increasing positive page numbers")
    if any(not s.get("title") for s in cfg.get("sections", [])):
        raise ValueError("Each section needs a title")
    if cfg.get("chapter_count") is not None:
        if cfg["chapter_count"] != sum(s.get("kind") == "chapter" for s in cfg.get("sections", [])):
            raise ValueError("chapter_count differs from configured chapter sections")
    return cfg


class Workspace:
    def __init__(self, root, cfg):
        self.root = Path(root).resolve()
        self.cfg = cfg
        inputs = [Path(cfg[k]).resolve() for k in ("pdf", "ocr")]
        for source in inputs:
            if (self.root == source or self.root in source.parents
                    or source.is_dir() and source in self.root.parents):
                raise ValueError("Output directory contains a source file")
        owner = self.root / "run.json"
        identity = {"config": cfg, "sources": {str(p): self._hash(p) for p in inputs}}
        self.identity = signature(identity)
        if self.root.exists() and not owner.exists() and any(self.root.iterdir()):
            raise ValueError("Output directory is not owned by bookocr; choose a new directory")
        if owner.exists() and read_json(owner).get("identity") != self.identity:
            raise ValueError("Input/config changed; choose a new output directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if not owner.exists():
            write_json(owner, {"identity": self.identity, **identity})
        self.code = signature({p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})

    @staticmethod
    def _hash(path):
        if path.is_dir():
            return signature({p.name: digest(p) for p in sorted(path.glob("page_*.json"))})
        return digest(path)

    def cached_json(self, relative, key, build):
        path = self.root / relative
        expected = signature([self.identity, self.code, key])
        if path.exists():
            try:
                saved = read_json(path)
                if (saved.get("cache_key") == expected
                        and saved.get("checksum") == signature(saved.get("data"))):
                    return saved["data"], True
            except (ValueError, OSError):
                pass
        data = build()
        write_json(path, {"cache_key": expected, "checksum": signature(data), "data": data})
        return data, False
