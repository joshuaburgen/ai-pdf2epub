"""Pandoc publication and independent package/link/content auditing."""
import hashlib
import posixpath
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .storage import atomic_text, write_json


CSS = """body { line-height: 1.7; font-family: serif; }
p { margin: .5em 0; text-indent: 2em; }
h1, h2, h3 { text-align: left; line-height: 1.4; }
figure { margin: 1em 0; text-align: center; }
img { max-width: 100%; height: auto; }
.footnotes p { text-indent: 0; }
"""


def build_epub(markdown, target, cfg, pdf, converter):
    root = Path(markdown).parent
    cover = root / "cover.png"
    pdf[cfg.get("cover_page", 1)-1].get_pixmap().save(cover)
    css = root / "book.css"
    atomic_text(css, CSS)
    metadata = {"title": converter.convert(cfg["title"]), "lang": cfg.get("language", "zh-CN")}
    for key in ("author", "translator", "publisher", "identifier"):
        if cfg.get(key):
            metadata[key] = (converter.convert(cfg[key]) if isinstance(cfg[key], str)
                             else [converter.convert(v) for v in cfg[key]] if isinstance(cfg[key], list) else cfg[key])
    meta = root / "metadata.json"
    write_json(meta, metadata)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.epub")
    command = [cfg.get("pandoc", "pandoc"), str(Path(markdown).resolve()),
               "--to=epub3", "--from=markdown-tex_math_dollars-raw_tex", "--toc",
               f"--toc-depth={cfg.get('toc_depth', 2)}", f"--split-level={cfg.get('split_level', 2)}",
               f"--resource-path={root.resolve()}", f"--css={css.resolve()}",
               f"--epub-cover-image={cover.resolve()}", f"--metadata-file={meta.resolve()}",
               "--output", str(temporary)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise RuntimeError(f"Pandoc failed ({result.returncode}): {result.stderr}")
    temporary.replace(target)
    return {"command": command, "stderr": result.stderr, "metadata": metadata}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def audit_epub(path, expected_title=None, expected_sections=None):
    errors, warnings, documents = [], [], {}
    report = {"errors": errors, "warnings": warnings, "status": "failed"}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            files = set(names)
            if len(names) != len(files):
                errors.append("Duplicate ZIP entries")
            if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
                errors.append("Unsafe ZIP entry path")
            bad_crc = archive.testzip()
            if bad_crc:
                errors.append(f"Bad ZIP CRC: {bad_crc}")
            if not names or names[0] != "mimetype" or archive.read("mimetype") != b"application/epub+zip":
                errors.append("Invalid EPUB mimetype entry/order")
            elif archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED:
                errors.append("EPUB mimetype entry must be uncompressed")
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            package_path = next(e.attrib["full-path"] for e in container.iter() if local_name(e.tag) == "rootfile")
            base = posixpath.dirname(package_path)
            package = ET.fromstring(archive.read(package_path))
            metadata = {local_name(e.tag): e.text for e in package.iter()
                        if local_name(e.tag) in {"title", "creator", "language", "identifier"}}
            report["metadata"] = metadata
            for key in ("title", "language", "identifier"):
                if not metadata.get(key):
                    errors.append(f"Missing metadata: {key}")
            if expected_title and metadata.get("title") != expected_title:
                errors.append("Title differs from configuration")
            manifest, nav_files = {}, []
            for item in package.iter():
                if local_name(item.tag) != "item":
                    continue
                name = posixpath.normpath(posixpath.join(base, unquote(item.attrib["href"])))
                if name not in files:
                    errors.append(f"Missing manifest resource: {name}")
                    continue
                manifest[item.attrib["id"]] = name
                if "nav" in item.get("properties", "").split():
                    nav_files.append(name)
                if item.get("media-type") == "application/xhtml+xml":
                    documents[name] = ET.fromstring(archive.read(name))
            if not nav_files:
                errors.append("No EPUB navigation document")
            toc_links = []
            for name in nav_files:
                for node in documents.get(name, ET.Element("empty")).iter():
                    if local_name(node.tag) == "nav" and "toc" in node.get("{http://www.idpf.org/2007/ops}type", "").split():
                        toc_links.extend(e.get("href") for e in node.iter() if local_name(e.tag) == "a")
            if not toc_links:
                errors.append("Navigation TOC is missing or empty")
            spine = []
            for item in package.iter():
                if local_name(item.tag) == "itemref":
                    if item.get("idref") not in manifest:
                        errors.append(f"Unknown spine ID: {item.get('idref')}")
                    else:
                        spine.append(manifest[item.get("idref")])
            if not spine:
                errors.append("Empty spine")
            id_sets = {}
            for name, doc in documents.items():
                ids = [e.get("id") for e in doc.iter() if e.get("id")]
                if len(ids) != len(set(ids)):
                    errors.append(f"Duplicate XHTML IDs: {name}")
                id_sets[name] = set(ids)
            heading_count, paragraphs, duplicate_texts, footnotes = 0, [], {}, 0
            for name, doc in documents.items():
                for node in doc.iter():
                    for attr in ("href", "src", "{http://www.w3.org/1999/xlink}href"):
                        ref = node.get(attr)
                        if not ref:
                            continue
                        url = urlsplit(ref)
                        if url.scheme or url.netloc:
                            continue
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path))) if url.path else name
                        if target not in files:
                            errors.append(f"Broken resource/link: {name}: {ref}")
                        elif url.fragment and target in id_sets and unquote(url.fragment) not in id_sets[target]:
                            errors.append(f"Broken anchor: {name}: {ref}")
                    if "noteref" in node.get("{http://www.idpf.org/2007/ops}type", "").split():
                        footnotes += 1
                if name not in spine:
                    continue
                heading_count += sum(local_name(e.tag) == "h2" for e in doc.iter())
                text = "".join(doc.itertext())
                if "\ufffd" in text:
                    warnings.append(f"Replacement character: {name}")
                body = next((e for e in doc.iter() if local_name(e.tag) == "body"), None)
                if body is not None and not "".join(body.itertext()).strip() and not any(local_name(e.tag) in {"img", "image"} for e in body.iter()):
                    errors.append(f"Empty spine document: {name}")
                for e in doc.iter():
                    if local_name(e.tag) == "p":
                        value = "".join(e.itertext()).strip()
                        paragraphs.append(value)
                        if len(value) >= 60:
                            key = hashlib.sha256(value.encode()).hexdigest()
                            if key in duplicate_texts:
                                warnings.append(f"Duplicate long paragraph: {name} and {duplicate_texts[key]}")
                            duplicate_texts[key] = name
            if expected_sections is not None and heading_count != expected_sections:
                errors.append(f"Section count mismatch: expected {expected_sections}, got {heading_count}")
            short = sum(0 < len(p) < 8 for p in paragraphs)
            if short:
                warnings.append(f"Short paragraphs requiring review: {short}")
            report.update(spine_documents=len(spine), section_headings=heading_count,
                          toc_links=len(toc_links), footnote_references=footnotes,
                          images=sum(n.lower().endswith((".jpg", ".jpeg", ".png", ".svg")) for n in names))
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError, StopIteration) as exc:
        errors.append(f"Cannot parse EPUB: {type(exc).__name__}: {exc}")
    report["status"] = "failed" if errors else "package_checks_passed"
    report["content_acceptance"] = "not_approved_requires_source_image_and_reader_review"
    return report


def write_audit(root, report):
    root = Path(root)
    write_json(root / "audit_report.json", report)
    lines = ["# EPUB 审计报告", "", f"包结构检查：{report['status']}",
             "", "内容尚未验收，需要原图对照与阅读器检查。", "", "## 错误", ""]
    lines += [f"- {v}" for v in report["errors"]] or ["无。"]
    lines += ["", "## 待审候选", ""]
    lines += [f"- {v}" for v in report["warnings"]] or ["无。"]
    atomic_text(root / "audit_report.md", "\n".join(lines) + "\n")
