"""Plain text, Markdown and ALTO/PAGE XML export.

ALTO and PAGE matter more than they look: they are the interchange formats of
eScriptorium, Transkribus and the wider OCR ground-truth world. Supporting them keeps a
book's verified transcription portable instead of locked inside this application.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from ..project import PageStatus, Project
from .document import BookDocument


@dataclass
class TextOptions:
    page_separator: str = "\n\n"
    include_page_markers: bool = False
    page_marker_format: str = "--- {label} ---"
    line_width: int = 0  # 0 leaves lines as they are


def export_text(
    document: BookDocument, output_path: str | Path, options: TextOptions | None = None
) -> Path:
    options = options or TextOptions()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    if document.title:
        parts.append(document.title)
        if document.author:
            parts.append(document.author)
        parts.append("")

    seen_pages: set[int] = set()
    for chapter in document.chapters:
        for block in chapter.blocks:
            if options.include_page_markers and block.page_idx not in seen_pages:
                seen_pages.add(block.page_idx)
                label = next(
                    (lbl for idx, lbl in document.page_breaks if idx == block.page_idx),
                    str(block.page_idx + 1),
                )
                parts.append(options.page_marker_format.format(label=label))
            parts.append(block.text)

    output.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
    return output


def export_markdown(document: BookDocument, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if document.title:
        lines += [f"# {document.title}", ""]
    if document.author:
        lines += [f"*{document.author}*", ""]

    for chapter in document.chapters:
        for block in chapter.blocks:
            if block.kind == "chapter":
                lines += [f"## {block.text}", ""]
            elif block.kind == "heading":
                lines += [f"### {block.text}", ""]
            elif block.kind == "subheading":
                lines += [f"#### {block.text}", ""]
            elif block.kind == "verse":
                lines += ["> " + ln for ln in block.text.split("\n")] + [""]
            elif block.kind == "quote":
                lines += [f"> {block.text}", ""]
            elif block.kind == "footnote":
                lines += [f"[^note]: {block.text}", ""]
            else:
                lines += [block.text, ""]

    output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output


# --------------------------------------------------------------------------------------
# ALTO
# --------------------------------------------------------------------------------------

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def export_alto(project: Project, output_dir: str | Path) -> list[Path]:
    """Write one ALTO 4 XML file per page, with word geometry and confidences."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for page in project.iter_pages():
        if page.status is PageStatus.SKIPPED:
            continue

        root = ET.Element("alto", {"xmlns": ALTO_NS})
        description = ET.SubElement(root, "Description")
        ET.SubElement(description, "MeasurementUnit").text = "pixel"
        source = ET.SubElement(description, "sourceImageInformation")
        ET.SubElement(source, "fileName").text = page.image_rel or page.label
        agency = ET.SubElement(description, "Processing", {"ID": "P1"})
        ET.SubElement(agency, "processingDateTime").text = datetime.now(timezone.utc).isoformat()
        step = ET.SubElement(agency, "processingStepSettings")
        step.text = f"engine={page.engine or 'manual'}"

        layout = ET.SubElement(root, "Layout")
        alto_page = ET.SubElement(
            layout,
            "Page",
            {
                "ID": f"page_{page.idx}",
                "PHYSICAL_IMG_NR": str(page.idx + 1),
                "WIDTH": str(page.width),
                "HEIGHT": str(page.height),
            },
        )
        print_space = ET.SubElement(
            alto_page,
            "PrintSpace",
            {"HPOS": "0", "VPOS": "0", "WIDTH": str(page.width), "HEIGHT": str(page.height)},
        )
        block = ET.SubElement(print_space, "TextBlock", {"ID": f"block_{page.idx}"})

        text = project.get_edited(page.id)
        words_by_line: dict[int, list] = {}
        for word in project.get_words(page.id):
            words_by_line.setdefault(word.line_idx, []).append(word)

        for line_idx in sorted(words_by_line):
            members = words_by_line[line_idx]
            x0 = min(w.x for w in members)
            y0 = min(w.y for w in members)
            x1 = max(w.x + w.w for w in members)
            y1 = max(w.y + w.h for w in members)
            text_line = ET.SubElement(
                block,
                "TextLine",
                {
                    "ID": f"line_{page.idx}_{line_idx}",
                    "HPOS": str(x0),
                    "VPOS": str(y0),
                    "WIDTH": str(x1 - x0),
                    "HEIGHT": str(y1 - y0),
                },
            )
            for word in members:
                content = (
                    text[word.start : word.end]
                    if 0 <= word.start < word.end <= len(text)
                    else word.text
                )
                ET.SubElement(
                    text_line,
                    "String",
                    {
                        "ID": f"w_{page.idx}_{line_idx}_{word.word_idx}",
                        "CONTENT": content,
                        "HPOS": str(word.x),
                        "VPOS": str(word.y),
                        "WIDTH": str(word.w),
                        "HEIGHT": str(word.h),
                        "WC": f"{word.conf:.3f}",
                    },
                )

        target = directory / f"{page.idx:05d}.xml"
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
        written.append(target)

    return written


def export_hocr(project: Project, output_path: str | Path) -> Path:
    """Single hOCR file for the whole book — the format Tesseract tooling expects."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'
        ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="am" lang="am">',
        "<head><meta charset='utf-8'/>"
        "<meta name='ocr-system' content='amharic-studio'/>"
        "<meta name='ocr-capabilities' content='ocr_page ocr_line ocrx_word'/>"
        f"<title>{html.escape(project.title)}</title></head>",
        "<body>",
    ]

    for page in project.iter_pages():
        if page.status is PageStatus.SKIPPED:
            continue
        text = project.get_edited(page.id)
        parts.append(
            f"<div class='ocr_page' id='page_{page.idx}' "
            f"title='image \"{html.escape(page.image_rel or '')}\"; "
            f"bbox 0 0 {page.width} {page.height}; ppageno {page.idx}'>"
        )
        words_by_line: dict[int, list] = {}
        for word in project.get_words(page.id):
            words_by_line.setdefault(word.line_idx, []).append(word)

        for line_idx in sorted(words_by_line):
            members = words_by_line[line_idx]
            x0 = min(w.x for w in members)
            y0 = min(w.y for w in members)
            x1 = max(w.x + w.w for w in members)
            y1 = max(w.y + w.h for w in members)
            parts.append(
                f"<span class='ocr_line' id='line_{page.idx}_{line_idx}' "
                f"title='bbox {x0} {y0} {x1} {y1}'>"
            )
            for word in members:
                content = (
                    text[word.start : word.end]
                    if 0 <= word.start < word.end <= len(text)
                    else word.text
                )
                parts.append(
                    f"<span class='ocrx_word' id='w_{page.idx}_{line_idx}_{word.word_idx}' "
                    f"title='bbox {word.x} {word.y} {word.x + word.w} {word.y + word.h}; "
                    f"x_wconf {int(word.conf * 100)}'>{html.escape(content)}</span> "
                )
            parts.append("</span>")
        parts.append("</div>")

    parts += ["</body>", "</html>"]
    output.write_text("\n".join(parts), encoding="utf-8")
    return output
