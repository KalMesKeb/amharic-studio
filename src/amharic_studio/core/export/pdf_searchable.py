"""Scan-faithful searchable PDF.

The page keeps the original scan exactly as it looks — the printer's typeface, the paper,
the marginalia — while an invisible text layer sits precisely over the ink so the book can
be searched, selected and copied. For a scholarly edition this is often the more valuable
of the two PDF outputs, because nothing about the artefact is reinterpreted.

The text layer uses the *corrected* text, not the raw OCR, with each word scaled
horizontally to fit the rectangle it was recognized in so that selection highlights land
on the right ink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

from .. import fonts
from ..fonts import FontInfo
from ..project import PageStatus, Project, WordBox
from .pdfutil import set_pdf_language

#: PDF text rendering mode 3 draws nothing but keeps the text selectable.
INVISIBLE_TEXT = 3


@dataclass
class SearchableOptions:
    dpi: int = 300
    #: Re-encode page images at this DPI. 0 keeps them at full resolution.
    target_dpi: int = 0
    image_format: str = "auto"  # auto | jpeg | png
    jpeg_quality: int = 78
    #: Draw the word rectangles visibly. Debugging aid for checking layer alignment.
    debug_boxes: bool = False
    add_outline: bool = True
    font: FontInfo | None = None
    #: Fraction of the box height used as the font size for the hidden text.
    text_height_ratio: float = 0.82


@dataclass
class SearchableReport:
    pages: int = 0
    words_placed: int = 0
    pages_without_geometry: int = 0
    output_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        size_mb = self.output_bytes / (1024 * 1024)
        out = f"{self.pages} pages, {self.words_placed:,} words in the text layer, {size_mb:.1f} MB"
        if self.pages_without_geometry:
            out += (
                f" — {self.pages_without_geometry} page(s) had no word geometry and carry "
                "page-level text only"
            )
        return out


def export_searchable_pdf(
    project: Project, output_path: str | Path, options: SearchableOptions | None = None
) -> tuple[Path, SearchableReport]:
    """Write a searchable PDF preserving the original page images."""
    options = options or SearchableOptions()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = SearchableReport()

    font_info = options.font or fonts.require_font()
    font_name = fonts.register_with_reportlab(font_info)

    pdf = rl_canvas.Canvas(str(output))
    pdf.setTitle(project.title)
    if project.author:
        pdf.setAuthor(project.author)
    set_pdf_language(pdf, project.get_meta("language", "am"))

    outline_entries: list[tuple[str, str, int]] = []

    for page in project.iter_pages():
        if page.status is PageStatus.SKIPPED:
            continue
        image_path = project.image_path(page)
        if image_path is None or not image_path.exists():
            report.warnings.append(f"{page.label}: no image, page skipped")
            continue

        image = Image.open(image_path)
        image, scale_applied = _maybe_downsample(image, page.dpi or options.dpi, options)
        dpi = (page.dpi or options.dpi) / scale_applied

        width_pt = image.width * 72.0 / dpi
        height_pt = image.height * 72.0 / dpi
        pdf.setPageSize((width_pt, height_pt))

        pdf.drawImage(
            _image_reader(image, options),
            0, 0, width=width_pt, height=height_pt,
            preserveAspectRatio=False,
            anchor="sw",
        )

        text = project.get_edited(page.id)
        boxes = project.get_words(page.id)
        if boxes:
            placed = _draw_text_layer(
                pdf, boxes, text, font_name, dpi, height_pt, scale_applied, options
            )
            report.words_placed += placed
        elif text.strip():
            _draw_fallback_text(pdf, text, font_name, height_pt)
            report.pages_without_geometry += 1

        if options.add_outline:
            for mark in project.get_structure(page.id):
                if mark.kind in {"chapter", "heading"}:
                    key = f"p{page.idx}_{mark.start}"
                    pdf.bookmarkPage(key)
                    label = mark.label or text[mark.start : mark.end].strip()[:80]
                    outline_entries.append((label or page.label, key, 0 if mark.kind == "chapter" else 1))

        pdf.showPage()
        report.pages += 1

    for label, key, level in outline_entries:
        pdf.addOutlineEntry(label, key, level=level)
    if outline_entries:
        pdf.showOutline()

    pdf.save()
    report.output_bytes = output.stat().st_size
    return output, report


# --------------------------------------------------------------------------------------


def _maybe_downsample(
    image: Image.Image, source_dpi: int, options: SearchableOptions
) -> tuple[Image.Image, float]:
    if not options.target_dpi or not source_dpi or options.target_dpi >= source_dpi:
        return image, 1.0
    scale = options.target_dpi / source_dpi
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS), scale


def _image_reader(image: Image.Image, options: SearchableOptions) -> ImageReader:
    """Pick an encoding: JPEG for continuous tone, lossless for bilevel scans.

    Bilevel pages must never go through JPEG — ringing artefacts around fidel strokes are
    exactly the kind of damage that makes a diacritic ambiguous.
    """
    fmt = options.image_format
    if fmt == "auto":
        extrema = image.convert("L").getextrema()
        is_bilevel = image.mode == "1" or (extrema and extrema[0] > 200) or _looks_bilevel(image)
        fmt = "png" if is_bilevel else "jpeg"

    if fmt == "jpeg" and image.mode not in ("L", "RGB"):
        image = image.convert("L")
    if fmt == "png" and image.mode == "L":
        image = image.convert("1")
    return ImageReader(image)


def _looks_bilevel(image: Image.Image, sample: int = 4096) -> bool:
    gray = image.convert("L")
    histogram = gray.histogram()
    total = sum(histogram) or 1
    extremes = sum(histogram[:24]) + sum(histogram[232:])
    return extremes / total > 0.95


def _draw_text_layer(
    pdf: rl_canvas.Canvas,
    boxes: list[WordBox],
    page_text: str,
    font_name: str,
    dpi: float,
    page_height_pt: float,
    image_scale: float,
    options: SearchableOptions,
) -> int:
    """Place one invisible text run per word box, scaled to fit that box."""
    px_to_pt = 72.0 / dpi * image_scale
    placed = 0

    for box in boxes:
        # Prefer the corrected text at this offset over whatever the recognizer read.
        word = page_text[box.start : box.end] if 0 <= box.start < box.end <= len(page_text) else box.text
        word = word.strip()
        if not word:
            continue

        x = box.x * px_to_pt
        box_width = max(box.w * px_to_pt, 0.5)
        box_height = max(box.h * px_to_pt, 0.5)
        baseline = page_height_pt - (box.y + box.h) * px_to_pt + box_height * 0.16

        size = max(1.0, box_height * options.text_height_ratio)
        natural = pdfmetrics.stringWidth(word, font_name, size)
        horizontal_scale = 100.0 * box_width / natural if natural > 0 else 100.0
        horizontal_scale = min(400.0, max(15.0, horizontal_scale))

        text_object = pdf.beginText()
        text_object.setTextRenderMode(INVISIBLE_TEXT)
        text_object.setFont(font_name, size)
        text_object.setHorizScale(horizontal_scale)
        text_object.setTextOrigin(x, baseline)
        text_object.textOut(word)
        pdf.drawText(text_object)
        placed += 1

        if options.debug_boxes:
            pdf.saveState()
            pdf.setStrokeColorRGB(1, 0, 0)
            pdf.setLineWidth(0.3)
            pdf.rect(x, page_height_pt - (box.y + box.h) * px_to_pt, box_width, box_height)
            pdf.restoreState()

    return placed


def _draw_fallback_text(
    pdf: rl_canvas.Canvas, text: str, font_name: str, page_height_pt: float
) -> None:
    """Attach page text with no geometry, so the page is at least searchable.

    Selection will not align with the ink, but the words are findable — better than a page
    that is invisible to search entirely.
    """
    text_object = pdf.beginText()
    text_object.setTextRenderMode(INVISIBLE_TEXT)
    text_object.setFont(font_name, 8)
    text_object.setTextOrigin(4, page_height_pt - 12)
    for line in text.split("\n"):
        text_object.textLine(line)
    pdf.drawText(text_object)
