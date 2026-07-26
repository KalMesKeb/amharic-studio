"""Typeset PDF export — a clean, reflowed edition of the corrected text.

Drawn on a ReportLab canvas rather than through Platypus, because Platypus wraps on
spaces and traditional Amharic text has none; the Ethiopic-aware breaker in
:mod:`..linebreak` supplies the line breaks and this module places them.

ReportLab is sufficient for Ethiopic despite having no complex-script shaping engine: the
script is a precomposed syllabary with no reordering, no contextual forms and no
mandatory ligatures, so one glyph per codepoint in logical order is correct output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import A4, A5, LETTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

from .. import fonts, linebreak
from ..fonts import FontInfo
from .document import Block, BookDocument
from .pdfutil import set_pdf_language

PAGE_SIZES = {"A4": A4, "A5": A5, "Letter": LETTER}


@dataclass
class TypesetOptions:
    page_size: str = "A5"  # A5 suits the proportions of most Ethiopian book printing
    margin_top: float = 18 * mm
    margin_bottom: float = 20 * mm
    margin_inner: float = 20 * mm
    margin_outer: float = 15 * mm
    body_size: float = 11.5
    #: Fidel need more leading than Latin at the same size; 1.62 is a comfortable minimum.
    leading_ratio: float = 1.62
    paragraph_indent: float = 5 * mm
    space_between_paragraphs: float = 0.0
    justify: bool = True
    #: Cap on how far a space may stretch before a line is left ragged instead.
    max_stretch_ratio: float = 3.0
    running_heads: bool = True
    page_numbers: bool = True
    page_number_size: float = 9.0
    title_page: bool = True
    chapter_starts_new_page: bool = True
    mirror_margins: bool = True  # wider gutter on the bound edge
    font: FontInfo | None = None
    heading_scale: tuple[float, float, float] = (1.55, 1.28, 1.12)
    embed_subset: bool = True


@dataclass
class TypesetReport:
    pages: int = 0
    lines: int = 0
    overfull_lines: int = 0
    missing_glyphs: set[str] = field(default_factory=set)
    font_used: str = ""

    def describe(self) -> str:
        out = f"{self.pages} pages, {self.lines} lines, set in {self.font_used}"
        if self.missing_glyphs:
            sample = "".join(sorted(self.missing_glyphs)[:12])
            out += f" — WARNING: {len(self.missing_glyphs)} characters have no glyph ({sample})"
        return out


class _Typesetter:
    def __init__(self, document: BookDocument, options: TypesetOptions, output: Path) -> None:
        self.document = document
        self.options = options
        self.report = TypesetReport()

        font_info = options.font or fonts.require_font()
        self.font_name = fonts.register_with_reportlab(font_info)
        self.bold_name = f"{self.font_name}-Bold"
        self.report.font_used = font_info.family

        self.page_size = PAGE_SIZES.get(options.page_size, A5)
        self.canvas = rl_canvas.Canvas(str(output), pagesize=self.page_size)
        self.canvas.setTitle(document.title)
        if document.author:
            self.canvas.setAuthor(document.author)
        self.canvas.setSubject("Amharic text edition")
        set_pdf_language(self.canvas, document.language or "am")

        self.page_width, self.page_height = self.page_size
        self.page_number = 1
        self.current_chapter = ""
        self.y = 0.0
        #: Chapter openings and the title page carry no running head, by convention.
        self._bare_page = True
        self._start_page(first=True)

    # -- geometry -------------------------------------------------------------------------

    @property
    def is_recto(self) -> bool:
        return self.page_number % 2 == 1

    @property
    def left_margin(self) -> float:
        if not self.options.mirror_margins:
            return self.options.margin_inner
        return self.options.margin_inner if self.is_recto else self.options.margin_outer

    @property
    def right_margin(self) -> float:
        if not self.options.mirror_margins:
            return self.options.margin_outer
        return self.options.margin_outer if self.is_recto else self.options.margin_inner

    @property
    def text_width(self) -> float:
        return self.page_width - self.left_margin - self.right_margin

    @property
    def bottom_limit(self) -> float:
        return self.options.margin_bottom

    def measure(self, text: str, size: float | None = None) -> float:
        return pdfmetrics.stringWidth(text, self.font_name, size or self.options.body_size)

    # -- page furniture ---------------------------------------------------------------------

    def _start_page(self, first: bool = False) -> None:
        if not first:
            self.canvas.showPage()
            self.page_number += 1
        self.y = self.page_height - self.options.margin_top
        self._bare_page = False
        self.report.pages = self.page_number

    def _finish_page(self) -> None:
        options = self.options
        if options.running_heads and self.current_chapter and self.page_number > 1 and not self._bare_page:
            self.canvas.setFont(self.font_name, options.page_number_size)
            self.canvas.setFillGray(0.35)
            head = self.current_chapter
            while self.measure(head, options.page_number_size) > self.text_width and len(head) > 4:
                head = head[:-2] + "…"
            x = self.left_margin if self.is_recto else self.page_width - self.right_margin
            align = self.canvas.drawString if self.is_recto else self.canvas.drawRightString
            align(x, self.page_height - options.margin_top + 7 * mm, head)
            self.canvas.setFillGray(0.0)

        if options.page_numbers and self.page_number > 1:
            self.canvas.setFont(self.font_name, options.page_number_size)
            self.canvas.setFillGray(0.35)
            self.canvas.drawCentredString(
                self.page_width / 2, options.margin_bottom - 9 * mm, str(self.page_number)
            )
            self.canvas.setFillGray(0.0)

    def _new_page(self) -> None:
        self._finish_page()
        self._start_page()

    def _ensure_space(self, needed: float) -> None:
        if self.y - needed < self.bottom_limit:
            self._new_page()

    # -- drawing -----------------------------------------------------------------------------

    def _draw_line(
        self, line: linebreak.Line, size: float, x_offset: float = 0.0, justify: bool = False,
        centered: bool = False, font: str | None = None,
    ) -> None:
        font = font or self.font_name
        leading = size * self.options.leading_ratio
        self._ensure_space(leading)
        self.y -= leading

        text = line.text
        if centered:
            width = pdfmetrics.stringWidth(text, font, size)
            self.canvas.setFont(font, size)
            self.canvas.drawString(self.left_margin + (self.text_width - width) / 2, self.y, text)
            return

        available = self.text_width - x_offset
        should_justify = justify and not line.is_last and line.break_count > 0
        if not should_justify:
            self.canvas.setFont(font, size)
            self.canvas.drawString(self.left_margin + x_offset, self.y, text)
            if pdfmetrics.stringWidth(text, font, size) > available + 0.5:
                self.report.overfull_lines += 1
            return

        natural = pdfmetrics.stringWidth(text, font, size)
        slack = available - natural
        gaps = line.break_count
        extra = slack / gaps if gaps else 0.0

        # A line needing absurd stretch looks worse justified than ragged.
        space_width = pdfmetrics.stringWidth(" ", font, size) or size * 0.25
        if extra > space_width * self.options.max_stretch_ratio:
            self.canvas.setFont(font, size)
            self.canvas.drawString(self.left_margin + x_offset, self.y, text)
            return

        cursor = self.left_margin + x_offset
        self.canvas.setFont(font, size)
        for index, seg in enumerate(line.segments):
            piece = seg.full if index < len(line.segments) - 1 else seg.text
            self.canvas.drawString(cursor, self.y, piece)
            cursor += pdfmetrics.stringWidth(piece, font, size)
            if index < len(line.segments) - 1:
                cursor += extra

    def _draw_paragraph(
        self, text: str, size: float | None = None, indent: bool = True, justify: bool | None = None,
        left_inset: float = 0.0, centered: bool = False, font: str | None = None,
    ) -> None:
        size = size or self.options.body_size
        justify = self.options.justify if justify is None else justify
        font = font or self.font_name

        first_indent = self.options.paragraph_indent if indent else 0.0
        measure = lambda s: pdfmetrics.stringWidth(s, font, size)  # noqa: E731

        # The first line is shorter by its indent, so it is wrapped separately.
        lines = linebreak.wrap(text, self.text_width - left_inset - first_indent, measure)
        if lines and first_indent:
            head, *rest = lines
            remaining = text[len(head.text) :].lstrip()
            lines = [head]
            if remaining:
                tail = linebreak.wrap(remaining, self.text_width - left_inset, measure)
                head.is_last = False
                lines.extend(tail)
            else:
                head.is_last = True
            del rest

        for index, line in enumerate(lines):
            offset = left_inset + (first_indent if index == 0 else 0.0)
            self._draw_line(line, size, offset, justify, centered, font)
            self.report.lines += 1

    # -- blocks -------------------------------------------------------------------------------

    def draw_block(self, block: Block) -> None:
        options = self.options
        size = options.body_size

        if block.kind == "chapter":
            if options.chapter_starts_new_page and self.report.lines:
                self._new_page()
            self.current_chapter = block.text
            self._bare_page = True
            self.y -= size * 1.6
            self._draw_paragraph(
                block.text, size * options.heading_scale[0], indent=False, justify=False,
                centered=True, font=self.bold_name,
            )
            self.y -= size * 0.9
        elif block.kind in {"heading", "subheading"}:
            scale = options.heading_scale[1 if block.kind == "heading" else 2]
            self.y -= size * 0.9
            self._draw_paragraph(
                block.text, size * scale, indent=False, justify=False, centered=True,
                font=self.bold_name,
            )
            self.y -= size * 0.4
        elif block.kind == "verse":
            for line in block.text.split("\n"):
                self._draw_paragraph(line, size, indent=False, justify=False, left_inset=8 * mm)
        elif block.kind == "quote":
            self._draw_paragraph(block.text, size * 0.95, indent=False, left_inset=8 * mm)
        elif block.kind == "caption":
            self._draw_paragraph(block.text, size * 0.9, indent=False, justify=False, centered=True)
        elif block.kind == "footnote":
            self.y -= size * 0.5
            self._draw_paragraph(block.text, size * 0.85, indent=False, justify=False)
        else:
            self._draw_paragraph(block.text, size, indent=True)
            if options.space_between_paragraphs:
                self.y -= options.space_between_paragraphs

    # -- title page ------------------------------------------------------------------------------

    def draw_title_page(self) -> None:
        document = self.document
        self.y = self.page_height * 0.66
        self._bare_page = True
        self._draw_paragraph(
            document.title, self.options.body_size * 2.0, indent=False, justify=False,
            centered=True, font=self.bold_name,
        )
        if document.author:
            self.y -= 12 * mm
            self._draw_paragraph(
                document.author, self.options.body_size * 1.15, indent=False, justify=False,
                centered=True,
            )
        self._new_page()
        self.report.lines = 0  # so the first chapter does not force another blank page

    # -- run ----------------------------------------------------------------------------------------

    def run(self) -> TypesetReport:
        font_path = (self.options.font or fonts.require_font()).path
        self.report.missing_glyphs = fonts.missing_characters(font_path, self.document.full_text)

        if self.options.title_page and self.document.title:
            self.draw_title_page()

        for chapter in self.document.chapters:
            for block in chapter.blocks:
                self.draw_block(block)

        self._finish_page()
        self.canvas.save()
        self.report.pages = self.page_number
        return self.report


def export_pdf(
    document: BookDocument, output_path: str | Path, options: TypesetOptions | None = None
) -> tuple[Path, TypesetReport]:
    """Typeset the corrected text into a clean PDF."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    typesetter = _Typesetter(document, options or TypesetOptions(), output)
    return output, typesetter.run()
