"""Bringing books into a project.

Three routes in, because "badly OCR'd Amharic books" arrive in three different states:

* **Scans** — images or image-only PDFs. Rasterize, preprocess, recognize.
* **Text-layer PDFs** — someone already ran OCR and baked the result in. The text is the
  thing to repair, and character boxes can be read straight out of the PDF, which gives a
  full side-by-side view with no recognition step at all.
* **Loose text** — bad OCR output with the scans long gone. No image pane; the original
  text becomes the left-hand reference and stays frozen.

Rendering uses pypdfium2 (BSD/Apache). PyMuPDF is deliberately avoided: it is AGPL, which
would follow anyone who redistributes this.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image

from .imaging import PreprocessOptions, image_dpi, load_image, make_thumbnail, preprocess
from .ocr.base import OcrWord, assemble_text, lines_from_words
from .project import LineBox, PageStatus, Project, WordBox

IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pnm", ".ppm", ".jp2"}
)

ProgressFn = Callable[[int, int, str], None]


class SourceKind(str, Enum):
    IMAGES = "images"
    PDF_SCANNED = "pdf_scanned"
    PDF_TEXT_LAYER = "pdf_text_layer"
    TEXT = "text"


@dataclass
class ImportReport:
    pages_added: int = 0
    kind: SourceKind = SourceKind.IMAGES
    text_layer_chars: int = 0
    probe: TextLayerProbe | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def describe(self) -> str:
        base = f"{self.pages_added} page{'s' if self.pages_added != 1 else ''} imported ({self.kind.value})"
        if self.text_layer_chars:
            base += f", {self.text_layer_chars:,} characters recovered from the text layer"
        if self.probe is not None and self.probe.reason:
            base += f". {self.probe.reason}"
        return base

    @property
    def needs_recognition(self) -> bool:
        """True when pages came in as images and no text has been read for them yet."""
        return self.pages_added > 0 and self.kind in (
            SourceKind.IMAGES,
            SourceKind.PDF_SCANNED,
        )


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------


def import_images(
    project: Project,
    paths: Sequence[str | Path],
    options: PreprocessOptions | None = None,
    apply_preprocessing: bool = True,
    progress: ProgressFn | None = None,
) -> ImportReport:
    """Add image files as pages, storing the cleaned image and a thumbnail."""
    report = ImportReport(kind=SourceKind.IMAGES)
    ordered = sorted((Path(p) for p in paths), key=_natural_key)
    total = len(ordered)

    for i, path in enumerate(ordered):
        if progress:
            progress(i, total, path.name)
        try:
            image = load_image(path)
        except OSError as exc:
            report.warnings.append(f"{path.name}: {exc}")
            continue

        dpi = image_dpi(image)
        page = project.add_page(path, label=path.stem, dpi=dpi)
        _store_derived_images(project, page.id, image, options, apply_preprocessing, dpi)
        report.pages_added += 1

    if progress:
        progress(total, total, "done")
    return report


def _store_derived_images(
    project: Project,
    page_id: int,
    image: Image.Image,
    options: PreprocessOptions | None,
    apply_preprocessing: bool,
    dpi: int,
) -> None:
    """Write the working (preprocessed) image and thumbnail beside the original."""
    project.update_page_geometry(page_id, image.width, image.height, dpi)

    if apply_preprocessing:
        opts = options or PreprocessOptions()
        if dpi:
            opts.source_dpi = dpi
        processed, _report = preprocess(image, opts)
        target = project.path / "images" / f"{page_id:05d}_work.png"
        processed.save(target)
        project.conn.execute(
            "UPDATE pages SET image_rel = ? WHERE id = ?",
            (f"images/{target.name}", page_id),
        )

    thumb_path = project.path / "thumbs" / f"{page_id:05d}.png"
    make_thumbnail(image).save(thumb_path)
    project.conn.execute(
        "UPDATE pages SET thumb_rel = ? WHERE id = ?", (f"thumbs/{thumb_path.name}", page_id)
    )
    project.conn.commit()


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------


#: A transcription of an Amharic page is written in Ethiopic. Scanner stamps, watermarks
#: and channel handles ("Scanned by CamScanner", "t.me/…") are pure ASCII, and they are
#: the usual reason an image-only PDF appears to already carry text. Set low rather than
#: at a majority so a book with Latin front matter or heavy citation still qualifies.
MIN_ETHIOPIC_RATIO = 0.30

#: A real page of a book runs to hundreds of characters. A stamp runs to a couple of
#: dozen, and that is the whole distinction being drawn here.
MIN_CHARS_PER_PAGE = 120

#: A line printed on this fraction of the sampled pages is furniture, not content.
BOILERPLATE_SHARE = 0.6


@dataclass
class TextLayerProbe:
    """What a PDF's existing text layer actually contains.

    Deciding whether to trust a text layer by counting its characters is how a two-line
    scanner watermark gets imported as the transcription of a 78-page book — and because
    that marks every page as recognized, it also stops OCR from ever running. So the
    question is not "is there text" but "is this text a transcription of these pages".
    """

    pages: int = 0
    sampled: int = 0
    chars_per_page: float = 0.0
    ethiopic_ratio: float = 0.0
    boilerplate: tuple[str, ...] = ()
    usable: bool = False
    reason: str = ""

    def describe(self) -> str:
        return self.reason


def _sample_indices(count: int, sample: int = 8) -> list[int]:
    """Spread the sample through the book.

    Taking the first few pages reads the cover, the title page and whatever blank leaf
    follows — the least representative pages in any scan.
    """
    if count <= sample:
        return list(range(count))
    step = count / sample
    return sorted({min(count - 1, int(i * step)) for i in range(sample)})


def _ethiopic_ratio(text: str) -> float:
    """Share of the letters that are Ethiopic. Digits and punctuation do not vote."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u1200" <= c <= "\u137f" or "\u2d80" <= c <= "\u2dde") / len(
        letters
    )


def _find_boilerplate(pages: Sequence[str], share: float = BOILERPLATE_SHARE) -> tuple[str, ...]:
    """Lines that recur across pages: watermarks, running heads, channel stamps."""
    if len(pages) < 3:
        return ()
    counts: dict[str, int] = {}
    for text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(3, int(len(pages) * share))
    return tuple(sorted(line for line, n in counts.items() if n >= threshold))


def _strip_boilerplate(text: str, boilerplate: Iterable[str]) -> str:
    drop = {b.strip() for b in boilerplate}
    if not drop:
        return text
    return "\n".join(ln for ln in text.splitlines() if ln.strip() not in drop)


def probe_pdf(path: str | Path, sample: int = 8) -> TextLayerProbe:
    """Judge whether a PDF's text layer is worth importing, without importing it."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return _probe_text_layer(pdf, _sample_indices(len(pdf), sample), len(pdf))
    finally:
        pdf.close()


def _probe_text_layer(pdf: object, indices: Sequence[int], pages: int) -> TextLayerProbe:
    texts = [_page_text(pdf[i]) for i in indices]  # type: ignore[index]
    probe = TextLayerProbe(pages=pages, sampled=len(texts))
    if not texts:
        probe.reason = "The file has no pages."
        return probe

    probe.boilerplate = _find_boilerplate(texts)
    content = [_strip_boilerplate(t, probe.boilerplate).strip() for t in texts]
    if not any(content):
        # Every line repeated on every page, so there was no contrast to separate
        # furniture from content — sampled pages that are genuine duplicates look exactly
        # like a page that is nothing but a stamp. Judge the text as it stands instead;
        # a stamp is still far too short to pass, and a real page still passes.
        content = [t.strip() for t in texts]
    joined = "\n".join(content)

    probe.chars_per_page = len(joined) / len(texts)
    probe.ethiopic_ratio = _ethiopic_ratio(joined)

    if probe.chars_per_page < MIN_CHARS_PER_PAGE:
        if probe.boilerplate:
            probe.reason = (
                f"The text layer is only a repeated stamp ({probe.boilerplate[0][:40]!r}); "
                "the pages themselves will be recognized."
            )
        else:
            probe.reason = (
                f"The text layer holds about {probe.chars_per_page:.0f} characters per page, "
                "too little to be a transcription; the pages will be recognized."
            )
        return probe

    if probe.ethiopic_ratio < MIN_ETHIOPIC_RATIO:
        probe.reason = (
            f"The text layer is {probe.ethiopic_ratio:.0%} Ethiopic, so it is not a "
            "transcription of an Amharic page; the pages will be recognized."
        )
        return probe

    probe.usable = True
    probe.reason = (
        f"Using the existing text layer: about {probe.chars_per_page:.0f} characters per "
        f"page, {probe.ethiopic_ratio:.0%} Ethiopic."
    )
    return probe


def import_pdf(
    project: Project,
    path: str | Path,
    dpi: int = 300,
    use_text_layer: bool | None = None,
    render_pages: bool = True,
    options: PreprocessOptions | None = None,
    apply_preprocessing: bool = True,
    page_range: Iterable[int] | None = None,
    progress: ProgressFn | None = None,
) -> ImportReport:
    """Import a PDF.

    ``use_text_layer`` defaults to autodetection: if the file already carries a plausible
    transcription, that text is imported as the recognizer output to be repaired, along
    with word boxes read from the PDF itself. Pass ``False`` to ignore it and re-OCR from
    the rendered images, or ``True`` to force it.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    report = ImportReport()
    try:
        indices = list(page_range) if page_range is not None else list(range(len(pdf)))
        total = len(indices)

        sampled = [indices[i] for i in _sample_indices(len(indices))]
        probe = _probe_text_layer(pdf, sampled, len(pdf))
        report.probe = probe
        if use_text_layer is None:
            use_text_layer = probe.usable
        # Even a genuine text layer carries the scanner's stamp on every page.
        boilerplate = probe.boilerplate if use_text_layer else ()
        report.kind = SourceKind.PDF_TEXT_LAYER if use_text_layer else SourceKind.PDF_SCANNED

        scale = dpi / 72.0
        for position, index in enumerate(indices):
            if progress:
                progress(position, total, f"page {index + 1}")
            pdf_page = pdf[index]

            image: Image.Image | None = None
            if render_pages:
                image = pdf_page.render(scale=scale).to_pil().convert("L")

            page = project.add_page(
                None, label=f"p. {index + 1}", dpi=dpi, copy_image=False
            )
            if image is not None:
                original = project.path / "images" / f"{page.id:05d}.png"
                image.save(original, dpi=(dpi, dpi))
                project.conn.execute(
                    "UPDATE pages SET image_rel = ? WHERE id = ?",
                    (f"images/{original.name}", page.id),
                )
                project.conn.commit()
                _store_derived_images(project, page.id, image, options, apply_preprocessing, dpi)

            if use_text_layer:
                chars = _import_text_layer(project, page.id, pdf_page, scale, boilerplate)
                report.text_layer_chars += chars
                if chars:
                    project.set_page_status(page.id, PageStatus.RECOGNIZED)

            report.pages_added += 1

        if progress:
            progress(total, total, "done")
        return report
    finally:
        pdf.close()


def _page_text(pdf_page: object) -> str:
    textpage = pdf_page.get_textpage()  # type: ignore[attr-defined]
    try:
        for method in ("get_text_bounded", "get_text_range"):
            getter = getattr(textpage, method, None)
            if getter is not None:
                return getter()
        return ""
    finally:
        textpage.close()


def _import_text_layer(
    project: Project,
    page_id: int,
    pdf_page: object,
    scale: float,
    boilerplate: Iterable[str] = (),
) -> int:
    """Pull text and per-character boxes out of a PDF, and group them into word boxes.

    PDF character boxes are in points with the origin at the bottom left; the rendered
    image is in pixels from the top left, so every box is scaled and flipped to match what
    the canvas will draw.
    """
    textpage = pdf_page.get_textpage()  # type: ignore[attr-defined]
    try:
        text = ""
        for method in ("get_text_bounded", "get_text_range"):
            getter = getattr(textpage, method, None)
            if getter is not None:
                text = getter()
                break
        if not text.strip():
            return 0

        page_height = pdf_page.get_height()  # type: ignore[attr-defined]
        boxes: list[tuple[str, float, float, float, float]] = []
        try:
            count = textpage.count_chars()
            for i in range(count):
                char = textpage.get_text_range(i, 1)
                if not char or char.isspace():
                    boxes.append((char, 0, 0, 0, 0))
                    continue
                left, bottom, right, top = textpage.get_charbox(i)
                boxes.append((char, left, bottom, right, top))
        except (AttributeError, ValueError):
            boxes = []

        words = _words_from_charboxes(boxes, page_height, scale) if boxes else []
        words = _drop_boilerplate_words(words, boilerplate)
        if words:
            rebuilt = assemble_text(words)
            project.set_raw_text(page_id, rebuilt)
            project.set_words(
                page_id,
                [
                    WordBox(w.text, w.x, w.y, w.w, w.h, w.conf, w.line_idx, w.word_idx,
                            w.start, w.end)
                    for w in words
                ],
                engine="pdf-text-layer",
            )
            project.set_lines(
                page_id,
                [
                    LineBox(ln.line_idx, ln.x, ln.y, ln.w, ln.h, ln.text, ln.conf, ln.baseline)
                    for ln in lines_from_words(words)
                ],
            )
            return len(rebuilt)

        text = _strip_boilerplate(text, boilerplate).strip()
        if not text:
            return 0
        project.set_raw_text(page_id, text)
        return len(text)
    finally:
        textpage.close()


def _drop_boilerplate_words(words: list[OcrWord], boilerplate: Iterable[str]) -> list[OcrWord]:
    """Remove whole lines that match a known stamp, and close the gap in the numbering."""
    drop = {b.strip() for b in boilerplate if b.strip()}
    if not drop or not words:
        return words

    keep_lines = {
        line.line_idx for line in lines_from_words(words) if line.text.strip() not in drop
    }
    kept = [w for w in words if w.line_idx in keep_lines]

    # line_idx is used as a dense index downstream, so it has to be renumbered.
    renumbered = {old: new for new, old in enumerate(sorted(keep_lines))}
    for word in kept:
        word.line_idx = renumbered[word.line_idx]
    return kept


def _words_from_charboxes(
    boxes: list[tuple[str, float, float, float, float]],
    page_height: float,
    scale: float,
    line_tolerance: float = 0.6,
) -> list[OcrWord]:
    """Group characters into words and lines using their geometry."""
    words: list[OcrWord] = []
    buffer: list[tuple[str, float, float, float, float]] = []
    line_idx = 0
    word_idx = 0
    last_baseline: float | None = None
    last_right: float | None = None

    def flush() -> None:
        nonlocal buffer, word_idx
        if not buffer:
            return
        text = "".join(c for c, *_ in buffer)
        lefts = [b[1] for b in buffer]
        bottoms = [b[2] for b in buffer]
        rights = [b[3] for b in buffer]
        tops = [b[4] for b in buffer]
        x0, y1 = min(lefts), max(tops)
        x1, y0 = max(rights), min(bottoms)
        words.append(
            OcrWord(
                text=text,
                x=int(x0 * scale),
                y=int((page_height - y1) * scale),
                w=max(1, int((x1 - x0) * scale)),
                h=max(1, int((y1 - y0) * scale)),
                conf=1.0,
                line_idx=line_idx,
                word_idx=word_idx,
            )
        )
        word_idx += 1
        buffer = []

    for char, left, bottom, right, top in boxes:
        if char.isspace() or not char:
            flush()
            if char == "\n":
                line_idx += 1
                word_idx = 0
            continue

        baseline = bottom
        height = max(1e-6, top - bottom)
        if last_baseline is not None and abs(baseline - last_baseline) > height * line_tolerance:
            flush()
            line_idx += 1
            word_idx = 0
        elif last_right is not None and (left - last_right) > height * 0.55:
            # A gap wider than roughly half a character height reads as a word break.
            flush()

        buffer.append((char, left, bottom, right, top))
        last_baseline = baseline
        last_right = right

    flush()
    return words


# --------------------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------------------

_PAGE_BREAK = re.compile(r"\f|^\s*(?:---+\s*page\s*\d*\s*---+|\[page\s*\d+\])\s*$", re.IGNORECASE | re.MULTILINE)


def import_text(
    project: Project,
    path: str | Path,
    encoding: str = "utf-8",
    lines_per_page: int = 0,
    progress: ProgressFn | None = None,
) -> ImportReport:
    """Import loose OCR text with no scans — the repair-only path.

    Pages are split on form feeds or ``--- page N ---`` markers when present, otherwise on
    ``lines_per_page``, otherwise the whole file becomes one page.
    """
    raw = Path(path).read_text(encoding=encoding, errors="replace")
    chunks = [c for c in _PAGE_BREAK.split(raw) if c and c.strip()]

    if len(chunks) <= 1 and lines_per_page > 0:
        lines = raw.split("\n")
        chunks = [
            "\n".join(lines[i : i + lines_per_page]) for i in range(0, len(lines), lines_per_page)
        ]
    if not chunks:
        chunks = [raw]

    report = ImportReport(kind=SourceKind.TEXT)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if progress:
            progress(i, total, f"page {i + 1}")
        page = project.add_page(None, label=f"p. {i + 1}")
        project.set_raw_text(page.id, chunk.strip("\n"))
        project.set_page_status(page.id, PageStatus.RECOGNIZED)
        report.pages_added += 1
        report.text_layer_chars += len(chunk)

    if progress:
        progress(total, total, "done")
    return report


# --------------------------------------------------------------------------------------


def import_any(
    project: Project,
    paths: Sequence[str | Path],
    progress: ProgressFn | None = None,
    **kwargs: object,
) -> ImportReport:
    """Dispatch on file type, so the UI can accept a mixed drop of files."""
    paths = [Path(p) for p in paths]
    images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    pdfs = [p for p in paths if p.suffix.lower() == ".pdf"]
    texts = [p for p in paths if p.suffix.lower() in {".txt", ".text", ".md"}]

    combined = ImportReport()
    if images:
        r = import_images(project, images, progress=progress)
        combined.pages_added += r.pages_added
        combined.warnings.extend(r.warnings)
        combined.kind = r.kind
    for pdf in pdfs:
        r = import_pdf(project, pdf, progress=progress, **kwargs)  # type: ignore[arg-type]
        combined.pages_added += r.pages_added
        combined.text_layer_chars += r.text_layer_chars
        combined.warnings.extend(r.warnings)
        combined.kind = r.kind
    for text in texts:
        r = import_text(project, text, progress=progress)
        combined.pages_added += r.pages_added
        combined.text_layer_chars += r.text_layer_chars
        combined.kind = r.kind
    return combined


def _natural_key(path: Path) -> tuple:
    """Sort page_2 before page_10."""
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)
