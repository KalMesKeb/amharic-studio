"""An intermediate document model shared by every exporter.

Exporters should not each re-derive a book's structure from pages and structure marks, so
that work happens once here. The model also keeps the original page boundaries, which lets
the EPUB carry a ``page-list`` that maps back to the printed pagination — the thing that
makes a digital edition citable against the paper one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import fidel
from ..project import FURNITURE_KINDS, PageStatus, Project


@dataclass
class Block:
    """One typographic unit: a paragraph, heading, verse line, footnote or caption."""

    kind: str
    text: str
    level: int = 1
    page_idx: int = 0
    label: str = ""

    @property
    def is_heading(self) -> bool:
        return self.kind in {"chapter", "heading", "subheading"}


@dataclass
class Chapter:
    title: str
    level: int = 1
    blocks: list[Block] = field(default_factory=list)
    start_page_idx: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class BookDocument:
    title: str = ""
    author: str = ""
    language: str = "am"
    chapters: list[Chapter] = field(default_factory=list)
    #: ``(page_idx, printed_label)`` for the EPUB page-list and PDF page mapping.
    page_breaks: list[tuple[int, str]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def blocks(self) -> list[Block]:
        return [b for chapter in self.chapters for b in chapter.blocks]

    @property
    def full_text(self) -> str:
        return "\n\n".join(c.text for c in self.chapters)

    @property
    def character_set(self) -> str:
        """Every distinct character in the book — what font subsetting is driven from."""
        return "".join(sorted(set(self.full_text + self.title + self.author)))

    def statistics(self) -> dict[str, int | float]:
        text = self.full_text
        return {
            "chapters": len(self.chapters),
            "blocks": len(self.blocks),
            "characters": len(text),
            "words": len(fidel.words(text)),
            "pages": len(self.page_breaks),
            "ethiopic_ratio": fidel.ethiopic_ratio(text),
        }


# --------------------------------------------------------------------------------------


def build_document(
    project: Project,
    include_furniture: bool = False,
    split_on_chapters: bool = True,
    pages_per_chunk: int = 25,
) -> BookDocument:
    """Assemble a :class:`BookDocument` from a project's corrected text and structure marks.

    Chapters come from structure marks when the editor has made them. Without any marks,
    the book is chunked into fixed page groups so exports still produce a navigable file
    rather than one enormous document.
    """
    document = BookDocument(
        title=project.title or project.path.stem,
        author=project.author,
        language=project.get_meta("language", "am"),
    )

    current = Chapter(title=document.title or "Text", level=1, start_page_idx=0)
    chapters: list[Chapter] = []

    for page in project.iter_pages():
        if page.status is PageStatus.SKIPPED:
            continue
        text = project.get_edited(page.id)
        if not text.strip():
            continue

        marks = project.get_structure(page.id)
        document.page_breaks.append((page.idx, page.label or str(page.idx + 1)))

        if not include_furniture:
            text = _remove_spans(text, [m for m in marks if m.kind in FURNITURE_KINDS])
            marks = [m for m in marks if m.kind not in FURNITURE_KINDS]

        for block in _blocks_for_page(text, marks, page.idx):
            starts_chapter = split_on_chapters and block.kind == "chapter"
            if starts_chapter and current.blocks:
                chapters.append(current)
                current = Chapter(title=block.text.strip(), level=1, start_page_idx=page.idx)
                current.blocks.append(block)
                continue
            if starts_chapter and not current.blocks:
                current.title = block.text.strip()
                current.start_page_idx = page.idx
            current.blocks.append(block)

        if not split_on_chapters and len(document.page_breaks) % pages_per_chunk == 0:
            chapters.append(current)
            current = Chapter(title=f"{document.title} ({page.idx + 1})", start_page_idx=page.idx)

    if current.blocks:
        chapters.append(current)

    document.chapters = chapters or [Chapter(title=document.title)]
    return document


def _remove_spans(text: str, marks: list) -> str:
    for mark in sorted(marks, key=lambda m: m.start, reverse=True):
        text = text[: mark.start] + text[mark.end :]
    return "\n".join(line for line in text.split("\n") if line.strip())


def _blocks_for_page(text: str, marks: list, page_idx: int) -> list[Block]:
    """Turn a page's text plus its structure marks into ordered blocks.

    Marked spans become their marked kind; everything between them is split into
    paragraphs on blank lines.
    """
    marks = sorted((m for m in marks if m.start < m.end <= len(text)), key=lambda m: m.start)
    blocks: list[Block] = []
    cursor = 0

    for mark in marks:
        if mark.start > cursor:
            blocks.extend(_paragraphs(text[cursor : mark.start], page_idx))
        blocks.append(
            Block(mark.kind, text[mark.start : mark.end].strip(), mark.level, page_idx, mark.label)
        )
        cursor = mark.end

    if cursor < len(text):
        blocks.extend(_paragraphs(text[cursor:], page_idx))
    return [b for b in blocks if b.text.strip()]


def _paragraphs(chunk: str, page_idx: int) -> list[Block]:
    """Split into paragraphs on blank lines, joining wrapped lines within a paragraph.

    Scanned pages arrive as one text line per printed line, which must be reflowed for an
    EPUB. Lines are joined with a space, except where a line already ends in ፡, since that
    separator does its own spacing.
    """
    out: list[Block] = []
    for raw in chunk.split("\n\n"):
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = lines[0]
        for line in lines[1:]:
            joined += "" if joined.endswith(fidel.WORDSPACE) else " "
            joined += line
        out.append(Block("paragraph", joined, 1, page_idx))
    return out
