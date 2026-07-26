"""Common types for OCR backends.

Every backend returns the same shape: page text plus word boxes carrying the character
offsets of each word within that text. Those offsets are what tie the editor to the scan —
click a word in the text pane and the canvas can highlight the exact rectangle it came
from, and vice versa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

from PIL import Image

from .. import fidel


class PageSegMode(IntEnum):
    """Tesseract page segmentation modes, reused as a common vocabulary."""

    OSD_ONLY = 0
    AUTO_OSD = 1
    AUTO = 3
    SINGLE_COLUMN = 4
    SINGLE_BLOCK_VERTICAL = 5
    SINGLE_BLOCK = 6  # the usual choice for a cropped body-text region
    SINGLE_LINE = 7
    SINGLE_WORD = 8
    SPARSE = 11
    SPARSE_OSD = 12


@dataclass
class OcrWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 0.0  # 0–1
    line_idx: int = 0
    word_idx: int = 0
    start: int = -1  # character offset into OcrResult.text
    end: int = -1

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class OcrLine:
    line_idx: int
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    conf: float = 0.0
    baseline: int = 0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class OcrResult:
    text: str = ""
    words: list[OcrWord] = field(default_factory=list)
    lines: list[OcrLine] = field(default_factory=list)
    engine: str = ""
    language: str = ""
    duration_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def label(self) -> str:
        """Engine plus model, e.g. ``tesseract/amh``.

        Running one engine over two language models is a normal way to get a second
        opinion, so reports that identify a run by engine name alone would show the same
        name twice and hide which model actually produced a reading.
        """
        return f"{self.engine}/{self.language}" if self.language else self.engine

    @property
    def mean_confidence(self) -> float:
        scored = [w.conf for w in self.words if w.conf > 0]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def low_confidence_words(self) -> list[OcrWord]:
        return [w for w in self.words if 0 < w.conf < 0.72]

    def confidence_spans(self) -> list[tuple[int, int, float]]:
        return [(w.start, w.end, w.conf) for w in self.words if w.start >= 0]


class OcrBackend(ABC):
    """Interface every recognizer implements."""

    name: str = "base"
    display_name: str = "Base"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now on this machine."""

    @abstractmethod
    def languages(self) -> list[str]:
        """Language or model identifiers this backend can use."""

    @abstractmethod
    def recognize(
        self,
        image: Image.Image,
        language: str = "amh",
        psm: PageSegMode = PageSegMode.SINGLE_BLOCK,
        **options: object,
    ) -> OcrResult:
        """Recognize a page image."""

    def unavailable_reason(self) -> str:
        return "" if self.available() else f"{self.display_name} is not installed"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} available={self.available()}>"


# --------------------------------------------------------------------------------------
# Text assembly
# --------------------------------------------------------------------------------------


def assemble_text(
    words: list[OcrWord],
    line_separator: str = "\n",
    word_separator: str = " ",
) -> str:
    """Join word boxes into page text, writing character offsets back onto each word."""
    parts: list[str] = []
    cursor = 0
    current_line = words[0].line_idx if words else 0

    for i, word in enumerate(words):
        if i > 0:
            separator = line_separator if word.line_idx != current_line else word_separator
            parts.append(separator)
            cursor += len(separator)
            current_line = word.line_idx
        word.start = cursor
        word.end = cursor + len(word.text)
        parts.append(word.text)
        cursor = word.end

    return "".join(parts)


def split_wordspace_boxes(words: list[OcrWord]) -> list[OcrWord]:
    """Split boxes that contain ፡ separators into one box per word.

    Traditional Amharic typesetting puts ፡ between words with no spaces, so a recognizer
    hands back an entire run as a single token. Subdividing the box proportionally by
    character count gives the editor usable per-word geometry. The estimate is approximate
    — fidel are near-monospaced in most book faces, which is what makes it hold up.
    """
    out: list[OcrWord] = []
    for word in words:
        if fidel.WORDSPACE not in word.text or len(word.text) < 3:
            out.append(word)
            continue

        pieces = word.text.split(fidel.WORDSPACE)
        total = len(word.text)
        offset = 0
        sub_index = 0
        for piece in pieces:
            if piece:
                x = word.x + round(word.w * offset / total)
                width = round(word.w * len(piece) / total)
                out.append(
                    OcrWord(
                        piece, x, word.y, max(1, width), word.h, word.conf,
                        word.line_idx, word.word_idx * 100 + sub_index,
                    )
                )
                sub_index += 1
            offset += len(piece) + 1
    return out


def lines_from_words(words: list[OcrWord]) -> list[OcrLine]:
    """Derive line boxes by unioning the word boxes on each line."""
    grouped: dict[int, list[OcrWord]] = {}
    for word in words:
        grouped.setdefault(word.line_idx, []).append(word)

    lines: list[OcrLine] = []
    for line_idx in sorted(grouped):
        members = grouped[line_idx]
        x0 = min(w.x for w in members)
        y0 = min(w.y for w in members)
        x1 = max(w.x + w.w for w in members)
        y1 = max(w.y + w.h for w in members)
        conf = sum(w.conf for w in members) / len(members)
        lines.append(
            OcrLine(
                line_idx, x0, y0, x1 - x0, y1 - y0,
                " ".join(w.text for w in members), conf, baseline=y1,
            )
        )
    return lines
