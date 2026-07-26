"""Line breaking for Amharic text.

Generic wrappers break on spaces, which fails badly on traditional Ethiopic typesetting
where ፡ separates words and spaces may not appear at all — a paragraph would come out as
one unbreakable line running off the page.

Per UAX #14, Ethiopic syllables are ordinary alphabetic characters (class AL) with no
break between them, while the Ethiopic punctuation marks ፡ ። ፣ ፤ ፥ ፦ ፧ ፨ are "break
after" (class BA). So legal break points are: after a space, and after any of those marks.
An emergency character-level break is kept for the pathological case of a single token
wider than the measure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .fidel import ETHIOPIC_PUNCTUATION

#: Marks that permit a break immediately after them.
BREAK_AFTER = frozenset(ETHIOPIC_PUNCTUATION) | frozenset(",.;:!?)]}»”’/-–—")

#: Characters that must not start a line.
NO_START = frozenset(ETHIOPIC_PUNCTUATION) | frozenset(",.;:!?)]}»”’")

MeasureFn = Callable[[str], float]


@dataclass(frozen=True)
class Segment:
    """An unbreakable run of text, plus whatever separator followed it."""

    text: str
    #: Trailing space that is dropped when the line breaks here.
    trailing_space: str = ""

    @property
    def full(self) -> str:
        return self.text + self.trailing_space

    @property
    def breakable_after(self) -> bool:
        return True


def segment(text: str) -> list[Segment]:
    """Split a paragraph into unbreakable runs at legal break opportunities."""
    segments: list[Segment] = []
    buffer: list[str] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]
        buffer.append(ch)

        if ch in (" ", "\t", "\u00a0"):
            spaces = ch
            i += 1
            while i < n and text[i] in (" ", "\t", "\u00a0"):
                spaces += text[i]
                i += 1
            buffer.pop()
            segments.append(Segment("".join(buffer), spaces))
            buffer = []
            continue

        if ch in BREAK_AFTER:
            # Keep a run of marks together (።፣ or ...) rather than breaking between them.
            while i + 1 < n and text[i + 1] in NO_START:
                i += 1
                buffer.append(text[i])
            trailing = ""
            j = i + 1
            while j < n and text[j] in (" ", "\t", "\u00a0"):
                trailing += text[j]
                j += 1
            segments.append(Segment("".join(buffer), trailing))
            buffer = []
            i = j
            continue

        i += 1

    if buffer:
        segments.append(Segment("".join(buffer)))
    return segments


@dataclass
class Line:
    segments: list[Segment]
    width: float
    is_last: bool = False

    @property
    def text(self) -> str:
        """The rendered text, with the final trailing space dropped."""
        if not self.segments:
            return ""
        body = "".join(s.full for s in self.segments[:-1])
        return body + self.segments[-1].text

    @property
    def break_count(self) -> int:
        """Gaps available for justification: the break opportunities inside the line."""
        return max(0, len(self.segments) - 1)


def wrap(text: str, max_width: float, measure: MeasureFn) -> list[Line]:
    """Greedily wrap one paragraph to ``max_width``.

    Greedy rather than Knuth–Plass: without hyphenation (Amharic does not hyphenate) the
    two produce nearly identical results, and greedy keeps repagination fast enough to
    stay interactive in the export preview.
    """
    lines: list[Line] = []
    current: list[Segment] = []
    current_width = 0.0

    for seg in segment(text):
        piece_width = measure(seg.text)
        gap_width = measure(seg.trailing_space) if seg.trailing_space else 0.0
        prospective = current_width + piece_width

        if current and prospective > max_width:
            lines.append(Line(current, current_width))
            current, current_width = [], 0.0
            prospective = piece_width

        if piece_width > max_width and not current:
            # A single run too wide for the measure: break it by character.
            for chunk, chunk_width in _break_by_character(seg.text, max_width, measure):
                lines.append(Line([Segment(chunk)], chunk_width))
            if lines:
                tail = lines.pop()
                current = list(tail.segments)
                current_width = tail.width + gap_width
                current[-1] = Segment(current[-1].text, seg.trailing_space)
            continue

        current.append(seg)
        current_width = prospective + gap_width

    if current:
        lines.append(Line(current, current_width))
    if lines:
        lines[-1].is_last = True
    return lines


def _break_by_character(
    text: str, max_width: float, measure: MeasureFn
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    buffer = ""
    for ch in text:
        candidate = buffer + ch
        if buffer and measure(candidate) > max_width:
            out.append((buffer, measure(buffer)))
            buffer = ch
        else:
            buffer = candidate
    if buffer:
        out.append((buffer, measure(buffer)))
    return out


def wrap_paragraphs(text: str, max_width: float, measure: MeasureFn) -> list[list[Line]]:
    """Wrap a multi-paragraph block, one list of lines per paragraph."""
    return [wrap(para, max_width, measure) for para in text.split("\n") if para.strip()]
