"""Text normalization for Ge'ez-script documents.

Two policies, chosen by the user per project:

``FAITHFUL``
    Diplomatic transcription. ሠ ኀ ዐ ፀ stay exactly as printed and ፡ remains a word
    separator. Only unambiguous OCR damage is repaired — invisible characters, ASCII
    punctuation standing in for Ethiopic punctuation, whitespace noise.

``MODERN``
    Everything ``FAITHFUL`` does, plus orthographic modernization: homophone families
    collapse to their canonical member and ፡ becomes an ordinary space.

Rules are individually toggleable and every rule reports what it changed, so nothing
happens to a book that the editor cannot see and undo.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from . import fidel
from .fidel import (
    COMMA,
    FULL_STOP,
    PREFACE_COLON,
    QUESTION_MARK,
    SEMICOLON,
    WORDSPACE,
)


class Policy(str, Enum):
    FAITHFUL = "faithful"
    MODERN = "modern"


class Severity(str, Enum):
    #: Reverses damage that cannot be intentional (invisible characters, NFC).
    SAFE = "safe"
    #: Very likely correct, but a scholar transcribing exactly may want it off.
    LIKELY = "likely"
    #: Changes the orthography itself. Only ever on under the MODERN policy.
    EDITORIAL = "editorial"


@dataclass
class Change:
    rule: str
    before: str
    after: str
    count: int = 1
    severity: Severity = Severity.SAFE

    def describe(self) -> str:
        shown_before = self.before.replace("\n", "\\n") or "∅"
        shown_after = self.after.replace("\n", "\\n") or "∅"
        return f"{shown_before} → {shown_after} ×{self.count}"


@dataclass
class NormalizationResult:
    text: str
    changes: list[Change] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return sum(c.count for c in self.changes)

    def summary(self) -> str:
        if not self.changes:
            return "no changes"
        parts = [f"{c.rule}: {c.describe()}" for c in self.changes]
        return "; ".join(parts)


# --------------------------------------------------------------------------------------
# Character sets
# --------------------------------------------------------------------------------------

#: Zero-width and formatting characters that silently break search, sorting and EPUB
#: validation. Never legitimate in Amharic body text.
INVISIBLE = {
    "\u200b": "zero width space",
    "\u200c": "zero width non-joiner",
    "\u200d": "zero width joiner",
    "\u200e": "left-to-right mark",
    "\u200f": "right-to-left mark",
    "\ufeff": "byte order mark",
    "\u00ad": "soft hyphen",
    "\u2060": "word joiner",
    "\u180e": "mongolian vowel separator",
}

#: ASCII punctuation OCR emits in place of Ethiopic punctuation. Only applied when the
#: surrounding text is Ethiopic, so Latin quotations inside a book are left alone.
ASCII_PUNCT_EQUIVALENT = {
    ":": WORDSPACE,
    ";": SEMICOLON,
    ",": COMMA,
    "?": QUESTION_MARK,
}


# --------------------------------------------------------------------------------------
# Normalizer
# --------------------------------------------------------------------------------------

ALL_RULES = (
    "nfc",
    "strip_invisible",
    "normalize_newlines",
    "double_wordspace",
    "ascii_full_stop",
    "ascii_punctuation",
    "dehyphenate",
    "punctuation_spacing",
    "collapse_whitespace",
    "strip_line_edges",
    "modern_orthography",
    "wordspace_to_space",
)

#: Rules that only make sense under the MODERN policy.
_EDITORIAL_RULES = frozenset({"modern_orthography", "wordspace_to_space"})


class Normalizer:
    def __init__(
        self,
        policy: Policy = Policy.FAITHFUL,
        disabled: set[str] | None = None,
        fold_kha: bool = False,
    ) -> None:
        self.policy = policy
        self.disabled = set(disabled or ())
        self.fold_kha = fold_kha
        if policy is Policy.FAITHFUL:
            self.disabled |= _EDITORIAL_RULES

    def enabled(self, rule: str) -> bool:
        return rule not in self.disabled

    def normalize(self, text: str) -> NormalizationResult:
        changes: list[Change] = []
        for rule in ALL_RULES:
            if not self.enabled(rule):
                continue
            text = getattr(self, f"_rule_{rule}")(text, changes)
        return NormalizationResult(text, changes)

    # -- individual rules ---------------------------------------------------------------

    def _rule_nfc(self, text: str, changes: list[Change]) -> str:
        out = unicodedata.normalize("NFC", text)
        if out != text:
            changes.append(Change("nfc", "decomposed", "composed", 1, Severity.SAFE))
        return out

    def _rule_strip_invisible(self, text: str, changes: list[Change]) -> str:
        counts = Counter(ch for ch in text if ch in INVISIBLE)
        if not counts:
            return text
        for ch, n in counts.items():
            changes.append(Change("strip_invisible", INVISIBLE[ch], "", n, Severity.SAFE))
        return text.translate({ord(ch): None for ch in counts})

    def _rule_normalize_newlines(self, text: str, changes: list[Change]) -> str:
        out = text.replace("\r\n", "\n").replace("\r", "\n")
        if out != text:
            changes.append(Change("normalize_newlines", "CRLF", "LF", 1, Severity.SAFE))
        return out

    def _rule_double_wordspace(self, text: str, changes: list[Change]) -> str:
        """፡፡ is the traditional way of writing ።, and OCR splits ። into two ፡."""
        n = text.count("\u1361\u1361")
        if n:
            text = text.replace("\u1361\u1361", FULL_STOP)
            changes.append(Change("double_wordspace", "፡፡", "።", n, Severity.LIKELY))
        return text

    def _rule_ascii_full_stop(self, text: str, changes: list[Change]) -> str:
        """'::' between Ethiopic text is a typewriter-era stand-in for ።."""
        pattern = re.compile(r"(?<=[\u1200-\u137f])\s*::\s*")
        n = len(pattern.findall(text))
        if n:
            text = pattern.sub(FULL_STOP + " ", text)
            changes.append(Change("ascii_full_stop", "::", "።", n, Severity.LIKELY))
        return text

    def _rule_ascii_punctuation(self, text: str, changes: list[Change]) -> str:
        """Replace ASCII punctuation that sits between two Ethiopic characters.

        The lookarounds matter: a colon inside a Latin citation or a page reference must
        survive untouched, so only colons with fidel on both sides are converted.
        """
        counts: Counter[str] = Counter()
        for ascii_ch, ethiopic_ch in ASCII_PUNCT_EQUIVALENT.items():
            if ascii_ch == ":":
                pattern = re.compile(r"(?<=[\u1200-\u137f])\s*:\s*(?=[\u1200-\u137f])")
                replacement = ethiopic_ch
            else:
                pattern = re.compile(
                    rf"(?<=[\u1200-\u137f])\s*{re.escape(ascii_ch)}(?=\s|$|[\u1200-\u137f])"
                )
                replacement = ethiopic_ch
            found = len(pattern.findall(text))
            if found:
                text = pattern.sub(replacement, text)
                counts[ascii_ch] = found

        for ascii_ch, n in counts.items():
            changes.append(
                Change(
                    "ascii_punctuation",
                    ascii_ch,
                    ASCII_PUNCT_EQUIVALENT[ascii_ch],
                    n,
                    Severity.LIKELY,
                )
            )
        return text

    def _rule_dehyphenate(self, text: str, changes: list[Change]) -> str:
        """Rejoin a word broken across lines by a hyphen."""
        pattern = re.compile(r"(\w)[-\u2010\u2011]\n\s*(\w)")
        n = len(pattern.findall(text))
        if n:
            text = pattern.sub(r"\1\2", text)
            changes.append(Change("dehyphenate", "-\\n", "", n, Severity.LIKELY))
        return text

    def _rule_punctuation_spacing(self, text: str, changes: list[Change]) -> str:
        """Ethiopic punctuation hugs the preceding word and is followed by a space."""
        total = 0

        # No space before punctuation.
        marks = "".join(re.escape(p) for p in fidel.ETHIOPIC_PUNCTUATION - {WORDSPACE})
        before = re.compile(rf"[ \t]+([{marks}])")
        total += len(before.findall(text))
        text = before.sub(r"\1", text)

        # A space after sentence-level punctuation, unless the line ends there.
        after = re.compile(
            rf"([{re.escape(FULL_STOP + QUESTION_MARK + COMMA + SEMICOLON + PREFACE_COLON)}])"
            r"(?=[\u1200-\u137f])"
        )
        total += len(after.findall(text))
        text = after.sub(r"\1 ", text)

        # ፡ is a separator: it must never be padded with spaces.
        padded = re.compile(rf"[ \t]*{re.escape(WORDSPACE)}[ \t]*")
        hits = [m for m in padded.finditer(text) if m.group(0) != WORDSPACE]
        if hits:
            total += len(hits)
            text = padded.sub(WORDSPACE, text)

        if total:
            changes.append(Change("punctuation_spacing", "spacing", "tidied", total, Severity.SAFE))
        return text

    def _rule_collapse_whitespace(self, text: str, changes: list[Change]) -> str:
        pattern = re.compile(r"[ \t\u00a0]{2,}")
        n = len(pattern.findall(text))
        if n:
            text = pattern.sub(" ", text)
            changes.append(Change("collapse_whitespace", "  ", " ", n, Severity.SAFE))
        # Never more than one blank line between paragraphs.
        blanks = re.compile(r"\n{3,}")
        m = len(blanks.findall(text))
        if m:
            text = blanks.sub("\n\n", text)
            changes.append(Change("collapse_whitespace", "blank lines", "\\n\\n", m, Severity.SAFE))
        return text

    def _rule_strip_line_edges(self, text: str, changes: list[Change]) -> str:
        lines = text.split("\n")
        stripped = [ln.strip(" \t\u00a0") for ln in lines]
        n = sum(1 for a, b in zip(lines, stripped, strict=True) if a != b)
        if n:
            changes.append(Change("strip_line_edges", "padding", "", n, Severity.SAFE))
        return "\n".join(stripped)

    def _rule_modern_orthography(self, text: str, changes: list[Change]) -> str:
        """Collapse homophone families to their canonical member. MODERN policy only."""
        policy = fidel.FoldPolicy(
            fold_homophone_families=True,
            fold_ha_orders=False,  # ሀ and ሃ are distinct letters; only families collapse
            fold_kha=self.fold_kha,
            fold_labiovelar=False,
        )
        out_chars: list[str] = []
        counts: Counter[tuple[str, str]] = Counter()
        for ch in text:
            folded = fidel.fold(ch, policy)
            if folded != ch:
                counts[(ch, folded)] += 1
            out_chars.append(folded)
        for (before, after), n in counts.items():
            changes.append(Change("modern_orthography", before, after, n, Severity.EDITORIAL))
        return "".join(out_chars)

    def _rule_wordspace_to_space(self, text: str, changes: list[Change]) -> str:
        n = text.count(WORDSPACE)
        if n:
            text = text.replace(WORDSPACE, " ")
            changes.append(Change("wordspace_to_space", "፡", "␠", n, Severity.EDITORIAL))
        return text


# --------------------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------------------


def normalize(text: str, policy: Policy = Policy.FAITHFUL) -> str:
    return Normalizer(policy).normalize(text).text


def detect_orthography(text: str) -> Policy:
    """Guess which orthography a book uses, to pick sensible project defaults.

    Traditional printing separates words with ፡; modern printing uses spaces. A high
    ratio of ፡ to spaces is the clearest signal, with archaic letters as a tiebreaker.
    """
    wordspaces = text.count(WORDSPACE)
    spaces = text.count(" ")
    if wordspaces >= 3 and wordspaces > spaces:
        return Policy.FAITHFUL

    def family_count(bases: tuple[int, ...]) -> int:
        return sum(text.count(chr(cp)) for base in bases for cp in range(base, base + 8))

    archaic = family_count((0x1220, 0x1280, 0x12D0, 0x1340))
    modern = family_count((0x1230, 0x1200, 0x12A0, 0x1338))
    if archaic and archaic > modern * 0.25:
        return Policy.FAITHFUL
    return Policy.MODERN
