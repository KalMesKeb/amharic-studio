"""Quality assessment and review triage.

The practical problem with a 400-page book is not correcting a page — it is knowing which
page to open. These metrics exist to answer that: every page gets a score, and the review
queue is sorted worst first, so effort lands where the recognition actually failed instead
of being spread evenly across pages that were already fine.

The signals deliberately do not include recognizer confidence alone. Confidence is
self-reported and an engine can be confidently wrong; unknown-word rate and impossible
character usage are external checks that catch failures confidence misses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import fidel
from .lexicon import Lexicon
from .project import PageStatus, Project
from .suggest import IssueKind, Suggester


@dataclass
class PageQuality:
    page_id: int
    page_idx: int
    label: str
    characters: int = 0
    words: int = 0
    unknown_rate: float = 0.0
    mean_confidence: float = 0.0
    ethiopic_ratio: float = 0.0
    issue_counts: Counter[str] = field(default_factory=Counter)
    suspect_characters: Counter[str] = field(default_factory=Counter)
    flags: list[str] = field(default_factory=list)
    status: PageStatus = PageStatus.NEW

    @property
    def high_severity_issues(self) -> int:
        return self.issue_counts.get("high", 0)

    @property
    def score(self) -> float:
        """0–1, higher is better. Used to sort the review queue.

        Unknown-word rate carries the most weight because it is the signal least
        correlated with the recognizer's own opinion of itself.
        """
        if not self.words:
            return 1.0 if self.status is PageStatus.SKIPPED else 0.0

        lexical = 1.0 - min(1.0, self.unknown_rate)
        confidence = self.mean_confidence if self.mean_confidence > 0 else 0.75
        script = self.ethiopic_ratio
        penalty = min(0.35, 0.02 * self.high_severity_issues + 0.05 * len(self.suspect_characters))

        return max(0.0, 0.5 * lexical + 0.25 * confidence + 0.25 * script - penalty)

    @property
    def grade(self) -> str:
        score = self.score
        if score >= 0.92:
            return "excellent"
        if score >= 0.82:
            return "good"
        if score >= 0.65:
            return "needs review"
        return "poor"

    def describe(self) -> str:
        return (
            f"{self.label}: {self.grade} ({self.score:.0%}) — "
            f"{self.unknown_rate:.0%} unknown words, {self.words} words"
            + (f", flags: {', '.join(self.flags)}" if self.flags else "")
        )


@dataclass
class BookQuality:
    pages: list[PageQuality] = field(default_factory=list)
    character_histogram: Counter[str] = field(default_factory=Counter)
    total_words: int = 0
    total_characters: int = 0

    @property
    def mean_score(self) -> float:
        scored = [p for p in self.pages if p.words]
        return sum(p.score for p in scored) / len(scored) if scored else 0.0

    @property
    def unknown_rate(self) -> float:
        total = sum(p.words for p in self.pages)
        if not total:
            return 0.0
        return sum(p.unknown_rate * p.words for p in self.pages) / total

    @property
    def verified_pages(self) -> int:
        return sum(1 for p in self.pages if p.status is PageStatus.VERIFIED)

    def worst_pages(self, limit: int = 20) -> list[PageQuality]:
        """Review queue: lowest-scoring pages first, verified pages excluded."""
        candidates = [p for p in self.pages if p.status is not PageStatus.VERIFIED and p.words]
        return sorted(candidates, key=lambda p: p.score)[:limit]

    def rare_characters(self, threshold: int = 3) -> list[tuple[str, int]]:
        """Characters used only a handful of times across the whole book.

        In a book of any length a fidel appearing twice is far more likely to be a
        misrecognition of a common one than a genuinely rare letter, which makes this a
        productive place to hunt for systematic errors.
        """
        rare = [
            (ch, n)
            for ch, n in self.character_histogram.items()
            if n <= threshold and fidel.is_amharic_syllable(ch)
        ]
        return sorted(rare, key=lambda pair: pair[1])

    def impossible_characters(self) -> list[tuple[str, int]]:
        """Ethiopic characters outside the range Amharic uses at all."""
        return sorted(
            ((ch, n) for ch, n in self.character_histogram.items() if fidel.is_non_amharic_ethiopic(ch)),
            key=lambda pair: -pair[1],
        )

    def summary(self) -> str:
        return (
            f"{len(self.pages)} pages · quality {self.mean_score:.0%} · "
            f"{self.unknown_rate:.1%} unknown words · "
            f"{self.verified_pages}/{len(self.pages)} verified"
        )


# --------------------------------------------------------------------------------------


def assess_page(
    project: Project,
    page_id: int,
    lexicon: Lexicon,
    suggester: Suggester | None = None,
) -> PageQuality:
    page = project.get_page(page_id)
    if page is None:
        raise KeyError(f"no page {page_id}")

    text = project.get_edited(page_id)
    tokens = fidel.words(text)
    quality = PageQuality(
        page_id=page_id,
        page_idx=page.idx,
        label=page.label,
        characters=len(text),
        words=len(tokens),
        mean_confidence=page.mean_conf,
        ethiopic_ratio=fidel.ethiopic_ratio(text),
        status=page.status,
    )
    if tokens:
        quality.unknown_rate = sum(
            1 for t in tokens if not lexicon.recognizes(t.text)
        ) / len(tokens)

    for ch in text:
        if fidel.is_non_amharic_ethiopic(ch):
            quality.suspect_characters[ch] += 1

    if suggester is not None:
        for issue in suggester.analyze(text):
            quality.issue_counts[issue.severity.value] += 1
            quality.issue_counts[issue.kind.value] += 1

    quality.flags = _flags_for(quality, text)
    return quality


def _flags_for(quality: PageQuality, text: str) -> list[str]:
    flags: list[str] = []
    if quality.words and quality.unknown_rate > 0.4:
        flags.append("very high unknown-word rate")
    if quality.ethiopic_ratio < 0.6 and quality.words > 5:
        flags.append("mostly non-Ethiopic text")
    if quality.suspect_characters:
        flags.append(f"{sum(quality.suspect_characters.values())} impossible characters")
    if quality.words and quality.characters / max(1, quality.words) > 24:
        flags.append("very long words — word separators may be missing")
    if 0 < quality.mean_confidence < 0.6:
        flags.append("low recognizer confidence")
    if quality.issue_counts.get(IssueKind.RUN_TOGETHER.value):
        flags.append("run-together words")
    if text and not text.strip():
        flags.append("blank")
    return flags


def assess_book(
    project: Project,
    lexicon: Lexicon,
    suggester: Suggester | None = None,
    progress: object = None,
) -> BookQuality:
    book = BookQuality()
    pages = project.pages()
    for index, page in enumerate(pages):
        if callable(progress):
            progress(index, len(pages), page.label)
        quality = assess_page(project, page.id, lexicon, suggester)
        book.pages.append(quality)
        book.total_words += quality.words
        book.total_characters += quality.characters
        book.character_histogram.update(
            ch for ch in project.get_edited(page.id) if not ch.isspace()
        )
    if callable(progress):
        progress(len(pages), len(pages), "done")
    return book
