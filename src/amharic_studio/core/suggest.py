"""Suggestion engine: find suspect spans and propose corrections.

Nothing here edits a document. Each finding is an :class:`Issue` with an offset range and
a ranked list of :class:`Suggestion` objects for the editor to accept or reject. That
separation is deliberate — automatic rewriting of a book you cannot audit is how silent
corruption gets baked into an edition.

Signals combined, in rough order of reliability:

1. **Character sanity** — invisible characters, Ethiopic blocks Amharic never uses, Latin
   letters stranded inside a fidel word.
2. **Lexicon** — unknown words, with candidates retrieved through the skeleton index and
   ranked by confusion-weighted distance.
3. **Segmentation** — a space OCR inserted into the middle of a word, or one it dropped
   between two words.
4. **Context** — bigram frequency, used to reorder otherwise comparable candidates.
5. **Recognizer confidence** — per-word confidence and disagreement between OCR engines,
   supplied by the caller when available.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from . import confusion, fidel, normalize
from .confusion import ConfusionModel
from .lexicon import Lexicon


class IssueKind(str, Enum):
    UNKNOWN_WORD = "unknown_word"
    SUSPECT_CHARACTER = "suspect_character"
    NON_AMHARIC_SCRIPT = "non_amharic_script"
    MIXED_SCRIPT = "mixed_script"
    SPLIT_WORD = "split_word"
    RUN_TOGETHER = "run_together"
    PUNCTUATION = "punctuation"
    LOW_CONFIDENCE = "low_confidence"
    ENGINE_DISAGREEMENT = "engine_disagreement"
    REPEATED_WORD = "repeated_word"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


@dataclass(frozen=True)
class Suggestion:
    text: str
    score: float  # lower is better
    source: str
    explanation: str = ""
    #: Whether this replacement is independently corroborated — either a word the lexicon
    #: knows, or a deterministic repair such as deleting a zero-width character.
    #: Unverified suggestions are offered to the editor but never applied automatically.
    verified: bool = False

    @property
    def confidence(self) -> float:
        """0–1 confidence, for colouring the UI."""
        return 1.0 / (1.0 + max(0.0, self.score))


@dataclass
class Issue:
    kind: IssueKind
    start: int
    end: int
    text: str
    message: str
    severity: Severity = Severity.MEDIUM
    suggestions: list[Suggestion] = field(default_factory=list)
    page_id: int | None = None
    #: Set when the flagged text is itself a known word, which bars auto-application:
    #: replacing a real word on a similarity score alone is how silent corruption happens.
    observed_is_known: bool = False

    @property
    def best(self) -> Suggestion | None:
        return self.suggestions[0] if self.suggestions else None

    @property
    def auto_applicable(self) -> bool:
        """Whether a batch pass may apply this without asking.

        Three conditions, all necessary. The suggestion must be a word the lexicon
        actually knows; the flagged text must not already be a known word; and the top
        candidate must be both cheap and clearly ahead of the runner-up. Anything else
        stays a decision for the editor.
        """
        if self.observed_is_known or not self.suggestions:
            return False
        top = self.suggestions[0]
        if not top.verified or top.score > 0.35:
            return False
        if len(self.suggestions) > 1 and self.suggestions[1].score - top.score < 0.25:
            return False
        return True

    def sort_key(self) -> tuple[int, int]:
        return (_SEVERITY_RANK[self.severity], self.start)


# --------------------------------------------------------------------------------------


@dataclass
class SuggestOptions:
    max_candidates: int = 6
    max_cost: float = 1.3
    check_segmentation: bool = True
    check_characters: bool = True
    check_punctuation: bool = True
    check_repeats: bool = True
    #: Words shorter than this rarely produce trustworthy candidates.
    min_word_length: int = 2
    #: Weight applied to bigram context when reordering candidates.
    context_weight: float = 0.25
    low_confidence_threshold: float = 0.72


class Suggester:
    def __init__(
        self,
        lexicon: Lexicon,
        model: ConfusionModel | None = None,
        options: SuggestOptions | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.model = model or confusion.DEFAULT_MODEL
        self.options = options or SuggestOptions()

    # -- entry point ---------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        word_confidences: Sequence[tuple[int, int, float]] | None = None,
        alternates: Sequence[tuple[int, int, list[str]]] | None = None,
    ) -> list[Issue]:
        """Analyse a page.

        ``word_confidences`` and ``alternates`` are optional OCR metadata: per-word
        confidence, and competing readings from other engines, each as
        ``(start, end, value)``.
        """
        issues: list[Issue] = []
        tokens = fidel.tokenize(text)
        word_tokens = [t for t in tokens if t.is_word]

        if self.options.check_characters:
            issues.extend(self._character_issues(text))
        if self.options.check_punctuation:
            issues.extend(self._punctuation_issues(text))
        if self.options.check_repeats:
            issues.extend(self._repeat_issues(word_tokens))

        issues.extend(self._lexicon_issues(word_tokens))
        if self.options.check_segmentation:
            issues.extend(self._segmentation_issues(word_tokens))
        if word_confidences:
            issues.extend(self._confidence_issues(text, word_confidences))
        if alternates:
            issues.extend(self._disagreement_issues(text, alternates))

        return self._dedupe(issues)

    # -- character level -------------------------------------------------------------------

    def _character_issues(self, text: str) -> list[Issue]:
        out: list[Issue] = []
        for i, ch in enumerate(text):
            if ch in normalize.INVISIBLE:
                out.append(
                    Issue(
                        IssueKind.SUSPECT_CHARACTER,
                        i,
                        i + 1,
                        ch,
                        f"Invisible character ({normalize.INVISIBLE[ch]})",
                        Severity.HIGH,
                        [Suggestion("", 0.0, "normalize", "Remove", verified=True)],
                    )
                )
            elif fidel.is_non_amharic_ethiopic(ch):
                name = unicodedata.name(ch, "unknown")
                suggestions = [
                    Suggestion(alt, cost, "confusion", "Visually similar Amharic fidel")
                    for alt, cost in self.model.neighbors(ch, max_cost=0.6, limit=4)
                    if fidel.is_amharic_syllable(alt)
                ]
                out.append(
                    Issue(
                        IssueKind.NON_AMHARIC_SCRIPT,
                        i,
                        i + 1,
                        ch,
                        f"{name} is outside the Amharic range — almost certainly a misread",
                        Severity.HIGH,
                        suggestions,
                    )
                )

        out.extend(self._mixed_script_issues(text))
        return out

    def _mixed_script_issues(self, text: str) -> list[Issue]:
        """Latin letters or digits stranded inside an otherwise Ethiopic word."""
        out: list[Issue] = []
        for token in fidel.words(text):
            word = token.text
            if len(word) < 2:
                continue
            ethiopic = sum(1 for c in word if fidel.is_ethiopic(c))
            latin = sum(1 for c in word if c.isascii() and c.isalpha())
            if ethiopic and latin and ethiopic >= latin:
                out.append(
                    Issue(
                        IssueKind.MIXED_SCRIPT,
                        token.start,
                        token.end,
                        word,
                        "Latin letters inside an Amharic word",
                        Severity.HIGH,
                        self._candidates_for(word),
                    )
                )
        return out

    # -- punctuation -----------------------------------------------------------------------

    _RE_STRAY_COLON = re.compile(r"(?<=[\u1200-\u137f])\s+:\s+")
    _RE_DOUBLED_PUNCT = re.compile(r"([\u1362-\u1368])\1+")
    _RE_LEADING_PUNCT = re.compile(r"^[ \t]*([\u1362-\u1368])", re.MULTILINE)

    def _punctuation_issues(self, text: str) -> list[Issue]:
        out: list[Issue] = []
        for m in self._RE_DOUBLED_PUNCT.finditer(text):
            out.append(
                Issue(
                    IssueKind.PUNCTUATION,
                    m.start(),
                    m.end(),
                    m.group(0),
                    "Repeated punctuation mark",
                    Severity.MEDIUM,
                    [Suggestion(m.group(1), 0.1, "punctuation", "Collapse to one", verified=True)],
                )
            )
        for m in self._RE_LEADING_PUNCT.finditer(text):
            out.append(
                Issue(
                    IssueKind.PUNCTUATION,
                    m.start(1),
                    m.end(1),
                    m.group(1),
                    "Line begins with punctuation — the preceding word may be lost",
                    Severity.LOW,
                )
            )
        for m in self._RE_STRAY_COLON.finditer(text):
            out.append(
                Issue(
                    IssueKind.PUNCTUATION,
                    m.start(),
                    m.end(),
                    m.group(0),
                    "Isolated colon between Amharic words",
                    Severity.MEDIUM,
                    [
                        Suggestion(
                            fidel.WORDSPACE, 0.05, "punctuation",
                            "Read as ፡ (word separator)", verified=True,
                        )
                    ],
                )
            )
        return out

    # -- repeats -----------------------------------------------------------------------------

    def _repeat_issues(self, tokens: Sequence[fidel.Token]) -> list[Issue]:
        """A word repeated back-to-back is usually a scan or line-stitching artefact."""
        out: list[Issue] = []
        for a, b in itertools.pairwise(tokens):
            if a.text == b.text and len(a.text) >= 2:
                out.append(
                    Issue(
                        IssueKind.REPEATED_WORD,
                        a.start,
                        b.end,
                        f"{a.text} {b.text}",
                        f"'{a.text}' appears twice in a row",
                        Severity.LOW,
                        [Suggestion(a.text, 0.3, "repeat", "Keep one")],
                    )
                )
        return out

    # -- lexicon ------------------------------------------------------------------------------

    def _lexicon_issues(self, tokens: Sequence[fidel.Token]) -> list[Issue]:
        out: list[Issue] = []
        for index, token in enumerate(tokens):
            word = token.text
            if len(word) < self.options.min_word_length:
                continue
            if not any(fidel.is_ethiopic(c) for c in word):
                continue  # leave Latin passages to a Latin spellchecker
            # Amharic inflects by gluing affixes on, so ``በቤታችን`` is correct text even
            # though only ``ቤት`` is in the wordlist. Flagging every inflected form would
            # bury the real errors.
            if self.lexicon.recognizes(word):
                continue

            previous = tokens[index - 1].text if index > 0 else None
            following = tokens[index + 1].text if index + 1 < len(tokens) else None
            suggestions = self._candidates_for(word, previous, following)
            severity = Severity.HIGH if suggestions and suggestions[0].score < 0.4 else Severity.MEDIUM
            out.append(
                Issue(
                    IssueKind.UNKNOWN_WORD,
                    token.start,
                    token.end,
                    word,
                    "Not in the lexicon",
                    severity,
                    suggestions,
                )
            )
        return out

    def _candidates_for(
        self, word: str, previous: str | None = None, following: str | None = None
    ) -> list[Suggestion]:
        raw = self.lexicon.candidates(
            word,
            self.model,
            max_results=self.options.max_candidates * 2,
            max_cost=self.options.max_cost,
        )
        scored: list[Suggestion] = []
        for candidate in raw:
            score = candidate.score
            bonus = self._context_bonus(candidate.surface, previous, following)
            score -= self.options.context_weight * bonus
            explanation = _explain(candidate.source, word, candidate.surface)
            # Everything the lexicon returns is by definition a known word.
            scored.append(
                Suggestion(candidate.surface, score, candidate.source, explanation, verified=True)
            )
        scored.sort(key=lambda s: s.score)
        return scored[: self.options.max_candidates]

    def _context_bonus(self, candidate: str, previous: str | None, following: str | None) -> float:
        total = 0
        if previous:
            total += self.lexicon.bigram_frequency(previous, candidate)
        if following:
            total += self.lexicon.bigram_frequency(candidate, following)
        return math.log1p(total)

    # -- segmentation --------------------------------------------------------------------------

    def _segmentation_issues(self, tokens: Sequence[fidel.Token]) -> list[Issue]:
        out: list[Issue] = []

        # A space OCR invented: two unknown fragments that form one known word.
        for a, b in itertools.pairwise(tokens):
            if self.lexicon.recognizes(a.text) and self.lexicon.recognizes(b.text):
                continue
            joined = a.text + b.text
            if len(joined) < 3 or not self.lexicon.recognizes(joined):
                continue
            out.append(
                Issue(
                    IssueKind.SPLIT_WORD,
                    a.start,
                    b.end,
                    f"{a.text} {b.text}",
                    "A word appears to be split by a stray space",
                    Severity.MEDIUM,
                    [Suggestion(joined, 0.2, "segmentation", "Join into one word", verified=True)],
                )
            )

        # A space OCR dropped: one unknown word that splits into two known words.
        for token in tokens:
            word = token.text
            if len(word) < 4 or self.lexicon.recognizes(word):
                continue
            for cut in range(2, len(word) - 1):
                left, right = word[:cut], word[cut:]
                if self.lexicon.recognizes(left) and self.lexicon.recognizes(right):
                    out.append(
                        Issue(
                            IssueKind.RUN_TOGETHER,
                            token.start,
                            token.end,
                            word,
                            "Two words may have run together",
                            Severity.MEDIUM,
                            [
                                Suggestion(
                                    f"{left} {right}", 0.25, "segmentation", "Insert a space",
                                    verified=True,
                                )
                            ],
                        )
                    )
                    break
        return out

    # -- recognizer signals ----------------------------------------------------------------------

    def _confidence_issues(
        self, text: str, confidences: Sequence[tuple[int, int, float]]
    ) -> list[Issue]:
        out: list[Issue] = []
        for start, end, conf in confidences:
            if conf >= self.options.low_confidence_threshold:
                continue
            word = text[start:end]
            if not word.strip():
                continue
            known = self.lexicon.recognizes(word)
            out.append(
                Issue(
                    IssueKind.LOW_CONFIDENCE,
                    start,
                    end,
                    word,
                    f"Recognizer confidence {conf:.0%}",
                    Severity.LOW if known else (Severity.MEDIUM if conf < 0.5 else Severity.LOW),
                    self._candidates_for(word),
                    observed_is_known=known,
                )
            )
        return out

    def _disagreement_issues(
        self, text: str, alternates: Sequence[tuple[int, int, list[str]]]
    ) -> list[Issue]:
        """Where engines disagree, surface every reading and let the lexicon arbitrate.

        Disagreement is a better attention signal than raw confidence: an engine can be
        confidently wrong, but two engines rarely fail the same way.
        """
        out: list[Issue] = []
        for start, end, readings in alternates:
            observed = text[start:end]
            distinct = {r for r in readings if r} | {observed}
            if len(distinct) < 2:
                continue

            observed_known = self.lexicon.recognizes(observed)
            ranked: list[Suggestion] = []
            for reading in distinct:
                if reading == observed:
                    continue
                known = self.lexicon.recognizes(reading)
                # A rival reading that is not a word cannot improve on one that is.
                if observed_known and not known:
                    continue
                cost = confusion.distance(observed, reading, self.model)
                ranked.append(
                    Suggestion(
                        reading,
                        cost - (0.5 if known else 0.0),
                        "engine",
                        "Another engine read this" + (" (in lexicon)" if known else ""),
                        verified=known,
                    )
                )

            if observed_known and not ranked:
                continue  # engines differed, but the current reading is the only real word

            ranked.sort(key=lambda s: s.score)
            out.append(
                Issue(
                    IssueKind.ENGINE_DISAGREEMENT,
                    start,
                    end,
                    observed,
                    f"{len(distinct)} engines disagree here",
                    Severity.MEDIUM if observed_known else Severity.HIGH,
                    ranked,
                    observed_is_known=observed_known,
                )
            )
        return out

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def _keep_rank(issue: Issue) -> tuple[int, int, int, float]:
        """How much a reader gains from seeing this issue rather than a rival for the same span.

        Severity leads, but severity alone is not enough. Several detectors can reach the
        same word, and the one that arrives first is often the least informative: a run
        of two words with no space is *also* an unknown word, and 'not in the lexicon'
        with no candidates loses to 'insert a space here' every time.
        """
        best = issue.best
        return (
            _SEVERITY_RANK[issue.severity],
            0 if (best is not None and best.verified) else 1,
            0 if issue.suggestions else 1,
            best.score if best is not None else 999.0,
        )

    @classmethod
    def _dedupe(cls, issues: Iterable[Issue]) -> list[Issue]:
        """Keep the single most useful issue per span, then order for reading."""
        best: dict[tuple[int, int], Issue] = {}
        for issue in issues:
            key = (issue.start, issue.end)
            existing = best.get(key)
            if existing is None or cls._keep_rank(issue) < cls._keep_rank(existing):
                best[key] = issue
        return sorted(best.values(), key=lambda i: i.sort_key())


def _explain(source: str, observed: str, candidate: str) -> str:
    if source == "order":
        # Not strict: an "order" candidate can be a character longer or shorter, and the
        # shared prefix is still what the explanation is about.
        changed = [
            f"{a}→{b}"
            for a, b in zip(observed, candidate, strict=False)
            if a != b and fidel.same_family(a, b)
        ]
        if changed:
            return "Vowel order: " + ", ".join(changed)
        return "Same consonant skeleton"
    if source == "family":
        return "Visually similar consonant"
    if source == "indel":
        return "A character was added or lost"
    return source


# --------------------------------------------------------------------------------------
# Applying suggestions
# --------------------------------------------------------------------------------------


def apply_issue(text: str, issue: Issue, replacement: str | None = None) -> str:
    """Return ``text`` with one issue's span replaced."""
    if replacement is None:
        if issue.best is None:
            return text
        replacement = issue.best.text
    return text[: issue.start] + replacement + text[issue.end :]


def apply_all(text: str, issues: Sequence[Issue], choices: dict[int, str] | None = None) -> str:
    """Apply several issues at once, working backwards so offsets stay valid.

    ``choices`` maps the index of an issue to the replacement chosen for it; issues
    absent from the mapping use their top suggestion.
    """
    choices = choices or {}
    ordered = sorted(enumerate(issues), key=lambda pair: pair[1].start, reverse=True)
    for index, issue in ordered:
        replacement = choices.get(index)
        if replacement is None and issue.best is None:
            continue
        text = apply_issue(text, issue, replacement)
    return text


def auto_apply(text: str, issues: Sequence[Issue]) -> tuple[str, list[Issue]]:
    """Apply only the unambiguous issues. Returns the new text and what was applied."""
    applied = [i for i in issues if i.auto_applicable]
    return apply_all(text, applied), applied
