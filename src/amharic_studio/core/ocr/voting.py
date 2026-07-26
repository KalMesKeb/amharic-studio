"""Reconcile the output of several recognizers.

Running two or three engines and comparing them buys two things. The smaller one is a
modest accuracy gain from picking the better reading. The larger one is an *attention
signal*: an engine can be confidently wrong, but two engines rarely fail identically, so
the places where they disagree are the places a human should look first. That signal
routes review far better than confidence scores do.

Where engines agree, the reading is accepted. Where they differ, the lexicon arbitrates,
and any remaining disagreement is passed to the editor as alternates rather than being
silently resolved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .. import confusion
from ..confusion import ConfusionModel
from ..lexicon import Lexicon
from .base import OcrResult, OcrWord, assemble_text, lines_from_words


@dataclass
class VoteOutcome:
    result: OcrResult
    #: ``(start, end, [competing readings])`` for every span where engines differed.
    disagreements: list[tuple[int, int, list[str]]] = field(default_factory=list)
    agreement_rate: float = 1.0
    engines: list[str] = field(default_factory=list)

    @property
    def disagreement_count(self) -> int:
        return len(self.disagreements)


def vote(
    results: list[OcrResult],
    lexicon: Lexicon | None = None,
    model: ConfusionModel | None = None,
) -> VoteOutcome:
    """Combine results, keeping the geometry of the first (primary) engine.

    The primary engine owns the layout because word boxes are not comparable across
    engines; only the *readings* are voted on.
    """
    usable = [r for r in results if r.ok and r.words]
    if not usable:
        failed = results[0] if results else OcrResult(error="no OCR results")
        return VoteOutcome(failed, [], 1.0, [r.label for r in results])
    if len(usable) == 1:
        return VoteOutcome(usable[0], [], 1.0, [usable[0].label])

    model = model or confusion.DEFAULT_MODEL
    primary, *others = usable
    primary_tokens = [w.text for w in primary.words]

    # For each primary word, gather what the other engines read at the same place.
    alternates: list[list[str]] = [[] for _ in primary_tokens]
    for other in others:
        other_tokens = [w.text for w in other.words]
        matcher = SequenceMatcher(a=primary_tokens, b=other_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    alternates[i1 + offset].append(other_tokens[j1 + offset])
            elif tag == "replace":
                # Unequal spans: attach the whole competing run to each primary word so
                # nothing the other engine saw is thrown away.
                replacement = " ".join(other_tokens[j1:j2])
                for index in range(i1, i2):
                    alternates[index].append(replacement)
            elif tag == "delete":
                for index in range(i1, i2):
                    alternates[index].append("")

    chosen: list[OcrWord] = []
    disagreements: list[tuple[int, int, list[str]]] = []
    agreements = 0

    for index, word in enumerate(primary.words):
        readings = [word.text] + [a for a in alternates[index] if a]
        counts = Counter(readings)
        distinct = set(counts)

        if len(distinct) == 1:
            agreements += 1
            chosen.append(_clone(word, word.text, word.conf))
            continue

        winner, confidence = _arbitrate(word.text, counts, lexicon, model)
        chosen.append(_clone(word, winner, confidence))
        disagreements.append((index, index, sorted(distinct)))

    text = assemble_text(chosen)
    # Disagreement indices were word positions; convert them to character offsets now
    # that offsets exist.
    spans = [
        (chosen[i].start, chosen[i].end, readings)
        for i, _j, readings in disagreements
        if chosen[i].start >= 0
    ]

    combined = OcrResult(
        text=text,
        words=chosen,
        lines=lines_from_words(chosen),
        engine="+".join(r.label for r in usable),
        language=primary.language,
        duration_s=sum(r.duration_s for r in usable),
    )
    rate = agreements / len(primary.words) if primary.words else 1.0
    return VoteOutcome(combined, spans, rate, [r.label for r in usable])


def _arbitrate(
    primary_reading: str,
    counts: Counter[str],
    lexicon: Lexicon | None,
    model: ConfusionModel,
) -> tuple[str, float]:
    """Choose between competing readings, returning the winner and a confidence."""
    total = sum(counts.values())
    scored: list[tuple[float, str]] = []

    for reading, votes in counts.items():
        score = votes / total
        if lexicon is not None and lexicon.contains(reading):
            # A dictionary hit outweighs a bare majority: two engines making the same
            # mistake is common, and both producing the same *real word* is not.
            score += 0.75
        if reading == primary_reading:
            score += 0.10  # tie-break toward the engine that owns the geometry
        # Readings wildly unlike the primary are likely alignment slippage, not a rival
        # reading of the same ink.
        if reading and primary_reading:
            similarity = confusion.similarity(primary_reading, reading, model)
            score += 0.2 * similarity
        scored.append((score, reading))

    scored.sort(reverse=True)
    best_score, best_reading = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    confidence = min(1.0, max(0.3, (best_score - runner_up) + 0.5))
    return best_reading, confidence


def _clone(word: OcrWord, text: str, conf: float) -> OcrWord:
    return OcrWord(
        text=text,
        x=word.x,
        y=word.y,
        w=word.w,
        h=word.h,
        conf=conf,
        line_idx=word.line_idx,
        word_idx=word.word_idx,
    )
