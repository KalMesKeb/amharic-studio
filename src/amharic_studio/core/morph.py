"""Lightweight Amharic affix stripping, so the lexicon is not fooled by agglutination.

Amharic glues a great many grammatical markers onto a stem. ``ኢትዮጵያ`` is a word; so are
``የኢትዮጵያ`` (of Ethiopia), ``በኢትዮጵያ`` (in Ethiopia), ``ኢትዮጵያን`` (Ethiopia, accusative) and
``ኢትዮጵያውያን`` (Ethiopians). A wordlist can never enumerate those forms, so a plain
membership test reports most perfectly good text as unknown — and an unknown word is what
drives this program to propose a correction. Without affix awareness the suggester spends
its time recommending that ``የኢትዮጵያ`` be "fixed" into ``ኢትዮጵያ``, which is nonsense and
exactly the kind of confident wrong answer that makes a correction tool untrustworthy.

Two things make the stripping less trivial than slicing characters off the ends.

**Suffixes fuse into the final syllable.** The script has no bare consonants, so a
consonant-final stem carries a sixth-order (ሳድስ, ``ə``) syllable that absorbs the suffix
vowel: ``ቤት`` + definite ``-u`` is written ``ቤቱ``, not ``ቤት`` + ``ኡ``. Recovering the stem
means putting the final syllable back into sixth order rather than deleting anything.

**Stripping must be conservative.** Every accepted analysis is an issue the user is never
shown, so an over-eager affix list hides real OCR errors. Each candidate stem is therefore
only accepted if it is *itself in the lexicon*: this module proposes, the lexicon disposes.

This is deliberately not a morphological analyser. It has no verb paradigms, no root
extraction, and it will happily mis-analyse a word whose surface happens to look affixed.
It only needs to answer one question well: is this string a plausible inflection of
something I already know?
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from . import fidel

SADIS = 5  # sixth order, the vowel-less-sounding ə form that consonant-final stems end in
GEEZ = 0


# --------------------------------------------------------------------------------------
# Affix inventory
# --------------------------------------------------------------------------------------

#: Proclitics, longest first so that ``እንደ`` wins over ``እ``. Each is (form, gloss).
#: Restricted to the prepositional and relative markers that attach to ordinary nouns;
#: verb agreement prefixes are left out because single characters over-strip wildly.
PREFIXES: tuple[tuple[str, str], ...] = (
    ("እንደማይ", "as-not-"),
    ("እንደሚ", "as-which-"),
    ("ስለማይ", "because-not-"),
    ("ከየ", "from-each-"),
    ("እንደተ", "as-passive-"),
    ("ስለሚ", "because-which-"),
    ("የማይ", "which-not-"),
    ("ባል", "in-not-"),
    ("እስከ", "until-"),
    ("እንደ", "like-"),
    ("ስለ", "about-"),
    ("ወደ", "toward-"),
    ("ያለ", "without-"),
    ("እየ", "while-"),
    ("የተ", "which-was-"),
    ("የሚ", "which-"),
    ("በየ", "each-"),
    ("ከሚ", "from-which-"),
    ("ሳይ", "before-"),
    ("የ", "of-"),
    ("በ", "in-"),
    ("ለ", "for-"),
    ("ከ", "from-"),
    ("እ", "at-"),
    ("ስ", "when-"),
    ("ም", "also-"),
)

#: Enclitics that are written as their own syllables, longest first.
SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ዎቻችን", "-our-plural"),
    ("ዎቻችሁ", "-your-plural"),
    ("ዎቻቸው", "-their-plural"),
    ("ዎችን", "-plural-object"),
    ("ኦችን", "-plural-object"),
    ("ዊያን", "-ians"),
    ("ውያን", "-ians"),
    ("ዎችም", "-plural-also"),
    ("ያንም", "-plural-also"),
    ("ችንም", "-our-also"),
    ("ነታችን", "-our-ness"),
    ("ዎችና", "-plural-and"),
    ("ዎች", "-plural"),
    ("ኦች", "-plural"),
    ("የው", "-the"),
    ("ዋት", "-plural"),
    ("ያት", "-plural"),
    ("ታት", "-plural"),
    ("ኣት", "-plural"),
    ("ያን", "-plural-object"),
    ("ዎቹ", "-the-plural"),
    ("ችን", "-our"),
    ("ችሁ", "-your-plural"),
    ("ቸው", "-their"),
    ("ነት", "-ness"),
    ("ዊት", "-ian-feminine"),
    ("አዊ", "-ian"),
    ("ዋን", "-her-object"),
    ("ውን", "-the-object"),
    ("ኛም", "-th-also"),
    ("ኛ", "-th"),
    ("ዊ", "-ian"),
    ("ም", "-also"),
    ("ን", "-object"),
    ("ና", "-and"),
    ("ው", "-the"),
    ("ዉ", "-the"),
    ("ዋ", "-her"),
    ("ሽ", "-your-feminine"),
    ("ህ", "-your-masculine"),
    ("ዬ", "-my"),
    ("ቹ", "-the-plural"),
    ("ች", "-plural"),
    # Last, and deliberately so. A bare ት is the tail of the broken plural (መንግሥት →
    # መንግሥታት) but it is also an extremely common ordinary final syllable, so it is only
    # reached once every more specific reading has failed.
    ("ት", "-plural"),
)

#: Final-syllable vowel changes that encode a suffix rather than adding a character.
#: ``ቤት`` → ``ቤቱ`` (the house), ``ቤቴ`` (my house), ``ቤታ`` (her house). Mapping each back to
#: sixth order recovers the stem.
FUSED_ORDERS: dict[int, str] = {
    1: "-the",             # ቤቱ
    2: "-construct",       # ቤቲ
    3: "-her",             # ቤታ
    4: "-my",              # ቤቴ
    6: "-vocative",        # ቤቶ
    7: "-her",             # ቤቷ, labialised
}

MIN_STEM = 2
MAX_DEPTH = 3


# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """One way of reading a surface form as stem plus affixes."""

    stem: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    glosses: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.prefixes) + len(self.suffixes)

    @property
    def is_bare(self) -> bool:
        return self.depth == 0

    def describe(self) -> str:
        """A short gloss for the Why column, e.g. ``የ- + ኢትዮጵያ (of-)``."""
        if self.is_bare:
            return self.stem
        pre = "".join(f"{p}-" for p in self.prefixes)
        suf = "".join(f"-{s}" for s in self.suffixes)
        return f"{pre}{self.stem}{suf}"


def _to_sadis(word: str) -> tuple[str, str] | None:
    """Rewrite a fused final syllable back to sixth order, returning (stem, gloss)."""
    if not word:
        return None
    entry = fidel.decompose(word[-1])
    if entry is None:
        return None
    family, order = entry
    gloss = FUSED_ORDERS.get(order)
    if gloss is None:
        return None
    if family.labiovelar and order == GEEZ:
        return None
    restored = fidel.compose(family.base, SADIS)
    if restored is None:
        return None
    return word[:-1] + restored, gloss


def segmentations(word: str, max_depth: int = MAX_DEPTH) -> Iterator[Analysis]:
    """Every plausible stem for ``word``, shallowest analysis first.

    Yields the bare word before anything is stripped, so a caller that stops at the first
    lexicon hit naturally prefers "already a word" over "a word once I remove three
    affixes". Ordering matters more than completeness here.
    """
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    frontier: list[Analysis] = [Analysis(word)]

    for _ in range(max_depth + 1):
        if not frontier:
            return
        nxt: list[Analysis] = []
        for analysis in frontier:
            key = (analysis.stem, analysis.prefixes, analysis.suffixes)
            if key in seen:
                continue
            seen.add(key)
            yield analysis

            stem = analysis.stem
            for form, gloss in PREFIXES:
                if stem.startswith(form) and len(stem) - len(form) >= MIN_STEM:
                    nxt.append(
                        Analysis(
                            stem[len(form) :],
                            (*analysis.prefixes, form),
                            analysis.suffixes,
                            (*analysis.glosses, gloss),
                        )
                    )
            for form, gloss in SUFFIXES:
                if stem.endswith(form) and len(stem) - len(form) >= MIN_STEM:
                    shorter = stem[: -len(form)]
                    nxt.append(
                        Analysis(
                            shorter,
                            analysis.prefixes,
                            (*analysis.suffixes, form),
                            (*analysis.glosses, gloss),
                        )
                    )
                    # A suffix often leaves the stem's own final consonant in a fused
                    # order too: ቤቶችን → ቤቶች → ቤት.
                    fused = _to_sadis(shorter)
                    if fused and len(fused[0]) >= MIN_STEM:
                        nxt.append(
                            Analysis(
                                fused[0],
                                analysis.prefixes,
                                (*analysis.suffixes, form),
                                (*analysis.glosses, gloss),
                            )
                        )
            fused = _to_sadis(stem)
            if fused and len(fused[0]) >= MIN_STEM:
                nxt.append(
                    Analysis(
                        fused[0],
                        analysis.prefixes,
                        (*analysis.suffixes, "◌"),
                        (*analysis.glosses, fused[1]),
                    )
                )
        frontier = nxt


def analyze(word: str, known: Callable[[str], bool], max_depth: int = MAX_DEPTH) -> Analysis | None:
    """The shallowest analysis of ``word`` whose stem satisfies ``known``, or None.

    ``known`` is normally ``Lexicon.contains``. Keeping it a callable lets the caller
    decide what counts as a stem — a plain wordlist, a stem-only list, or a user glossary.
    """
    if not word:
        return None
    for analysis in segmentations(word, max_depth):
        if known(analysis.stem):
            return analysis
    return None


def is_inflection(word: str, known: Callable[[str], bool], max_depth: int = MAX_DEPTH) -> bool:
    """True if ``word`` is a known word or a plausible inflection of one."""
    return analyze(word, known, max_depth) is not None


