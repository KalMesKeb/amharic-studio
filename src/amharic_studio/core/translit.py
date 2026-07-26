"""Phonetic Latin-to-fidel transliteration, for typing corrections.

Correcting a book means typing thousands of fidel, and the Windows Amharic keyboard layout
is slow for anyone who has not committed it to muscle memory. A phonetic input method
removes that friction entirely: ``selam`` becomes ሰላም as you type.

The scheme is SERA (System for Ethiopic Representation in ASCII), the established
convention. Its virtue is that it is unambiguous and complete — every syllable in the
script is reachable, which matters because a corrector needs the rare ones far more often
than an ordinary writer does.

Typing model: a bare consonant immediately produces its ሳድስ (sixth-order) form, and a
following vowel rewrites that one character in place.

    s        →  ስ
    se       →  ሰ
    sel      →  ሰል
    sela     →  ሰላ
    selam    →  ሰላም
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fidel import compose

#: Vowel key to vowel order, when a consonant is already open. ``he`` is ሀ, ``ha`` is ሃ.
VOWEL_ORDER: dict[str, int] = {
    "e": 0,   # ግዕዝ    ä
    "u": 1,   # ካዕብ    u
    "i": 2,   # ሣልስ    i
    "a": 3,   # ራብዕ    a
    "E": 4,   # ኃምስ    e
    "I": 5,   # ሳድስ    ə  (explicit)
    "o": 6,   # ሳብዕ    o
}

#: A vowel typed on its own is a syllable of the አ (alf) family, which has no consonant.
#: SERA spells these with the vowel alone, so ``a`` is አ and ``A`` is ኣ — a different
#: mapping from the post-consonant one above.
STANDALONE_VOWEL_ORDER: dict[str, int] = {
    "a": 0,   # አ
    "e": 0,   # አ — the ግዕዝ vowel key, so a stray 'e' never leaks Latin into the text
    "u": 1,   # ኡ
    "i": 2,   # ኢ
    "A": 3,   # ኣ
    "E": 4,   # ኤ
    "I": 5,   # እ
    "o": 6,   # ኦ
}

ALF_BASE = 0x12A0

#: The eighth, labialised slot (ሏ, ሟ, ቧ …).
LABIALIZED_KEY = "WA"

#: Consonant keys to family base codepoints. Longer keys are matched first, so ``sh``
#: wins over ``s`` and ``Ch`` over ``C``.
CONSONANT_BASE: dict[str, int] = {
    "h": 0x1200,
    "l": 0x1208,
    "H": 0x1210,   # ሐ  hawt
    "m": 0x1218,
    "S": 0x1220,   # ሠ  sawt
    "r": 0x1228,
    "s": 0x1230,
    "sh": 0x1238,
    "x": 0x1238,   # SERA's alternative for ሸ
    "q": 0x1240,
    "Q": 0x1250,   # ቐ
    "b": 0x1260,
    "v": 0x1268,
    "t": 0x1270,
    "ch": 0x1278,
    "c": 0x1278,
    "X": 0x1280,   # ኀ  harm
    "n": 0x1290,
    "ny": 0x1298,
    "N": 0x1298,
    "k": 0x12A8,
    "kh": 0x12B8,  # ኸ
    "K": 0x12B8,
    "w": 0x12C8,
    "`": 0x12D0,   # ዐ  ayn — SERA marks it with a backtick to keep it clear of አ
    "z": 0x12D8,
    "zh": 0x12E0,
    "Z": 0x12E0,
    "y": 0x12E8,
    "d": 0x12F0,
    "D": 0x12F8,   # ዸ
    "j": 0x1300,
    "g": 0x1308,
    "G": 0x1318,   # ጘ
    "T": 0x1320,
    "C": 0x1328,   # ጨ
    "P": 0x1330,   # ጰ
    "ts": 0x1338,  # ጸ  tsaday
    "TS": 0x1338,
    "tz": 0x1340,  # ፀ  tzappa
    "f": 0x1348,
    "p": 0x1350,
}

#: Punctuation and separators, typed with a leading colon or as doubled ASCII.
PUNCTUATION_KEYS: dict[str, str] = {
    "::": "\u1362",  # ።
    ":": "\u1361",   # ፡
    ",": "\u1363",   # ፣
    ";": "\u1364",   # ፤
    "-:": "\u1365",  # ፥
    ":-": "\u1366",  # ፦
    "?": "\u1367",   # ፧
    "::-": "\u1368", # ፨
}

_MAX_CONSONANT_LEN = max(len(k) for k in CONSONANT_BASE)


@dataclass
class Emission:
    """What the editor should do with one keystroke."""

    #: Characters to delete backwards before inserting.
    delete: int = 0
    #: Text to insert.
    insert: str = ""
    #: True when the key was consumed by the transliterator.
    handled: bool = False


@dataclass
class Transliterator:
    """Incremental Latin-to-fidel converter.

    Holds only the state needed to rewrite the single character just emitted, so it stays
    correct when the user clicks elsewhere — call :meth:`reset` on any cursor move.
    """

    #: The consonant key currently open for a vowel, e.g. ``"s"`` after typing ``s``.
    pending: str = ""
    #: Whether a vowel has already been applied to ``pending``.
    vowel_applied: bool = False
    enabled: bool = True
    #: The punctuation key emitted by the previous keystroke, so that a multi-character
    #: mark can still be typed one key at a time.
    _punctuation: str = field(default="", repr=False)

    def reset(self) -> None:
        self.pending = ""
        self.vowel_applied = False
        self._punctuation = ""

    # -- main entry point ----------------------------------------------------------------

    def feed(self, key: str) -> Emission:
        """Process one typed character."""
        if not self.enabled or len(key) != 1:
            self.reset()
            return Emission(handled=False)

        # Anything that is not itself punctuation ends a punctuation run.
        punctuation, self._punctuation = self._punctuation, ""

        # Extend the pending consonant into a digraph: s -> sh, t -> ts, c -> ch.
        if self.pending and not self.vowel_applied:
            digraph = self.pending + key
            if digraph in CONSONANT_BASE:
                syllable = compose(CONSONANT_BASE[digraph], 5)
                if syllable:
                    self.pending = digraph
                    return Emission(delete=1, insert=syllable, handled=True)

        # A vowel rewrites the syllable just emitted.
        if key in VOWEL_ORDER and self.pending and not self.vowel_applied:
            syllable = compose(CONSONANT_BASE[self.pending], VOWEL_ORDER[key])
            if syllable:
                self.vowel_applied = True
                return Emission(delete=1, insert=syllable, handled=True)

        # 'wa' after a vowel-completed syllable reaches the labialised eighth slot.
        if key in ("W", "w") and self.pending and self.vowel_applied:
            labialized = compose(CONSONANT_BASE[self.pending], 7)
            if labialized:
                self.vowel_applied = False
                self.pending = ""
                return Emission(delete=1, insert=labialized, handled=True)

        # A new consonant.
        if key in CONSONANT_BASE:
            syllable = compose(CONSONANT_BASE[key], 5)
            if syllable:
                self.pending = key
                self.vowel_applied = False
                return Emission(insert=syllable, handled=True)

        # A bare vowel with nothing pending is an አ-family syllable.
        if key in STANDALONE_VOWEL_ORDER:
            syllable = compose(ALF_BASE, STANDALONE_VOWEL_ORDER[key])
            if syllable:
                self.pending = ""
                self.vowel_applied = False
                return Emission(insert=syllable, handled=True)

        # A second colon turns the word separator ፡ just emitted into the full stop ።.
        # Every mark is one character wide, so extending one always deletes exactly one.
        extended = punctuation + key
        if punctuation and extended in PUNCTUATION_KEYS:
            self._punctuation = extended
            return Emission(delete=1, insert=PUNCTUATION_KEYS[extended], handled=True)

        if key in PUNCTUATION_KEYS:
            self.reset()
            self._punctuation = key
            return Emission(insert=PUNCTUATION_KEYS[key], handled=True)

        self.reset()
        return Emission(handled=False)


# --------------------------------------------------------------------------------------
# Batch conversion
# --------------------------------------------------------------------------------------


def transliterate(text: str) -> str:
    """Convert a whole SERA string at once — used by the search box and tests."""
    out: list[str] = []
    i, n = 0, len(text)

    while i < n:
        consonant = ""
        for length in range(min(_MAX_CONSONANT_LEN, n - i), 0, -1):
            candidate = text[i : i + length]
            if candidate in CONSONANT_BASE:
                consonant = candidate
                break

        if consonant:
            i += len(consonant)
            order = 5
            if i < n and text[i] in VOWEL_ORDER:
                order = VOWEL_ORDER[text[i]]
                i += 1
                if text[i : i + 2] in ("wa", "WA"):
                    order = 7
                    i += 2
            out.append(compose(CONSONANT_BASE[consonant], order) or consonant)
            continue

        if text[i] in STANDALONE_VOWEL_ORDER:
            out.append(compose(ALF_BASE, STANDALONE_VOWEL_ORDER[text[i]]) or text[i])
            i += 1
            continue

        for key in sorted(PUNCTUATION_KEYS, key=len, reverse=True):
            if text.startswith(key, i):
                out.append(PUNCTUATION_KEYS[key])
                i += len(key)
                break
        else:
            out.append(text[i])
            i += 1

    return "".join(out)


def keyboard_reference() -> list[tuple[str, str, str]]:
    """``(key, fidel, description)`` rows for the in-app help panel."""
    rows: list[tuple[str, str, str]] = []
    seen: set[int] = set()
    for key, base in CONSONANT_BASE.items():
        if base in seen:
            continue
        seen.add(base)
        syllable = compose(base, 0) or ""
        orders = " ".join(compose(base, o) or "·" for o in range(7))
        rows.append((key, syllable, orders))
    return rows
