"""Ge'ez script (fidel) model: syllabary structure, orthographic variants, punctuation.

The Ethiopic block is laid out as families of eight consecutive codepoints, one per
vowel order. That regularity is what makes principled OCR correction possible: most
recognition errors are a *wrong order within the correct family* (ቀ read as ቁ), because
the orders differ only by a small appendage on an otherwise identical glyph.

Nothing here mutates text. Folding produces a shadow form used for dictionary lookup,
search and scoring only; the stored reading always stays exactly as printed.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

# --------------------------------------------------------------------------------------
# Unicode ranges
# --------------------------------------------------------------------------------------

ETHIOPIC_MAIN = (0x1200, 0x137F)
ETHIOPIC_SUPPLEMENT = (0x1380, 0x139F)
ETHIOPIC_EXTENDED = (0x2D80, 0x2DDF)
ETHIOPIC_EXTENDED_A = (0xAB00, 0xAB2F)
ETHIOPIC_EXTENDED_B = (0x1E7E0, 0x1E7FF)

ALL_ETHIOPIC_RANGES = (
    ETHIOPIC_MAIN,
    ETHIOPIC_SUPPLEMENT,
    ETHIOPIC_EXTENDED,
    ETHIOPIC_EXTENDED_A,
    ETHIOPIC_EXTENDED_B,
)

# Amharic uses only the main block. Anything from the supplement or extended blocks in an
# Amharic book is a near-certain OCR error (those blocks encode Sebat Bet Gurage, Me'en,
# Basketo and similar), so QA flags them rather than silently accepting them.
NON_AMHARIC_ETHIOPIC_RANGES = (
    ETHIOPIC_SUPPLEMENT,
    ETHIOPIC_EXTENDED,
    ETHIOPIC_EXTENDED_A,
    ETHIOPIC_EXTENDED_B,
)

# --------------------------------------------------------------------------------------
# Vowel orders
# --------------------------------------------------------------------------------------

ORDER_COUNT = 8


@dataclass(frozen=True)
class Order:
    index: int
    amharic: str
    latin: str
    vowel: str


ORDERS: tuple[Order, ...] = (
    Order(0, "ግዕዝ", "geez", "ä"),
    Order(1, "ካዕብ", "kaib", "u"),
    Order(2, "ሣልስ", "salis", "i"),
    Order(3, "ራብዕ", "rabi", "a"),
    Order(4, "ኃምስ", "hamis", "e"),
    Order(5, "ሳድስ", "sadis", "ə"),
    Order(6, "ሳብዕ", "sabi", "o"),
    # The eighth slot holds the labialised -wa form (ሏ, ሟ, ቧ …) in most families.
    Order(7, "ዘመደ-ግዕዝ", "labialized", "wa"),
)


@dataclass(frozen=True)
class Family:
    """One consonant family: a base codepoint plus its vowel orders."""

    base: int
    latin: str
    label: str
    labiovelar: bool = False
    #: Offsets from ``base`` that are actually assigned in Unicode.
    offsets: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    @property
    def geez(self) -> str:
        return chr(self.base)


# Sparse labiovelar rows carry only five members: ä, i, a, e, ə.
_LABIOVELAR_OFFSETS = (0, 2, 3, 4, 5)

# ቐ, ኸ and ዐ stop at the seventh order: Unicode leaves U+1257, U+12BF and U+12D7
# unassigned, so these families have no labialised eighth form.
_NO_LABIALIZED = (0, 1, 2, 3, 4, 5, 6)

FAMILIES: tuple[Family, ...] = (
    Family(0x1200, "h", "ሀ (hoy)"),
    Family(0x1208, "l", "ለ (lawe)"),
    Family(0x1210, "ḥ", "ሐ (hawt)"),
    Family(0x1218, "m", "መ (may)"),
    Family(0x1220, "ś", "ሠ (sawt)"),
    Family(0x1228, "r", "ረ (res)"),
    Family(0x1230, "s", "ሰ (sat)"),
    Family(0x1238, "š", "ሸ (sha)"),
    Family(0x1240, "q", "ቀ (qaf)"),
    Family(0x1248, "qʷ", "ቈ (qaf labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x1250, "qʼ", "ቐ (qha)", offsets=_NO_LABIALIZED),
    Family(0x1258, "qʼʷ", "ቘ (qha labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x1260, "b", "በ (bet)"),
    Family(0x1268, "v", "ቨ (ve)"),
    Family(0x1270, "t", "ተ (tawe)"),
    Family(0x1278, "č", "ቸ (cha)"),
    Family(0x1280, "ḫ", "ኀ (harm)"),
    Family(0x1288, "ḫʷ", "ኈ (harm labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x1290, "n", "ነ (nahas)"),
    Family(0x1298, "ñ", "ኘ (nya)"),
    Family(0x12A0, "ʾ", "አ (alf)"),
    Family(0x12A8, "k", "ከ (kaf)"),
    Family(0x12B0, "kʷ", "ኰ (kaf labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x12B8, "ḵ", "ኸ (kha)", offsets=_NO_LABIALIZED),
    Family(0x12C0, "ḵʷ", "ዀ (kha labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x12C8, "w", "ወ (wawe)"),
    Family(0x12D0, "ʿ", "ዐ (ayn)", offsets=_NO_LABIALIZED),
    Family(0x12D8, "z", "ዘ (zay)"),
    Family(0x12E0, "ž", "ዠ (zha)"),
    Family(0x12E8, "y", "የ (yaman)"),
    Family(0x12F0, "d", "ደ (dent)"),
    Family(0x12F8, "ḍ", "ዸ (dda)"),
    Family(0x1300, "ǧ", "ጀ (ja)"),
    Family(0x1308, "g", "ገ (geml)"),
    Family(0x1310, "gʷ", "ጐ (geml labiovelar)", True, _LABIOVELAR_OFFSETS),
    Family(0x1318, "gʼ", "ጘ (gga)"),
    Family(0x1320, "ṭ", "ጠ (tait)"),
    Family(0x1328, "č̣", "ጨ (cha)"),
    Family(0x1330, "ṗ", "ጰ (psa)"),
    Family(0x1338, "ṣ", "ጸ (tsaday)"),
    Family(0x1340, "ṣ́", "ፀ (tzappa)"),
    Family(0x1348, "f", "ፈ (af)"),
    Family(0x1350, "p", "ፐ (psa)"),
)

#: Isolated syllables that do not belong to an eight-slot family.
STANDALONE_SYLLABLES = {0x1358: "rya", 0x1359: "mya", 0x135A: "fya"}

# --------------------------------------------------------------------------------------
# Punctuation, numerals, marks
# --------------------------------------------------------------------------------------

SECTION_MARK = "\u1360"  # ፠
WORDSPACE = "\u1361"  # ፡  separates words in traditional orthography
FULL_STOP = "\u1362"  # ።
COMMA = "\u1363"  # ፣
SEMICOLON = "\u1364"  # ፤
COLON = "\u1365"  # ፥
PREFACE_COLON = "\u1366"  # ፦
QUESTION_MARK = "\u1367"  # ፧
PARAGRAPH_SEPARATOR = "\u1368"  # ፨

ETHIOPIC_PUNCTUATION = frozenset(
    [
        SECTION_MARK,
        WORDSPACE,
        FULL_STOP,
        COMMA,
        SEMICOLON,
        COLON,
        PREFACE_COLON,
        QUESTION_MARK,
        PARAGRAPH_SEPARATOR,
    ]
)

SENTENCE_ENDERS = frozenset([FULL_STOP, QUESTION_MARK, PARAGRAPH_SEPARATOR, ".", "?", "!"])

#: Combining gemination / vowel-length marks. Extremely rare in printed Amharic.
COMBINING_MARKS = frozenset("\u135d\u135e\u135f")

ETHIOPIC_DIGITS: dict[str, int] = {
    "\u1369": 1, "\u136a": 2, "\u136b": 3, "\u136c": 4, "\u136d": 5,
    "\u136e": 6, "\u136f": 7, "\u1370": 8, "\u1371": 9,
    "\u1372": 10, "\u1373": 20, "\u1374": 30, "\u1375": 40, "\u1376": 50,
    "\u1377": 60, "\u1378": 70, "\u1379": 80, "\u137a": 90,
    "\u137b": 100, "\u137c": 10000,
}  # fmt: skip

HUNDRED = "\u137b"
TEN_THOUSAND = "\u137c"

# --------------------------------------------------------------------------------------
# Derived lookup tables
# --------------------------------------------------------------------------------------


def _build_tables() -> tuple[dict[int, tuple[Family, int]], dict[tuple[int, int], int]]:
    """Map codepoint -> (family, order) and (family base, order) -> codepoint.

    Only codepoints actually assigned in the running interpreter's Unicode data are
    included, so the table can never hand out reserved slots such as U+1249.
    """
    by_char: dict[int, tuple[Family, int]] = {}
    by_slot: dict[tuple[int, int], int] = {}
    for fam in FAMILIES:
        for offset in fam.offsets:
            cp = fam.base + offset
            try:
                unicodedata.name(chr(cp))
            except ValueError:
                continue  # unassigned slot
            by_char[cp] = (fam, offset)
            by_slot[(fam.base, offset)] = cp
    return by_char, by_slot


_BY_CHAR, _BY_SLOT = _build_tables()

FAMILY_BY_BASE: dict[int, Family] = {f.base: f for f in FAMILIES}

#: Plain family <-> labiovelar family. OCR routinely drops or invents the -w mark.
LABIOVELAR_PARTNER: dict[int, int] = {
    0x1240: 0x1248, 0x1250: 0x1258, 0x1280: 0x1288,
    0x12A8: 0x12B0, 0x12B8: 0x12C0, 0x1308: 0x1310,
}  # fmt: skip
LABIOVELAR_PARTNER.update({v: k for k, v in LABIOVELAR_PARTNER.items()})


# --------------------------------------------------------------------------------------
# Orthographic variant groups
# --------------------------------------------------------------------------------------

#: Families that spell the same Amharic sound. Ge'ez distinguished these; Amharic does
#: not, so nineteenth and twentieth century printing uses them somewhat freely and OCR
#: confuses them constantly. The first entry of each group is the modern canonical form.
HOMOPHONE_FAMILY_GROUPS: tuple[tuple[int, ...], ...] = (
    (0x1200, 0x1210, 0x1280),  # ሀ ሐ ኀ  — all /h/
    (0x1230, 0x1220),  # ሰ ሠ        — all /s/
    (0x12A0, 0x12D0),  # አ ዐ        — all glottal /ʔ/
    (0x1338, 0x1340),  # ጸ ፀ        — all /tsʼ/
)

#: Optional extra fold: ኸ /kh/ is an allophone of ከ /k/ for most speakers.
SOFT_HOMOPHONE_FAMILY_GROUPS: tuple[tuple[int, ...], ...] = ((0x12A8, 0x12B8),)

_FAMILY_FOLD: dict[int, int] = {}
for _group in HOMOPHONE_FAMILY_GROUPS:
    for _member in _group:
        _FAMILY_FOLD[_member] = _group[0]

_SOFT_FAMILY_FOLD: dict[int, int] = {}
for _group in SOFT_HOMOPHONE_FAMILY_GROUPS:
    for _member in _group:
        _SOFT_FAMILY_FOLD[_member] = _group[0]

#: Within the /h/ families the ግዕዝ and ራብዕ orders are also homophonous (ሀ and ሃ are both
#: /ha/), which is why ሀ/ሃ/ሐ/ሓ/ኀ/ኃ are the single most confused set in Amharic OCR.
_HA_FAMILIES = frozenset({0x1200, 0x1210, 0x1280})


@dataclass(frozen=True)
class FoldPolicy:
    """Which orthographic distinctions to ignore when *matching* text.

    Storage is never affected. ``faithful`` output keeps ሠ, ኀ, ዐ, ፀ and ፡ exactly as
    printed while these folds let the same text match a modern dictionary.
    """

    fold_homophone_families: bool = True
    fold_ha_orders: bool = True
    fold_kha: bool = False
    fold_labiovelar: bool = False
    fold_gemination_marks: bool = True

    @classmethod
    def strict(cls) -> FoldPolicy:
        """No folding at all — compare glyph for glyph."""
        return cls(False, False, False, False, False)

    @classmethod
    def aggressive(cls) -> FoldPolicy:
        return cls(True, True, True, True, True)


DEFAULT_FOLD = FoldPolicy()


# --------------------------------------------------------------------------------------
# Character-level queries
# --------------------------------------------------------------------------------------


def is_ethiopic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ALL_ETHIOPIC_RANGES)


def is_amharic_syllable(ch: str) -> bool:
    """True for a syllable in the main block that belongs to a known family."""
    return ord(ch) in _BY_CHAR


def is_non_amharic_ethiopic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in NON_AMHARIC_ETHIOPIC_RANGES)


def is_ethiopic_punctuation(ch: str) -> bool:
    return ch in ETHIOPIC_PUNCTUATION


def is_ethiopic_digit(ch: str) -> bool:
    return ch in ETHIOPIC_DIGITS


def decompose(ch: str) -> tuple[Family, int] | None:
    """Split a syllable into its consonant family and vowel order."""
    return _BY_CHAR.get(ord(ch))


def compose(family_base: int, order: int) -> str | None:
    """Rebuild a syllable from a family and an order, or None if that slot is unassigned."""
    cp = _BY_SLOT.get((family_base, order))
    return chr(cp) if cp is not None else None


def family_of(ch: str) -> Family | None:
    entry = _BY_CHAR.get(ord(ch))
    return entry[0] if entry else None


def order_of(ch: str) -> int | None:
    entry = _BY_CHAR.get(ord(ch))
    return entry[1] if entry else None


def same_family(a: str, b: str) -> bool:
    fa, fb = family_of(a), family_of(b)
    return fa is not None and fa is fb


def order_variants(ch: str) -> list[str]:
    """Every other vowel order of the same consonant — the usual OCR candidate set."""
    entry = _BY_CHAR.get(ord(ch))
    if entry is None:
        return []
    fam, order = entry
    out = []
    for offset in fam.offsets:
        if offset == order:
            continue
        alt = compose(fam.base, offset)
        if alt:
            out.append(alt)
    return out


def homophone_variants(ch: str, policy: FoldPolicy = DEFAULT_FOLD) -> list[str]:
    """Alternative spellings of the same sound (ሰላም / ሠላም)."""
    entry = _BY_CHAR.get(ord(ch))
    if entry is None:
        return []
    fam, order = entry
    groups: list[tuple[int, ...]] = list(HOMOPHONE_FAMILY_GROUPS)
    if policy.fold_kha:
        groups += list(SOFT_HOMOPHONE_FAMILY_GROUPS)

    out: list[str] = []
    for group in groups:
        if fam.base not in group:
            continue
        for base in group:
            for alt_order in _ha_order_alternatives(base, order, policy):
                alt = compose(base, alt_order)
                if alt and alt != ch:
                    out.append(alt)
    return out


def _ha_order_alternatives(base: int, order: int, policy: FoldPolicy) -> tuple[int, ...]:
    if policy.fold_ha_orders and base in _HA_FAMILIES and order in (0, 3):
        return (0, 3)
    return (order,)


# --------------------------------------------------------------------------------------
# Folding (matching form)
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _fold_char(ch: str, policy: FoldPolicy) -> str:
    if policy.fold_gemination_marks and ch in COMBINING_MARKS:
        return ""

    entry = _BY_CHAR.get(ord(ch))
    if entry is None:
        return ch
    fam, order = entry
    base = fam.base

    if policy.fold_labiovelar and base in LABIOVELAR_PARTNER and fam.labiovelar:
        partner = LABIOVELAR_PARTNER[base]
        # Labiovelar rows only carry five orders; map onto the plain family's same order.
        if compose(partner, order):
            base = partner

    if policy.fold_homophone_families:
        base = _FAMILY_FOLD.get(base, base)
    if policy.fold_kha:
        base = _SOFT_FAMILY_FOLD.get(base, base)

    if policy.fold_ha_orders and base in _HA_FAMILIES and order == 3:
        order = 0

    return compose(base, order) or ch


def fold(text: str, policy: FoldPolicy = DEFAULT_FOLD) -> str:
    """Return the matching form of ``text``.

    Use this for dictionary lookup, search and scoring. Never write the result back into
    the document: ሠላም folds to ሰላም so it can match the lexicon, but the page must keep
    the ሠ the printer actually set.
    """
    return "".join(_fold_char(ch, policy) for ch in text)


# --------------------------------------------------------------------------------------
# Ethiopic numerals
# --------------------------------------------------------------------------------------


def ethiopic_to_int(text: str) -> int | None:
    """Parse an Ethiopic numeral. Returns None if the string is not a valid numeral.

    Ethiopic numerals are additive within a group and multiplicative across ፻ (100) and
    ፼ (10000): ፳፻፸፭ is 20×100 + 70 + 5 = 2075.
    """
    if not text or any(ch not in ETHIOPIC_DIGITS for ch in text):
        return None

    total = 0
    group = 0
    for ch in text:
        value = ETHIOPIC_DIGITS[ch]
        if ch == TEN_THOUSAND:
            total = (total + (group or 1)) * 10000
            group = 0
        elif ch == HUNDRED:
            group = (group or 1) * 100
        else:
            group += value
    return total + group


def int_to_ethiopic(value: int) -> str:
    """Render a positive integer in Ethiopic numerals."""
    if value <= 0:
        raise ValueError("Ethiopic numerals represent positive integers only")

    units = {v: k for k, v in ETHIOPIC_DIGITS.items() if 1 <= v <= 9}
    tens = {v: k for k, v in ETHIOPIC_DIGITS.items() if 10 <= v <= 90}

    def two_digits(n: int) -> str:
        out = ""
        if n >= 10:
            out += tens[(n // 10) * 10]
        if n % 10:
            out += units[n % 10]
        return out

    if value >= 10000:
        high, low = divmod(value, 10000)
        prefix = int_to_ethiopic(high) if high > 1 else ""
        return prefix + TEN_THOUSAND + (int_to_ethiopic(low) if low else "")

    out = ""
    hundreds, rest = divmod(value, 100)
    if hundreds:
        out += (two_digits(hundreds) if hundreds > 1 else "") + HUNDRED
    if rest:
        out += two_digits(rest)
    return out


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------

#: Characters that end a word. ፡ is a *separator*, not punctuation attached to the word.
_WORD_BREAKS = frozenset({WORDSPACE, " ", "\t", "\n", "\r", "\u00a0", "\u200b"})


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    kind: str  # "word" | "space" | "punct" | "number" | "other"

    @property
    def is_word(self) -> bool:
        return self.kind == "word"


def tokenize(text: str) -> list[Token]:
    """Split into words, separators and punctuation, preserving offsets.

    Handles both orthographies: modern space-separated text and traditional text where
    ፡ (U+1361) stands between every word. Offsets are preserved so the editor can map a
    token back to its exact position on the page.
    """
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        start = i

        if ch in _WORD_BREAKS:
            while i < n and text[i] in _WORD_BREAKS:
                i += 1
            tokens.append(Token(text[start:i], start, i, "space"))
        elif ch in ETHIOPIC_PUNCTUATION or unicodedata.category(ch).startswith("P"):
            while i < n and (
                text[i] in ETHIOPIC_PUNCTUATION - {WORDSPACE}
                or (unicodedata.category(text[i]).startswith("P") and text[i] not in _WORD_BREAKS)
            ):
                i += 1
            tokens.append(Token(text[start:i], start, i, "punct"))
        elif ch in ETHIOPIC_DIGITS or ch.isdigit():
            while i < n and (text[i] in ETHIOPIC_DIGITS or text[i].isdigit()):
                i += 1
            tokens.append(Token(text[start:i], start, i, "number"))
        else:
            while i < n:
                c = text[i]
                if (
                    c in _WORD_BREAKS
                    or c in ETHIOPIC_PUNCTUATION
                    or c in ETHIOPIC_DIGITS
                    or c.isdigit()
                    or (unicodedata.category(c).startswith("P") and c not in COMBINING_MARKS)
                ):
                    break
                i += 1
            kind = "word" if i > start else "other"
            if i == start:  # defensive: always consume at least one character
                i += 1
            tokens.append(Token(text[start:i], start, i, kind))
    return tokens


def words(text: str) -> list[Token]:
    return [t for t in tokenize(text) if t.is_word]


def split_sentences(text: str) -> list[str]:
    """Split on ። ፧ ፨ (and Latin equivalents), keeping the terminator attached."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_ENDERS:
            out.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def ethiopic_ratio(text: str) -> float:
    """Share of letter characters that are Ethiopic — the basic language sanity check."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_ethiopic(c)) / len(letters)
