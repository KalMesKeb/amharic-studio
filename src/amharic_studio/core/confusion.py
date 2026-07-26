"""Confusion-weighted edit distance for Ge'ez text.

A plain Levenshtein distance treats ቀ→ቁ and ቀ→መ as equally likely, which is badly wrong
for this script. ቀ and ቁ are the same glyph differing by one small appendage, and that
substitution accounts for a large share of real OCR errors, while ቀ→መ essentially never
happens. Weighting substitutions by *visual* similarity is what makes candidate ranking
useful instead of noise.

The starting weights below encode the structure of the script. They are only priors: the
model records every correction the editor makes and shifts its own costs toward what this
particular book, typeface and scan quality actually produce.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import fidel
from .fidel import COMMA, FULL_STOP, QUESTION_MARK, SEMICOLON, WORDSPACE

# --------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------

DEFAULT_SUB_COST = 1.0
DEFAULT_INDEL_COST = 0.9

#: Cost of confusing two vowel orders within the same consonant family, keyed by the
#: unordered pair of order indices. Values reflect how much of the glyph actually differs:
#: ä/a is a leg lengthening, u/i are both small right-side strokes, and so on.
ORDER_COST: dict[tuple[int, int], float] = {
    (0, 3): 0.18,
    (0, 5): 0.22,
    (1, 2): 0.20,
    (0, 1): 0.25,
    (0, 2): 0.28,
    (2, 4): 0.28,
    (1, 6): 0.30,
    (3, 4): 0.30,
    (5, 6): 0.30,
    (2, 3): 0.30,
    (4, 5): 0.32,
    (3, 6): 0.35,
    (1, 5): 0.35,
    (2, 5): 0.33,
    (0, 6): 0.36,
    (4, 6): 0.34,
    (1, 3): 0.34,
    (1, 4): 0.36,
    (0, 4): 0.38,
    (2, 6): 0.38,
    (3, 5): 0.33,
}
SAME_FAMILY_DEFAULT = 0.42
LABIALIZED_ORDER_COST = 0.45  # eighth slot (ሏ, ሟ …) against anything else

#: Consonant families whose glyphs are easily confused. Most of these are palatalization
#: pairs where the second letter is the first plus a stroke on top — exactly the kind of
#: fine detail that dies in a bad scan.
VISUAL_FAMILY_PAIRS: tuple[tuple[int, int, float], ...] = (
    (0x1230, 0x1238, 0.30),  # ሰ ሸ
    (0x1240, 0x1250, 0.30),  # ቀ ቐ
    (0x1270, 0x1278, 0.30),  # ተ ቸ
    (0x1290, 0x1298, 0.30),  # ነ ኘ
    (0x12A8, 0x12B8, 0.30),  # ከ ኸ
    (0x12D8, 0x12E0, 0.30),  # ዘ ዠ
    (0x1320, 0x1328, 0.30),  # ጠ ጨ
    (0x1308, 0x1318, 0.32),  # ገ ጘ
    (0x1260, 0x1268, 0.28),  # በ ቨ
    (0x12F0, 0x12F8, 0.32),  # ደ ዸ
    (0x12F0, 0x1300, 0.38),  # ደ ጀ
    (0x1300, 0x12F8, 0.40),  # ጀ ዸ
    (0x1278, 0x1328, 0.42),  # ቸ ጨ
    (0x1298, 0x1318, 0.44),  # ኘ ጘ
    (0x1330, 0x1350, 0.30),  # ጰ ፐ
    (0x1348, 0x1350, 0.38),  # ፈ ፐ
    (0x1210, 0x1220, 0.34),  # ሐ ሠ  — both three-legged
    (0x1210, 0x1218, 0.40),  # ሐ መ
    (0x1220, 0x1218, 0.42),  # ሠ መ
    (0x12C8, 0x12D0, 0.40),  # ወ ዐ
    (0x1200, 0x1210, 0.36),  # ሀ ሐ
    (0x1280, 0x1210, 0.36),  # ኀ ሐ
    (0x1338, 0x1340, 0.30),  # ጸ ፀ
)

HOMOPHONE_FAMILY_COST = 0.12
LABIOVELAR_COST = 0.30

#: Punctuation the OCR mixes up. ፡ against ASCII colon is nearly free — they are the same
#: two dots to a recognizer trained mostly on Latin.
PUNCT_COST: dict[tuple[str, str], float] = {
    (WORDSPACE, ":"): 0.05,
    (WORDSPACE, FULL_STOP): 0.30,
    (WORDSPACE, "."): 0.35,
    (FULL_STOP, ":"): 0.18,
    (FULL_STOP, "."): 0.20,
    (FULL_STOP, "\u1368"): 0.30,
    (COMMA, ","): 0.06,
    (SEMICOLON, ";"): 0.06,
    (QUESTION_MARK, "?"): 0.08,
    (COMMA, SEMICOLON): 0.25,
    (COMMA, WORDSPACE): 0.30,
    ("\u1365", ":"): 0.10,
    ("\u1366", ":"): 0.15,
}

#: Ethiopic numerals against the Latin digits OCR prefers to emit.
DIGIT_COST: dict[tuple[str, str], float] = {
    ("\u1369", "1"): 0.5,
    ("\u136a", "2"): 0.5,
    ("\u136b", "3"): 0.5,
    ("\u1372", "10"): 0.6,
}

#: Deletions and insertions that are cheaper than average, because OCR adds or drops them
#: constantly: separators, spaces, and the rare combining marks.
CHEAP_INDEL: dict[str, float] = {
    WORDSPACE: 0.40,
    " ": 0.45,
    "\u135d": 0.30,
    "\u135e": 0.30,
    "\u135f": 0.30,
    ".": 0.55,
    ",": 0.55,
    ":": 0.50,
    "-": 0.55,
    "'": 0.50,
    "`": 0.45,
}

#: OCR loses faint marks more often than it invents them, so reading a plain ግዕዝ glyph
#: where a marked order was printed is likelier than the reverse.
MARK_LOSS_BIAS = 0.85


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class ConfusionModel:
    """Substitution / insertion / deletion costs between characters.

    Costs are directional: ``sub_cost(observed, intended)`` is the cost of believing that
    the recognizer printed ``observed`` when the page actually read ``intended``.
    """

    #: Corrections observed from the editor, as (observed, intended) -> count.
    observations: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    #: How strongly learned evidence overrides the priors.
    learning_weight: float = 0.6
    _cache: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    # -- prior ---------------------------------------------------------------------------

    def _prior_sub_cost(self, observed: str, intended: str) -> float:
        if observed == intended:
            return 0.0

        pair = (observed, intended)
        reverse = (intended, observed)
        for table in (PUNCT_COST, DIGIT_COST):
            if pair in table:
                return table[pair]
            if reverse in table:
                return table[reverse]

        a = fidel.decompose(observed)
        b = fidel.decompose(intended)
        if a is None or b is None:
            # One side is not a syllable. Case-insensitive Latin near-match stays cheap.
            if observed.lower() == intended.lower():
                return 0.3
            return DEFAULT_SUB_COST

        fam_a, order_a = a
        fam_b, order_b = b

        if fam_a.base == fam_b.base:
            cost = self._order_cost(order_a, order_b)
        else:
            cost = self._family_cost(fam_a.base, fam_b.base)
            # Different family *and* different order compounds the unlikeliness.
            if order_a != order_b:
                cost = min(DEFAULT_SUB_COST, cost + 0.5 * self._order_cost(order_a, order_b))

        if order_a == 0 and order_b != 0:
            cost *= MARK_LOSS_BIAS
        return round(cost, 4)

    @staticmethod
    def _order_cost(order_a: int, order_b: int) -> float:
        if order_a == order_b:
            return 0.0
        if 7 in (order_a, order_b):
            return LABIALIZED_ORDER_COST
        key = (min(order_a, order_b), max(order_a, order_b))
        return ORDER_COST.get(key, SAME_FAMILY_DEFAULT)

    @staticmethod
    def _family_cost(base_a: int, base_b: int) -> float:
        for x, y, cost in VISUAL_FAMILY_PAIRS:
            if {x, y} == {base_a, base_b}:
                return cost

        if fidel.LABIOVELAR_PARTNER.get(base_a) == base_b:
            return LABIOVELAR_COST

        for group in fidel.HOMOPHONE_FAMILY_GROUPS:
            if base_a in group and base_b in group:
                return HOMOPHONE_FAMILY_COST

        return DEFAULT_SUB_COST

    # -- learned -------------------------------------------------------------------------

    def observe(self, observed: str, intended: str, count: int = 1) -> None:
        """Record a correction the editor actually made."""
        if observed == intended:
            return
        self.observations[(observed, intended)] += count
        self._cache.clear()

    def observe_alignment(self, before: str, after: str) -> int:
        """Learn from a corrected string by aligning it against the original."""
        learned = 0
        for op, observed, intended in align(before, after, self):
            if op == "sub":
                self.observe(observed, intended)
                learned += 1
        return learned

    def sub_cost(self, observed: str, intended: str) -> float:
        if observed == intended:
            return 0.0
        key = (observed, intended)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cost = self._prior_sub_cost(observed, intended)
        seen = self.observations.get(key, 0)
        if seen:
            # Each confirmed sighting pulls the cost toward zero with diminishing returns,
            # so one stray correction cannot collapse a cost to nothing.
            pull = self.learning_weight * (1.0 - 1.0 / (1.0 + math.log1p(seen)))
            cost *= 1.0 - pull

        self._cache[key] = cost
        return cost

    def del_cost(self, ch: str) -> float:
        """Cost of the recognizer having invented ``ch``."""
        return CHEAP_INDEL.get(ch, DEFAULT_INDEL_COST)

    def ins_cost(self, ch: str) -> float:
        """Cost of the recognizer having missed ``ch``."""
        return CHEAP_INDEL.get(ch, DEFAULT_INDEL_COST)

    # -- neighbours ----------------------------------------------------------------------

    def neighbors(self, ch: str, max_cost: float = 0.45, limit: int = 16) -> list[tuple[str, float]]:
        """Characters ``ch`` is plausibly a misreading of, cheapest first."""
        out: list[tuple[str, float]] = []

        for alt in fidel.order_variants(ch):
            cost = self.sub_cost(ch, alt)
            if cost <= max_cost:
                out.append((alt, cost))

        entry = fidel.decompose(ch)
        if entry is not None:
            fam, order = entry
            related: set[int] = set()
            for x, y, _ in VISUAL_FAMILY_PAIRS:
                if fam.base == x:
                    related.add(y)
                elif fam.base == y:
                    related.add(x)
            partner = fidel.LABIOVELAR_PARTNER.get(fam.base)
            if partner is not None:
                related.add(partner)
            for group in fidel.HOMOPHONE_FAMILY_GROUPS:
                if fam.base in group:
                    related.update(group)
            related.discard(fam.base)

            for base in related:
                for alt_order in {order, 0, 3, 5}:
                    alt = fidel.compose(base, alt_order)
                    if not alt or alt == ch:
                        continue
                    cost = self.sub_cost(ch, alt)
                    if cost <= max_cost:
                        out.append((alt, cost))

        for (a, b), cost in {**PUNCT_COST, **DIGIT_COST}.items():
            if cost > max_cost:
                continue
            if a == ch:
                out.append((b, cost))
            elif b == ch:
                out.append((a, cost))

        for (observed, intended), _ in self.observations.items():
            if observed == ch:
                cost = self.sub_cost(ch, intended)
                if cost <= max_cost:
                    out.append((intended, cost))

        best: dict[str, float] = {}
        for alt, cost in out:
            if alt != ch and (alt not in best or cost < best[alt]):
                best[alt] = cost
        return sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:limit]

    # -- persistence ---------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "learning_weight": self.learning_weight,
                "observations": [
                    [observed, intended, count]
                    for (observed, intended), count in sorted(self.observations.items())
                ],
            },
            ensure_ascii=False,
            indent=1,
        )

    @classmethod
    def from_json(cls, payload: str) -> ConfusionModel:
        data = json.loads(payload)
        model = cls(learning_weight=data.get("learning_weight", 0.6))
        for observed, intended, count in data.get("observations", []):
            model.observations[(observed, intended)] = count
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ConfusionModel:
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text(encoding="utf-8"))

    def top_confusions(self, limit: int = 20) -> list[tuple[str, str, int]]:
        rows = [(o, i, c) for (o, i), c in self.observations.items()]
        rows.sort(key=lambda r: -r[2])
        return rows[:limit]


DEFAULT_MODEL = ConfusionModel()


# --------------------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------------------


def distance(
    observed: str,
    intended: str,
    model: ConfusionModel | None = None,
    cutoff: float = float("inf"),
) -> float:
    """Weighted edit distance. Returns ``inf`` once the best path exceeds ``cutoff``."""
    model = model or DEFAULT_MODEL
    if observed == intended:
        return 0.0

    previous = [0.0]
    for ch in intended:
        previous.append(previous[-1] + model.ins_cost(ch))

    for obs_ch in observed:
        current = [previous[0] + model.del_cost(obs_ch)]
        row_best = current[0]
        for j, int_ch in enumerate(intended, start=1):
            cost = min(
                previous[j] + model.del_cost(obs_ch),
                current[j - 1] + model.ins_cost(int_ch),
                previous[j - 1] + model.sub_cost(obs_ch, int_ch),
            )
            current.append(cost)
            row_best = min(row_best, cost)
        if row_best > cutoff:
            return float("inf")
        previous = current

    return previous[-1]


def align(
    observed: str, intended: str, model: ConfusionModel | None = None
) -> list[tuple[str, str, str]]:
    """Align two strings, returning ``(op, observed_char, intended_char)`` triples.

    Used both to learn confusion weights from corrections and to drive the diff view.
    """
    model = model or DEFAULT_MODEL
    n, m = len(observed), len(intended)
    table = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        table[i][0] = table[i - 1][0] + model.del_cost(observed[i - 1])
    for j in range(1, m + 1):
        table[0][j] = table[0][j - 1] + model.ins_cost(intended[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = min(
                table[i - 1][j] + model.del_cost(observed[i - 1]),
                table[i][j - 1] + model.ins_cost(intended[j - 1]),
                table[i - 1][j - 1] + model.sub_cost(observed[i - 1], intended[j - 1]),
            )

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = table[i - 1][j - 1] + model.sub_cost(observed[i - 1], intended[j - 1])
            if abs(table[i][j] - sub) < 1e-9:
                op = "match" if observed[i - 1] == intended[j - 1] else "sub"
                ops.append((op, observed[i - 1], intended[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(table[i][j] - (table[i - 1][j] + model.del_cost(observed[i - 1]))) < 1e-9:
            ops.append(("del", observed[i - 1], ""))
            i -= 1
            continue
        ops.append(("ins", "", intended[j - 1]))
        j -= 1

    ops.reverse()
    return ops


def similarity(observed: str, intended: str, model: ConfusionModel | None = None) -> float:
    """Normalized 0–1 similarity, where 1.0 means identical."""
    if not observed and not intended:
        return 1.0
    d = distance(observed, intended, model)
    return max(0.0, 1.0 - d / max(len(observed), len(intended)))
