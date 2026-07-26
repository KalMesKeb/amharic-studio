"""Word store and fuzzy candidate retrieval for Amharic.

Retrieval is built around a property specific to this script. The overwhelmingly common
OCR error is a *wrong vowel order inside the correct consonant family* — ቀ read as ቁ,
ተ read as ታ — because those glyphs differ only by a small appendage. Strip the vowel
orders off a word and you get its consonant **skeleton**, and that skeleton normally
survives the error untouched:

    ጠይቀዉ  →  skeleton ጠ-የ-ቀ-ወ
    ጣይቃዋ  →  skeleton ጠ-የ-ቀ-ወ   (same bucket, four order errors)

So a single indexed lookup on the skeleton retrieves the right word directly, where a
generic edit-distance index would have to scan a huge neighbourhood. Family-level errors
(ሰ read as ሸ) are handled by expanding the skeleton through the confusion model, and
insertions and deletions by a small deletion index.
"""

from __future__ import annotations

import itertools
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from . import confusion, fidel, morph
from .confusion import ConfusionModel
from .fidel import DEFAULT_FOLD, FoldPolicy

SCHEMA_VERSION = 1

#: Membership caches are cleared wholesale once past this size. A book's vocabulary is far
#: smaller than this, so in practice the caches never turn over during a session.
_CACHE_LIMIT = 200_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS words (
    surface  TEXT PRIMARY KEY,
    folded   TEXT NOT NULL,
    skeleton TEXT NOT NULL,
    freq     INTEGER NOT NULL DEFAULT 1,
    source   TEXT NOT NULL DEFAULT 'corpus'
);
CREATE INDEX IF NOT EXISTS idx_words_folded   ON words(folded);
CREATE INDEX IF NOT EXISTS idx_words_skeleton ON words(skeleton);
CREATE INDEX IF NOT EXISTS idx_words_freq     ON words(freq DESC);

-- Deletion index: one row per single-character deletion of a folded word. Catches the
-- insertion and deletion errors that the skeleton index cannot see.
CREATE TABLE IF NOT EXISTS deletes (
    key     TEXT NOT NULL,
    surface TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deletes_key ON deletes(key);

CREATE TABLE IF NOT EXISTS bigrams (
    w1   TEXT NOT NULL,
    w2   TEXT NOT NULL,
    freq INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (w1, w2)
);
"""


# --------------------------------------------------------------------------------------
# Skeletons
# --------------------------------------------------------------------------------------


def skeleton(word: str, policy: FoldPolicy = DEFAULT_FOLD) -> str:
    """Consonant skeleton: every syllable reduced to its family's ግዕዝ form.

    ``ቁም`` and ``ቃም`` and ``ቅም`` all reduce to ``ቀም``. Non-syllable characters pass
    through unchanged so that Latin and digits still bucket sensibly.
    """
    out: list[str] = []
    for ch in word:
        entry = fidel.decompose(ch)
        if entry is None:
            out.append(ch)
            continue
        fam, _order = entry
        base = fam.base
        if policy.fold_homophone_families:
            for group in fidel.HOMOPHONE_FAMILY_GROUPS:
                if base in group:
                    base = group[0]
                    break
        if policy.fold_labiovelar:
            base = fidel.LABIOVELAR_PARTNER.get(base, base) if fam.labiovelar else base
        out.append(fidel.compose(base, 0) or ch)
    return "".join(out)


def deletion_keys(word: str, max_len: int = 24) -> set[str]:
    """Every single-character deletion of ``word``."""
    if len(word) > max_len or len(word) < 2:
        return set()
    return {word[:i] + word[i + 1 :] for i in range(len(word))}


# --------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    surface: str
    cost: float
    frequency: int
    source: str

    @property
    def score(self) -> float:
        """Lower is better. Frequency breaks ties between equally plausible readings."""
        import math

        return self.cost - 0.06 * math.log1p(self.frequency)


# --------------------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------------------


class Lexicon:
    """SQLite-backed word store. Pass ``:memory:`` for a scratch lexicon."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        fold_policy: FoldPolicy = DEFAULT_FOLD,
        build_deletes: bool = True,
        deletes_min_freq: int = 1,
    ) -> None:
        self.path = str(path)
        self.fold_policy = fold_policy
        self.build_deletes = build_deletes
        # The deletion index costs one row per character of every word it covers, so over
        # a corpus wordlist it dwarfs the words themselves. It only ever supplies
        # correction *targets*, and a word seen twice in a billion is not a plausible
        # thing to correct towards, so rare words are indexed for recognition only.
        self.deletes_min_freq = deletes_min_freq
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # The UI runs recognition, analysis and quality assessment on worker threads, and
        # every one of them consults the lexicon. A connection is bound to its creating
        # thread unless told otherwise, so without this the entire background pipeline
        # fails with a ProgrammingError. Serializing on a lock is enough: the queries are
        # short, and SQLite serializes writes regardless.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()
        self._total_freq: int | None = None
        # Membership is asked thousands of times per page — once per token directly, then
        # again for every affix-stripped candidate — and each miss is a round trip to
        # SQLite. The answers only change when a word is added, so cache them.
        self._contains_cache: dict[str, bool] = {}
        self._recognizes_cache: dict[str, morph.Analysis | None] = {}

    # -- population ----------------------------------------------------------------------

    def add(self, surface: str, freq: int = 1, source: str = "corpus", commit: bool = True) -> None:
        surface = unicodedata.normalize("NFC", surface).strip()
        if not surface:
            return
        folded = fidel.fold(surface, self.fold_policy)
        skel = skeleton(surface, self.fold_policy)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO words(surface, folded, skeleton, freq, source) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(surface) DO UPDATE SET freq = freq + excluded.freq
                """,
                (surface, folded, skel, freq, source),
            )
            if self.build_deletes and freq >= self.deletes_min_freq:
                rows = [(key, surface) for key in deletion_keys(folded)]
                if rows:
                    self.conn.executemany("INSERT INTO deletes(key, surface) VALUES (?, ?)", rows)
            if commit:
                self.conn.commit()
            self._total_freq = None
            # A new word can turn any cached "unknown" into a hit, including indirectly
            # via an affix analysis, so both caches have to go.
            if not self._contains_cache.get(folded, False):
                self._contains_cache.clear()
                self._recognizes_cache.clear()

    def add_many(self, items: Iterable[tuple[str, int]] | Iterable[str], source: str = "corpus") -> int:
        n = 0
        with self._lock:
            for item in items:
                if isinstance(item, str):
                    self.add(item, 1, source, commit=False)
                else:
                    self.add(item[0], item[1], source, commit=False)
                n += 1
            self.conn.commit()
        return n

    def bulk_add(
        self,
        items: Iterable[tuple[str, int]],
        source: str = "corpus",
        chunk: int = 20_000,
        progress: Callable[[int], None] | None = None,
    ) -> int:
        """Load a corpus wordlist.

        ``add`` issues several statements per word, which is fine for the handful a user
        types but not for the hundreds of thousands in a frequency list — first run would
        take minutes. This batches into ``executemany`` and commits per chunk, so a
        cancelled load leaves a usable partial lexicon rather than nothing.
        """
        total = 0
        words: list[tuple[str, str, str, int, str]] = []
        deletes: list[tuple[str, str]] = []

        def flush() -> None:
            if not words:
                return
            with self._lock:
                self.conn.executemany(
                    """
                    INSERT INTO words(surface, folded, skeleton, freq, source)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(surface) DO UPDATE SET freq = freq + excluded.freq
                    """,
                    words,
                )
                if deletes:
                    self.conn.executemany(
                        "INSERT INTO deletes(key, surface) VALUES (?, ?)", deletes
                    )
                self.conn.commit()
            words.clear()
            deletes.clear()

        for surface, freq in items:
            surface = unicodedata.normalize("NFC", surface).strip()
            if not surface:
                continue
            folded = fidel.fold(surface, self.fold_policy)
            words.append((surface, folded, skeleton(surface, self.fold_policy), freq, source))
            if self.build_deletes and freq >= self.deletes_min_freq:
                deletes.extend((key, surface) for key in deletion_keys(folded))
            total += 1
            if len(words) >= chunk:
                flush()
                if progress is not None:
                    progress(total)
        flush()

        with self._lock:
            self._total_freq = None
            self._contains_cache.clear()
            self._recognizes_cache.clear()
        if progress is not None:
            progress(total)
        return total

    def add_bigram(self, w1: str, w2: str, freq: int = 1) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO bigrams(w1, w2, freq) VALUES (?, ?, ?)
                ON CONFLICT(w1, w2) DO UPDATE SET freq = freq + excluded.freq
                """,
                (fidel.fold(w1, self.fold_policy), fidel.fold(w2, self.fold_policy), freq),
            )

    def ingest_text(self, text: str, source: str = "corpus") -> int:
        """Learn every word and word pair in a block of text."""
        tokens = [t.text for t in fidel.words(text)]
        with self._lock:
            for word in tokens:
                self.add(word, 1, source, commit=False)
            for a, b in itertools.pairwise(tokens):
                self.add_bigram(a, b)
            self.conn.commit()
            self._total_freq = None
        return len(tokens)

    # -- queries -------------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM words").fetchone()[0]

    @property
    def total_frequency(self) -> int:
        with self._lock:
            if self._total_freq is None:
                row = self.conn.execute("SELECT COALESCE(SUM(freq), 0) FROM words").fetchone()
                self._total_freq = int(row[0])
            return self._total_freq

    def contains(self, word: str) -> bool:
        """True if the word is known, comparing folded forms.

        Folding is what lets ሠላም match a lexicon that only knows ሰላም, without the stored
        text ever being rewritten.
        """
        folded = fidel.fold(word, self.fold_policy)
        with self._lock:
            hit = self._contains_cache.get(folded)
            if hit is None:
                row = self.conn.execute(
                    "SELECT 1 FROM words WHERE folded = ? LIMIT 1", (folded,)
                ).fetchone()
                hit = row is not None
                if len(self._contains_cache) > _CACHE_LIMIT:
                    self._contains_cache.clear()
                self._contains_cache[folded] = hit
            return hit

    def analyze(self, word: str) -> morph.Analysis | None:
        """Read ``word`` as a known stem plus affixes, or None if nothing fits.

        Amharic glues prepositions, articles and case markers onto stems, so ``contains``
        alone reports a great deal of correct text as unknown. Callers that are deciding
        whether to *flag* a word should use this; callers looking up a specific surface
        form should use ``contains``.
        """
        with self._lock:
            if word in self._recognizes_cache:
                return self._recognizes_cache[word]
            result = morph.analyze(word, self.contains)
            if len(self._recognizes_cache) > _CACHE_LIMIT:
                self._recognizes_cache.clear()
            self._recognizes_cache[word] = result
            return result

    def recognizes(self, word: str) -> bool:
        """True if the word is known outright or as an inflection of a known stem."""
        return self.contains(word) or self.analyze(word) is not None

    def frequency(self, word: str) -> int:
        folded = fidel.fold(word, self.fold_policy)
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(freq), 0) FROM words WHERE folded = ?", (folded,)
            ).fetchone()
        return int(row[0])

    def bigram_frequency(self, w1: str, w2: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT freq FROM bigrams WHERE w1 = ? AND w2 = ?",
                (fidel.fold(w1, self.fold_policy), fidel.fold(w2, self.fold_policy)),
            ).fetchone()
        return int(row[0]) if row else 0

    def _rows_for(self, column: str, values: Iterable[str]) -> Iterator[sqlite3.Row]:
        values = list(dict.fromkeys(values))
        if not values:
            return iter(())
        placeholders = ",".join("?" * len(values))
        with self._lock:
            return iter(
                self.conn.execute(
                    f"SELECT surface, freq FROM words WHERE {column} IN ({placeholders})", values
                ).fetchall()
            )

    def candidates(
        self,
        word: str,
        model: ConfusionModel | None = None,
        max_results: int = 8,
        max_cost: float = 1.4,
        expand_families: bool = True,
    ) -> list[Candidate]:
        """Plausible intended readings of ``word``, best first."""
        model = model or confusion.DEFAULT_MODEL
        folded = fidel.fold(word, self.fold_policy)
        skel = skeleton(word, self.fold_policy)

        pool: dict[str, str] = {}  # surface -> retrieval source

        def collect(rows: Iterable[sqlite3.Row], source: str) -> None:
            for row in rows:
                pool.setdefault(row["surface"], source)

        # 1. Same consonant skeleton — pure vowel-order errors.
        collect(self._rows_for("skeleton", [skel]), "order")

        # 2. Skeleton with one consonant family swapped for a visually similar one.
        if expand_families:
            collect(self._rows_for("skeleton", _family_variants(skel, model)), "family")

        # 3. Deletion index — insertions and deletions.
        keys = deletion_keys(folded) | {folded}
        placeholders = ",".join("?" * len(keys))
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT w.surface AS surface, w.freq AS freq
                FROM deletes d JOIN words w ON w.surface = d.surface
                WHERE d.key IN ({placeholders})
                """,
                list(keys),
            ).fetchall()
        collect(rows, "indel")
        collect(self._rows_for("folded", list(deletion_keys(folded))), "indel")

        if not pool:
            return []

        placeholders = ",".join("?" * len(pool))
        with self._lock:
            freqs = {
                row["surface"]: row["freq"]
                for row in self.conn.execute(
                    f"SELECT surface, freq FROM words WHERE surface IN ({placeholders})",
                    list(pool),
                ).fetchall()
            }

        out: list[Candidate] = []
        for surface, source in pool.items():
            if surface == word:
                continue
            cost = confusion.distance(word, surface, model, cutoff=max_cost)
            if cost > max_cost:
                continue
            out.append(Candidate(surface, cost, freqs.get(surface, 1), source))

        out.sort(key=lambda c: c.score)
        return out[:max_results]

    def unknown_words(self, text: str) -> list[str]:
        return [t.text for t in fidel.words(text) if not self.recognizes(t.text)]

    def unknown_rate(self, text: str) -> float:
        tokens = fidel.words(text)
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if not self.recognizes(t.text)) / len(tokens)

    # -- maintenance ---------------------------------------------------------------------

    def rebuild_deletes(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM deletes")
            rows = self.conn.execute("SELECT surface, folded FROM words").fetchall()
            payload: list[tuple[str, str]] = []
            for row in rows:
                payload.extend((key, row["surface"]) for key in deletion_keys(row["folded"]))
            self.conn.executemany("INSERT INTO deletes(key, surface) VALUES (?, ?)", payload)
            self.conn.commit()

    def vacuum(self) -> None:
        with self._lock:
            self.conn.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()

    def __enter__(self) -> Lexicon:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _family_variants(skel: str, model: ConfusionModel, limit_per_position: int = 4) -> list[str]:
    """Skeletons reachable by swapping one consonant for a visually similar one."""
    out: list[str] = []
    for i, ch in enumerate(skel):
        if fidel.decompose(ch) is None:
            continue
        for alt, _cost in model.neighbors(ch, max_cost=0.45, limit=limit_per_position):
            entry = fidel.decompose(alt)
            if entry is None or entry[1] != 0:
                continue  # skeletons only contain ግዕዝ forms
            out.append(skel[:i] + alt + skel[i + 1 :])
    return out


# --------------------------------------------------------------------------------------
# Seed data
# --------------------------------------------------------------------------------------


def default_lexicon_path() -> Path:
    from ..paths import user_data_dir

    return user_data_dir() / "lexicon" / "amharic.db"


#: Below this corpus frequency a word is recognised but never offered as a correction.
#: Corpus tails are mostly typos, transliterated names and OCR noise from the crawl, and
#: proposing them as fixes would be worse than proposing nothing.
CORPUS_DELETES_MIN_FREQ = 20


def load_or_create(
    path: str | Path | None = None,
    seed: bool = True,
    progress: Callable[[int], None] | None = None,
) -> Lexicon:
    """Open the lexicon, seeding it from the bundled wordlist the first time.

    Seeding reads a few hundred thousand words, so it takes a few seconds and should be
    given a ``progress`` callback and a worker thread when there is a UI to keep alive.
    """
    from ..data import wordlist
    from ..data.seed_words import SEED_WORDS

    target = Path(path) if path else default_lexicon_path()
    fresh = not Path(target).exists()
    lex = Lexicon(target, deletes_min_freq=CORPUS_DELETES_MIN_FREQ)
    if fresh and seed:
        # The seed list goes in at a high frequency regardless of the corpus, so the
        # everyday words it curates always outrank crawl noise when candidates tie.
        lex.add_many(SEED_WORDS, source="seed")
        if wordlist.available():
            lex.bulk_add(wordlist.entries(), source="corpus", progress=progress)
    return lex
