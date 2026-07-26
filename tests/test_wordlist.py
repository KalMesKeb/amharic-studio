"""The bundled corpus wordlist and the bulk loader that reads it."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from amharic_studio.core.lexicon import CORPUS_DELETES_MIN_FREQ, Lexicon, load_or_create
from amharic_studio.data import wordlist

SAMPLE = [
    ("ኢትዮጵያ", 5000),
    ("ጥንታዊት", 400),
    ("ይኖራል", 120),
    ("ነገሥታቱም", 12),
    ("ኧረጭንቀጥ", 3),  # deliberate junk, the kind a crawl always carries
]


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("# comment line the reader must skip\n")
        for word, freq in SAMPLE:
            fh.write(f"{word}\t{freq}\n")
    return path


class TestReading:
    def test_entries_round_trip(self, sample: Path) -> None:
        assert list(wordlist.entries(sample)) == SAMPLE

    def test_a_missing_file_is_empty_rather_than_an_error(self, tmp_path: Path) -> None:
        assert list(wordlist.entries(tmp_path / "nope.txt.gz")) == []

    def test_a_line_without_a_frequency_still_counts_as_a_word(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("ሰላም\n")
        assert list(wordlist.entries(path)) == [("ሰላም", 1)]


class TestBulkAdd:
    def test_every_word_is_loaded(self, sample: Path) -> None:
        lex = Lexicon(":memory:")
        assert lex.bulk_add(wordlist.entries(sample)) == len(SAMPLE)
        assert all(lex.contains(w) for w, _ in SAMPLE)

    def test_frequencies_survive_the_load(self, sample: Path) -> None:
        lex = Lexicon(":memory:")
        lex.bulk_add(wordlist.entries(sample))
        assert lex.frequency("ኢትዮጵያ") == 5000
        assert lex.frequency("ይኖራል") == 120

    def test_progress_is_reported(self, sample: Path) -> None:
        lex = Lexicon(":memory:")
        seen: list[int] = []
        lex.bulk_add(wordlist.entries(sample), progress=seen.append, chunk=2)
        assert seen and seen[-1] == len(SAMPLE)
        assert seen == sorted(seen)

    def test_it_matches_adding_one_at_a_time(self, sample: Path) -> None:
        bulk, single = Lexicon(":memory:"), Lexicon(":memory:")
        bulk.bulk_add(wordlist.entries(sample))
        for word, freq in SAMPLE:
            single.add(word, freq)
        assert len(bulk) == len(single)
        for word, _ in SAMPLE:
            assert bulk.contains(word) == single.contains(word)
            assert bulk.frequency(word) == single.frequency(word)

    def test_rare_words_are_recognized_but_not_offered_as_corrections(self, sample: Path) -> None:
        """The crawl tail has to be searchable without becoming a suggestion."""
        lex = Lexicon(":memory:", deletes_min_freq=CORPUS_DELETES_MIN_FREQ)
        lex.bulk_add(wordlist.entries(sample))
        junk = "ኧረጭንቀጥ"
        assert lex.contains(junk)
        # One character short of the junk word: only the deletion index could reach it.
        assert junk not in [c.surface for c in lex.candidates(junk[:-1])]

    def test_common_words_are_still_reachable_from_a_typo(self, sample: Path) -> None:
        lex = Lexicon(":memory:", deletes_min_freq=CORPUS_DELETES_MIN_FREQ)
        lex.bulk_add(wordlist.entries(sample))
        assert "ኢትዮጵያ" in [c.surface for c in lex.candidates("ኢትዮጵ")]


class TestFirstRun:
    def test_a_fresh_lexicon_knows_everyday_words(self, tmp_path: Path) -> None:
        if not wordlist.available():
            pytest.skip("no bundled wordlist; run tools/build_lexicon.py")
        lex = load_or_create(tmp_path / "fresh.db")
        # Inflected forms of ordinary words: exactly what a curated list always misses.
        for word in ("ጥንታዊት", "ይኖራል", "ሁለተኛው", "ተጻፈ", "ከዚያም"):
            assert lex.recognizes(word), word
        lex.close()

    def test_seed_words_keep_their_curated_weight(self, tmp_path: Path) -> None:
        """Seeding adds to the corpus count rather than being overwritten by it.

        The seed tiers are what stop a crawl artefact from outranking an everyday word
        when two candidates are equally close to the misreading.
        """
        if not wordlist.available():
            pytest.skip("no bundled wordlist; run tools/build_lexicon.py")
        from amharic_studio.data.seed_words import SEED_WORDS

        seeded = dict(SEED_WORDS)
        corpus = dict(wordlist.entries())
        lex = load_or_create(tmp_path / "fresh.db")
        try:
            word = "ሰላም"
            assert lex.frequency(word) == seeded[word] + corpus.get(word, 0)
            assert lex.frequency(word) > seeded[word]
        finally:
            lex.close()
