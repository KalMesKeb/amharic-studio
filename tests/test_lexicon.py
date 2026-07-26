"""The word store and its retrieval strategy."""

from __future__ import annotations

from amharic_studio.core.lexicon import Candidate, Lexicon, deletion_keys, skeleton

#: Well-formed Amharic words that the seed wordlist does not contain, so that tests about
#: adding and finding words start from a genuine absence.
ABSENT = "ቆርቆሮ"
ABSENT_2 = "ብርጭቆ"


class TestSkeletons:
    def test_vowel_orders_collapse_onto_the_family(self):
        # This is the whole retrieval idea: order errors do not move a word's bucket.
        assert skeleton("ቁም") == skeleton("ቃም") == skeleton("ቅም") == skeleton("ቀም")

    def test_different_consonants_land_in_different_buckets(self):
        assert skeleton("ቀም") != skeleton("ተም")

    def test_non_ethiopic_passes_through(self):
        assert skeleton("abc") == "abc"

    def test_deletion_keys_cover_every_position(self):
        assert deletion_keys("ሰላም") == {"ላም", "ሰም", "ሰላ"}

    def test_deletion_keys_skip_trivial_words(self):
        assert deletion_keys("ሰ") == set()


class TestMembership:
    def test_added_words_are_found(self, lexicon: Lexicon):
        lexicon.add(ABSENT)
        assert lexicon.contains(ABSENT)

    def test_membership_sees_through_homophone_spelling(self, lexicon: Lexicon):
        lexicon.add("ሰላም")
        assert lexicon.contains("ሠላም"), "old spelling must match a modern-spelt lexicon"

    def test_unknown_words_are_unknown(self, lexicon: Lexicon):
        assert not lexicon.contains("ቅቅቅቅ")

    def test_frequencies_accumulate(self, lexicon: Lexicon):
        before = lexicon.frequency(ABSENT)
        lexicon.add(ABSENT, freq=3)
        lexicon.add(ABSENT, freq=4)
        assert lexicon.frequency(ABSENT) == before + 7

    def test_recognizes_accepts_inflections(self, lexicon: Lexicon):
        assert lexicon.contains("ኢትዮጵያ")
        assert not lexicon.contains("የኢትዮጵያን")
        assert lexicon.recognizes("የኢትዮጵያን")

    def test_cache_is_invalidated_when_a_word_is_added(self, lexicon: Lexicon):
        assert not lexicon.contains(ABSENT)  # populates the negative cache
        lexicon.add(ABSENT)
        assert lexicon.contains(ABSENT), "a stale cache would hide the new word"

    def test_adding_a_stem_makes_its_inflections_recognizable(self, lexicon: Lexicon):
        assert not lexicon.recognizes(f"የ{ABSENT_2}ን")
        lexicon.add(ABSENT_2)
        assert lexicon.recognizes(f"የ{ABSENT_2}ን")


class TestCandidates:
    def test_a_vowel_order_error_retrieves_the_right_word(self, lexicon: Lexicon, model):
        lexicon.add("ጠይቀው", freq=20)
        candidates = lexicon.candidates("ጣይቃው", model)
        assert "ጠይቀው" in [c.surface for c in candidates]

    def test_the_word_itself_is_not_offered_as_a_correction(self, lexicon: Lexicon, model):
        lexicon.add("ሰላም", freq=99)
        assert "ሰላም" not in [c.surface for c in lexicon.candidates("ሰላም", model)]

    def test_candidates_come_back_best_first(self, lexicon: Lexicon, model):
        scores = [c.score for c in lexicon.candidates("ጣይቃው", model)]
        assert scores == sorted(scores)

    def test_a_deletion_is_retrievable(self, lexicon: Lexicon, model):
        lexicon.add("መጽሐፍ", freq=30)
        assert "መጽሐፍ" in [c.surface for c in lexicon.candidates("መጽሐ", model)]

    def test_an_insertion_is_retrievable(self, lexicon: Lexicon, model):
        lexicon.add("መጽሐፍ", freq=30)
        assert "መጽሐፍ" in [c.surface for c in lexicon.candidates("መጽሐፍፍ", model)]

    def test_nonsense_retrieves_nothing_plausible(self, lexicon: Lexicon, model):
        assert lexicon.candidates("ቅቅቅቅቅቅ", model, max_cost=0.9) == []

    def test_results_are_capped(self, lexicon: Lexicon, model):
        assert len(lexicon.candidates("ሰላም", model, max_results=3)) <= 3

    def test_frequency_only_breaks_ties(self):
        # Frequency nudges, it does not override. A common word must not beat a rarer one
        # that is a visibly closer match, or every error would be "corrected" to ነው.
        common = Candidate("ሀ", cost=0.9, frequency=100_000, source="order")
        close = Candidate("ለ", cost=0.2, frequency=1, source="order")
        assert close.score < common.score

        cheap = Candidate("ሀ", cost=0.4, frequency=1000, source="order")
        rare = Candidate("ለ", cost=0.4, frequency=1, source="order")
        assert cheap.score < rare.score


class TestCorpusStatistics:
    def test_ingesting_text_learns_words_and_pairs(self):
        lex = Lexicon(":memory:")
        lex.ingest_text("ሰላም ለሁሉም ሰው")
        assert lex.contains("ለሁሉም")
        assert lex.bigram_frequency("ሰላም", "ለሁሉም") == 1

    def test_unknown_rate_of_known_text_is_zero(self, lexicon: Lexicon):
        assert lexicon.unknown_rate("ሰላም ኢትዮጵያ") == 0.0

    def test_unknown_rate_of_gibberish_is_one(self, lexicon: Lexicon):
        assert lexicon.unknown_rate("ቅቅቅቅ ዠዠዠዠ") == 1.0

    def test_unknown_rate_of_empty_text_is_zero(self, lexicon: Lexicon):
        assert lexicon.unknown_rate("") == 0.0


class TestPersistence:
    def test_a_lexicon_survives_being_closed_and_reopened(self, tmp_path):
        path = tmp_path / "lex.db"
        with Lexicon(path) as lex:
            lex.add("ጨረቃ", freq=5)
        with Lexicon(path) as reopened:
            assert reopened.contains("ጨረቃ")
            assert reopened.frequency("ጨረቃ") == 5

    def test_rebuild_deletes_restores_the_index(self, tmp_path, model):
        with Lexicon(tmp_path / "lex.db") as lex:
            lex.add("መጽሐፍ", freq=30)
            lex.rebuild_deletes()
            assert "መጽሐፍ" in [c.surface for c in lex.candidates("መጽሐ", model)]
