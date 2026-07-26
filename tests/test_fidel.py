"""The script model. Everything else in the program is built on these being right."""

from __future__ import annotations

import pytest

from amharic_studio.core import fidel


class TestSyllables:
    def test_decompose_gives_family_and_order(self):
        family, order = fidel.decompose("ቱ")
        assert family.base == ord("ተ")
        assert order == 1  # ካዕብ

    def test_compose_inverts_decompose(self):
        # Every assigned slot must survive a round trip, or candidate generation would
        # silently drop characters.
        for family in fidel.FAMILIES:
            for offset in family.offsets:
                ch = fidel.compose(family.base, offset)
                assert ch is not None
                assert fidel.decompose(ch) == (family, offset)

    def test_every_declared_slot_exists_in_unicode(self):
        # The tables are hand-written and the Ethiopic block has real holes in it
        # (U+1257, U+12BF, U+12D7). A family that claims a slot Unicode never assigned
        # would hand downstream code an unnamed codepoint that no font can draw.
        import unicodedata

        for family in fidel.FAMILIES:
            for offset in family.offsets:
                cp = family.base + offset
                assert unicodedata.name(chr(cp), "") != "", (
                    f"{family.label} claims U+{cp:04X}, which is unassigned"
                )

    def test_compose_returns_none_for_unassigned_slot(self):
        # Not every family fills all eight orders; the gaps must be reported, not faked.
        gappy = [f for f in fidel.FAMILIES if len(f.offsets) < 8]
        assert gappy, "expected at least one family with unassigned orders"
        family = gappy[0]
        missing = set(range(8)) - set(family.offsets)
        assert fidel.compose(family.base, missing.pop()) is None

    def test_order_variants_excludes_the_input(self):
        variants = fidel.order_variants("ቀ")
        assert "ቀ" not in variants
        assert "ቁ" in variants and "ቃ" in variants

    def test_non_ethiopic_decomposes_to_nothing(self):
        assert fidel.decompose("a") is None
        assert fidel.decompose("።") is None


class TestClassification:
    @pytest.mark.parametrize("ch", ["ሀ", "ቀ", "ጵ", "ኋ"])
    def test_ethiopic_syllables(self, ch):
        assert fidel.is_ethiopic(ch)

    @pytest.mark.parametrize("ch", ["a", "1", " ", "،"])
    def test_non_ethiopic(self, ch):
        assert not fidel.is_ethiopic(ch)

    def test_punctuation_is_ethiopic_but_not_a_syllable(self):
        assert fidel.is_ethiopic_punctuation("።")
        assert not fidel.is_amharic_syllable("።")

    def test_geez_only_letters_are_flagged_as_non_amharic(self):
        # ሏ-family letters used only in Ge'ez or Tigrinya are a signal that the recognizer
        # wandered outside Amharic, so they must be distinguishable.
        assert fidel.is_non_amharic_ethiopic("ⶀ") or fidel.is_non_amharic_ethiopic("ᎀ")

    def test_ethiopic_ratio(self):
        assert fidel.ethiopic_ratio("ሰላም") == 1.0
        assert fidel.ethiopic_ratio("abc") == 0.0
        assert 0.4 < fidel.ethiopic_ratio("ሰላም abc") < 0.6


class TestNumerals:
    @pytest.mark.parametrize(
        ("numeral", "value"),
        [("፩", 1), ("፲", 10), ("፲፩", 11), ("፳", 20), ("፻", 100), ("፼", 10000)],
    )
    def test_ethiopic_to_int(self, numeral, value):
        assert fidel.ethiopic_to_int(numeral) == value

    @pytest.mark.parametrize("value", [1, 9, 10, 11, 19, 20, 99, 100, 101, 1000, 10000, 12345])
    def test_int_round_trip(self, value):
        assert fidel.ethiopic_to_int(fidel.int_to_ethiopic(value)) == value

    def test_invalid_numeral_returns_none(self):
        assert fidel.ethiopic_to_int("ሰላም") is None


class TestTokenizing:
    def test_words_ignores_punctuation(self):
        tokens = fidel.words("ሰላም ለሁሉም።")
        assert [t.text for t in tokens] == ["ሰላም", "ለሁሉም"]

    def test_token_offsets_index_the_original_string(self):
        text = "ሰላም ለሁሉም።"
        for token in fidel.words(text):
            assert text[token.start : token.end] == token.text

    def test_ethiopic_wordspace_separates_words(self):
        # ፡ is a word separator in older orthography, not punctuation inside a word.
        tokens = fidel.words("ሰላም፡ለሁሉም")
        assert [t.text for t in tokens] == ["ሰላም", "ለሁሉም"]

    def test_split_sentences_on_full_stop(self):
        parts = fidel.split_sentences("አንድ ነው። ሁለት ነው። ")
        assert len(parts) == 2


class TestFolding:
    def test_homophones_fold_together(self):
        # ሠላም and ሰላም are the same word in modern spelling; matching must see through it.
        assert fidel.fold("ሠላም") == fidel.fold("ሰላም")
        assert fidel.fold("ፀሐይ") == fidel.fold("ጸሐይ")

    def test_folding_does_not_change_the_stored_string(self):
        original = "ሠላም"
        fidel.fold(original)
        assert original == "ሠላም"

    def test_folding_is_idempotent(self):
        once = fidel.fold("ኀይል")
        assert fidel.fold(once) == once

    def test_distinct_words_do_not_fold_together(self):
        assert fidel.fold("በላ") != fidel.fold("በሉ")

    def test_homophone_variants_round_trip(self):
        variants = fidel.homophone_variants("ሰ")
        assert "ሠ" in variants
