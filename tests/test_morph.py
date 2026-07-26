"""Affix stripping.

The stakes here are asymmetric. A missed analysis costs the user one spurious suggestion;
a wrong analysis makes a real OCR error invisible. So recall on genuine inflections is
tested alongside a garbage set that must stay unrecognised.
"""

from __future__ import annotations

import pytest

from amharic_studio.core import morph

KNOWN = {
    "ኢትዮጵያ",
    "ቤት",
    "ሰው",
    "መንግሥት",
    "ተማሪ",
    "አገር",
    "መጽሐፍ",
    "ልጅ",
    "ከተማ",
    "ሀገር",
}


def known(word: str) -> bool:
    return word in KNOWN


class TestPrefixes:
    @pytest.mark.parametrize(
        ("surface", "stem"),
        [
            ("የኢትዮጵያ", "ኢትዮጵያ"),
            ("በኢትዮጵያ", "ኢትዮጵያ"),
            ("ለኢትዮጵያ", "ኢትዮጵያ"),
            ("ከኢትዮጵያ", "ኢትዮጵያ"),
            ("ወደኢትዮጵያ", "ኢትዮጵያ"),
            ("እንደኢትዮጵያ", "ኢትዮጵያ"),
            ("ስለኢትዮጵያ", "ኢትዮጵያ"),
        ],
    )
    def test_prepositional_prefixes(self, surface, stem):
        analysis = morph.analyze(surface, known)
        assert analysis is not None, surface
        assert analysis.stem == stem

    def test_longest_prefix_wins(self):
        # እንደ must be taken whole, not read as እ + ንደ.
        analysis = morph.analyze("እንደኢትዮጵያ", known)
        assert analysis.prefixes == ("እንደ",)


class TestSuffixes:
    @pytest.mark.parametrize(
        ("surface", "stem"),
        [
            ("ኢትዮጵያን", "ኢትዮጵያ"),
            ("ኢትዮጵያውያን", "ኢትዮጵያ"),
            ("ሰዎች", "ሰው"),
            ("ተማሪዎቹ", "ተማሪ"),
            ("ከተማዋ", "ከተማ"),
            ("አገራችን", "አገር"),
            ("ልጆች", "ልጅ"),
        ],
    )
    def test_enclitics(self, surface, stem):
        analysis = morph.analyze(surface, known)
        assert analysis is not None, surface
        assert analysis.stem == stem


class TestFusedSuffixes:
    """Suffixes that change the final syllable instead of adding one."""

    @pytest.mark.parametrize(
        ("surface", "stem"),
        [
            ("ቤቱ", "ቤት"),      # the house
            ("ቤቴ", "ቤት"),      # my house
            ("ቤታ", "ቤት"),      # her house
            ("መጽሐፉ", "መጽሐፍ"),  # the book
            ("መንግሥታት", "መንግሥት"),  # governments
        ],
    )
    def test_final_vowel_carries_the_suffix(self, surface, stem):
        analysis = morph.analyze(surface, known)
        assert analysis is not None, surface
        assert analysis.stem == stem


class TestStacking:
    @pytest.mark.parametrize(
        ("surface", "stem"),
        [
            ("የኢትዮጵያን", "ኢትዮጵያ"),
            ("በቤታችን", "ቤት"),
            ("ከከተማው", "ከተማ"),
            ("ለልጆች", "ልጅ"),
        ],
    )
    def test_prefix_and_suffix_together(self, surface, stem):
        analysis = morph.analyze(surface, known)
        assert analysis is not None, surface
        assert analysis.stem == stem


class TestConservatism:
    @pytest.mark.parametrize(
        "garbage", ["ዘየጠቀ", "ቅቅቅቅ", "ዠዠዠ", "ኺኺን", "ጨጨጨው", "ኰኰኰ", "ፐፐፐን", "ጏጏጏ"]
    )
    def test_garbage_is_not_analysed_into_a_known_stem(self, garbage):
        assert morph.analyze(garbage, known) is None, garbage

    def test_stem_must_survive_a_minimum_length(self):
        # Otherwise ``የቤ`` would be read as የ- plus a one-character stem and anything
        # starting with a preposition would look like a word.
        assert morph.analyze("የቤ", lambda w: True) is not None  # the bare form is allowed
        assert all(len(a.stem) >= morph.MIN_STEM for a in morph.segmentations("የቤት") if a.depth)

    def test_bare_word_is_offered_before_any_stripping(self):
        first = next(iter(morph.segmentations("የኢትዮጵያ")))
        assert first.is_bare and first.stem == "የኢትዮጵያ"

    def test_a_known_word_is_never_reanalysed(self):
        analysis = morph.analyze("ኢትዮጵያ", known)
        assert analysis.is_bare


class TestReporting:
    def test_describe_shows_the_segmentation(self):
        analysis = morph.analyze("የኢትዮጵያን", known)
        assert "ኢትዮጵያ" in analysis.describe()
        assert analysis.describe() != analysis.stem

    def test_segmentations_terminate_and_do_not_repeat(self):
        seen = [a for a in morph.segmentations("የቤቶቻችንም")]
        keys = [(a.stem, a.prefixes, a.suffixes) for a in seen]
        assert len(keys) == len(set(keys))
        assert len(seen) < 500  # bounded, so a page of unknown words stays cheap
