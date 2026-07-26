"""Normalization.

The governing rule is that the faithful policy must never change what the page says. It
may only remove things that are invisible or that no printed book could have contained.
"""

from __future__ import annotations

import unicodedata

import pytest

from amharic_studio.core.normalize import Normalizer, Policy, detect_orthography, normalize


class TestFaithfulPolicy:
    def test_leaves_old_orthography_alone(self):
        # A diplomatic transcription keeps ሠ and ኀ exactly as the book printed them.
        for text in ["ሠላም", "ኀይል", "ዐለም", "ፀሐይ"]:
            assert normalize(text, Policy.FAITHFUL) == text

    def test_applies_nfc(self):
        decomposed = unicodedata.normalize("NFD", "ካፌ")
        assert normalize(decomposed, Policy.FAITHFUL) == unicodedata.normalize("NFC", "ካፌ")

    @pytest.mark.parametrize("invisible", ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"])
    def test_strips_invisible_characters(self, invisible):
        assert normalize(f"ሰላ{invisible}ም", Policy.FAITHFUL) == "ሰላም"

    def test_converts_ascii_punctuation_that_stands_in_for_fidel(self):
        # OCR frequently emits :: or a colon where the page had ። or ፡.
        assert "።" in normalize("ሰላም ነው::", Policy.FAITHFUL)

    def test_collapses_runs_of_spaces(self):
        assert normalize("ሰላም    ነው", Policy.FAITHFUL) == "ሰላም ነው"

    def test_is_idempotent(self):
        once = normalize("ሰላም ነው:: ጤና   ይስጥልኝ", Policy.FAITHFUL)
        assert normalize(once, Policy.FAITHFUL) == once


class TestModernPolicy:
    @pytest.mark.parametrize(
        ("old", "modern"),
        [
            ("ሠላም", "ሰላም"),
            ("ዐለም", "አለም"),
            # Every member of the /h/ group collapses onto ሀ, so ኀ and ሐ both move.
            ("ኀይል", "ሀይል"),
            ("ፀሐይ", "ጸሀይ"),
        ],
    )
    def test_regularizes_homophones(self, old, modern):
        assert normalize(old, Policy.MODERN) == modern

    def test_vowel_order_is_preserved_when_the_family_changes(self):
        # ኃ is ኀ in fourth order. Regularizing the family must not also reset the vowel,
        # which would turn one word into another.
        assert normalize("ኃይል", Policy.MODERN) == "ሃይል"

    def test_word_separator_becomes_a_space(self):
        assert normalize("ሰላም፡ለሁሉም", Policy.MODERN) == "ሰላም ለሁሉም"

    def test_full_stop_survives(self):
        # ። is current usage; only ፡ is archaic. Removing it would destroy sentences.
        assert "።" in normalize("ሰላም ነው።", Policy.MODERN)


class TestChangeReporting:
    def test_every_change_is_reported(self):
        result = Normalizer(Policy.MODERN).normalize("ሠላም፡ለሁሉም\u200b")
        assert result.text != "ሠላም፡ለሁሉም\u200b"
        assert result.changes, "a silent rewrite is not acceptable"
        for change in result.changes:
            assert change.rule and change.before != change.after

    def test_a_clean_string_reports_no_changes(self):
        result = Normalizer(Policy.FAITHFUL).normalize("ሰላም ለሁሉም።")
        assert result.changes == []

    def test_rules_can_be_switched_off(self):
        enabled = Normalizer(Policy.MODERN)
        rule = next(c.rule for c in enabled.normalize("ሠላም").changes)
        disabled = Normalizer(Policy.MODERN, disabled={rule})
        assert disabled.normalize("ሠላም").text == "ሠላም"


class TestOrthographyDetection:
    def test_detects_old_spelling(self):
        assert detect_orthography("ሠላም፡ኀይል፡ዐለም፡ፀሐይ፡ሠናይ") == Policy.FAITHFUL

    def test_detects_modern_spelling(self):
        assert detect_orthography("ሰላም ለሁሉም ሰው ይሁን። ፍቅር ደግሞ ይብዛ።") == Policy.MODERN

    def test_empty_text_does_not_crash(self):
        assert detect_orthography("") in (Policy.FAITHFUL, Policy.MODERN)
