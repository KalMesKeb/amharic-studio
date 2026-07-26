"""SERA phonetic input.

The editor's Latin-to-fidel typing has to behave predictably keystroke by keystroke, so
both the batch function and the incremental machine are exercised, and required to agree.
"""

from __future__ import annotations

import pytest

from amharic_studio.core.translit import Transliterator, keyboard_reference, transliterate


class TestBatch:
    @pytest.mark.parametrize(
        ("latin", "fidel"),
        [
            ("selam", "ሰላም"),
            ("le", "ለ"),
            ("lu", "ሉ"),
            ("li", "ሊ"),
            ("la", "ላ"),
            ("lE", "ሌ"),
            ("l", "ል"),
            ("lo", "ሎ"),
        ],
    )
    def test_vowel_orders(self, latin, fidel):
        assert transliterate(latin) == fidel

    @pytest.mark.parametrize(
        ("latin", "fidel"),
        [("she", "ሸ"), ("che", "ቸ"), ("tse", "ጸ"), ("zhe", "ዠ"), ("nye", "ኘ")],
    )
    def test_digraphs(self, latin, fidel):
        assert transliterate(latin) == fidel

    def test_a_bare_vowel_becomes_an_alef_syllable(self):
        assert transliterate("a") == "አ"

    def test_case_selects_between_similar_consonants(self):
        # SERA leans on capitals to reach the second member of a homophone pair.
        assert transliterate("se") != transliterate("Se")

    def test_latin_that_is_not_a_syllable_passes_through(self):
        assert transliterate("1998") == "1998"

    def test_spaces_and_punctuation_survive(self):
        assert transliterate("selam lehulum") == "ሰላም ለሁሉም"

    def test_empty_input(self):
        assert transliterate("") == ""


class TestPunctuation:
    @pytest.mark.parametrize(
        ("latin", "fidel"),
        [("selam::", "ሰላም።"), ("selam:lehulum", "ሰላም፡ለሁሉም"), ("selam,", "ሰላም፣"), ("selam;", "ሰላም፤")],
    )
    def test_ethiopic_marks(self, latin, fidel):
        assert transliterate(latin) == fidel

    def test_a_bare_period_is_left_alone(self):
        # Needed for abbreviations, decimals and file names; ። is typed as ::.
        assert transliterate("3.14") == "3.14"


def type_out(latin: str) -> str:
    """Replay a string through the incremental machine the way the editor does."""
    machine = Transliterator()
    buffer = ""
    for ch in latin:
        emission = machine.feed(ch)
        if not emission.handled:
            buffer += ch
            continue
        if emission.delete:
            buffer = buffer[: -emission.delete]
        buffer += emission.insert
    return buffer


class TestIncremental:
    @pytest.mark.parametrize(
        "latin", ["selam", "lehulum", "ityoPya", "tsehay", "selam::", "selam:lehulum", "ale::-"]
    )
    def test_typing_matches_pasting(self, latin):
        # The editor types through feed(); the search box converts through transliterate().
        # If the two disagreed, the same keys would produce different text in each.
        assert type_out(latin) == transliterate(latin), latin

    def test_a_lone_consonant_shows_its_sixth_order_immediately(self):
        # Nothing is hidden while typing: 'l' shows ል, which becomes ለ once 'e' arrives.
        machine = Transliterator()
        assert machine.feed("l").insert == "ል"

    def test_the_following_vowel_rewrites_the_syllable_in_place(self):
        machine = Transliterator()
        machine.feed("l")
        emission = machine.feed("e")
        assert emission.delete == 1 and emission.insert == "ለ"

    def test_reset_discards_pending_state(self):
        machine = Transliterator()
        machine.feed("l")
        machine.reset()
        assert not machine.pending
        # With nothing pending, 'e' is a standalone vowel rather than a continuation.
        assert machine.feed("e").delete == 0

    def test_unhandled_keys_are_reported_so_the_editor_can_pass_them_through(self):
        assert not Transliterator().feed("\n").handled


class TestReference:
    def test_keyboard_reference_is_not_empty(self):
        reference = keyboard_reference()
        assert reference
        assert any("ሰ" in str(row) for row in reference)
