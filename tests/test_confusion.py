"""Confusion-weighted edit distance.

The point of the weighting is that a vowel-order slip inside one consonant family is a
far likelier reading error than a jump to an unrelated letter. The tests assert that
ordering rather than any particular number, so the cost table can be retuned freely.
"""

from __future__ import annotations

from amharic_studio.core import confusion
from amharic_studio.core.confusion import ConfusionModel


class TestDistance:
    def test_identical_strings_cost_nothing(self):
        assert confusion.distance("ሰላም", "ሰላም") == 0.0

    def test_a_vowel_order_slip_is_cheaper_than_an_unrelated_letter(self):
        model = ConfusionModel()
        order_slip = confusion.distance("ቀን", "ቁን", model)      # same family, wrong order
        unrelated = confusion.distance("ቀን", "ዘን", model)       # different family entirely
        assert order_slip < unrelated

    def test_visually_similar_families_are_cheaper_than_dissimilar_ones(self):
        model = ConfusionModel()
        similar = confusion.distance("ሰላም", "ሠላም", model)
        dissimilar = confusion.distance("ሰላም", "ጨላም", model)
        assert similar < dissimilar

    def test_cost_grows_with_the_number_of_errors(self):
        model = ConfusionModel()
        one = confusion.distance("ሰላም", "ሱላም", model)
        two = confusion.distance("ሰላም", "ሱሉም", model)
        assert two > one

    def test_cutoff_short_circuits_without_changing_the_verdict(self):
        model = ConfusionModel()
        full = confusion.distance("ሰላም", "ጨጨጨጨ", model)
        capped = confusion.distance("ሰላም", "ጨጨጨጨ", model, cutoff=0.5)
        assert capped >= 0.5 and full >= 0.5

    def test_empty_against_nonempty(self):
        assert confusion.distance("", "ሰላም") > 0


class TestAlignment:
    def test_identical_strings_align_as_all_matches(self):
        assert {op for op, _a, _b in confusion.align("ሰላም", "ሰላም")} == {"match"}

    def test_align_reports_a_substitution(self):
        assert ("sub", "ቀ", "ቁ") in confusion.align("ቀን", "ቁን")

    def test_align_reports_a_deletion(self):
        ops = confusion.align("ሰላም", "ሰላ")
        assert any(op == "del" for op, _a, _b in ops)

    def test_align_reports_an_insertion(self):
        ops = confusion.align("ሰላ", "ሰላም")
        assert any(op == "ins" for op, _a, _b in ops)

    def test_alignment_covers_both_strings(self):
        ops = confusion.align("ሰላም", "ሱላሙ")
        assert "".join(a for _op, a, _b in ops if a) == "ሰላም"
        assert "".join(b for _op, _a, b in ops if b) == "ሱላሙ"


class TestLearning:
    def test_an_observed_correction_becomes_cheaper(self):
        model = ConfusionModel()
        before = confusion.distance("ቀን", "ዘን", model)
        for _ in range(20):
            model.observe("ቀ", "ዘ")
        after = confusion.distance("ቀን", "ዘን", model)
        assert after < before, "the model should learn from what the user actually fixes"

    def test_learning_from_an_alignment(self):
        model = ConfusionModel()
        before = confusion.distance("ሰላም", "ሰላሙ", model)
        for _ in range(20):
            model.observe_alignment("ሰላም", "ሰላሙ")
        assert confusion.distance("ሰላም", "ሰላሙ", model) < before

    def test_neighbors_are_returned_cheapest_first(self):
        model = ConfusionModel()
        neighbors = model.neighbors("ቀ", max_cost=0.6, limit=6)
        costs = [c for _ch, c in neighbors]
        assert costs == sorted(costs)
        assert all(c <= 0.6 for c in costs)

    def test_neighbors_excludes_the_character_itself(self):
        assert "ቀ" not in [ch for ch, _c in ConfusionModel().neighbors("ቀ")]


class TestSimilarity:
    def test_identical_strings_are_maximally_similar(self):
        assert confusion.similarity("ሰላም", "ሰላም") == 1.0

    def test_similarity_is_bounded(self):
        assert 0.0 <= confusion.similarity("ሰላም", "ጨጨጨ") <= 1.0
