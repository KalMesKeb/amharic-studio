"""The suggestion engine.

Two properties matter more than raw recall, and both are about not being wrong:

* clean text produces no suggestions, so the issue list stays worth reading;
* nothing is applied automatically unless it is a verified improvement, because a silent
  wrong "fix" is worse than the OCR error it replaced.
"""

from __future__ import annotations

from amharic_studio.core import suggest
from amharic_studio.core.suggest import IssueKind, Severity, Suggester


def kinds(issues) -> set[IssueKind]:
    return {i.kind for i in issues}


class TestNoFalseAlarms:
    def test_clean_known_text_is_left_alone(self, suggester: Suggester):
        assert suggester.analyze("ሰላም ለሁሉም።") == []

    def test_inflected_forms_are_not_flagged(self, suggester: Suggester):
        # The single most common false alarm in Amharic: a correct word carrying affixes.
        text = "የኢትዮጵያ ታሪክ በኢትዮጵያን ውስጥ"
        flagged = {i.text for i in suggester.analyze(text) if i.kind is IssueKind.UNKNOWN_WORD}
        assert "የኢትዮጵያ" not in flagged
        assert "በኢትዮጵያን" not in flagged

    def test_empty_text_is_handled(self, suggester: Suggester):
        assert suggester.analyze("") == []

    def test_latin_passages_are_left_to_a_latin_checker(self, suggester: Suggester):
        issues = suggester.analyze("Addis Ababa University Press")
        assert IssueKind.UNKNOWN_WORD not in kinds(issues)


class TestDetection:
    def test_a_nonsense_word_is_flagged(self, suggester: Suggester):
        issues = suggester.analyze("ሰላም ቅቅቅቅ ነው")
        assert any(i.text == "ቅቅቅቅ" for i in issues)

    def test_a_vowel_order_error_gets_the_right_suggestion(self, lexicon, model):
        lexicon.add("ጠይቀው", freq=40)
        issues = Suggester(lexicon, model).analyze("ጣይቃው ነው")
        target = next(i for i in issues if i.text == "ጣይቃው")
        assert target.best is not None
        assert target.best.text == "ጠይቀው"

    def test_mixed_script_inside_one_word_is_flagged(self, suggester: Suggester):
        issues = suggester.analyze("ሰላod ነው")
        assert IssueKind.MIXED_SCRIPT in kinds(issues)

    def test_low_confidence_spans_are_surfaced(self, suggester: Suggester):
        text = "ሰላም ለሁሉም"
        issues = suggester.analyze(text, word_confidences=[(0, 4, 0.21)])
        assert IssueKind.LOW_CONFIDENCE in kinds(issues)

    def test_engine_disagreement_is_surfaced(self, suggester: Suggester):
        text = "ሰላም ለሁሉም"
        issues = suggester.analyze(text, alternates=[(0, 4, ["ሰላም", "ሠላሙ"])])
        assert IssueKind.ENGINE_DISAGREEMENT in kinds(issues)

    def test_issue_offsets_point_at_what_they_describe(self, suggester: Suggester):
        text = "ሰላም ቅቅቅቅ ነው"
        for issue in suggester.analyze(text):
            assert text[issue.start : issue.end] == issue.text


class TestSegmentation:
    def test_a_stray_space_inside_a_word_is_detected(self, lexicon, model):
        lexicon.add("መጽሐፍ", freq=40)
        issues = Suggester(lexicon, model).analyze("መጽ ሐፍ ቅዱስ")
        assert IssueKind.SPLIT_WORD in kinds(issues)

    def test_a_missing_space_between_words_is_detected(self, lexicon, model):
        lexicon.add("ሰላም", freq=40)
        lexicon.add("ለሁሉም", freq=40)
        issues = Suggester(lexicon, model).analyze("ሰላምለሁሉም")
        assert IssueKind.RUN_TOGETHER in kinds(issues)


class TestAutoApplySafety:
    def test_a_word_already_in_the_lexicon_is_never_auto_replaced(self, lexicon, model):
        # The regression this guards: an engine offers a visually similar rival reading
        # for a word that is already correct, and it gets applied without anyone looking.
        text = "ሰላም ለሁሉም"
        issues = Suggester(lexicon, model).analyze(
            text, alternates=[(0, 3, ["ሰላም", "ሠላሙ"])]
        )
        corrected, applied = suggest.auto_apply(text, issues)
        assert corrected == text
        assert applied == []

    def test_unverified_suggestions_are_never_auto_applied(self, suggester: Suggester):
        issues = suggester.analyze("ሰላም ቅቅቅቅ ነው")
        for issue in issues:
            if issue.auto_applicable:
                assert issue.best.verified

    def test_auto_apply_returns_the_text_unchanged_when_nothing_qualifies(self, suggester):
        text = "ሰላም ለሁሉም።"
        corrected, applied = suggest.auto_apply(text, suggester.analyze(text))
        assert corrected == text and applied == []


class TestApplying:
    def test_apply_issue_substitutes_at_the_right_offsets(self, suggester: Suggester):
        text = "ሰላም ቅቅቅቅ ነው"
        issue = next(i for i in suggester.analyze(text) if i.text == "ቅቅቅቅ")
        assert suggest.apply_issue(text, issue, "ሰላም") == "ሰላም ሰላም ነው"

    def test_applying_several_issues_keeps_offsets_consistent(self, suggester: Suggester):
        # Replacements of different lengths must not corrupt the offsets of later ones.
        text = "ቅቅቅቅ ሰላም ዠዠዠዠዠዠ"
        issues = [i for i in suggester.analyze(text) if i.kind is IssueKind.UNKNOWN_WORD]
        assert len(issues) == 2
        result = suggest.apply_all(text, issues, {0: "ሀ", 1: "ሁለት ሦስት"})
        assert result == "ሀ ሰላም ሁለት ሦስት"


class TestRanking:
    def test_issues_come_back_in_document_order(self, suggester: Suggester):
        issues = suggester.analyze("ቅቅቅቅ ሰላም ዠዠዠዠ")
        assert [i.start for i in issues] == sorted(i.start for i in issues)

    def test_severity_is_assigned(self, suggester: Suggester):
        for issue in suggester.analyze("ሰላም ቅቅቅቅ ነው"):
            assert isinstance(issue.severity, Severity)
