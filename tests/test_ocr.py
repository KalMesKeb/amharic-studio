"""Recognizer plumbing: word geometry, text assembly, and multi-engine voting.

Tesseract itself is not exercised here — these are the parts that decide what the rest of
the program sees, and they must behave the same whether or not an engine is installed.
"""

from __future__ import annotations

import itertools

from amharic_studio.core.ocr.base import (
    OcrResult,
    OcrWord,
    assemble_text,
    lines_from_words,
    split_wordspace_boxes,
)
from amharic_studio.core.ocr.registry import (
    all_backends,
    available_backends,
    default_backend,
    get_backend,
)
from amharic_studio.core.ocr.voting import vote


def word(text: str, x: int, line: int = 0, idx: int = 0, conf: float = 0.9) -> OcrWord:
    return OcrWord(text=text, x=x, y=line * 50, w=len(text) * 20, h=30, conf=conf,
                   line_idx=line, word_idx=idx)


class TestTextAssembly:
    def test_words_join_with_spaces_and_lines_with_newlines(self):
        words = [word("ሰላም", 0, 0, 0), word("ለሁሉም", 80, 0, 1), word("ጤና", 0, 1, 0)]
        assert assemble_text(words) == "ሰላም ለሁሉም\nጤና"

    def test_assembly_records_character_offsets(self):
        words = [word("ሰላም", 0, 0, 0), word("ለሁሉም", 80, 0, 1)]
        text = assemble_text(words)
        for w in words:
            assert text[w.start : w.end] == w.text

    def test_offsets_stay_correct_across_a_line_break(self):
        words = [word("ሰላም", 0, 0, 0), word("ጤና", 0, 1, 0)]
        text = assemble_text(words)
        assert text[words[1].start : words[1].end] == "ጤና"

    def test_no_words_gives_empty_text(self):
        assert assemble_text([]) == ""


class TestWordspaceSplitting:
    def test_a_box_containing_the_ethiopic_word_separator_is_split(self):
        # Tesseract often returns ሰላም፡ለሁሉም as a single box; downstream everything is
        # word-based, so it has to become two.
        boxes = split_wordspace_boxes([OcrWord("ሰላም፡ለሁሉም", 0, 0, 180, 30, 0.9, 0, 0)])
        assert [b.text for b in boxes] == ["ሰላም", "ለሁሉም"]

    def test_the_split_boxes_partition_the_original_width(self):
        original = OcrWord("ሰላም፡ለሁሉም", 100, 0, 180, 30, 0.9, 0, 0)
        boxes = split_wordspace_boxes([original])
        assert boxes[0].x == 100
        assert boxes[-1].x + boxes[-1].w <= original.x + original.w
        for left, right in itertools.pairwise(boxes):
            assert left.x + left.w <= right.x

    def test_confidence_is_inherited_by_both_halves(self):
        boxes = split_wordspace_boxes([OcrWord("ሰላም፡ለሁሉም", 0, 0, 180, 30, 0.42, 0, 0)])
        assert all(b.conf == 0.42 for b in boxes)

    def test_ordinary_words_pass_through_untouched(self):
        original = [word("ሰላም", 0)]
        assert split_wordspace_boxes(original) == original

    def test_a_trailing_separator_does_not_create_an_empty_word(self):
        boxes = split_wordspace_boxes([OcrWord("ሰላም፡", 0, 0, 90, 30, 0.9, 0, 0)])
        assert [b.text for b in boxes] == ["ሰላም"]


class TestLineGeometry:
    def test_a_line_box_encloses_its_words(self):
        words = [word("ሰላም", 0, 0, 0), word("ለሁሉም", 100, 0, 1)]
        line = lines_from_words(words)[0]
        assert line.x <= min(w.x for w in words)
        assert line.x + line.w >= max(w.x + w.w for w in words)

    def test_one_line_box_per_line(self):
        words = [word("ሰላም", 0, 0), word("ጤና", 0, 1), word("ደህና", 0, 2)]
        assert len(lines_from_words(words)) == 3

    def test_line_text_is_the_words_joined(self):
        words = [word("ሰላም", 0, 0, 0), word("ለሁሉም", 100, 0, 1)]
        assert lines_from_words(words)[0].text == "ሰላም ለሁሉም"

    def test_line_confidence_summarizes_its_words(self):
        words = [word("ሰላም", 0, 0, 0, conf=0.4), word("ለሁሉም", 100, 0, 1, conf=0.8)]
        assert 0.4 <= lines_from_words(words)[0].conf <= 0.8


class TestResult:
    def test_label_distinguishes_two_runs_of_one_engine(self):
        # Running Tesseract twice with different models is a normal second opinion; the
        # report must not just say "tesseract" twice.
        a = OcrResult(engine="tesseract", language="amh")
        b = OcrResult(engine="tesseract", language="script/Ethiopic")
        assert a.label != b.label

    def test_mean_confidence_ignores_unscored_words(self):
        result = OcrResult(words=[word("ሀ", 0, conf=0.8), word("ለ", 50, conf=0.0)])
        assert result.mean_confidence == 0.8

    def test_a_result_with_an_error_is_not_ok(self):
        assert not OcrResult(error="Tesseract is not installed").ok

    def test_confidence_spans_skip_words_with_no_offsets(self):
        words = [word("ሀ", 0)]
        assert OcrResult(words=words).confidence_spans() == []


def result(engine: str, texts: list[str], conf: float = 0.9) -> OcrResult:
    words = [word(t, i * 100, 0, i, conf) for i, t in enumerate(texts)]
    text = assemble_text(words)
    return OcrResult(text=text, words=words, engine=engine, language="amh")


class TestVoting:
    def test_one_engine_passes_straight_through(self):
        only = result("tesseract", ["ሰላም", "ለሁሉም"])
        outcome = vote([only])
        assert outcome.result.text == only.text
        assert outcome.agreement_rate == 1.0

    def test_engines_that_agree_produce_no_disagreements(self):
        outcome = vote([result("a", ["ሰላም", "ለሁሉም"]), result("b", ["ሰላም", "ለሁሉም"])])
        assert outcome.disagreements == []
        assert outcome.agreement_rate == 1.0

    def test_a_disagreement_is_reported_as_an_attention_signal(self):
        outcome = vote([result("a", ["ሰላም", "ለሁሉም"]), result("b", ["ሰላም", "ለሁላም"])])
        assert outcome.disagreements
        assert outcome.agreement_rate < 1.0

    def test_the_lexicon_arbitrates_a_disagreement(self, lexicon):
        # Two engines differ; only one of them produced a real word. That one wins even
        # though neither has a majority.
        lexicon.add("ለሁሉም", freq=100)
        outcome = vote(
            [result("a", ["ሰላም", "ለሁላም"]), result("b", ["ሰላም", "ለሁሉም"])], lexicon=lexicon
        )
        assert "ለሁሉም" in outcome.result.text

    def test_failed_engines_are_ignored(self):
        good = result("a", ["ሰላም"])
        outcome = vote([OcrResult(engine="broken", error="not installed"), good])
        assert outcome.result.text == good.text

    def test_all_engines_failing_returns_the_error(self):
        outcome = vote([OcrResult(engine="a", error="boom"), OcrResult(engine="b", error="bang")])
        assert not outcome.result.ok

    def test_no_results_at_all_does_not_crash(self):
        assert not vote([]).result.ok

    def test_the_combined_result_names_every_engine_that_contributed(self):
        outcome = vote([result("a", ["ሰላም"]), result("b", ["ሰላም"])])
        assert len(outcome.engines) == 2


class TestRegistry:
    def test_the_builtin_backend_is_registered(self):
        assert get_backend("tesseract") is not None

    def test_availability_is_reported_without_raising(self):
        # The UI calls this on a machine that may have no engine installed at all.
        for backend in all_backends():
            assert isinstance(backend.available(), bool)

    def test_available_backends_is_a_subset_of_all_backends(self):
        assert set(available_backends()) <= set(all_backends())

    def test_an_unknown_backend_is_none_rather_than_an_error(self):
        assert get_backend("kraken") is None

    def test_default_backend_is_available_if_it_exists(self):
        backend = default_backend()
        assert backend is None or backend.available()
