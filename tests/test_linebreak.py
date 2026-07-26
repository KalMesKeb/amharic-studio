"""Amharic-aware line breaking for the typeset PDF.

Ethiopic punctuation is a legitimate break opportunity and older text may use ፡ instead
of a space, so a wrapper that only knows about U+0020 sets long unbreakable lines.
A fixed-width measure keeps the arithmetic checkable by hand.
"""

from __future__ import annotations

from amharic_studio.core.linebreak import segment, wrap, wrap_paragraphs


def measure(text: str) -> float:
    """One unit per character."""
    return float(len(text))


def rendered(lines) -> list[str]:
    return ["".join(s.text + (" " if s.trailing_space else "") for s in line.segments).rstrip()
            for line in lines]


class TestSegmenting:
    def test_spaces_separate_segments(self):
        assert [s.text for s in segment("ሰላም ለሁሉም")] == ["ሰላም", "ለሁሉም"]

    def test_a_space_is_recorded_on_the_preceding_segment(self):
        segments = segment("ሰላም ለሁሉም")
        assert segments[0].trailing_space
        assert not segments[-1].trailing_space

    def test_ethiopic_punctuation_stays_attached_to_its_word(self):
        # A break before ። would leave the full stop orphaned at the start of a line.
        segments = segment("ሰላም። ለሁሉም")
        assert segments[0].text == "ሰላም።"

    def test_the_word_separator_is_a_break_opportunity(self):
        assert [s.text for s in segment("ሰላም፡ለሁሉም")] == ["ሰላም፡", "ለሁሉም"]

    def test_segmenting_is_lossless(self):
        for text in ["ሰላም ለሁሉም", "ሰላም፡ለሁሉም።", "አንድ ሁለት፣ ሦስት።"]:
            joined = "".join(s.text + (" " if s.trailing_space else "") for s in segment(text))
            assert joined.replace("\u00a0", " ").strip() == text.replace("፡", "፡").strip()

    def test_empty_text_has_no_segments(self):
        assert segment("") == []


class TestWrapping:
    def test_short_text_stays_on_one_line(self):
        assert len(wrap("ሰላም ለሁሉም", 100, measure)) == 1

    def test_text_is_broken_when_it_does_not_fit(self):
        assert len(wrap("ሰላም ለሁሉም ጤና ይስጥልኝ ደህና ናችሁ", 12, measure)) > 1

    def test_no_line_exceeds_the_measure(self):
        lines = wrap("ሰላም ለሁሉም ጤና ይስጥልኝ ደህና ናችሁ ወዳጆቼ", 14, measure)
        for line in lines:
            assert line.width <= 14 or len(line.segments) == 1

    def test_wrapping_loses_no_words(self):
        text = "ሰላም ለሁሉም ጤና ይስጥልኝ ደህና ናችሁ ወዳጆቼ"
        lines = wrap(text, 14, measure)
        assert " ".join(rendered(lines)).split() == text.split()

    def test_only_the_final_line_is_marked_last(self):
        lines = wrap("ሰላም ለሁሉም ጤና ይስጥልኝ ደህና ናችሁ", 12, measure)
        assert [line.is_last for line in lines] == [False] * (len(lines) - 1) + [True]

    def test_a_single_word_longer_than_the_measure_is_not_dropped(self):
        # Better an overlong line than silently losing text.
        lines = wrap("ሰላምለሁሉምጤናይስጥልኝ", 4, measure)
        assert "".join(rendered(lines)).strip() == "ሰላምለሁሉምጤናይስጥልኝ"

    def test_empty_text_wraps_to_nothing(self):
        assert wrap("", 40, measure) == []


class TestParagraphs:
    def test_each_newline_starts_a_new_paragraph(self):
        # By the time text reaches the typesetter, the document builder has already
        # rejoined the scan's per-line breaks, so a surviving newline is deliberate.
        assert len(wrap_paragraphs("አንድ ነው።\nሁለት ነው።", 40, measure)) == 2

    def test_blank_lines_do_not_produce_empty_paragraphs(self):
        assert len(wrap_paragraphs("አንድ ነው።\n\n\nሁለት ነው።", 40, measure)) == 2

    def test_every_paragraph_ends_with_a_last_line(self):
        for lines in wrap_paragraphs("አንድ ነው።\n\nሁለት ሦስት አራት አምስት ስድስት።", 10, measure):
            assert lines[-1].is_last
