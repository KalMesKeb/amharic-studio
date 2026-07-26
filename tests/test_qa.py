"""Quality assessment and the worst-pages-first review queue.

The scoring exists to answer one question: which page should the user open next? So the
tests are mostly about ordering — a bad page must rank worse than a good one — rather
than about any particular score.
"""

from __future__ import annotations

from amharic_studio.core import qa
from amharic_studio.core.project import PageStatus, Project, WordBox

GOOD = "ሰላም ለሁሉም ሰው ይሁን። ኢትዮጵያ ሀገር ናት።"
BAD = "ቅቅቅቅ ዠዠዠዠ ጨጨጨ ኰኰኰ ፐፐፐ"


def make_page(project: Project, idx: int, text: str, conf: float = 0.9) -> int:
    page = project.add_page(idx=idx, label=f"page_{idx + 1}")
    project.set_raw_text(page.id, text)
    words = [
        WordBox(w, i * 60, 0, 50, 30, conf, 0, i) for i, w in enumerate(text.split())
    ]
    project.set_words(page.id, words, engine="tesseract/amh")
    project.set_page_status(page.id, PageStatus.RECOGNIZED)
    return page.id


class TestPageMetrics:
    def test_clean_text_has_a_low_unknown_rate(self, project, lexicon):
        page_id = make_page(project, 0, GOOD)
        assert qa.assess_page(project, page_id, lexicon).unknown_rate < 0.3

    def test_gibberish_has_a_high_unknown_rate(self, project, lexicon):
        page_id = make_page(project, 0, BAD)
        assert qa.assess_page(project, page_id, lexicon).unknown_rate > 0.8

    def test_amharic_text_is_reported_as_ethiopic(self, project, lexicon):
        page_id = make_page(project, 0, GOOD)
        assert qa.assess_page(project, page_id, lexicon).ethiopic_ratio > 0.9

    def test_latin_contamination_lowers_the_ethiopic_ratio(self, project, lexicon):
        page_id = make_page(project, 0, "ሰላም hello world foo bar")
        assert qa.assess_page(project, page_id, lexicon).ethiopic_ratio < 0.6

    def test_confidence_is_carried_through_from_the_recognizer(self, project, lexicon):
        page_id = make_page(project, 0, GOOD, conf=0.42)
        assert abs(qa.assess_page(project, page_id, lexicon).mean_confidence - 0.42) < 0.01

    def test_word_and_character_counts_are_reported(self, project, lexicon):
        page_id = make_page(project, 0, GOOD)
        quality = qa.assess_page(project, page_id, lexicon)
        assert quality.words == len(GOOD.split())
        assert quality.characters == len(GOOD)

    def test_an_empty_page_does_not_crash_the_metrics(self, project, lexicon):
        page = project.add_page(idx=0, label="blank")
        quality = qa.assess_page(project, page.id, lexicon)
        assert quality.words == 0
        assert quality.unknown_rate == 0.0


class TestSuspectCharacters:
    def test_non_amharic_ethiopic_letters_are_flagged(self, project, lexicon):
        page_id = make_page(project, 0, "ሰላም ⶀⶀⶀ ነው")
        assert qa.assess_page(project, page_id, lexicon).suspect_characters

    def test_clean_amharic_flags_nothing(self, project, lexicon):
        page_id = make_page(project, 0, GOOD)
        assert not qa.assess_page(project, page_id, lexicon).suspect_characters


class TestScoring:
    def test_a_good_page_scores_better_than_a_bad_one(self, project, lexicon):
        good = qa.assess_page(project, make_page(project, 0, GOOD), lexicon)
        bad = qa.assess_page(project, make_page(project, 1, BAD, conf=0.3), lexicon)
        assert good.score > bad.score

    def test_scores_are_bounded(self, project, lexicon):
        for i, text in enumerate([GOOD, BAD]):
            quality = qa.assess_page(project, make_page(project, i, text), lexicon)
            assert 0.0 <= quality.score <= 1.0

    def test_every_page_gets_a_grade_and_a_description(self, project, lexicon):
        quality = qa.assess_page(project, make_page(project, 0, GOOD), lexicon)
        assert quality.grade
        assert quality.describe()


class TestBookReport:
    def test_the_review_queue_puts_the_worst_page_first(self, project, lexicon):
        make_page(project, 0, GOOD)
        bad_id = make_page(project, 1, BAD, conf=0.3)
        make_page(project, 2, GOOD)
        report = qa.assess_book(project, lexicon)
        assert report.worst_pages()[0].page_id == bad_id

    def test_totals_add_up_across_pages(self, project, lexicon):
        make_page(project, 0, GOOD)
        make_page(project, 1, GOOD)
        report = qa.assess_book(project, lexicon)
        assert report.total_words == len(GOOD.split()) * 2
        assert len(report.pages) == 2

    def test_the_character_histogram_covers_the_book(self, project, lexicon):
        make_page(project, 0, "ሰላም")
        report = qa.assess_book(project, lexicon)
        assert report.character_histogram["ሰ"] == 1

    def test_impossible_characters_are_collected_for_the_dashboard(self, project, lexicon):
        make_page(project, 0, "ሰላም ⶀⶀ ነው")
        assert qa.assess_book(project, lexicon).impossible_characters()

    def test_verified_pages_are_counted(self, project, lexicon):
        page_id = make_page(project, 0, GOOD)
        make_page(project, 1, GOOD)
        project.set_page_status(page_id, PageStatus.VERIFIED)
        assert qa.assess_book(project, lexicon).verified_pages == 1

    def test_an_empty_book_reports_nothing_rather_than_dividing_by_zero(
        self, project, lexicon
    ):
        report = qa.assess_book(project, lexicon)
        assert report.pages == []
        assert report.mean_score == 0.0
        assert report.summary()

    def test_progress_is_reported_for_every_page(self, project, lexicon):
        for i in range(3):
            make_page(project, i, GOOD)
        seen: list[tuple[int, int, str]] = []
        qa.assess_book(project, lexicon, progress=lambda i, n, label: seen.append((i, n, label)))
        # One call per page, then a final call so a progress bar reaches the end.
        assert [i for i, _n, _label in seen] == [0, 1, 2, 3]
        assert all(n == 3 for _i, n, _label in seen)
