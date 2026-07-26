"""Deciding whether a PDF's text layer is a transcription or a scanner's signature.

Scanned books are routinely stamped — "Scanned by CamScanner", a Telegram channel
handle, a library watermark. That stamp is real text in the PDF, and taking its presence
as proof the book has been OCR'd imports two lines of ASCII as the transcription of a
whole book, marks every page recognized, and quietly stops OCR from ever running.
"""

from __future__ import annotations

from amharic_studio.core import importers
from amharic_studio.core.importers import (
    TextLayerProbe,
    _drop_boilerplate_words,
    _ethiopic_ratio,
    _find_boilerplate,
    _sample_indices,
    _strip_boilerplate,
)
from amharic_studio.core.ocr.base import OcrWord

STAMP = "Scanned by CamScanner\nt.me/OLDBOOKSPDF"

_LINES = [
    "ሸክላ ተሰብሮ ፍልጥ ጥርሱ ሲደማ",
    "ውሽማዬ ቁጭ ብላ በጐን ታለቅሳለች",
    "እንዳለቀ ሻማ ፍቅር ሲያቅማማ",
    "አይኗ ተቀዶ ጥቁር ውሃ ሲያፈስ",
    "እንዳፍታቃሪ ሁሉ ወጥቼ ከሜዳ",
    "ጥንታዊት ሀገር ናት ብዙ ሕዝብ በውስጧ ይኖራል",
    "የአክሱም መንግሥት ታላቅ ነበር ነገሥታቱም ብዙ ሠሩ",
    "መጽሐፍ ቅዱስ በግዕዝ ቋንቋ ተጻፈ ተማሪዎቹ አነበቡት",
]


def amharic_pages(n: int = 8) -> list[str]:
    """Pages that differ from each other, the way pages of a real book do."""
    return [
        "\n".join(f"{line} {i}" for line in _LINES[i % len(_LINES) :] + _LINES[: i % len(_LINES)])
        for i in range(n)
    ]


AMHARIC_PAGE = "\n".join(_LINES)


def probe(pages: list[str]) -> TextLayerProbe:
    """Run the real detector over canned page text, with no PDF involved."""

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeDoc:
        def __init__(self, texts: list[str]) -> None:
            self.pages = [FakePage(t) for t in texts]

        def __getitem__(self, i: int) -> FakePage:
            return self.pages[i]

    original = importers._page_text
    importers._page_text = lambda page: page.text
    try:
        return importers._probe_text_layer(
            FakeDoc(pages), list(range(len(pages))), len(pages)
        )
    finally:
        importers._page_text = original


class TestEthiopicRatio:
    def test_amharic_is_all_ethiopic(self):
        assert _ethiopic_ratio("ሰላም ለሁሉም") == 1.0

    def test_a_scanner_stamp_has_none(self):
        assert _ethiopic_ratio(STAMP) == 0.0

    def test_digits_and_punctuation_do_not_vote(self):
        assert _ethiopic_ratio("ገጽ 17 ፡ ።") == 1.0

    def test_empty_text_is_not_a_division_by_zero(self):
        assert _ethiopic_ratio("123 ...") == 0.0


class TestBoilerplate:
    def test_a_line_on_every_page_is_boilerplate(self):
        found = _find_boilerplate([STAMP] * 6)
        assert set(found) == {"Scanned by CamScanner", "t.me/OLDBOOKSPDF"}

    def test_a_line_on_one_page_is_not(self):
        pages = [STAMP, STAMP, STAMP, STAMP + "\nመቅድም"]
        assert "መቅድም" not in _find_boilerplate(pages)

    def test_too_few_pages_to_tell(self):
        assert _find_boilerplate([STAMP, STAMP]) == ()

    def test_stripping_leaves_the_real_text(self):
        page = f"Scanned by CamScanner\n{AMHARIC_PAGE}\nt.me/OLDBOOKSPDF"
        cleaned = _strip_boilerplate(page, ["Scanned by CamScanner", "t.me/OLDBOOKSPDF"])
        assert "CamScanner" not in cleaned
        assert "t.me" not in cleaned
        assert "ጥንታዊት ሀገር ናት ብዙ ሕዝብ በውስጧ ይኖራል" in cleaned


class TestVerdict:
    def test_a_stamped_scan_is_sent_to_ocr(self):
        """The bug this file exists for: 78 pages of images behind a two-line stamp."""
        result = probe([STAMP] * 8)
        assert not result.usable
        assert "stamp" in result.reason

    def test_a_real_amharic_transcription_is_used(self):
        result = probe(amharic_pages())
        assert result.usable
        assert result.ethiopic_ratio > 0.9

    def test_a_transcription_is_still_used_when_it_is_stamped_too(self):
        result = probe([f"Scanned by CamScanner\n{p}" for p in amharic_pages()])
        assert result.usable
        assert "Scanned by CamScanner" in result.boilerplate

    def test_identical_sampled_pages_are_still_judged_on_their_content(self):
        """With no contrast between pages, every line looks like furniture."""
        result = probe([AMHARIC_PAGE] * 8)
        assert result.usable

    def test_a_latin_text_layer_is_not_a_transcription_of_an_amharic_page(self):
        latin = [
            "\n".join(f"This is line {j} of English page {i}, with plenty of text on it."
                      for j in range(12))
            for i in range(8)
        ]
        result = probe(latin)
        assert not result.usable
        assert "Ethiopic" in result.reason

    def test_a_page_number_alone_is_not_a_transcription(self):
        result = probe([f"{i}" for i in range(8)])
        assert not result.usable

    def test_no_pages_is_not_a_crash(self):
        assert not probe([]).usable

    def test_the_old_character_count_rule_would_have_been_fooled(self):
        """Guards the actual regression rather than just the new behaviour."""
        pages = [STAMP] * 8
        assert sum(len(p) for p in pages[:5]) > 40  # what the old rule measured
        assert not probe(pages).usable


class TestSampling:
    def test_a_short_book_is_read_whole(self):
        assert _sample_indices(5, sample=8) == [0, 1, 2, 3, 4]

    def test_a_long_book_is_sampled_across_its_span(self):
        picks = _sample_indices(800, sample=8)
        assert len(picks) == 8
        assert picks[0] == 0
        # Front matter is the least representative part of a scan, so the sample must
        # not be just the first few pages.
        assert picks[-1] > 600

    def test_no_index_runs_off_the_end(self):
        for count in (1, 2, 7, 9, 78, 1231):
            assert max(_sample_indices(count)) < count


class TestWordFiltering:
    @staticmethod
    def words(lines: list[list[str]]) -> list[OcrWord]:
        out = []
        for line_idx, line in enumerate(lines):
            for word_idx, text in enumerate(line):
                out.append(
                    OcrWord(text, x=word_idx * 50, y=line_idx * 30, w=40, h=20,
                            conf=1.0, line_idx=line_idx, word_idx=word_idx)
                )
        return out

    def test_a_stamped_line_is_dropped_whole(self):
        words = self.words([["Scanned", "by", "CamScanner"], ["ሰላም", "ለሁሉም"]])
        kept = _drop_boilerplate_words(words, ["Scanned by CamScanner"])
        assert [w.text for w in kept] == ["ሰላም", "ለሁሉም"]

    def test_line_numbering_stays_dense_afterwards(self):
        words = self.words([["Scanned", "by", "CamScanner"], ["ሰላም"], ["ለሁሉም"]])
        kept = _drop_boilerplate_words(words, ["Scanned by CamScanner"])
        assert sorted({w.line_idx for w in kept}) == [0, 1]

    def test_nothing_to_drop_leaves_the_words_alone(self):
        words = self.words([["ሰላም", "ለሁሉም"]])
        assert _drop_boilerplate_words(words, []) == words
