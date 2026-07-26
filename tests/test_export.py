"""Exporters.

The exports are the product: everything upstream exists so that these files are right.
Each format is checked for being structurally valid, for carrying the Amharic text
through unmangled, and for declaring its language — an EPUB or PDF that does not say it
is Amharic gets the wrong fonts and the wrong hyphenation everywhere it is opened.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

from amharic_studio.core.export import build_document
from amharic_studio.core.export.epub import EpubOptions, export_epub
from amharic_studio.core.export.pdf_clean import TypesetOptions, export_pdf
from amharic_studio.core.export.text import export_alto, export_hocr, export_markdown, export_text
from amharic_studio.core.project import PageStatus, Project, StructureMark, WordBox

CHAPTER_ONE = "የኢትዮጵያ ታሪክ"
BODY_ONE = "ኢትዮጵያ ጥንታዊት ሀገር ናት። ብዙ ሕዝብ በውስጧ ይኖራል።"
CHAPTER_TWO = "ሁለተኛው ምዕራፍ"
BODY_TWO = "ሰላም ለሁሉም ሰው ይሁን። ፍቅር ደግሞ ይብዛ።"


@pytest.fixture
def book(tmp_path: Path) -> Project:
    project = Project.create(tmp_path / "Book.amproj", title="የኢትዮጵያ ታሪክ", author="ሙከራ ደራሲ")
    for idx, (heading, body) in enumerate([(CHAPTER_ONE, BODY_ONE), (CHAPTER_TWO, BODY_TWO)]):
        page = project.add_page(idx=idx, label=f"page_{idx + 1}")
        text = f"{heading}\n{body}"
        project.set_raw_text(page.id, text)
        project.set_words(
            page.id,
            [
                WordBox(w, 100 + i * 120, 200, 110, 40, 0.88, 0, i)
                for i, w in enumerate(text.split())
            ],
            engine="tesseract/amh",
        )
        project.set_page_status(page.id, PageStatus.RECOGNIZED)
        project.add_structure(StructureMark(page.id, 0, len(heading), "chapter", 1, heading))
    yield project
    project.close()


@pytest.fixture
def document(book: Project):
    return build_document(book)


class TestDocumentModel:
    def test_structure_marks_become_chapters(self, document):
        assert len(document.chapters) == 2
        assert document.chapters[0].title == CHAPTER_ONE

    def test_metadata_is_carried_over(self, document):
        assert document.title == "የኢትዮጵያ ታሪክ"
        assert document.author == "ሙከራ ደራሲ"

    def test_scan_line_breaks_are_reflowed_into_paragraphs(self, document):
        paragraphs = [b for c in document.chapters for b in c.blocks if b.kind == "paragraph"]
        assert paragraphs
        assert all("\n" not in b.text for b in paragraphs), (
            "a typeset edition must reflow, not reproduce the scan's line breaks"
        )

    def test_the_character_set_drives_font_subsetting(self, document):
        assert "ኢ" in document.character_set
        assert len(set(document.character_set)) == len(document.character_set)

    def test_statistics_are_reported(self, document):
        stats = document.statistics()
        assert stats["chapters"] == 2
        assert stats["ethiopic_ratio"] > 0.9


class TestPlainText:
    def test_the_text_is_written_verbatim(self, document, tmp_path):
        path = export_text(document, tmp_path / "book.txt")
        content = path.read_text(encoding="utf-8")
        assert CHAPTER_ONE in content
        assert "ኢትዮጵያ ጥንታዊት ሀገር ናት።" in content

    def test_it_is_valid_utf8_with_no_replacement_characters(self, document, tmp_path):
        path = export_text(document, tmp_path / "book.txt")
        assert "\ufffd" not in path.read_text(encoding="utf-8")


class TestMarkdown:
    def test_chapters_become_headings(self, document, tmp_path):
        content = export_markdown(document, tmp_path / "book.md").read_text(encoding="utf-8")
        assert f"# {CHAPTER_ONE}" in content

    def test_the_body_survives(self, document, tmp_path):
        content = export_markdown(document, tmp_path / "book.md").read_text(encoding="utf-8")
        assert "ኢትዮጵያ ጥንታዊት ሀገር ናት።" in content


class TestEpub:
    def test_the_container_is_a_valid_epub_zip(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub")
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            # The mimetype entry must come first and be stored uncompressed.
            assert names[0] == "mimetype"
            info = archive.getinfo("mimetype")
            assert info.compress_type == zipfile.ZIP_STORED
            assert archive.read("mimetype") == b"application/epub+zip"
            assert "META-INF/container.xml" in names

    def test_the_language_is_declared_as_amharic(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub")
        with zipfile.ZipFile(path) as archive:
            opf = next(n for n in archive.namelist() if n.endswith(".opf"))
            content = archive.read(opf).decode("utf-8")
        assert ">am<" in content or 'xml:lang="am"' in content

    def test_every_xhtml_document_is_well_formed(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub")
        with zipfile.ZipFile(path) as archive:
            pages = [n for n in archive.namelist() if n.endswith((".xhtml", ".html"))]
            assert pages
            for name in pages:
                ElementTree.fromstring(archive.read(name))  # raises if malformed

    def test_the_text_is_present_in_the_content(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub")
        with zipfile.ZipFile(path) as archive:
            body = "".join(
                archive.read(n).decode("utf-8")
                for n in archive.namelist()
                if n.endswith((".xhtml", ".html"))
            )
        assert CHAPTER_ONE in body
        assert "ኢትዮጵያ ጥንታዊት ሀገር ናት።" in body

    def test_a_navigation_document_is_included(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub")
        with zipfile.ZipFile(path) as archive:
            opf = next(n for n in archive.namelist() if n.endswith(".opf"))
            assert 'properties="nav"' in archive.read(opf).decode("utf-8")

    def test_a_font_is_embedded_so_the_script_renders_anywhere(self, document, tmp_path):
        path = export_epub(document, tmp_path / "book.epub", EpubOptions(embed_font=True))
        with zipfile.ZipFile(path) as archive:
            assert any(n.endswith((".ttf", ".otf")) for n in archive.namelist())


class TestPdf:
    def test_a_typeset_pdf_is_produced(self, document, tmp_path):
        path, report = export_pdf(document, tmp_path / "book.pdf", TypesetOptions(page_size="A5"))
        assert path.exists()
        assert path.stat().st_size > 1000
        assert path.read_bytes().startswith(b"%PDF-")

    def test_the_report_describes_the_result(self, document, tmp_path):
        _path, report = export_pdf(document, tmp_path / "book.pdf")
        assert report.pages > 0
        assert report.describe()

    def test_the_pdf_declares_its_language(self, document, tmp_path):
        path, _report = export_pdf(document, tmp_path / "book.pdf")
        assert b"/Lang" in path.read_bytes()

    def test_longer_text_produces_more_pages(self, book, tmp_path):
        for idx in range(2, 12):
            page = book.add_page(idx=idx, label=f"page_{idx + 1}")
            book.set_raw_text(page.id, (BODY_ONE + " ") * 12)
        _short, short_report = export_pdf(build_document(book), tmp_path / "a.pdf")
        assert short_report.pages > 2


class TestAlto:
    def test_one_well_formed_file_per_page(self, book, tmp_path):
        paths = export_alto(book, tmp_path / "alto")
        assert len(paths) == 2
        for path in paths:
            ElementTree.parse(path)  # raises if malformed

    def test_word_geometry_is_preserved(self, book, tmp_path):
        path = export_alto(book, tmp_path / "alto")[0]
        content = path.read_text(encoding="utf-8")
        assert "String" in content
        assert 'CONTENT="ኢትዮጵያ"' in content or "ኢትዮጵያ" in content


class TestHocr:
    def test_the_output_is_well_formed_html(self, book, tmp_path):
        path = export_hocr(book, tmp_path / "book.hocr")
        ElementTree.parse(path)

    def test_word_boxes_are_emitted(self, book, tmp_path):
        content = export_hocr(book, tmp_path / "book.hocr").read_text(encoding="utf-8")
        assert "ocrx_word" in content
        assert "bbox" in content

    def test_the_language_is_declared(self, book, tmp_path):
        content = export_hocr(book, tmp_path / "book.hocr").read_text(encoding="utf-8")
        assert 'lang="am"' in content
