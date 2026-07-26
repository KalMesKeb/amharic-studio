"""The .amproj container.

This is the only thing standing between a user and losing a month of correction work, so
the tests care mostly about durability: what is written comes back, the raw OCR is never
overwritten by an edit, and every change leaves a trace in the ledger.
"""

from __future__ import annotations

from pathlib import Path

from amharic_studio.core.project import (
    PageStatus,
    Project,
    StructureMark,
    WordBox,
)


def add_page(project: Project, idx: int = 0, label: str = "") -> int:
    return project.add_page(idx=idx, label=label or f"page_{idx + 1}").id


class TestContainer:
    def test_create_makes_a_directory_bundle(self, tmp_path: Path):
        path = tmp_path / "Book.amproj"
        with Project.create(path, title="ርዕስ", author="ደራሲ") as project:
            assert project.title == "ርዕስ"
        assert path.is_dir()

    def test_metadata_survives_reopening(self, tmp_path: Path):
        path = tmp_path / "Book.amproj"
        Project.create(path, title="ርዕስ", author="ደራሲ").close()
        with Project(path) as reopened:
            assert reopened.title == "ርዕስ"
            assert reopened.author == "ደራሲ"

    def test_amharic_metadata_round_trips_exactly(self, tmp_path: Path):
        path = tmp_path / "Book.amproj"
        title = "የኢትዮጵያ ታሪክ ፲፱፻፸፪"
        Project.create(path, title=title).close()
        with Project(path) as reopened:
            assert reopened.title == title


class TestPages:
    def test_pages_come_back_in_order(self, project: Project):
        for i in range(5):
            add_page(project, i)
        assert [p.idx for p in project.pages()] == [0, 1, 2, 3, 4]

    def test_get_page_by_id(self, project: Project):
        page_id = add_page(project, 0, "አንድ")
        assert project.get_page(page_id).label == "አንድ"

    def test_get_page_returns_none_for_a_missing_id(self, project: Project):
        assert project.get_page(9999) is None

    def test_status_is_recorded(self, project: Project):
        page_id = add_page(project)
        project.set_page_status(page_id, PageStatus.RECOGNIZED)
        assert project.get_page(page_id).status is PageStatus.RECOGNIZED


class TestText:
    def test_raw_text_round_trips(self, project: Project):
        page_id = add_page(project)
        project.set_raw_text(page_id, "ሰላም ለሁሉም")
        raw, edited = project.get_text(page_id)
        assert raw == "ሰላም ለሁሉም"
        # The editable copy starts life as a duplicate, so the user edits real text
        # rather than an empty pane.
        assert edited == "ሰላም ለሁሉም"

    def test_re_recognizing_can_leave_the_edited_copy_alone(self, project: Project):
        page_id = add_page(project)
        project.set_raw_text(page_id, "ሰላም")
        project.set_edited_text(page_id, "ሠላም ተስተካክሏል")
        project.set_raw_text(page_id, "ሰላም ለሁሉም", seed_edited=False)
        raw, edited = project.get_text(page_id)
        assert raw == "ሰላም ለሁሉም"
        assert edited == "ሠላም ተስተካክሏል", "re-running OCR must not discard corrections"

    def test_re_recognizing_an_untouched_page_updates_what_you_see(self, project: Project):
        """Re-running OCR has to change the editor, or it looks like it did nothing.

        Seeding the working copy only when it is empty is not enough: after the first
        recognition it is never empty again. A book imported with a junk text layer would
        keep showing the junk no matter how many times it was recognized.
        """
        page_id = add_page(project)
        project.set_raw_text(page_id, "Scanned by CamScanner")
        project.set_raw_text(page_id, "ሰላም ለሁሉም")
        raw, edited = project.get_text(page_id)
        assert raw == "ሰላም ለሁሉም"
        assert edited == "ሰላም ለሁሉም"

    def test_re_recognizing_a_corrected_page_keeps_the_correction(self, project: Project):
        page_id = add_page(project)
        project.set_raw_text(page_id, "ሰላም")
        project.set_edited_text(page_id, "ሠላም ተስተካክሏል")
        project.set_raw_text(page_id, "ሰላም ለሁሉም")
        raw, edited = project.get_text(page_id)
        assert raw == "ሰላም ለሁሉም"
        assert edited == "ሠላም ተስተካክሏል", "re-running OCR must not discard corrections"

    def test_an_edit_does_not_destroy_the_original_ocr(self, project: Project):
        # The scan-versus-text comparison and the training data both depend on this.
        page_id = add_page(project)
        project.set_raw_text(page_id, "ሰላም ለሁሉም")
        project.set_edited_text(page_id, "ሰላም ለሁላችሁም")
        raw, edited = project.get_text(page_id)
        assert raw == "ሰላም ለሁሉም"
        assert edited == "ሰላም ለሁላችሁም"

    def test_get_edited_falls_back_to_raw(self, project: Project):
        page_id = add_page(project)
        project.set_raw_text(page_id, "ሰላም")
        assert project.get_edited(page_id) == "ሰላም"

    def test_text_survives_reopening(self, tmp_path: Path):
        path = tmp_path / "Book.amproj"
        with Project.create(path) as project:
            page_id = project.add_page(idx=0, label="p1").id
            project.set_raw_text(page_id, "ሰላም")
            project.set_edited_text(page_id, "ሠላም")
        with Project(path) as reopened:
            assert reopened.get_text(page_id) == ("ሰላም", "ሠላም")


class TestGeometry:
    def test_word_boxes_round_trip(self, project: Project):
        page_id = add_page(project)
        boxes = [
            WordBox("ሰላም", 10, 20, 100, 40, conf=0.91, line_idx=0, word_idx=0, start=0, end=3),
            WordBox("ለሁሉም", 120, 20, 120, 40, conf=0.55, line_idx=0, word_idx=1, start=4, end=8),
        ]
        project.set_words(page_id, boxes, engine="tesseract/amh")
        stored = project.get_words(page_id)
        assert [b.text for b in stored] == ["ሰላም", "ለሁሉም"]
        assert stored[0].rect == (10, 20, 100, 40)
        assert stored[1].conf == 0.55

    def test_replacing_word_boxes_does_not_accumulate(self, project: Project):
        page_id = add_page(project)
        project.set_words(page_id, [WordBox("ሀ", 0, 0, 1, 1)], engine="e")
        project.set_words(page_id, [WordBox("ለ", 0, 0, 1, 1)], engine="e")
        assert [b.text for b in project.get_words(page_id)] == ["ለ"]

    def test_the_engine_is_recorded_on_the_page(self, project: Project):
        page_id = add_page(project)
        project.set_words(page_id, [WordBox("ሀ", 0, 0, 1, 1)], engine="tesseract/amh")
        assert project.get_page(page_id).engine == "tesseract/amh"


class TestStructure:
    def test_marks_round_trip(self, project: Project):
        page_id = add_page(project)
        project.set_raw_text(page_id, "ምዕራፍ አንድ\nጽሑፍ")
        project.add_structure(StructureMark(page_id, 0, 9, "chapter", 1, "ምዕራፍ አንድ"))
        marks = project.get_structure(page_id)
        assert len(marks) == 1
        assert marks[0].kind == "chapter"
        assert marks[0].label == "ምዕራፍ አንድ"

    def test_a_mark_can_be_deleted(self, project: Project):
        page_id = add_page(project)
        project.add_structure(StructureMark(page_id, 0, 4, "heading", 1, "ራስ"))
        mark = project.get_structure(page_id)[0]
        project.delete_structure(mark.id)
        assert project.get_structure(page_id) == []


class TestLedger:
    def test_an_edit_is_recorded(self, project: Project):
        page_id = add_page(project)
        project.record_edit(page_id, "ሠላም", "ሰላም", "accepted", "suggestion")
        history = project.edit_history()
        assert len(history) == 1
        assert history[0]["before"] == "ሠላም"
        assert history[0]["after"] == "ሰላም"

    def test_history_is_capped(self, project: Project):
        page_id = add_page(project)
        for i in range(20):
            project.record_edit(page_id, f"ሀ{i}", f"ለ{i}", "accepted", "manual")
        assert len(project.edit_history(limit=5)) == 5


class TestStats:
    def test_counts_reflect_what_was_stored(self, project: Project):
        for i in range(3):
            page_id = add_page(project, i)
            project.set_raw_text(page_id, "ሰላም ለሁሉም")
            project.set_words(page_id, [WordBox("ሰላም", 0, 0, 1, 1)], engine="e")
            if i < 2:
                project.set_page_status(page_id, PageStatus.RECOGNIZED)
        stats = project.stats()
        assert stats.pages == 3
        assert stats.recognized == 2
        assert stats.words == 3

    def test_verified_fraction_of_an_empty_project_is_zero(self, project: Project):
        assert project.stats().verified_fraction == 0.0
