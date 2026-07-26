"""The stores are driven from worker threads, so they have to survive being shared.

Every long job in the app — recognition, analysis, QA, export — runs off the UI thread
while the UI keeps reading the same project and lexicon. These tests exist because the
rest of the suite runs single-threaded and happily passed while the real app was
throwing ``SQLite objects created in a thread can only be used in that same thread``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from amharic_studio.core.lexicon import Lexicon
from amharic_studio.core.project import Project, WordBox

WORDS = ("ኢትዮጵያ", "መጽሐፍ", "ትምህርት", "ቋንቋ", "ታሪክ", "ከተማ")


def drain(executor: ThreadPoolExecutor, jobs) -> list:
    """Run everything and re-raise the first failure instead of swallowing it."""
    return [f.result() for f in [executor.submit(j) for j in jobs]]


class TestLexiconAcrossThreads:
    def test_a_worker_can_read_a_lexicon_built_on_the_main_thread(self, lexicon: Lexicon) -> None:
        result: dict[str, bool] = {}

        def worker() -> None:
            result["known"] = lexicon.recognizes("ኢትዮጵያ")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert result["known"]

    def test_a_worker_can_write_and_the_main_thread_sees_it(self, lexicon: Lexicon) -> None:
        word = "ጨረቃዬ"
        assert not lexicon.contains(word)

        thread = threading.Thread(target=lexicon.add, args=(word,), kwargs={"source": "user"})
        thread.start()
        thread.join()

        assert lexicon.contains(word)

    def test_concurrent_writers_do_not_lose_words(self, lexicon: Lexicon) -> None:
        new = [f"ሙከራ{i}" for i in range(60)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            drain(pool, [lambda w=w: lexicon.add(w, source="test") for w in new])
        assert all(lexicon.contains(w) for w in new)

    def test_lookups_stay_correct_while_words_are_added(self, lexicon: Lexicon) -> None:
        """The read caches are invalidated on write; a racing reader must not pin a stale miss."""
        word = "የማይታወቅቃል"
        stop = threading.Event()
        seen: list[bool] = []

        def reader() -> None:
            while not stop.is_set():
                seen.append(lexicon.contains(word))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            lexicon.add(word, source="user")
        finally:
            stop.set()
            thread.join()

        assert lexicon.contains(word)
        assert seen, "the reader never got to run"


class TestProjectAcrossThreads:
    def test_a_worker_can_write_recognition_output(self, project: Project) -> None:
        page = project.add_page(label="p. 1")

        def recognize() -> None:
            project.set_raw_text(page.id, "ሰላም ለሁሉም")
            project.set_words(page.id, [WordBox("ሰላም", 0, 0, 40, 20, 0.9, 0, 0, 0, 3)])

        thread = threading.Thread(target=recognize)
        thread.start()
        thread.join()

        raw, _ = project.get_text(page.id)
        assert raw == "ሰላም ለሁሉም"
        assert [w.text for w in project.get_words(page.id)] == ["ሰላም"]

    def test_pages_added_concurrently_all_get_distinct_indexes(self, project: Project) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pages = drain(pool, [lambda: project.add_page() for _ in range(40)])
        assert len({p.idx for p in pages}) == len(pages)
        assert project.page_count() == len(pages)

    def test_a_reader_never_sees_a_page_mid_rewrite(self, project: Project) -> None:
        """``set_words`` deletes then re-inserts; the gap must not be observable."""
        page = project.add_page()
        boxes = [WordBox(w, i * 50, 0, 40, 20, 0.9, 0, i, 0, 3) for i, w in enumerate(WORDS)]
        project.set_words(page.id, boxes)

        stop = threading.Event()
        counts: list[int] = []

        def reader() -> None:
            while not stop.is_set():
                counts.append(len(project.get_words(page.id)))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for _ in range(30):
                project.set_words(page.id, boxes)
        finally:
            stop.set()
            thread.join()

        assert counts, "the reader never got to run"
        assert set(counts) == {len(boxes)}

    def test_the_bundle_survives_a_reopen_after_threaded_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "Threaded.amproj"
        project = Project.create(path, title="ክር")
        page = project.add_page()
        with ThreadPoolExecutor(max_workers=4) as pool:
            drain(pool, [lambda i=i: project.set_edited_text(page.id, f"ጽሑፍ {i}") for i in range(20)])
        project.close()

        reopened = Project(path)
        try:
            assert reopened.page_count() == 1
            assert reopened.get_edited(page.id).startswith("ጽሑፍ ")
        finally:
            reopened.close()
