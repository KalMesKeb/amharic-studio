"""The ``.amproj`` project container.

A project is a *directory* rather than a single file. Books run to hundreds of pages of
scans, and a bundle keeps the images as ordinary files on disk — recoverable with a file
manager if anything ever goes wrong with the index — while SQLite holds the text, the
word geometry, the structure marks and the correction ledger.

    Book.amproj/
        project.db      index, text, geometry, ledger
        images/         page scans as imported
        thumbs/         cached thumbnails for the page strip
        lines/          cropped line images, harvested as OCR training data
        models/         book-specific fine-tuned recognizer
        exports/        generated PDF and EPUB

Two things here are load-bearing for the rest of the app. Every page keeps both the
original recognizer output and the edited text, so the diff view can always show what
changed. And every accepted correction is written to the ``edits`` ledger, which is what
the confusion model learns from and what the batch "apply everywhere" pass reads.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .normalize import Policy

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".amproj"

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    idx         INTEGER NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    image_rel   TEXT,
    thumb_rel   TEXT,
    width       INTEGER DEFAULT 0,
    height      INTEGER DEFAULT 0,
    dpi         INTEGER DEFAULT 0,
    rotation    INTEGER DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'new',
    engine      TEXT DEFAULT '',
    mean_conf   REAL DEFAULT 0.0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pages_idx ON pages(idx);

CREATE TABLE IF NOT EXISTS page_text (
    page_id     INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
    raw_text    TEXT NOT NULL DEFAULT '',
    edited_text TEXT NOT NULL DEFAULT '',
    version     INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL
);

-- Word geometry: the link between a character offset in the text and a rectangle on the
-- scan. This is what makes click-to-locate work in both directions.
CREATE TABLE IF NOT EXISTS words (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    line_idx INTEGER NOT NULL DEFAULT 0,
    word_idx INTEGER NOT NULL DEFAULT 0,
    start    INTEGER NOT NULL DEFAULT -1,
    end      INTEGER NOT NULL DEFAULT -1,
    text     TEXT NOT NULL DEFAULT '',
    x        INTEGER NOT NULL DEFAULT 0,
    y        INTEGER NOT NULL DEFAULT 0,
    w        INTEGER NOT NULL DEFAULT 0,
    h        INTEGER NOT NULL DEFAULT 0,
    conf     REAL NOT NULL DEFAULT 0.0,
    engine   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_words_page ON words(page_id, line_idx, word_idx);

CREATE TABLE IF NOT EXISTS lines (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id   INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    line_idx  INTEGER NOT NULL,
    x         INTEGER NOT NULL DEFAULT 0,
    y         INTEGER NOT NULL DEFAULT 0,
    w         INTEGER NOT NULL DEFAULT 0,
    h         INTEGER NOT NULL DEFAULT 0,
    baseline  INTEGER NOT NULL DEFAULT 0,
    text      TEXT NOT NULL DEFAULT '',
    conf      REAL NOT NULL DEFAULT 0.0,
    image_rel TEXT
);
CREATE INDEX IF NOT EXISTS idx_lines_page ON lines(page_id, line_idx);

-- Structural marks drive the EPUB table of contents and let exports drop running heads.
CREATE TABLE IF NOT EXISTS structure (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    start   INTEGER NOT NULL,
    end     INTEGER NOT NULL,
    kind    TEXT NOT NULL,
    level   INTEGER NOT NULL DEFAULT 1,
    label   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_structure_page ON structure(page_id, start);

-- Correction ledger: training data for the confusion model and the source of truth for
-- document-wide propagation of a fix.
CREATE TABLE IF NOT EXISTS edits (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER REFERENCES pages(id) ON DELETE SET NULL,
    ts      REAL NOT NULL,
    before  TEXT NOT NULL,
    after   TEXT NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'manual',
    source  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_edits_before ON edits(before);

-- Verified (line image, text) pairs harvested for in-book recognizer fine-tuning.
CREATE TABLE IF NOT EXISTS ground_truth (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id    INTEGER REFERENCES pages(id) ON DELETE CASCADE,
    line_id    INTEGER REFERENCES lines(id) ON DELETE CASCADE,
    image_rel  TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class PageStatus(str, Enum):
    NEW = "new"  # imported, not yet recognized
    RECOGNIZED = "recognized"  # OCR run, untouched by a human
    IN_REVIEW = "in_review"  # partially corrected
    VERIFIED = "verified"  # human-confirmed; eligible as training data
    SKIPPED = "skipped"  # blank or intentionally excluded


@dataclass
class Page:
    id: int
    idx: int
    label: str = ""
    image_rel: str | None = None
    thumb_rel: str | None = None
    width: int = 0
    height: int = 0
    dpi: int = 0
    rotation: int = 0
    status: PageStatus = PageStatus.NEW
    engine: str = ""
    mean_conf: float = 0.0

    @property
    def has_image(self) -> bool:
        return bool(self.image_rel)


@dataclass
class WordBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 0.0
    line_idx: int = 0
    word_idx: int = 0
    start: int = -1
    end: int = -1
    id: int | None = None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class LineBox:
    line_idx: int
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    conf: float = 0.0
    baseline: int = 0
    image_rel: str | None = None
    id: int | None = None


@dataclass
class StructureMark:
    page_id: int
    start: int
    end: int
    kind: str  # chapter | heading | subheading | footnote | verse | running_head | page_number
    level: int = 1
    label: str = ""
    id: int | None = None


STRUCTURE_KINDS = (
    "chapter",
    "heading",
    "subheading",
    "paragraph",
    "verse",
    "footnote",
    "running_head",
    "page_number",
    "caption",
    "quote",
)

#: Marks that describe page furniture rather than content, and are dropped from a
#: reflowed export.
FURNITURE_KINDS = frozenset({"running_head", "page_number"})


@dataclass
class ProjectStats:
    pages: int = 0
    recognized: int = 0
    verified: int = 0
    words: int = 0
    ground_truth_lines: int = 0
    edits: int = 0

    @property
    def verified_fraction(self) -> float:
        return self.verified / self.pages if self.pages else 0.0


# --------------------------------------------------------------------------------------


class Project:
    """Open or create a project bundle."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix != PROJECT_SUFFIX:
            self.path = self.path.with_suffix(PROJECT_SUFFIX)
        self.path.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "thumbs", "lines", "models", "exports"):
            (self.path / sub).mkdir(exist_ok=True)

        self._db = self.path / "project.db"
        self._local = threading.local()
        self._open_connections: list[sqlite3.Connection] = []
        self.lock = threading.RLock()

        self.conn.executescript(_SCHEMA)
        self._init_meta()

    # -- lifecycle -----------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection to the bundle.

        Recognition, import, export and QA all run on worker threads while the UI keeps
        reading the same project, so one shared connection will not do: statements issued
        on a shared connection share its transaction, which means a reader can observe a
        writer's half-finished work — a page in the instant between ``DELETE FROM words``
        and the re-insert looks like a page with no words at all. A connection per thread
        gives each one its own transaction, and WAL lets readers carry on against the last
        committed state while a write is in flight.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # Concurrent writers serialise on the file rather than failing outright.
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
            with self.lock:
                self._open_connections.append(conn)
        return conn

    @classmethod
    def create(
        cls,
        path: str | Path,
        title: str = "",
        author: str = "",
        policy: Policy = Policy.FAITHFUL,
    ) -> Project:
        project = cls(path)
        project.set_meta("title", title or project.path.stem)
        project.set_meta("author", author)
        project.set_meta("policy", policy.value)
        return project

    def close(self) -> None:
        self.conn.commit()
        with self.lock:
            connections, self._open_connections = self._open_connections, []
        for conn in connections:
            # Worker threads are gone by the time a project closes, so their connections
            # have to be reaped from here.
            conn.close()
        self._local = threading.local()

    def __enter__(self) -> Project:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _init_meta(self) -> None:
        defaults = {
            "schema_version": str(SCHEMA_VERSION),
            "title": self.path.stem,
            "author": "",
            "language": "am",
            "policy": Policy.FAITHFUL.value,
            "created_at": str(time.time()),
        }
        for key, value in defaults.items():
            self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    # -- metadata ------------------------------------------------------------------------

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self.conn.commit()

    @property
    def title(self) -> str:
        return self.get_meta("title")

    @property
    def author(self) -> str:
        return self.get_meta("author")

    @property
    def policy(self) -> Policy:
        return Policy(self.get_meta("policy", Policy.FAITHFUL.value))

    def get_settings(self) -> dict:
        raw = self.get_meta("settings", "{}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def update_settings(self, **kwargs: object) -> dict:
        settings = self.get_settings()
        settings.update(kwargs)
        self.set_meta("settings", json.dumps(settings, ensure_ascii=False))
        return settings

    # -- pages ---------------------------------------------------------------------------

    def add_page(
        self,
        image_path: str | Path | None = None,
        label: str = "",
        idx: int | None = None,
        copy_image: bool = True,
        width: int = 0,
        height: int = 0,
        dpi: int = 0,
    ) -> Page:
        """Add a page, optionally copying its scan into the bundle.

        ``image_path`` may be None for the text-only repair path, where a book exists as
        bad OCR output with no usable scans.
        """
        now = time.time()
        # Claim the index and the row first so two importer threads cannot land on the
        # same page number; the scan is copied afterwards, outside the lock, because a
        # large file copy would otherwise stall every reader.
        with self.lock:
            if idx is None:
                row = self.conn.execute(
                    "SELECT COALESCE(MAX(idx), -1) + 1 AS n FROM pages"
                ).fetchone()
                idx = int(row["n"])
            cursor = self.conn.execute(
                """
                INSERT INTO pages(idx, label, image_rel, width, height, dpi, status,
                                  created_at, updated_at)
                VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (idx, label or f"p. {idx + 1}", width, height, dpi,
                 PageStatus.NEW.value, now, now),
            )
            page_id = int(cursor.lastrowid)
            self.conn.execute(
                "INSERT INTO page_text(page_id, raw_text, edited_text, updated_at)"
                " VALUES (?, '', '', ?)",
                (page_id, now),
            )
            self.conn.commit()

        if image_path is not None:
            source = Path(image_path)
            if copy_image:
                target = self.path / "images" / f"{idx:05d}{source.suffix.lower()}"
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                image_rel = f"images/{target.name}"
            else:
                image_rel = str(source)
            with self.lock:
                self.conn.execute(
                    "UPDATE pages SET image_rel = ? WHERE id = ?", (image_rel, page_id)
                )
                self.conn.commit()

        return self.get_page(page_id)  # type: ignore[return-value]

    def get_page(self, page_id: int) -> Page | None:
        row = self.conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        return _page_from_row(row) if row else None

    def page_at(self, idx: int) -> Page | None:
        row = self.conn.execute("SELECT * FROM pages WHERE idx = ?", (idx,)).fetchone()
        return _page_from_row(row) if row else None

    def pages(self) -> list[Page]:
        rows = self.conn.execute("SELECT * FROM pages ORDER BY idx").fetchall()
        return [_page_from_row(r) for r in rows]

    def iter_pages(self) -> Iterator[Page]:
        for row in self.conn.execute("SELECT * FROM pages ORDER BY idx"):
            yield _page_from_row(row)

    def page_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0])

    def image_path(self, page: Page) -> Path | None:
        if not page.image_rel:
            return None
        candidate = Path(page.image_rel)
        return candidate if candidate.is_absolute() else self.path / page.image_rel

    def set_page_status(self, page_id: int, status: PageStatus) -> None:
        self.conn.execute(
            "UPDATE pages SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, time.time(), page_id),
        )
        self.conn.commit()

    def update_page_geometry(self, page_id: int, width: int, height: int, dpi: int = 0) -> None:
        self.conn.execute(
            "UPDATE pages SET width = ?, height = ?, dpi = ?, updated_at = ? WHERE id = ?",
            (width, height, dpi, time.time(), page_id),
        )
        self.conn.commit()

    def delete_page(self, page_id: int) -> None:
        self.conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))
        self.conn.commit()

    def reorder_page(self, page_id: int, new_idx: int) -> None:
        page = self.get_page(page_id)
        if page is None or page.idx == new_idx:
            return
        with self.lock:
            # Park the moving page outside the range so the unique index never collides.
            self.conn.execute("UPDATE pages SET idx = -1 WHERE id = ?", (page_id,))
            if new_idx < page.idx:
                self.conn.execute(
                    "UPDATE pages SET idx = idx + 1 WHERE idx >= ? AND idx < ?", (new_idx, page.idx)
                )
            else:
                self.conn.execute(
                    "UPDATE pages SET idx = idx - 1 WHERE idx > ? AND idx <= ?", (page.idx, new_idx)
                )
            self.conn.execute("UPDATE pages SET idx = ? WHERE id = ?", (new_idx, page_id))
            self.conn.commit()

    # -- text ----------------------------------------------------------------------------

    def get_text(self, page_id: int) -> tuple[str, str]:
        """Return ``(raw_text, edited_text)`` — the recognizer output and the current text."""
        row = self.conn.execute(
            "SELECT raw_text, edited_text FROM page_text WHERE page_id = ?", (page_id,)
        ).fetchone()
        return (row["raw_text"], row["edited_text"]) if row else ("", "")

    def get_edited(self, page_id: int) -> str:
        raw, edited = self.get_text(page_id)
        return edited or raw

    def set_raw_text(self, page_id: int, text: str, seed_edited: bool = True) -> None:
        """Store recognizer output. The edited copy starts as a duplicate of it."""
        now = time.time()
        with self.lock:
            row = self.conn.execute(
                "SELECT edited_text FROM page_text WHERE page_id = ?", (page_id,)
            ).fetchone()
            edited = (
                text
                if (seed_edited and (row is None or not row["edited_text"]))
                else row["edited_text"]
            )
            self.conn.execute(
                """
                INSERT INTO page_text(page_id, raw_text, edited_text, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    raw_text = excluded.raw_text,
                    edited_text = excluded.edited_text,
                    updated_at = excluded.updated_at
                """,
                (page_id, text, edited, now),
            )
            self.conn.commit()

    def set_edited_text(self, page_id: int, text: str, bump_status: bool = True) -> None:
        now = time.time()
        with self.lock:
            self.conn.execute(
                """
                UPDATE page_text SET edited_text = ?, version = version + 1, updated_at = ?
                WHERE page_id = ?
                """,
                (text, now, page_id),
            )
            if bump_status:
                row = self.conn.execute(
                    "SELECT status FROM pages WHERE id = ?", (page_id,)
                ).fetchone()
                if row and row["status"] in (PageStatus.NEW.value, PageStatus.RECOGNIZED.value):
                    self.conn.execute(
                        "UPDATE pages SET status = ?, updated_at = ? WHERE id = ?",
                        (PageStatus.IN_REVIEW.value, now, page_id),
                    )
            self.conn.commit()

    def full_text(self, separator: str = "\n\n", skip_furniture: bool = False) -> str:
        parts: list[str] = []
        for page in self.iter_pages():
            if page.status is PageStatus.SKIPPED:
                continue
            text = self.get_edited(page.id)
            if skip_furniture:
                text = self.strip_furniture(page.id, text)
            if text.strip():
                parts.append(text)
        return separator.join(parts)

    # -- word and line geometry ------------------------------------------------------------

    def set_words(self, page_id: int, boxes: Sequence[WordBox], engine: str = "") -> None:
        with self.lock:
            self.conn.execute("DELETE FROM words WHERE page_id = ?", (page_id,))
            self.conn.executemany(
                """
                INSERT INTO words(page_id, line_idx, word_idx, start, end, text,
                                  x, y, w, h, conf, engine)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (page_id, b.line_idx, b.word_idx, b.start, b.end, b.text,
                     b.x, b.y, b.w, b.h, b.conf, engine)
                    for b in boxes
                ],
            )
            if boxes:
                mean_conf = sum(b.conf for b in boxes) / len(boxes)
                self.conn.execute(
                    "UPDATE pages SET mean_conf = ?, engine = ? WHERE id = ?",
                    (mean_conf, engine, page_id),
                )
            self.conn.commit()

    def get_words(self, page_id: int) -> list[WordBox]:
        rows = self.conn.execute(
            "SELECT * FROM words WHERE page_id = ? ORDER BY line_idx, word_idx", (page_id,)
        ).fetchall()
        return [
            WordBox(r["text"], r["x"], r["y"], r["w"], r["h"], r["conf"],
                    r["line_idx"], r["word_idx"], r["start"], r["end"], r["id"])
            for r in rows
        ]

    def word_at_offset(self, page_id: int, offset: int) -> WordBox | None:
        """Find the word box covering a character offset — click in text, locate on scan."""
        row = self.conn.execute(
            "SELECT * FROM words WHERE page_id = ? AND start <= ? AND end > ? LIMIT 1",
            (page_id, offset, offset),
        ).fetchone()
        if row is None:
            return None
        return WordBox(row["text"], row["x"], row["y"], row["w"], row["h"], row["conf"],
                       row["line_idx"], row["word_idx"], row["start"], row["end"], row["id"])

    def set_lines(self, page_id: int, lines: Sequence[LineBox]) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM lines WHERE page_id = ?", (page_id,))
            self.conn.executemany(
                """
                INSERT INTO lines(page_id, line_idx, x, y, w, h, baseline, text, conf, image_rel)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (page_id, ln.line_idx, ln.x, ln.y, ln.w, ln.h, ln.baseline,
                     ln.text, ln.conf, ln.image_rel)
                    for ln in lines
                ],
            )
            self.conn.commit()

    def get_lines(self, page_id: int) -> list[LineBox]:
        rows = self.conn.execute(
            "SELECT * FROM lines WHERE page_id = ? ORDER BY line_idx", (page_id,)
        ).fetchall()
        return [
            LineBox(r["line_idx"], r["x"], r["y"], r["w"], r["h"], r["text"],
                    r["conf"], r["baseline"], r["image_rel"], r["id"])
            for r in rows
        ]

    # -- structure -------------------------------------------------------------------------

    def add_structure(self, mark: StructureMark) -> int:
        cursor = self.conn.execute(
            "INSERT INTO structure(page_id, start, end, kind, level, label) VALUES (?, ?, ?, ?, ?, ?)",
            (mark.page_id, mark.start, mark.end, mark.kind, mark.level, mark.label),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_structure(self, page_id: int) -> list[StructureMark]:
        rows = self.conn.execute(
            "SELECT * FROM structure WHERE page_id = ? ORDER BY start", (page_id,)
        ).fetchall()
        return [
            StructureMark(r["page_id"], r["start"], r["end"], r["kind"], r["level"],
                          r["label"], r["id"])
            for r in rows
        ]

    def delete_structure(self, mark_id: int) -> None:
        self.conn.execute("DELETE FROM structure WHERE id = ?", (mark_id,))
        self.conn.commit()

    def strip_furniture(self, page_id: int, text: str) -> str:
        """Remove running heads and page numbers from a page's text."""
        marks = [m for m in self.get_structure(page_id) if m.kind in FURNITURE_KINDS]
        for mark in sorted(marks, key=lambda m: m.start, reverse=True):
            text = text[: mark.start] + text[mark.end :]
        return "\n".join(line for line in text.split("\n") if line.strip())

    def toc(self) -> list[tuple[int, int, str]]:
        """Chapter and heading marks as ``(page_idx, level, label)``, in reading order."""
        rows = self.conn.execute(
            """
            SELECT p.idx AS page_idx, s.level AS level, s.label AS label, s.kind AS kind
            FROM structure s JOIN pages p ON p.id = s.page_id
            WHERE s.kind IN ('chapter', 'heading', 'subheading')
            ORDER BY p.idx, s.start
            """
        ).fetchall()
        return [(r["page_idx"], r["level"], r["label"]) for r in rows]

    # -- correction ledger ------------------------------------------------------------------

    def record_edit(
        self, page_id: int | None, before: str, after: str, kind: str = "manual", source: str = ""
    ) -> None:
        if before == after:
            return
        self.conn.execute(
            "INSERT INTO edits(page_id, ts, before, after, kind, source) VALUES (?, ?, ?, ?, ?, ?)",
            (page_id, time.time(), before, after, kind, source),
        )
        self.conn.commit()

    def edit_history(self, limit: int = 500) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edits ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()

    def known_corrections(self, min_count: int = 1) -> dict[str, tuple[str, int]]:
        """Corrections applied at least ``min_count`` times, for propagation across a book."""
        rows = self.conn.execute(
            """
            SELECT before, after, COUNT(*) AS n FROM edits
            WHERE kind IN ('manual', 'accepted')
            GROUP BY before, after HAVING n >= ?
            ORDER BY n DESC
            """,
            (min_count,),
        ).fetchall()
        out: dict[str, tuple[str, int]] = {}
        for row in rows:
            if row["before"] not in out:
                out[row["before"]] = (row["after"], int(row["n"]))
        return out

    def occurrences(self, needle: str) -> list[tuple[int, int, int]]:
        """Every ``(page_id, page_idx, offset)`` where ``needle`` appears in edited text."""
        out: list[tuple[int, int, int]] = []
        for page in self.iter_pages():
            text = self.get_edited(page.id)
            start = text.find(needle)
            while start != -1:
                out.append((page.id, page.idx, start))
                start = text.find(needle, start + 1)
        return out

    # -- ground truth ------------------------------------------------------------------------

    def add_ground_truth(self, page_id: int, line_id: int | None, image_rel: str, text: str) -> None:
        self.conn.execute(
            "INSERT INTO ground_truth(page_id, line_id, image_rel, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (page_id, line_id, image_rel, text, time.time()),
        )
        self.conn.commit()

    def ground_truth_pairs(self) -> list[tuple[Path, str]]:
        rows = self.conn.execute("SELECT image_rel, text FROM ground_truth").fetchall()
        return [(self.path / r["image_rel"], r["text"]) for r in rows]

    # -- statistics ----------------------------------------------------------------------------

    def stats(self) -> ProjectStats:
        def scalar(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0])

        return ProjectStats(
            pages=scalar("SELECT COUNT(*) FROM pages"),
            recognized=scalar(
                "SELECT COUNT(*) FROM pages WHERE status != 'new'"
            ),
            verified=scalar("SELECT COUNT(*) FROM pages WHERE status = 'verified'"),
            words=scalar("SELECT COUNT(*) FROM words"),
            ground_truth_lines=scalar("SELECT COUNT(*) FROM ground_truth"),
            edits=scalar("SELECT COUNT(*) FROM edits"),
        )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        id=row["id"],
        idx=row["idx"],
        label=row["label"],
        image_rel=row["image_rel"],
        thumb_rel=row["thumb_rel"],
        width=row["width"],
        height=row["height"],
        dpi=row["dpi"],
        rotation=row["rotation"],
        status=PageStatus(row["status"]),
        engine=row["engine"] or "",
        mean_conf=row["mean_conf"] or 0.0,
    )
