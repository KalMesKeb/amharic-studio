"""Dockable side panels: pages, issues, quality, structure and the diff view."""

from __future__ import annotations

import html
from difflib import SequenceMatcher

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import fidel
from ..core.project import Page, PageStatus, Project, StructureMark
from ..core.qa import BookQuality, PageQuality
from ..core.suggest import Issue
from . import theme

# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------


class PageListPanel(QWidget):
    """Thumbnail strip with a status dot per page."""

    pageSelected = Signal(int)  # page id

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItems(
            ["All pages", "Not recognized", "Needs review", "Verified", "Worst first"]
        )
        controls.addWidget(self.filter, 1)
        layout.addLayout(controls)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.ListMode)
        self.list.setIconSize(QSize(52, 70))
        self.list.setSpacing(1)
        self.list.setUniformItemSizes(False)
        self.list.currentItemChanged.connect(self._on_selection)
        layout.addWidget(self.list, 1)

        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(self.summary)

        self._quality: dict[int, PageQuality] = {}
        self._pages: list[Page] = []
        self.filter.currentIndexChanged.connect(lambda _i: self.rebuild())

    def set_pages(self, project: Project, quality: dict[int, PageQuality] | None = None) -> None:
        self._project = project
        self._pages = project.pages()
        self._quality = quality or {}
        self.rebuild()

    def rebuild(self) -> None:
        mode = self.filter.currentText()
        pages = list(self._pages)

        if mode == "Not recognized":
            pages = [p for p in pages if p.status is PageStatus.NEW]
        elif mode == "Needs review":
            pages = [p for p in pages if p.status in (PageStatus.RECOGNIZED, PageStatus.IN_REVIEW)]
        elif mode == "Verified":
            pages = [p for p in pages if p.status is PageStatus.VERIFIED]
        elif mode == "Worst first":
            pages = sorted(pages, key=lambda p: self._quality.get(p.id, _NO_QUALITY).score)

        current_id = self.current_page_id()
        self.list.blockSignals(True)
        self.list.clear()
        for page in pages:
            item = QListWidgetItem(self._label_for(page))
            item.setData(Qt.ItemDataRole.UserRole, page.id)
            item.setIcon(self._icon_for(page))
            item.setToolTip(self._tooltip_for(page))
            self.list.addItem(item)
            if page.id == current_id:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)

        verified = sum(1 for p in self._pages if p.status is PageStatus.VERIFIED)
        self.summary.setText(f"{len(self._pages)} pages · {verified} verified")

    def _label_for(self, page: Page) -> str:
        quality = self._quality.get(page.id)
        if quality and quality.words:
            return f"{page.label}\n{quality.grade} · {quality.unknown_rate:.0%} unknown"
        return f"{page.label}\n{page.status.value.replace('_', ' ')}"

    def _tooltip_for(self, page: Page) -> str:
        quality = self._quality.get(page.id)
        lines = [page.label, f"status: {page.status.value}"]
        if page.engine:
            lines.append(f"engine: {page.engine}")
        if quality:
            lines.append(f"quality: {quality.score:.0%} ({quality.grade})")
            lines.extend(f"⚠ {flag}" for flag in quality.flags)
        return "\n".join(lines)

    def _icon_for(self, page: Page) -> QIcon:
        thumb_path = None
        if page.thumb_rel:
            candidate = self._project.path / page.thumb_rel
            if candidate.exists():
                thumb_path = str(candidate)

        pixmap = QPixmap(thumb_path) if thumb_path else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(52, 70)
            pixmap.fill(QColor(self.palette_.surface_alt))
        else:
            pixmap = pixmap.scaled(
                52, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )

        painter = QPainter(pixmap)
        painter.setBrush(theme.status_color(self.palette_, page.status.value))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(3, 3, 8, 8)
        painter.end()
        return QIcon(pixmap)

    def current_page_id(self) -> int | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def select_page(self, page_id: int) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == page_id:
                self.list.setCurrentItem(item)
                self.list.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
                return

    def _on_selection(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self.pageSelected.emit(current.data(Qt.ItemDataRole.UserRole))


_NO_QUALITY = PageQuality(page_id=-1, page_idx=-1, label="")


# --------------------------------------------------------------------------------------
# Issues
# --------------------------------------------------------------------------------------


class IssuesPanel(QWidget):
    """Flagged spans on the current page, grouped by severity."""

    issueActivated = Signal(object)  # Issue
    acceptRequested = Signal(object)  # Issue
    acceptAllRequested = Signal()
    ignoreRequested = Signal(object)

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Text", "Suggestion", "Why"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self.accept_button = QPushButton("Accept selected")
        self.accept_button.clicked.connect(self._accept_selected)
        buttons.addWidget(self.accept_button)

        self.accept_all_button = QPushButton("Accept all confident")
        self.accept_all_button.setToolTip(
            "Apply only unambiguous fixes: a dictionary-verified candidate, clearly ahead "
            "of the runner-up, replacing something that is not already a word."
        )
        self.accept_all_button.clicked.connect(self.acceptAllRequested.emit)
        buttons.addWidget(self.accept_all_button)
        layout.addLayout(buttons)

        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(self.summary)

    def set_issues(self, issues: list[Issue]) -> None:
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        counts = {"high": 0, "medium": 0, "low": 0}

        for issue in issues:
            severity = issue.severity.value
            counts[severity] = counts.get(severity, 0) + 1
            if severity not in groups:
                parent = QTreeWidgetItem([severity.upper(), "", ""])
                parent.setForeground(0, theme.severity_color(self.palette_, severity))
                parent.setExpanded(True)
                self.tree.addTopLevelItem(parent)
                groups[severity] = parent

            best = issue.best
            item = QTreeWidgetItem(
                [
                    issue.text if len(issue.text) < 28 else issue.text[:26] + "…",
                    best.text if best else "—",
                    best.explanation if best else issue.message,
                ]
            )
            item.setFont(0, theme.ethiopic_font(11))
            item.setFont(1, theme.ethiopic_font(11))
            item.setData(0, Qt.ItemDataRole.UserRole, issue)
            item.setToolTip(0, issue.message)
            if best and not best.verified:
                item.setForeground(1, QColor(self.palette_.text_muted))
            groups[severity].addChild(item)

        for severity, parent in groups.items():
            parent.setText(0, f"{severity.upper()}  ({parent.childCount()})")

        auto = sum(1 for i in issues if i.auto_applicable)
        self.summary.setText(
            f"{len(issues)} issues · {counts.get('high', 0)} high · {auto} safe to auto-apply"
        )
        self.accept_all_button.setEnabled(auto > 0)

    def _selected_issue(self) -> Issue | None:
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        issue = item.data(0, Qt.ItemDataRole.UserRole)
        if issue is not None:
            self.issueActivated.emit(issue)

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        issue = item.data(0, Qt.ItemDataRole.UserRole)
        if issue is not None and issue.best is not None:
            self.acceptRequested.emit(issue)

    def _accept_selected(self) -> None:
        issue = self._selected_issue()
        if issue is not None and issue.best is not None:
            self.acceptRequested.emit(issue)


# --------------------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------------------


class QualityPanel(QWidget):
    """Book-level metrics and the worst-first review queue."""

    pageRequested = Signal(int)
    refreshRequested = Signal()

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.headline = QLabel("Run an assessment to see quality metrics.")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.headline)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("quality %p%")
        layout.addWidget(self.bar)

        layout.addWidget(_section_label("Review queue — worst pages first", palette))
        self.queue = QTreeWidget()
        self.queue.setHeaderLabels(["Page", "Quality", "Unknown", "Flags"])
        self.queue.itemActivated.connect(self._on_activated)
        self.queue.itemClicked.connect(self._on_activated)
        self.queue.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.queue, 2)

        layout.addWidget(_section_label("Suspicious characters", palette))
        self.characters = QTreeWidget()
        self.characters.setHeaderLabels(["Character", "Count", "Note"])
        self.characters.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.characters, 1)

        refresh = QPushButton("Reassess book")
        refresh.clicked.connect(self.refreshRequested.emit)
        layout.addWidget(refresh)

    def set_quality(self, book: BookQuality) -> None:
        self.headline.setText(book.summary())
        self.bar.setValue(int(book.mean_score * 100))

        self.queue.clear()
        for quality in book.worst_pages(30):
            item = QTreeWidgetItem(
                [
                    quality.label,
                    f"{quality.score:.0%}",
                    f"{quality.unknown_rate:.0%}",
                    ", ".join(quality.flags) or "—",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, quality.page_id)
            item.setForeground(1, _grade_color(self.palette_, quality.grade))
            self.queue.addTopLevelItem(item)

        self.characters.clear()
        for ch, count in book.impossible_characters()[:20]:
            item = QTreeWidgetItem([ch, str(count), "outside the Amharic range"])
            item.setFont(0, theme.ethiopic_font(13))
            item.setForeground(0, QColor(self.palette_.high))
            self.characters.addTopLevelItem(item)
        for ch, count in book.rare_characters()[:20]:
            item = QTreeWidgetItem([ch, str(count), _rare_note(ch)])
            item.setFont(0, theme.ethiopic_font(13))
            self.characters.addTopLevelItem(item)

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        page_id = item.data(0, Qt.ItemDataRole.UserRole)
        if page_id is not None:
            self.pageRequested.emit(page_id)


def _rare_note(ch: str) -> str:
    entry = fidel.decompose(ch)
    if entry is None:
        return "rare"
    family, order = entry
    return f"rare — {family.label}, {fidel.ORDERS[order].amharic} order"


def _grade_color(palette: theme.Palette, grade: str) -> QColor:
    return QColor(
        {
            "excellent": palette.verified,
            "good": palette.low,
            "needs review": palette.medium,
            "poor": palette.high,
        }.get(grade, palette.text)
    )


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


class StructurePanel(QWidget):
    """Mark headings, verse, footnotes and page furniture on the current page."""

    markRequested = Signal(str)  # kind
    markActivated = Signal(object)  # StructureMark
    deleteRequested = Signal(int)  # mark id

    KINDS = [
        ("chapter", "Chapter"),
        ("heading", "Heading"),
        ("subheading", "Subheading"),
        ("verse", "Verse / ቅኔ"),
        ("quote", "Quotation"),
        ("caption", "Caption"),
        ("footnote", "Footnote"),
        ("running_head", "Running head"),
        ("page_number", "Page number"),
    ]

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        hint = QLabel("Select text in the editor, then mark what it is.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(hint)

        grid = QHBoxLayout()
        self.kind_box = QComboBox()
        for kind, label in self.KINDS:
            self.kind_box.addItem(label, kind)
        grid.addWidget(self.kind_box, 1)
        add = QPushButton("Mark")
        add.clicked.connect(lambda: self.markRequested.emit(self.kind_box.currentData()))
        grid.addWidget(add)
        layout.addLayout(grid)

        note = QLabel(
            "Running heads and page numbers are dropped from reflowed exports "
            "but kept in the searchable PDF."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {palette.text_muted}; font-size: 7.5pt;")
        layout.addWidget(note)

        self.list = QListWidget()
        self.list.itemClicked.connect(self._on_clicked)
        layout.addWidget(self.list, 1)

        remove = QPushButton("Remove selected mark")
        remove.clicked.connect(self._on_remove)
        layout.addWidget(remove)

    def set_marks(self, marks: list[StructureMark], text: str) -> None:
        self.list.clear()
        for mark in marks:
            excerpt = text[mark.start : mark.end].strip().replace("\n", " ")
            if len(excerpt) > 40:
                excerpt = excerpt[:38] + "…"
            item = QListWidgetItem(f"[{mark.kind}]  {excerpt}")
            item.setFont(theme.ethiopic_font(11))
            item.setData(Qt.ItemDataRole.UserRole, mark)
            self.list.addItem(item)

    def _on_clicked(self, item: QListWidgetItem) -> None:
        self.markActivated.emit(item.data(Qt.ItemDataRole.UserRole))

    def _on_remove(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            mark = item.data(Qt.ItemDataRole.UserRole)
            if mark.id is not None:
                self.deleteRequested.emit(mark.id)


# --------------------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------------------


class DiffPanel(QWidget):
    """Word-level diff between the recognizer's output and the current text.

    The reference point that keeps a correction session honest: at any moment you can see
    every change made to a page, which is the difference between an edition and a guess.
    """

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(self.summary)

        self.view = QTextBrowser()
        self.view.setFont(theme.ethiopic_font(13))
        self.view.setOpenExternalLinks(False)
        layout.addWidget(self.view, 1)

    def set_texts(self, original: str, edited: str) -> None:
        original_tokens = _tokens(original)
        edited_tokens = _tokens(edited)
        matcher = SequenceMatcher(a=original_tokens, b=edited_tokens, autojunk=False)

        parts: list[str] = []
        changes = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                parts.append(html.escape("".join(original_tokens[i1:i2])))
                continue
            changes += 1
            if tag in ("replace", "delete"):
                removed = html.escape("".join(original_tokens[i1:i2]))
                parts.append(
                    f'<span style="background:{self.palette_.high}33;'
                    f'text-decoration:line-through;color:{self.palette_.high}">{removed}</span>'
                )
            if tag in ("replace", "insert"):
                added = html.escape("".join(edited_tokens[j1:j2]))
                parts.append(
                    f'<span style="background:{self.palette_.verified}33;'
                    f'color:{self.palette_.verified};font-weight:600">{added}</span>'
                )

        body = "".join(parts).replace("\n", "<br/>")
        self.view.setHtml(
            f'<div style="line-height:1.8; color:{self.palette_.text}">{body}</div>'
        )
        similarity = matcher.ratio()
        self.summary.setText(
            f"{changes} changed span{'s' if changes != 1 else ''} · {similarity:.1%} unchanged"
        )


def _tokens(text: str) -> list[str]:
    return [t.text for t in fidel.tokenize(text)]


# --------------------------------------------------------------------------------------


class SearchPanel(QWidget):
    """Book-wide find, with phonetic entry so queries can be typed in Latin."""

    searchRequested = Signal(str, bool)  # query, whole word
    resultActivated = Signal(int, int)  # page id, offset

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setFont(theme.ethiopic_font(13))
        self.query.setPlaceholderText("Search the book…")
        self.query.returnPressed.connect(self._emit_search)
        row.addWidget(self.query, 1)
        button = QPushButton("Find")
        button.clicked.connect(self._emit_search)
        row.addWidget(button)
        layout.addLayout(row)

        self.phonetic = QLineEdit()
        self.phonetic.setPlaceholderText("…or type phonetically: selam → ሰላም")
        self.phonetic.textChanged.connect(self._on_phonetic)
        layout.addWidget(self.phonetic)

        self.results = QTreeWidget()
        self.results.setHeaderLabels(["Page", "Context"])
        self.results.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results.itemClicked.connect(self._on_result)
        layout.addWidget(self.results, 1)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(self.status)

    def _on_phonetic(self, text: str) -> None:
        from ..core.translit import transliterate

        if text:
            self.query.setText(transliterate(text))

    def _emit_search(self) -> None:
        self.searchRequested.emit(self.query.text(), False)

    def set_results(self, rows: list[tuple[int, str, str, int]]) -> None:
        """``rows`` is ``(page_id, page_label, context, offset)``."""
        self.results.clear()
        for page_id, label, context, offset in rows:
            item = QTreeWidgetItem([label, context])
            item.setFont(1, theme.ethiopic_font(11))
            item.setData(0, Qt.ItemDataRole.UserRole, (page_id, offset))
            self.results.addTopLevelItem(item)
        self.status.setText(f"{len(rows)} match{'es' if len(rows) != 1 else ''}")

    def _on_result(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if payload:
            self.resultActivated.emit(*payload)


def _section_label(text: str, palette: theme.Palette) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {palette.text_muted}; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.6px;"
    )
    return label
