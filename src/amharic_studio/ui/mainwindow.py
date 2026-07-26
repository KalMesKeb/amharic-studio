"""The main window: layout, project lifecycle and the correction workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QWidget,
)

from .. import APP_NAME, __version__
from ..core import fidel, importers, qa, suggest
from ..core.confusion import ConfusionModel
from ..core.export import build_document
from ..core.export.epub import EpubOptions, export_epub
from ..core.export.pdf_clean import export_pdf
from ..core.export.pdf_searchable import SearchableOptions, export_searchable_pdf
from ..core.export.text import export_alto, export_hocr, export_markdown, export_text
from ..core.lexicon import Lexicon, load_or_create
from ..core.pipeline import Pipeline
from ..core.project import PROJECT_SUFFIX, PageStatus, Project, StructureMark
from ..core.suggest import Issue, Suggester, Suggestion
from ..paths import ensure_dirs
from . import dialogs, theme
from .editor import EditorPane
from .pagecanvas import PageCanvas
from .panels import (
    DiffPanel,
    IssuesPanel,
    PageListPanel,
    QualityPanel,
    SearchPanel,
    StructurePanel,
)
from .workers import WorkerManager

AUTOSAVE_MS = 1200


class MainWindow(QMainWindow):
    projectChanged = Signal()

    def __init__(self, palette: theme.Palette, lexicon: Lexicon | None = None) -> None:
        super().__init__()
        ensure_dirs()

        self.palette_ = palette
        self.settings = QSettings("AmharicStudio", "AmharicStudio")
        self.project: Project | None = None
        self.pipeline: Pipeline | None = None
        self.lexicon: Lexicon = lexicon if lexicon is not None else load_or_create()
        self.model = ConfusionModel()
        self.suggester = Suggester(self.lexicon, self.model)
        self.workers = WorkerManager(self)

        self.current_page_id: int | None = None
        self.current_issues: list[Issue] = []
        self.page_quality: dict[int, qa.PageQuality] = {}
        self._dirty = False
        self._loading = False
        self._dock_sizes: list[tuple[QDockWidget, int, bool]] = []
        self._restored_state = False
        self._layout_applied = False

        self.setWindowTitle(APP_NAME)
        self.resize(1680, 1000)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._restore_geometry()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self._flush_text)

        self._update_enabled_state()

    # ==================================================================================
    # Construction
    # ==================================================================================

    def _build_central(self) -> None:
        self.canvas = PageCanvas(self.palette_)
        self.editor_pane = EditorPane(self.palette_)
        self.editor = self.editor_pane.editor

        self.canvas.setMinimumWidth(260)
        self.editor_pane.setMinimumWidth(300)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.editor_pane)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self.canvas.wordClicked.connect(self._on_canvas_word_clicked)
        self.canvas.wordDoubleClicked.connect(self._on_canvas_word_double_clicked)
        self.editor.offsetChanged.connect(self._on_editor_offset)
        self.editor.textChanged.connect(self._on_text_changed)
        self.editor.issueAccepted.connect(self._on_issue_accepted)
        self.editor.editCommitted.connect(self._on_edit_committed)

    def _build_docks(self) -> None:
        self.pages_panel = PageListPanel(self.palette_)
        self.pages_panel.pageSelected.connect(self.load_page)
        self._add_dock("Pages", self.pages_panel, Qt.DockWidgetArea.LeftDockWidgetArea, 280)

        self.issues_panel = IssuesPanel(self.palette_)
        self.issues_panel.issueActivated.connect(self._on_issue_activated)
        self.issues_panel.acceptRequested.connect(self._accept_issue)
        self.issues_panel.acceptAllRequested.connect(self.accept_all_confident)

        self.quality_panel = QualityPanel(self.palette_)
        self.quality_panel.pageRequested.connect(self.load_page)
        self.quality_panel.refreshRequested.connect(self.assess_book)

        self.structure_panel = StructurePanel(self.palette_)
        self.structure_panel.markRequested.connect(self._add_structure_mark)
        self.structure_panel.markActivated.connect(self._on_mark_activated)
        self.structure_panel.deleteRequested.connect(self._delete_structure_mark)

        self.search_panel = SearchPanel(self.palette_)
        self.search_panel.searchRequested.connect(self.search_book)
        self.search_panel.resultActivated.connect(self._on_search_result)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.issues_panel, "Issues")
        self.right_tabs.addTab(self.quality_panel, "Quality")
        self.right_tabs.addTab(self.structure_panel, "Structure")
        self.right_tabs.addTab(self.search_panel, "Search")
        self._add_dock("Review", self.right_tabs, Qt.DockWidgetArea.RightDockWidgetArea, 400)

        self.diff_panel = DiffPanel(self.palette_)
        self.diff_dock = self._add_dock(
            "Compared with the original OCR",
            self.diff_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
            220,
        )
        self.diff_dock.hide()

    def _add_dock(
        self, title: str, widget: QWidget, area: Qt.DockWidgetArea, size: int
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(area, dock)
        horizontal = area in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        # A hard minimum would let a panel's size hint eat the central area, so the
        # preferred size is applied once via resizeDocks after the window is laid out.
        if horizontal:
            dock.setMinimumWidth(200)
        else:
            dock.setMinimumHeight(140)
        self._dock_sizes.append((dock, size, horizontal))
        return dock

    def _apply_default_layout(self) -> None:
        """Give the scan and the editor the bulk of the window on a first, unsaved run."""
        # The preferred dock widths assume a roomy display. On a smaller screen they would
        # leave the canvas and the editor at their minimums, so scale them down together.
        cap = max(0.35, self.width() / 1600.0)
        horizontal = [(d, int(s * min(1.0, cap))) for d, s, h in self._dock_sizes
                      if h and d.isVisible()]
        vertical = [(d, s) for d, s, h in self._dock_sizes if not h and d.isVisible()]
        if horizontal:
            self.resizeDocks(
                [d for d, _ in horizontal],
                [s for _, s in horizontal],
                Qt.Orientation.Horizontal,
            )
        if vertical:
            self.resizeDocks(
                [d for d, _ in vertical],
                [s for _, s in vertical],
                Qt.Orientation.Vertical,
            )
        # resizeDocks only takes effect on the next layout pass, so the central width is
        # still stale here; split the panes once it is not.
        QTimer.singleShot(0, self._apply_default_split)

    def _apply_default_split(self) -> None:
        width = max(self.splitter.width(), 800)
        canvas = int(width * 0.56)
        self.splitter.setSizes([canvas, width - canvas])

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._layout_applied:
            self._layout_applied = True
            if not self._restored_state:
                QTimer.singleShot(0, self._apply_default_layout)

    def _build_actions(self) -> None:
        def make(text, shortcut=None, handler=None, checkable=False, tip="") -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if handler:
                action.triggered.connect(handler)
            action.setCheckable(checkable)
            if tip:
                action.setToolTip(tip)
                action.setStatusTip(tip)
            return action

        self.act_new = make("New project", "Ctrl+N", self.new_project)
        self.act_open = make("Open project", "Ctrl+O", self.open_project)
        self.act_save = make("Save", "Ctrl+S", self.save_project)
        self.act_close = make("Close project", None, self.close_project)
        self.act_quit = make("Quit", "Ctrl+Q", self.close)

        self.act_import = make("Import pages", "Ctrl+Shift+I", self.import_pages)
        self.act_recognize = make("Recognize", "Ctrl+R", self.recognize_pages)
        self.act_export = make("Export", "Ctrl+Shift+X", self.export_book)

        self.act_next_page = make("Next page", "PgDown", lambda: self.step_page(1))
        self.act_prev_page = make("Previous page", "PgUp", lambda: self.step_page(-1))
        self.act_goto_page = make("Go to page…", "Ctrl+G", self.goto_page)

        self.act_next_issue = make("Next issue", "F8", lambda: self.editor.next_issue())
        self.act_prev_issue = make("Previous issue", "Shift+F8", lambda: self.editor.previous_issue())
        self.act_accept_all = make(
            "Accept all confident", "Ctrl+Shift+A", self.accept_all_confident,
            tip="Apply only unambiguous, dictionary-verified corrections on this page",
        )
        self.act_propagate = make(
            "Apply everywhere", "Ctrl+Shift+E", self.propagate_selection,
            tip="Replace the selected text throughout the book, with a review step first",
        )
        self.act_verify = make(
            "Mark verified", "Ctrl+Return", self.verify_page,
            tip="Confirm this page and harvest its lines as recognizer training data",
        )
        self.act_reanalyze = make("Re-analyze page", "F5", self.analyze_current_page)
        self.act_normalize = make("Normalize page", None, self.normalize_current_page)

        self.act_zoom_in = make("Zoom in", "Ctrl+=", lambda: self.canvas.zoom_by(1.2))
        self.act_zoom_out = make("Zoom out", "Ctrl+-", lambda: self.canvas.zoom_by(1 / 1.2))
        self.act_fit_width = make("Fit width", "Ctrl+0", lambda: self.canvas.set_fit_mode("width"))
        self.act_fit_page = make("Fit page", None, lambda: self.canvas.set_fit_mode("page"))

        self.act_boxes = make("Word boxes", "Ctrl+B", self._toggle_boxes, checkable=True)
        self.act_boxes.setChecked(True)
        self.act_heatmap = make(
            "Confidence heatmap", "Ctrl+H", self._toggle_heatmap, checkable=True,
            tip="Shade each word by how confident the recognizer was",
        )
        self.act_heatmap.setChecked(True)
        self.act_diff = make("Show diff", "Ctrl+D", self._toggle_diff, checkable=True)

        self.act_bigger_text = make("Larger text", "Ctrl+Shift+=", lambda: self._scale_text(1))
        self.act_smaller_text = make("Smaller text", "Ctrl+Shift+-", lambda: self._scale_text(-1))

        self.act_ime = make(
            "Phonetic input", "Ctrl+I", self._toggle_ime, checkable=True,
            tip="Type Amharic phonetically: selam → ሰላም",
        )
        self.act_keyboard_help = make("Keyboard reference", "F1", self.show_keyboard_help)
        self.act_about = make("About", None, self.show_about)

        self.act_theme_light = make("Light", None, lambda: self._set_theme("light"), checkable=True)
        self.act_theme_dark = make("Dark", None, lambda: self._set_theme("dark"), checkable=True)
        theme_group = QActionGroup(self)
        theme_group.addAction(self.act_theme_light)
        theme_group.addAction(self.act_theme_dark)
        (self.act_theme_dark if self.palette_.is_dark else self.act_theme_light).setChecked(True)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Main")
        bar.setObjectName("main_toolbar")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        for action in (self.act_import, self.act_recognize):
            bar.addAction(action)
        bar.addSeparator()
        for action in (self.act_prev_page, self.act_next_page):
            bar.addAction(action)
        bar.addSeparator()
        for action in (self.act_prev_issue, self.act_next_issue, self.act_accept_all):
            bar.addAction(action)
        bar.addSeparator()
        bar.addAction(self.act_verify)
        bar.addSeparator()
        for action in (self.act_boxes, self.act_heatmap, self.act_diff, self.act_ime):
            bar.addAction(action)
        bar.addSeparator()
        bar.addAction(self.act_export)

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        for action in (self.act_new, self.act_open, self.act_save, self.act_close):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.act_import)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("Undo", self.editor.undo, QKeySequence.StandardKey.Undo)
        edit_menu.addAction("Redo", self.editor.redo, QKeySequence.StandardKey.Redo)
        edit_menu.addSeparator()
        for action in (self.act_accept_all, self.act_propagate, self.act_normalize):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_ime)

        view_menu = menubar.addMenu("&View")
        for action in (self.act_zoom_in, self.act_zoom_out, self.act_fit_width, self.act_fit_page):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in (self.act_boxes, self.act_heatmap, self.act_diff):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in (self.act_bigger_text, self.act_smaller_text):
            view_menu.addAction(action)
        theme_menu = view_menu.addMenu("Theme")
        theme_menu.addAction(self.act_theme_light)
        theme_menu.addAction(self.act_theme_dark)

        go_menu = menubar.addMenu("&Go")
        for action in (
            self.act_prev_page, self.act_next_page, self.act_goto_page,
            self.act_prev_issue, self.act_next_issue,
        ):
            go_menu.addAction(action)

        book_menu = menubar.addMenu("&Book")
        book_menu.addAction(self.act_recognize)
        book_menu.addAction(self.act_reanalyze)
        book_menu.addAction(self.act_verify)
        book_menu.addSeparator()
        book_menu.addAction("Reassess quality", self.assess_book)
        book_menu.addAction("Replay correction ledger…", self.replay_ledger)
        book_menu.addAction("Training data status…", self.show_training_status)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_keyboard_help)
        help_menu.addAction(self.act_about)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()

        self.status_page = QLabel("No project")
        bar.addWidget(self.status_page)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        bar.addPermanentWidget(self.progress)

        self.status_quality = QLabel("")
        bar.addPermanentWidget(self.status_quality)

        self.status_engine = QLabel("")
        bar.addPermanentWidget(self.status_engine)

        self._refresh_engine_status()

    def _refresh_engine_status(self) -> None:
        from ..core.ocr import registry

        available = registry.available_backends()
        if not available:
            self.status_engine.setText("⚠ no OCR engine")
            self.status_engine.setToolTip(
                "Install Tesseract with Amharic data to recognize scans. "
                "Correcting existing text works without it."
            )
            return
        names = ", ".join(b.display_name for b in available)
        amharic = any(getattr(b, "has_amharic", lambda: False)() for b in available)
        self.status_engine.setText(f"{names}{'' if amharic else ' (no amh data)'}")

    # ==================================================================================
    # Project lifecycle
    # ==================================================================================

    def new_project(self) -> None:
        dialog = dialogs.NewProjectDialog(self.palette_, self)
        if dialog.exec() != dialogs.QDialog.DialogCode.Accepted:
            return
        path, title, author, policy = dialog.result_values()
        if path.exists():
            QMessageBox.warning(self, "Already exists", f"{path} already exists.")
            return
        self._open(Project.create(path, title, author, policy))
        self.import_pages()

    def open_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, f"Open project ({PROJECT_SUFFIX} folder)", str(Path.home())
        )
        if not directory:
            return
        path = Path(directory)
        if not (path / "project.db").exists():
            QMessageBox.warning(
                self, "Not a project", f"{path.name} does not contain a project.db file."
            )
            return
        self._open(Project(path))

    def _open(self, project: Project) -> None:
        self.close_project()
        self.project = project
        self.pipeline = Pipeline(project, self.lexicon, self.model, self.suggester)
        self.pipeline.load_model()
        self.model = self.pipeline.model
        self.suggester.model = self.model

        self.setWindowTitle(f"{project.title} — {APP_NAME}")
        self.pages_panel.set_pages(project)
        self._update_enabled_state()

        pages = project.pages()
        if pages:
            self.load_page(pages[0].id)
        self.statusBar().showMessage(f"Opened {project.path.name}", 4000)

    def close_project(self) -> None:
        if self.project is None:
            return
        self._flush_text()
        if self.pipeline is not None:
            self.pipeline.save_model()
        self.project.close()
        self.project = None
        self.pipeline = None
        self.current_page_id = None
        self.current_issues = []
        self.canvas.clear()
        self.editor.set_text("")
        self.editor.set_issues([])
        self.setWindowTitle(APP_NAME)
        self._update_enabled_state()

    def save_project(self) -> None:
        self._flush_text()
        if self.pipeline is not None:
            self.pipeline.save_model()
        self.statusBar().showMessage("Saved", 2500)

    def _update_enabled_state(self) -> None:
        has_project = self.project is not None
        has_page = self.current_page_id is not None
        for action in (
            self.act_save, self.act_close, self.act_import, self.act_recognize,
            self.act_export, self.act_goto_page,
        ):
            action.setEnabled(has_project)
        for action in (
            self.act_next_page, self.act_prev_page, self.act_next_issue, self.act_prev_issue,
            self.act_accept_all, self.act_propagate, self.act_verify, self.act_reanalyze,
            self.act_normalize,
        ):
            action.setEnabled(has_page)
        self.editor.setReadOnly(not has_page)

    # ==================================================================================
    # Import and recognition
    # ==================================================================================

    def import_pages(self) -> None:
        if self.project is None:
            return
        dialog = dialogs.ImportDialog(self.palette_, self)
        if dialog.exec() != dialogs.QDialog.DialogCode.Accepted or not dialog.paths:
            return

        project = self.project
        paths = dialog.paths
        options = dialog.preprocess_options()
        dpi = dialog.dpi.value()
        use_text_layer = dialog.text_layer.currentData()
        preprocess = dialog.preprocess.isChecked()

        def task(progress, _cancel):
            return importers.import_any(
                project,
                paths,
                progress=progress,
                dpi=dpi,
                use_text_layer=use_text_layer,
                options=options,
                apply_preprocessing=preprocess,
            )

        self._run(task, "Importing", self._on_import_done)

    def _on_import_done(self, report: importers.ImportReport) -> None:
        if self.project is None:
            return
        self.pages_panel.set_pages(self.project, self.page_quality)
        pages = self.project.pages()
        if pages and self.current_page_id is None:
            self.load_page(pages[0].id)
        self.statusBar().showMessage(report.describe(), 12000)
        if report.warnings:
            QMessageBox.warning(self, "Import warnings", "\n".join(report.warnings[:20]))

        if report.needs_recognition:
            # These pages are images with no text yet, and the point of importing them is
            # to read them. Asking here saves discovering the empty editor page by page.
            detail = report.probe.reason if report.probe is not None else ""
            answer = QMessageBox.question(
                self,
                "Recognize now?",
                f"{report.pages_added} pages were imported as scans."
                + (f"\n\n{detail}" if detail else "")
                + "\n\nRun OCR over them now?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.recognize_pages()

    def recognize_pages(self) -> None:
        if self.project is None or self.pipeline is None:
            return
        dialog = dialogs.RecognizeDialog(self.project.page_count(), self.palette_, self)
        if dialog.exec() != dialogs.QDialog.DialogCode.Accepted:
            return

        options = dialog.options(self.project.policy)
        scope = dialog.scope
        if scope == "page" and self.current_page_id is not None:
            page_ids = [self.current_page_id]
        elif scope == "book":
            page_ids = [p.id for p in self.project.pages()]
        else:
            page_ids = [p.id for p in self.project.pages() if p.status is PageStatus.NEW]

        if not page_ids:
            QMessageBox.information(self, "Nothing to do", "No pages match that scope.")
            return

        pipeline = self.pipeline
        self._flush_text()

        def task(progress, cancel):
            return pipeline.recognize_pages(page_ids, options, progress, cancel)

        self._run(task, f"Recognizing {len(page_ids)} pages", self._on_recognize_done)

    def _on_recognize_done(self, outcomes: list) -> None:
        if self.project is None:
            return
        failures = [o for o in outcomes if not o.ok]
        self.pages_panel.set_pages(self.project, self.page_quality)
        if self.current_page_id is not None:
            self.load_page(self.current_page_id)

        applied = sum(o.auto_applied for o in outcomes)
        message = f"Recognized {len(outcomes) - len(failures)} pages"
        if applied:
            message += f", {applied} corrections applied automatically"
        if failures:
            message += f", {len(failures)} failed"
        self.statusBar().showMessage(message, 9000)

        if failures:
            QMessageBox.warning(
                self,
                "Some pages failed",
                "\n".join(f"{o.label}: {o.error}" for o in failures[:12]),
            )

    # ==================================================================================
    # Page loading and sync
    # ==================================================================================

    def load_page(self, page_id: int) -> None:
        if self.project is None:
            return
        if self.current_page_id is not None and self.current_page_id != page_id:
            self._flush_text()

        page = self.project.get_page(page_id)
        if page is None:
            return

        self._loading = True
        self.current_page_id = page_id

        image_path = self.project.image_path(page)
        boxes = self.project.get_words(page_id)
        self.canvas.set_page(str(image_path) if image_path and image_path.exists() else None, boxes)

        raw, edited = self.project.get_text(page_id)
        self.editor.set_text(edited or raw)
        self.diff_panel.set_texts(raw, edited or raw)
        self.structure_panel.set_marks(self.project.get_structure(page_id), edited or raw)
        self.pages_panel.select_page(page_id)

        self._loading = False
        self._dirty = False
        self._update_enabled_state()

        self.status_page.setText(
            f"{page.label} · {page.status.value.replace('_', ' ')}"
            + (f" · {page.engine}" if page.engine else "")
            + ("" if image_path and image_path.exists() else " · no scan (text-only)")
        )
        self.analyze_current_page()

    def step_page(self, delta: int) -> None:
        if self.project is None or self.current_page_id is None:
            return
        pages = self.project.pages()
        index = next((i for i, p in enumerate(pages) if p.id == self.current_page_id), None)
        if index is None:
            return
        target = max(0, min(len(pages) - 1, index + delta))
        if target != index:
            self.load_page(pages[target].id)

    def goto_page(self) -> None:
        if self.project is None:
            return
        pages = self.project.pages()
        number, ok = QInputDialog.getInt(
            self, "Go to page", f"Page number (1–{len(pages)}):", 1, 1, len(pages)
        )
        if ok:
            self.load_page(pages[number - 1].id)

    def _on_editor_offset(self, offset: int) -> None:
        if self._loading:
            return
        self.canvas.highlight_offset(offset)
        issue = self.editor.issue_at(offset)
        word = _word_at(self.editor.toPlainText(), offset)
        parts = []
        if word:
            parts.append(word)
            if self.lexicon.contains(word):
                parts.append("in dictionary")
            else:
                parts.append("not in dictionary")
        if issue is not None:
            parts.append(issue.message)
        self.editor_pane.set_status("  ·  ".join(parts))

    def _on_canvas_word_clicked(self, start: int, end: int) -> None:
        self.editor.goto_offset(start, end)
        self.editor.setFocus()

    def _on_canvas_word_double_clicked(self, start: int, end: int) -> None:
        self.editor.goto_offset(start, end)
        issue = self.editor.issue_at(start)
        if issue is not None:
            self.editor.show_suggestions_for(issue)

    def _on_text_changed(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self.autosave_timer.start()

    def _flush_text(self) -> None:
        if self.project is None or self.current_page_id is None or not self._dirty:
            return
        self.project.set_edited_text(self.current_page_id, self.editor.toPlainText())
        self._dirty = False
        raw, edited = self.project.get_text(self.current_page_id)
        self.diff_panel.set_texts(raw, edited)

    # ==================================================================================
    # Analysis and correction
    # ==================================================================================

    def analyze_current_page(self) -> None:
        if self.project is None or self.current_page_id is None:
            return
        text = self.editor.toPlainText()
        suggester = self.suggester
        page_id = self.current_page_id

        def task(_progress, _cancel):
            return page_id, suggester.analyze(text)

        self._run(task, "", self._on_analysis_done, show_progress=False)

    def _on_analysis_done(self, payload) -> None:
        page_id, issues = payload
        if page_id != self.current_page_id:
            return  # the user moved on while analysis was running
        self.current_issues = issues
        self.editor.set_issues(issues)
        self.issues_panel.set_issues(issues)
        self._sync_canvas_flags()

    def _sync_canvas_flags(self) -> None:
        boxes = self.canvas._boxes  # noqa: SLF001 - the canvas owns them; read-only use
        flagged: dict[int, str] = {}
        for index, box in enumerate(boxes):
            if box.start < 0:
                continue
            for issue in self.current_issues:
                if issue.start < box.end and box.start < issue.end:
                    flagged[index] = issue.severity.value
                    break
        self.canvas.set_flagged(flagged)

    def _on_issue_activated(self, issue: Issue) -> None:
        self.editor.goto_offset(issue.start, issue.end)
        self.canvas.highlight_offset(issue.start)
        self.editor.show_suggestions_for(issue)

    def _accept_issue(self, issue: Issue) -> None:
        if issue.best is None:
            return
        self.editor.replace_span(issue.start, issue.end, issue.best.text)
        self._on_issue_accepted(issue, issue.best)

    def _on_issue_accepted(self, issue: Issue, suggestion: Suggestion) -> None:
        if self.pipeline is not None:
            self.pipeline.accept_correction(
                self.current_page_id, issue.text, suggestion.text, suggestion.source
            )
        self._flush_text()
        self.analyze_current_page()

    def _on_edit_committed(self, before: str, after: str) -> None:
        """A manual edit. Learn from it, but do not chase every keystroke."""
        if self.pipeline is None or not before.strip() or not after.strip():
            return
        if len(before) > 40 or len(after) > 40:
            return
        self.pipeline.accept_correction(self.current_page_id, before, after, "typed")

    def accept_all_confident(self) -> None:
        if not self.current_issues:
            return
        text = self.editor.toPlainText()
        new_text, applied = suggest.auto_apply(text, self.current_issues)
        if not applied:
            self.statusBar().showMessage("Nothing on this page is unambiguous enough", 5000)
            return

        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.select(cursor.SelectionType.Document)
        cursor.insertText(new_text)
        cursor.endEditBlock()

        if self.pipeline is not None:
            for issue in applied:
                if issue.best:
                    self.pipeline.accept_correction(
                        self.current_page_id, issue.text, issue.best.text, "auto"
                    )
        self._flush_text()
        self.analyze_current_page()
        self.statusBar().showMessage(f"Applied {len(applied)} confident corrections", 5000)

    def propagate_selection(self) -> None:
        if self.project is None or self.pipeline is None:
            return
        before = self.editor.textCursor().selectedText().replace("\u2029", "\n")
        if not before.strip():
            QMessageBox.information(
                self, "Select something first", "Select the text you want to replace everywhere."
            )
            return

        after, ok = QInputDialog.getText(self, "Apply everywhere", f"Replace “{before}” with:", text=before)
        if not ok or after == before:
            return

        self._flush_text()
        matches = self.pipeline.propagate(before, after, dry_run=True)
        if not matches:
            QMessageBox.information(self, "No matches", "That text does not occur anywhere else.")
            return

        labels = {p.id: p.label for p in self.project.pages()}
        dialog = dialogs.PropagateDialog(before, after, matches, labels, self.palette_, self)
        if dialog.exec() != dialogs.QDialog.DialogCode.Accepted:
            return

        changed = self.pipeline.propagate(
            before, after, whole_word=dialog.whole_word.isChecked(), dry_run=False
        )
        total = sum(count for _page, count in changed)
        if self.current_page_id is not None:
            self.load_page(self.current_page_id)
        self.statusBar().showMessage(f"Replaced {total} occurrences on {len(changed)} pages", 8000)

    def replay_ledger(self) -> None:
        if self.pipeline is None:
            return
        self._flush_text()
        preview = self.pipeline.apply_ledger(min_count=2, dry_run=True)
        if not preview:
            QMessageBox.information(
                self,
                "Nothing to replay",
                "No correction has been made often enough yet. Corrections you apply at "
                "least twice become candidates for replaying across the whole book.",
            )
            return

        total = sum(count for changes in preview.values() for _page, count in changes)
        answer = QMessageBox.question(
            self,
            "Replay corrections",
            f"{len(preview)} distinct corrections would be applied "
            f"{total} times across the book.\n\nApply them?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        applied = self.pipeline.apply_ledger(min_count=2, dry_run=False)
        if self.current_page_id is not None:
            self.load_page(self.current_page_id)
        self.statusBar().showMessage(f"Replayed {len(applied)} corrections", 8000)

    def normalize_current_page(self) -> None:
        if self.pipeline is None or self.current_page_id is None:
            return
        self._flush_text()
        result = self.pipeline.normalize_page(self.current_page_id)
        self.load_page(self.current_page_id)
        self.statusBar().showMessage(
            f"{result.total_changes} normalizations — {result.summary()}"[:220], 9000
        )

    def verify_page(self) -> None:
        if self.pipeline is None or self.current_page_id is None or self.project is None:
            return
        self._flush_text()
        harvested = self.pipeline.mark_verified(self.current_page_id)
        self.pages_panel.set_pages(self.project, self.page_quality)

        verified, lines, advice = self.pipeline.training_readiness()
        self.statusBar().showMessage(
            f"Verified · {harvested} lines harvested · {verified} pages, {lines} lines total · {advice}",
            12000,
        )
        self.step_page(1)

    def show_training_status(self) -> None:
        if self.pipeline is None:
            return
        verified, lines, advice = self.pipeline.training_readiness()
        QMessageBox.information(
            self,
            "Training data",
            f"{verified} pages verified\n{lines} ground-truth lines harvested\n\n{advice}\n\n"
            "Ground truth is written to the project's lines/ folder alongside its text, "
            "in the layout Tesseract and Kraken expect for fine-tuning.",
        )

    # ==================================================================================
    # Quality and search
    # ==================================================================================

    def assess_book(self) -> None:
        if self.project is None:
            return
        self._flush_text()
        project, lexicon, suggester = self.project, self.lexicon, self.suggester

        def task(progress, _cancel):
            return qa.assess_book(project, lexicon, suggester, progress)

        self._run(task, "Assessing quality", self._on_assessment_done)

    def _on_assessment_done(self, book: qa.BookQuality) -> None:
        self.page_quality = {p.page_id: p for p in book.pages}
        self.quality_panel.set_quality(book)
        if self.project is not None:
            self.pages_panel.set_pages(self.project, self.page_quality)
        self.status_quality.setText(f"quality {book.mean_score:.0%}")
        self.right_tabs.setCurrentWidget(self.quality_panel)

    def search_book(self, query: str, _whole_word: bool) -> None:
        if self.project is None or not query.strip():
            return
        self._flush_text()
        folded_query = fidel.fold(query)
        rows: list[tuple[int, str, str, int]] = []

        for page in self.project.iter_pages():
            text = self.project.get_edited(page.id)
            folded = fidel.fold(text)
            start = folded.find(folded_query)
            while start != -1 and len(rows) < 500:
                left = max(0, start - 28)
                right = min(len(text), start + len(query) + 28)
                context = text[left:right].replace("\n", " ")
                rows.append((page.id, page.label, f"…{context}…", start))
                start = folded.find(folded_query, start + 1)

        self.search_panel.set_results(rows)

    def _on_search_result(self, page_id: int, offset: int) -> None:
        if page_id != self.current_page_id:
            self.load_page(page_id)
        QTimer.singleShot(60, lambda: self.editor.goto_offset(offset, offset + 1))

    # ==================================================================================
    # Structure
    # ==================================================================================

    def _add_structure_mark(self, kind: str) -> None:
        if self.project is None or self.current_page_id is None:
            return
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            QMessageBox.information(
                self, "Select text first", "Select the text you want to mark, then press Mark."
            )
            return
        self._flush_text()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        label = self.editor.toPlainText()[start:end].strip()[:120]
        self.project.add_structure(
            StructureMark(self.current_page_id, start, end, kind, 1, label)
        )
        self.structure_panel.set_marks(
            self.project.get_structure(self.current_page_id), self.editor.toPlainText()
        )
        self.statusBar().showMessage(f"Marked as {kind}", 3000)

    def _on_mark_activated(self, mark: StructureMark) -> None:
        self.editor.goto_offset(mark.start, mark.end)

    def _delete_structure_mark(self, mark_id: int) -> None:
        if self.project is None or self.current_page_id is None:
            return
        self.project.delete_structure(mark_id)
        self.structure_panel.set_marks(
            self.project.get_structure(self.current_page_id), self.editor.toPlainText()
        )

    # ==================================================================================
    # Export
    # ==================================================================================

    def export_book(self) -> None:
        if self.project is None:
            return
        self._flush_text()
        dialog = dialogs.ExportDialog(self.project.path / "exports", self.palette_, self)
        if dialog.exec() != dialogs.QDialog.DialogCode.Accepted:
            return

        project = self.project
        directory = Path(dialog.directory.text())
        stem = _safe_stem(project.title or project.path.stem)
        include_furniture = dialog.furniture.isChecked()
        typeset_options = dialog.typeset_options()
        wants = {
            "epub": dialog.epub.isChecked(),
            "pdf_clean": dialog.pdf_clean.isChecked(),
            "pdf_searchable": dialog.pdf_searchable.isChecked(),
            "text": dialog.text.isChecked(),
            "markdown": dialog.markdown.isChecked(),
            "alto": dialog.alto.isChecked(),
        }

        def task(progress, _cancel):
            steps = [k for k, v in wants.items() if v]
            total = len(steps) or 1
            written: list[str] = []
            document = build_document(project, include_furniture=include_furniture)

            for index, step in enumerate(steps):
                progress(index, total, step)
                if step == "epub":
                    path = export_epub(document, directory / f"{stem}.epub", EpubOptions())
                    written.append(str(path))
                elif step == "pdf_clean":
                    path, report = export_pdf(
                        document, directory / f"{stem} (typeset).pdf", typeset_options
                    )
                    written.append(f"{path}  —  {report.describe()}")
                elif step == "pdf_searchable":
                    path, report = export_searchable_pdf(
                        project, directory / f"{stem} (searchable).pdf", SearchableOptions()
                    )
                    written.append(f"{path}  —  {report.describe()}")
                elif step == "text":
                    written.append(str(export_text(document, directory / f"{stem}.txt")))
                elif step == "markdown":
                    written.append(str(export_markdown(document, directory / f"{stem}.md")))
                elif step == "alto":
                    files = export_alto(project, directory / "alto")
                    written.append(f"{len(files)} ALTO files in {directory / 'alto'}")
                    written.append(str(export_hocr(project, directory / f"{stem}.hocr")))

            progress(total, total, "done")
            return written

        self._run(task, "Exporting", lambda written: self._on_export_done(written, directory))

    def _on_export_done(self, written: list[str], directory: Path) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Export complete")
        box.setText(f"Wrote {len(written)} item(s) to {directory}")
        box.setDetailedText("\n".join(written))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
        self.statusBar().showMessage(f"Exported to {directory}", 9000)

    # ==================================================================================
    # View toggles
    # ==================================================================================

    def _toggle_boxes(self, checked: bool) -> None:
        self.canvas.set_show_boxes(checked)

    def _toggle_heatmap(self, checked: bool) -> None:
        self.canvas.set_show_heatmap(checked)

    def _toggle_diff(self, checked: bool) -> None:
        self.diff_dock.setVisible(checked)
        if checked and self.project is not None and self.current_page_id is not None:
            raw, edited = self.project.get_text(self.current_page_id)
            self.diff_panel.set_texts(raw, edited or raw)

    def _toggle_ime(self, checked: bool) -> None:
        self.editor.set_transliteration(checked)
        self.editor_pane.set_ime_active(checked)
        if checked:
            self.statusBar().showMessage(
                "Phonetic input on — type selam for ሰላም. Press F1 for the full chart.", 7000
            )

    def _scale_text(self, delta: int) -> None:
        self.editor.set_font_size(self.editor.font_size + delta)

    def _set_theme(self, name: str) -> None:
        self.settings.setValue("theme", name)
        QMessageBox.information(
            self, "Theme", "The new theme will be applied the next time you start the app."
        )

    def show_keyboard_help(self) -> None:
        dialogs.KeyboardHelpDialog(self.palette_, self).exec()

    def show_about(self) -> None:
        dialogs.AboutDialog(__version__, self).exec()

    # ==================================================================================
    # Worker plumbing
    # ==================================================================================

    def _run(self, task, description: str, on_done, show_progress: bool = True) -> None:
        if show_progress:
            self.progress.show()
            self.progress.setRange(0, 0)
            if description:
                self.statusBar().showMessage(f"{description}…")

        def progress(done: int, total: int, message: str) -> None:
            if not show_progress:
                return
            if total:
                self.progress.setRange(0, total)
                self.progress.setValue(done)
            if description and message:
                self.statusBar().showMessage(f"{description}: {message} ({done}/{total})")

        def done(result) -> None:
            self.progress.hide()
            on_done(result)

        def failed(message: str) -> None:
            self.progress.hide()
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Something went wrong")
            box.setText(message.split("\n")[0])
            box.setDetailedText(message)
            box.exec()

        self.workers.start(task, description, progress, done, failed)

    # ==================================================================================
    # Window state
    # ==================================================================================

    def _restore_geometry(self) -> None:
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state and self.restoreState(state):
            self._restored_state = True
        splitter = self.settings.value("splitter")
        if splitter:
            self.splitter.restoreState(splitter)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._flush_text()
        self.workers.cancel_all()
        self.workers.wait_all(3000)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("splitter", self.splitter.saveState())
        if self.project is not None:
            if self.pipeline is not None:
                self.pipeline.save_model()
            self.project.close()
        self.lexicon.close()
        super().closeEvent(event)


def _word_at(text: str, offset: int) -> str:
    for token in fidel.words(text):
        if token.start <= offset <= token.end:
            return token.text
    return ""


def _safe_stem(name: str) -> str:
    cleaned = "".join(c for c in name if c not in '\\/:*?"<>|').strip()
    return cleaned or "book"


def _prepare_lexicon(app: QApplication, palette: theme.Palette) -> Lexicon:
    """Open the lexicon, showing a splash while the bundled wordlist is loaded.

    Only the very first launch pays for this: several hundred thousand corpus words go
    into SQLite, which takes long enough that a window appearing with no explanation
    would look like a hang. Afterwards the database is already there and this returns
    immediately, so no splash is shown at all.
    """
    from ..core.lexicon import default_lexicon_path

    if default_lexicon_path().exists():
        return load_or_create()

    splash = QLabel(
        f"{APP_NAME}\n\nPreparing the Amharic dictionary…\nThis happens once.",
        alignment=Qt.AlignCenter,
    )
    splash.setWindowFlags(Qt.SplashScreen | Qt.WindowStaysOnTopHint)
    splash.setFixedSize(420, 180)
    splash.setStyleSheet(
        f"background: {palette.surface}; color: {palette.text};"
        f" border: 1px solid {palette.border}; font-size: 11pt;"
    )
    splash.show()
    app.processEvents()

    def tick(loaded: int) -> None:
        splash.setText(
            f"{APP_NAME}\n\nPreparing the Amharic dictionary…\n{loaded:,} words"
        )
        # Loading runs on this thread on purpose — there is nothing else for the user to
        # do yet, and a worker would only add a race against the window opening.
        app.processEvents()

    try:
        return load_or_create(progress=tick)
    finally:
        splash.close()


def run(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("AmharicStudio")

    settings = QSettings("AmharicStudio", "AmharicStudio")
    palette = theme.PALETTES.get(str(settings.value("theme", "light")), theme.LIGHT)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.stylesheet(palette))
    app.setFont(theme.ui_font())

    window = MainWindow(palette, lexicon=_prepare_lexicon(app, palette))
    window.show()

    # Open a project passed on the command line.
    for argument in argv[1:]:
        candidate = Path(argument)
        if candidate.exists() and (candidate / "project.db").exists():
            window._open(Project(candidate))  # noqa: SLF001 - deliberate entry point
            break

    return app.exec()
