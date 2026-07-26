"""Dialogs: project creation, import, recognition, export, propagation and help."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import translit
from ..core.export.pdf_clean import TypesetOptions
from ..core.imaging import BinarizeMethod, PreprocessOptions
from ..core.normalize import Policy
from ..core.ocr import registry
from ..core.ocr.base import PageSegMode
from ..core.pipeline import EngineSpec, RecognizeOptions
from . import theme


class NewProjectDialog(QDialog):
    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.title = QLineEdit()
        self.title.setFont(theme.ethiopic_font(13))
        self.title.setPlaceholderText("Book title (Amharic or Latin)")
        form.addRow("Title", self.title)

        self.author = QLineEdit()
        self.author.setFont(theme.ethiopic_font(13))
        form.addRow("Author", self.author)

        location_row = QHBoxLayout()
        self.location = QLineEdit(str(Path.home() / "Documents"))
        location_row.addWidget(self.location, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        location_row.addWidget(browse)
        container = QWidget()
        container.setLayout(location_row)
        form.addRow("Location", container)

        self.policy = QComboBox()
        self.policy.addItem("Faithful — keep ሠ ኀ ዐ ፀ and ፡ exactly as printed", Policy.FAITHFUL)
        self.policy.addItem("Modern — normalize orthography and use spaces", Policy.MODERN)
        form.addRow("Orthography", self.policy)

        layout.addLayout(form)

        note = QLabel(
            "Faithful is the safe default for older printing. It never rewrites the "
            "letters on the page; homophone folding is still used behind the scenes so "
            "dictionary lookup and search work across spellings."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Project location", self.location.text())
        if directory:
            self.location.setText(directory)

    def result_values(self) -> tuple[Path, str, str, Policy]:
        title = self.title.text().strip() or "Untitled book"
        safe_name = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or "Untitled book"
        path = Path(self.location.text()) / f"{safe_name}.amproj"
        return path, title, self.author.text().strip(), self.policy.currentData()


class ImportDialog(QDialog):
    """Import options, with a preprocessing section that explains its own defaults."""

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import pages")
        self.setMinimumWidth(560)
        self.paths: list[Path] = []

        layout = QVBoxLayout(self)

        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        self.files_label = QLabel("No files chosen")
        self.files_label.setWordWrap(True)
        source_layout.addWidget(self.files_label)

        buttons_row = QHBoxLayout()
        for label, handler in (
            ("Choose images…", self._choose_images),
            ("Choose PDF…", self._choose_pdf),
            ("Choose text…", self._choose_text),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            buttons_row.addWidget(button)
        source_layout.addLayout(buttons_row)
        layout.addWidget(source_group)

        pdf_group = QGroupBox("PDF options")
        pdf_form = QFormLayout(pdf_group)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 900)
        self.dpi.setValue(300)
        self.dpi.setToolTip("300 dpi is the practical minimum for reliable fidel recognition.")
        pdf_form.addRow("Render at", self.dpi)

        self.text_layer = QComboBox()
        self.text_layer.addItem("Auto-detect", None)
        self.text_layer.addItem("Use the existing text layer (repair it)", True)
        self.text_layer.addItem("Ignore it and re-OCR from images", False)
        pdf_form.addRow("Existing text", self.text_layer)
        layout.addWidget(pdf_group)

        prep_group = QGroupBox("Preprocessing")
        prep_form = QFormLayout(prep_group)
        self.preprocess = QCheckBox("Clean up scans on import")
        self.preprocess.setChecked(True)
        prep_form.addRow(self.preprocess)

        self.binarize = QComboBox()
        for method, label in (
            (BinarizeMethod.SAUVOLA, "Sauvola (best for aged paper)"),
            (BinarizeMethod.OTSU, "Otsu (even, modern printing)"),
            (BinarizeMethod.ADAPTIVE_MEAN, "Adaptive mean"),
            (BinarizeMethod.NONE, "None — keep greyscale"),
        ):
            self.binarize.addItem(label, method)
        prep_form.addRow("Binarization", self.binarize)

        self.deskew = QCheckBox("Straighten skewed pages")
        self.deskew.setChecked(True)
        prep_form.addRow(self.deskew)

        self.despeckle = QCheckBox("Remove speckles")
        self.despeckle.setChecked(True)
        self.despeckle.setToolTip(
            "Conservative on purpose: Ethiopic diacritics are small components too, and an "
            "aggressive filter would erase the marks that identify a vowel order."
        )
        prep_form.addRow(self.despeckle)
        layout.addWidget(prep_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_paths(self, paths: list[str]) -> None:
        self.paths = [Path(p) for p in paths]
        if self.paths:
            names = ", ".join(p.name for p in self.paths[:4])
            extra = f" and {len(self.paths) - 4} more" if len(self.paths) > 4 else ""
            self.files_label.setText(f"{len(self.paths)} file(s): {names}{extra}")
        self.ok_button.setEnabled(bool(self.paths))

    def _choose_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose page images", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        )
        if paths:
            self._set_paths(paths)

    def _choose_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose PDF", "", "PDF (*.pdf)")
        if path:
            self._set_paths([path])

    def _choose_text(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose text file", "", "Text (*.txt *.md)")
        if path:
            self._set_paths([path])

    def preprocess_options(self) -> PreprocessOptions:
        return PreprocessOptions(
            deskew=self.deskew.isChecked(),
            binarize=self.binarize.currentData(),
            despeckle=self.despeckle.isChecked(),
        )


class RecognizeDialog(QDialog):
    """Choose engines and scope for a recognition run."""

    def __init__(self, page_count: int, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recognize pages")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        scope_group = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_group)
        self.scope_all = QRadioButton("All unrecognized pages")
        self.scope_all.setChecked(True)
        self.scope_book = QRadioButton(f"Every page in the book ({page_count})")
        self.scope_page = QRadioButton("Current page only")
        for button in (self.scope_all, self.scope_book, self.scope_page):
            scope_layout.addWidget(button)
        layout.addWidget(scope_group)

        engine_group = QGroupBox("Engines")
        engine_layout = QVBoxLayout(engine_group)
        self.engine_checks: list[tuple[QCheckBox, EngineSpec]] = []

        available = registry.available_backends()
        if not available:
            engine_layout.addWidget(
                QLabel(
                    "No OCR engine found. Install Tesseract and its Amharic language data, "
                    "then reopen this dialog."
                )
            )
        for backend in available:
            languages = backend.languages() or ["amh"]
            for language in languages:
                if "amh" not in language.lower() and "ethiopic" not in language.lower():
                    continue
                check = QCheckBox(f"{backend.display_name} — {language}")
                check.setChecked(True)
                engine_layout.addWidget(check)
                self.engine_checks.append((check, EngineSpec(backend.name, language)))

        note = QLabel(
            "Selecting more than one engine turns on voting. The real benefit is not the "
            "small accuracy gain but the disagreement marks: two engines rarely fail the "
            "same way, so where they differ is where a human should look first."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        engine_layout.addWidget(note)
        layout.addWidget(engine_group)

        options_group = QGroupBox("Options")
        options_form = QFormLayout(options_group)

        self.psm = QComboBox()
        for mode, label in (
            (PageSegMode.SINGLE_BLOCK, "Single block of text (usual for body pages)"),
            (PageSegMode.AUTO, "Automatic page segmentation"),
            (PageSegMode.SINGLE_COLUMN, "Single column"),
            (PageSegMode.SPARSE, "Sparse text"),
            (PageSegMode.SINGLE_LINE, "Single line"),
        ):
            self.psm.addItem(label, mode)
        options_form.addRow("Layout", self.psm)

        self.normalize = QCheckBox("Normalize obvious OCR damage")
        self.normalize.setChecked(True)
        options_form.addRow(self.normalize)

        self.auto_correct = QCheckBox("Apply unambiguous corrections automatically")
        self.auto_correct.setChecked(False)
        self.auto_correct.setToolTip(
            "Only applies a fix when the candidate is a dictionary word, clearly ahead of "
            "the runner-up, and replacing something that is not already a word."
        )
        options_form.addRow(self.auto_correct)

        self.overwrite = QCheckBox("Re-recognize pages already marked verified")
        options_form.addRow(self.overwrite)
        layout.addWidget(options_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self, policy: Policy) -> RecognizeOptions:
        engines = [spec for check, spec in self.engine_checks if check.isChecked()]
        return RecognizeOptions(
            engines=engines or [EngineSpec()],
            psm=self.psm.currentData(),
            normalize_text=self.normalize.isChecked(),
            policy=policy,
            auto_correct=self.auto_correct.isChecked(),
            overwrite_existing=self.overwrite.isChecked(),
        )

    @property
    def scope(self) -> str:
        if self.scope_page.isChecked():
            return "page"
        if self.scope_book.isChecked():
            return "book"
        return "unrecognized"


class ExportDialog(QDialog):
    def __init__(self, default_dir: Path, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.directory = QLineEdit(str(default_dir))
        row.addWidget(self.directory, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        wrapper = QWidget()
        wrapper.setLayout(row)
        form = QFormLayout()
        form.addRow("Folder", wrapper)
        layout.addLayout(form)

        formats = QGroupBox("Formats")
        formats_layout = QVBoxLayout(formats)
        self.epub = QCheckBox("EPUB 3 — reflowable, embedded font, printed-page map")
        self.epub.setChecked(True)
        self.pdf_clean = QCheckBox("Clean PDF — freshly typeset from the corrected text")
        self.pdf_clean.setChecked(True)
        self.pdf_searchable = QCheckBox(
            "Searchable PDF — original scans with an invisible corrected text layer"
        )
        self.pdf_searchable.setChecked(True)
        self.text = QCheckBox("Plain text")
        self.markdown = QCheckBox("Markdown")
        self.alto = QCheckBox("ALTO XML + hOCR (for eScriptorium / Transkribus)")
        for box in (self.epub, self.pdf_clean, self.pdf_searchable, self.text, self.markdown, self.alto):
            formats_layout.addWidget(box)
        layout.addWidget(formats)

        typeset = QGroupBox("Clean PDF layout")
        typeset_form = QFormLayout(typeset)
        self.page_size = QComboBox()
        self.page_size.addItems(["A5", "A4", "Letter"])
        typeset_form.addRow("Page size", self.page_size)

        self.body_size = QSpinBox()
        self.body_size.setRange(7, 24)
        self.body_size.setValue(12)
        typeset_form.addRow("Body size (pt)", self.body_size)

        self.justify = QCheckBox("Justify body text")
        self.justify.setChecked(True)
        typeset_form.addRow(self.justify)

        self.title_page = QCheckBox("Include a title page")
        self.title_page.setChecked(True)
        typeset_form.addRow(self.title_page)

        self.furniture = QCheckBox("Keep running heads and page numbers from the scans")
        self.furniture.setToolTip(
            "Off by default: the original page furniture is noise once the text reflows, "
            "and the export generates its own."
        )
        typeset_form.addRow(self.furniture)
        layout.addWidget(typeset)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export folder", self.directory.text())
        if directory:
            self.directory.setText(directory)

    def typeset_options(self) -> TypesetOptions:
        return TypesetOptions(
            page_size=self.page_size.currentText(),
            body_size=float(self.body_size.value()),
            justify=self.justify.isChecked(),
            title_page=self.title_page.isChecked(),
        )


class PropagateDialog(QDialog):
    """Review a correction before applying it across the whole book."""

    def __init__(
        self,
        before: str,
        after: str,
        matches: list[tuple[int, int]],
        page_labels: dict[int, str],
        palette: theme.Palette,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Apply everywhere")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        headline = QLabel(f"Replace “{before}” with “{after}”")
        headline.setFont(theme.ethiopic_font(14))
        layout.addWidget(headline)

        total = sum(count for _page, count in matches)
        summary = QLabel(
            f"{total} occurrence{'s' if total != 1 else ''} on {len(matches)} page"
            f"{'s' if len(matches) != 1 else ''}."
        )
        summary.setStyleSheet(f"color: {palette.text_muted};")
        layout.addWidget(summary)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Page", "Occurrences"])
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for page_id, count in matches:
            tree.addTopLevelItem(
                QTreeWidgetItem([page_labels.get(page_id, str(page_id)), str(count)])
            )
        layout.addWidget(tree, 1)

        self.whole_word = QCheckBox("Whole words only")
        self.whole_word.setChecked(True)
        layout.addWidget(self.whole_word)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply to all")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class KeyboardHelpDialog(QDialog):
    """Shortcut reference and the phonetic input chart."""

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard reference")
        self.resize(720, 620)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        shortcuts = QPlainTextEdit()
        shortcuts.setReadOnly(True)
        shortcuts.setPlainText(SHORTCUT_TEXT)
        tabs.addTab(shortcuts, "Shortcuts")

        table = QTreeWidget()
        table.setHeaderLabels(["Type", "Letter", "ግዕዝ  ካዕብ  ሣልስ  ራብዕ  ኃምስ  ሳድስ  ሳብዕ"])
        table.setFont(theme.ethiopic_font(13))
        table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for key, letter, orders in translit.keyboard_reference():
            table.addTopLevelItem(QTreeWidgetItem([key, letter, orders]))
        tabs.addTab(table, "Phonetic input (SERA)")

        vowels = QPlainTextEdit()
        vowels.setReadOnly(True)
        vowels.setFont(theme.ethiopic_font(13))
        vowels.setPlainText(VOWEL_TEXT)
        tabs.addTab(vowels, "Vowels and punctuation")

        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


SHORTCUT_TEXT = """\
NAVIGATION
  Page Down / Page Up       next / previous page
  F8 / Shift+F8             next / previous flagged word
  Ctrl+G                    go to page
  Ctrl+F                    search the book

CORRECTION
  Alt+1 … Alt+5             accept the numbered suggestion
  Alt+Enter                 accept the top suggestion
  Alt+Space                 open the suggestion list
  Ctrl+Shift+A              accept every confident suggestion on this page
  Ctrl+Shift+E              apply this correction everywhere in the book
  Ctrl+Enter                mark the page verified and harvest training data

VIEW
  Ctrl+= / Ctrl+-           zoom the scan
  Ctrl+0                    fit page width
  Ctrl+B                    show or hide word boxes
  Ctrl+H                    show or hide the confidence heatmap
  Ctrl+D                    show the diff against the original OCR
  Ctrl++ / Ctrl+_           larger / smaller editor text

INPUT
  Ctrl+I                    turn phonetic (SERA) input on or off

FILE
  Ctrl+N / Ctrl+O / Ctrl+S  new / open / save
  Ctrl+Shift+I              import pages
  Ctrl+R                    recognize pages
  Ctrl+Shift+X              export
"""

VOWEL_TEXT = """\
VOWEL ORDERS — type the consonant, then the vowel

  he  ሀ    ግዕዝ    ä
  hu  ሁ    ካዕብ    u
  hi  ሂ    ሣልስ    i
  ha  ሃ    ራብዕ    a
  hE  ሄ    ኃምስ    e
  h   ህ    ሳድስ    ə   (consonant alone)
  ho  ሆ    ሳብዕ    o
  hWA ኋ    labialized

A VOWEL ON ITS OWN gives the አ family:
  a አ    u ኡ    i ኢ    A ኣ    E ኤ    I እ    o ኦ

LETTERS THAT NEED A CAPITAL OR A DIGRAPH
  H ሐ    S ሠ    X ኀ    ` ዐ    Q ቐ    K/kh ኸ    D ዸ    G ጘ
  sh/x ሸ   ch/c ቸ   ny/N ኘ   zh/Z ዠ   T ጠ   C ጨ   P ጰ
  ts ጸ    tz ፀ

PUNCTUATION
  :    ፡   word separator
  ::   ።   full stop
  ,    ፣   comma
  ;    ፤   semicolon
  ?    ፧   question mark
"""


class AboutDialog(QDialog):
    def __init__(self, version: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Amharic Studio")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        title = QLabel("Amharic Studio")
        title.setStyleSheet("font-size: 16pt; font-weight: 700;")
        layout.addWidget(title)
        layout.addWidget(QLabel(f"Version {version}"))

        body = QLabel(
            "Offline digitization and correction of Ge'ez-script books.\n\n"
            "Everything runs on this machine. No text, image or correction is ever sent "
            "anywhere, and the application never requires a network connection."
        )
        body.setWordWrap(True)
        layout.addWidget(body)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
