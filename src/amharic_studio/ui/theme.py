"""Colours, fonts and stylesheet.

Colour is doing real work in this application, not decoration: severity of an issue,
recognizer confidence and page status are all encoded in it. The palette is therefore
built around a small set of semantic colours that stay distinguishable against both a
scanned page and a text background, and the severity ramp avoids red/green as the sole
distinction so it survives the most common forms of colour blindness.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QFontDatabase

from ..core import fonts


@dataclass(frozen=True)
class Palette:
    name: str
    window: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_text: str

    # Semantic
    high: str
    medium: str
    low: str
    verified: str
    warning: str

    @property
    def is_dark(self) -> bool:
        return QColor(self.window).lightness() < 128


LIGHT = Palette(
    name="light",
    window="#f5f4f1",       # warm off-white, easier beside a scanned page than pure grey
    surface="#ffffff",
    surface_alt="#efede8",
    border="#d5d1c8",
    text="#1c1b19",
    text_muted="#6d6a63",
    accent="#8a5a2b",       # ink brown, picks up the tone of aged paper
    accent_text="#ffffff",
    high="#c0392b",
    medium="#c9821a",
    low="#4a7c9e",
    verified="#3f7d46",
    warning="#b5432f",
)

DARK = Palette(
    name="dark",
    window="#1e1e21",
    surface="#26262a",
    surface_alt="#2e2e33",
    border="#3c3c43",
    text="#e8e6e1",
    text_muted="#9a968e",
    accent="#c8934f",
    accent_text="#1e1e21",
    high="#e06c5b",
    medium="#e0a955",
    low="#6fa8cc",
    verified="#6fbf78",
    warning="#e0806b",
)

PALETTES = {"light": LIGHT, "dark": DARK}


# --------------------------------------------------------------------------------------
# Semantic colours
# --------------------------------------------------------------------------------------


def severity_color(palette: Palette, severity: str) -> QColor:
    return QColor({"high": palette.high, "medium": palette.medium, "low": palette.low}.get(severity, palette.low))


def confidence_color(confidence: float, palette: Palette) -> QColor:
    """Heat colour for a recognizer confidence, from alarming to unobtrusive.

    Deliberately non-linear: everything above roughly 90% is treated as equally fine,
    because the interesting distinctions all live at the bottom of the range.
    """
    confidence = max(0.0, min(1.0, confidence))
    if confidence >= 0.90:
        color = QColor(palette.verified)
        color.setAlpha(38)
    elif confidence >= 0.75:
        color = QColor(palette.low)
        color.setAlpha(58)
    elif confidence >= 0.55:
        color = QColor(palette.medium)
        color.setAlpha(80)
    else:
        color = QColor(palette.high)
        color.setAlpha(105)
    return color


STATUS_COLORS = {
    "new": "text_muted",
    "recognized": "low",
    "in_review": "medium",
    "verified": "verified",
    "skipped": "border",
}


def status_color(palette: Palette, status: str) -> QColor:
    return QColor(getattr(palette, STATUS_COLORS.get(status, "text_muted")))


# --------------------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------------------

_ethiopic_family: str | None = None

#: Interface fonts, in preference order, before the Ethiopic fallback is appended.
UI_FAMILIES = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Cantarell", "Noto Sans", "sans-serif")
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono", "monospace")


def ethiopic_family() -> str:
    """The name of a font family that can draw fidel, loading the bundled face if needed."""
    global _ethiopic_family

    if _ethiopic_family is None:
        info = fonts.default_font()
        if info is not None:
            font_id = QFontDatabase.addApplicationFont(str(info.path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
            _ethiopic_family = families[0] if families else info.family
        else:
            _ethiopic_family = "Nyala"
    return _ethiopic_family


def ethiopic_font(size: int = 16, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """A font for book text, which is always Ethiopic."""
    font = QFont(ethiopic_family(), size)
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def _with_ethiopic_fallback(families: tuple[str, ...], size: int) -> QFont:
    """A font over ``families`` that falls back to fidel rather than drawing nothing.

    Interface fonts do not cover the Ethiopic block, and on Windows the missing glyph is
    *blank* rather than a visible box: the text reserves its width and then draws nothing
    at all. Any label, menu entry, tooltip or table cell holding a word from the book
    would silently disappear. Naming the Ethiopic family as an explicit fallback fixes
    every one of those at once, instead of each widget having to remember.
    """
    font = QFont()
    font.setFamilies([*families, ethiopic_family()])
    font.setPointSize(size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def ui_font(size: int = 9) -> QFont:
    return _with_ethiopic_fallback(UI_FAMILIES, size)


def mono_font(size: int = 9) -> QFont:
    return _with_ethiopic_fallback(MONO_FAMILIES, size)


# --------------------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------------------


def stylesheet(palette: Palette) -> str:
    p = palette
    return f"""
/* Deliberately no font-family or font-size here. A stylesheet font overrides
   QWidget.setFont(), which would silently replace the Ethiopic face on the editor and
   the fidel columns with a UI font that cannot draw the script. The base UI font is set
   once through QApplication.setFont() instead. */
QWidget {{
    background: {p.window};
    color: {p.text};
}}
QMainWindow::separator {{ background: {p.border}; width: 1px; height: 1px; }}

QToolBar {{
    background: {p.surface};
    border: none;
    border-bottom: 1px solid {p.border};
    padding: 4px 6px;
    spacing: 3px;
}}
QToolBar QToolButton {{
    padding: 5px 10px;
    border-radius: 4px;
    color: {p.text};
}}
QToolBar QToolButton:hover {{ background: {p.surface_alt}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: {p.accent}; color: {p.accent_text};
}}
QToolBar QToolButton:disabled {{ color: {p.text_muted}; }}

QDockWidget {{ titlebar-close-icon: none; font-weight: 600; }}
QDockWidget::title {{
    background: {p.surface_alt};
    padding: 6px 9px;
    border-bottom: 1px solid {p.border};
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.6px;
}}

QStatusBar {{
    background: {p.surface};
    border-top: 1px solid {p.border};
    color: {p.text_muted};
}}
QStatusBar QLabel {{ padding: 0 8px; }}

QMenuBar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {p.surface_alt}; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 3px; }}
QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 8px; }}

QListView, QTreeView, QTableView {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
    outline: none;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QListView::item, QTreeView::item {{ padding: 4px 6px; border-radius: 3px; }}
QListView::item:hover, QTreeView::item:hover {{ background: {p.surface_alt}; }}
QListView::item:selected, QTreeView::item:selected {{
    background: {p.accent}; color: {p.accent_text};
}}

QPlainTextEdit, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {p.accent}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}

QPushButton {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ background: {p.surface_alt}; }}
QPushButton:default, QPushButton[primary="true"] {{
    background: {p.accent}; color: {p.accent_text}; border-color: {p.accent};
}}
QPushButton:disabled {{ color: {p.text_muted}; background: {p.surface_alt}; }}

QProgressBar {{
    border: 1px solid {p.border};
    border-radius: 4px;
    background: {p.surface_alt};
    text-align: center;
    height: 16px;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 3px; }}

QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: 4px; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    padding: 6px 14px;
    border-bottom: 2px solid transparent;
    color: {p.text_muted};
}}
QTabBar::tab:selected {{ color: {p.text}; border-bottom-color: {p.accent}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {p.text}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {p.border}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {p.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {p.accent}; }}

QGroupBox {{
    border: 1px solid {p.border};
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; color: {p.text_muted}; }}

QToolTip {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    padding: 5px 7px;
}}

QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QCheckBox::indicator:unchecked {{ border: 1px solid {p.border}; border-radius: 3px; background: {p.surface}; }}
QCheckBox::indicator:checked {{ border: 1px solid {p.accent}; border-radius: 3px; background: {p.accent}; }}

QHeaderView::section {{
    background: {p.surface_alt};
    padding: 5px 7px;
    border: none;
    border-right: 1px solid {p.border};
    border-bottom: 1px solid {p.border};
    font-weight: 600;
}}
"""
