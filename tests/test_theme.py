"""Fonts and palette.

These need a QApplication, so they are skipped where Qt cannot open a display. They are
worth having anyway: the failure mode they guard is silent, and it makes the program look
broken in a way that is easy to mistake for missing data.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    if QGuiApplication.primaryScreen() is None:
        pytest.skip("no display available")
    return app


@pytest.fixture(scope="module")
def theme_module(qt_app):
    from amharic_studio.ui import theme

    return theme


def ink(font, text: str) -> int:
    """How many pixels a string actually puts on the page."""
    pixmap = QPixmap(220, 40)
    pixmap.fill()
    painter = QPainter(pixmap)
    painter.setFont(font)
    painter.drawText(4, 28, text)
    painter.end()
    image = pixmap.toImage()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).lightness() < 200
    )


class TestFonts:
    @pytest.mark.parametrize("which", ["ui_font", "mono_font", "ethiopic_font"])
    def test_every_font_can_actually_draw_fidel(self, theme_module, which):
        # An interface font with no Ethiopic coverage draws *nothing* on Windows rather
        # than a visible box: the text takes up its width and is simply invisible. Any
        # widget showing a word from the book would appear empty.
        assert ink(getattr(theme_module, which)(), "ሰላም") > 20, which

    @pytest.mark.parametrize("which", ["ui_font", "mono_font"])
    def test_interface_fonts_still_prefer_the_interface_face(self, theme_module, which):
        # The fidel fallback must come last, or the whole interface changes typeface.
        families = getattr(theme_module, which)().families()
        assert families[-1] == theme_module.ethiopic_family()
        assert len(families) > 1

    def test_latin_still_renders(self, theme_module):
        assert ink(theme_module.ui_font(), "Abc") > 20

    def test_the_ethiopic_family_resolves_to_something(self, theme_module):
        assert theme_module.ethiopic_family()


class TestPalette:
    def test_both_themes_define_every_colour(self, theme_module):
        light, dark = theme_module.LIGHT, theme_module.DARK
        assert set(vars(light)) == set(vars(dark))

    def test_confidence_colours_distinguish_good_from_bad(self, theme_module):
        good = theme_module.confidence_color(0.98, theme_module.LIGHT)
        bad = theme_module.confidence_color(0.20, theme_module.LIGHT)
        assert good.rgba() != bad.rgba()

    def test_severity_colours_are_distinct(self, theme_module):
        colors = {
            theme_module.severity_color(theme_module.LIGHT, s).name()
            for s in ("high", "medium", "low")
        }
        assert len(colors) == 3

    @pytest.mark.parametrize("palette_name", ["LIGHT", "DARK"])
    def test_the_stylesheet_builds_for_both_themes(self, theme_module, palette_name):
        sheet = theme_module.stylesheet(getattr(theme_module, palette_name))
        assert "QWidget" in sheet
        assert "{" in sheet and "}" in sheet

    def test_the_stylesheet_sets_no_font_family(self, theme_module):
        # A stylesheet font silently overrides QWidget.setFont(), which would replace the
        # Ethiopic face on the editor with something that cannot draw the script.
        sheet = re.sub(r"/\*.*?\*/", "", theme_module.stylesheet(theme_module.LIGHT), flags=re.S)
        assert "font-family" not in sheet
