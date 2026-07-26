"""Locating and subsetting Ethiopic fonts.

Font handling is not a detail here. Most systems have exactly one Ethiopic font installed
(or none), and a PDF or EPUB that does not *embed* one will render as a page of empty
boxes on the reader's machine. Every export therefore embeds a subsetted face.

Ethiopic is a precomposed syllabary: no contextual forms, no reordering, no mandatory
ligatures. That means a plain glyph-per-codepoint renderer such as ReportLab typesets it
correctly, and the heavy shaping stack that Arabic or Devanagari would require is simply
not needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..paths import bundled_fonts_dir

#: Preferred faces, best first. Abyssinica SIL and Noto Serif Ethiopic are the two
#: high-quality free options; Nyala ships with Windows and is the usual fallback.
PREFERRED_FONTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Abyssinica SIL", ("AbyssinicaSIL-Regular.ttf", "AbyssinicaSIL-R.ttf", "abyssinicasil-r.ttf")),
    ("Noto Serif Ethiopic", ("NotoSerifEthiopic-Regular.ttf", "NotoSerifEthiopic[wdth,wght].ttf")),
    ("Noto Sans Ethiopic", ("NotoSansEthiopic-Regular.ttf", "NotoSansEthiopic[wdth,wght].ttf")),
    ("Nyala", ("nyala.ttf", "Nyala.ttf")),
    ("Ebrima", ("ebrima.ttf",)),
    ("Kefa", ("Kefa.ttc",)),
)

#: A sample covering several families and orders, used to verify glyph coverage.
COVERAGE_SAMPLE = "ሀለሐመሠረሰሸቀበተኀነአከወዐዘየደገጠጨጰጸፀፈፐ፩፲፻።፣፤፥፦፧፨፡"


@dataclass(frozen=True)
class FontInfo:
    family: str
    path: Path
    bold_path: Path | None = None
    italic_path: Path | None = None

    @property
    def name(self) -> str:
        return self.family


def system_font_dirs() -> list[Path]:
    dirs: list[Path] = [bundled_fonts_dir()]
    if sys.platform == "win32":
        windir = Path(__import__("os").environ.get("WINDIR", r"C:\Windows"))
        dirs.append(windir / "Fonts")
        local = __import__("os").environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    else:
        dirs += [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local" / "share" / "fonts",
            Path.home() / ".fonts",
        ]
    return [d for d in dirs if d.exists()]


@lru_cache(maxsize=1)
def find_ethiopic_fonts() -> list[FontInfo]:
    """Every usable Ethiopic font on this machine, in preference order."""
    found: list[FontInfo] = []
    seen: set[Path] = set()

    for family, filenames in PREFERRED_FONTS:
        for directory in system_font_dirs():
            for filename in filenames:
                candidate = directory / filename
                if not candidate.exists():
                    # Directory listings are case-sensitive on Linux; fall back to a scan.
                    matches = list(directory.glob(filename)) or [
                        p for p in directory.iterdir()
                        if p.is_file() and p.name.lower() == filename.lower()
                    ]
                    candidate = matches[0] if matches else candidate
                if candidate.exists() and candidate not in seen:
                    seen.add(candidate)
                    found.append(FontInfo(family, candidate, _sibling(candidate, "bold")))
                    break

    if not found:
        found.extend(_scan_for_coverage())
    return found


def _sibling(path: Path, style: str) -> Path | None:
    """Find a bold or italic companion next to a regular face."""
    stem = path.stem.lower()
    for candidate in path.parent.glob(f"{path.stem.split('-')[0]}*{path.suffix}"):
        name = candidate.stem.lower()
        if name == stem:
            continue
        if style in name or (style == "bold" and name.endswith("bd")):
            return candidate
    return None


def _scan_for_coverage(limit: int = 400) -> list[FontInfo]:
    """Last resort: scan installed fonts for one that actually has Ethiopic glyphs."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:  # pragma: no cover
        return []

    out: list[FontInfo] = []
    checked = 0
    for directory in system_font_dirs():
        for path in directory.glob("*.tt[fc]"):
            if checked >= limit:
                return out
            checked += 1
            try:
                font = TTFont(path, fontNumber=0, lazy=True)
                cmap = font.getBestCmap()
                font.close()
            except Exception:  # noqa: BLE001 - font files are frequently malformed
                continue
            if 0x1200 in cmap and 0x1362 in cmap:
                out.append(FontInfo(path.stem, path))
    return out


def default_font() -> FontInfo | None:
    fonts = find_ethiopic_fonts()
    return fonts[0] if fonts else None


def coverage(font_path: Path, sample: str = COVERAGE_SAMPLE) -> float:
    """Fraction of ``sample`` the font can actually draw."""
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(font_path, fontNumber=0, lazy=True)
        cmap = font.getBestCmap()
        font.close()
    except Exception:  # noqa: BLE001
        return 0.0
    if not sample:
        return 0.0
    return sum(1 for ch in sample if ord(ch) in cmap) / len(sample)


def missing_characters(font_path: Path, text: str) -> set[str]:
    """Characters in ``text`` the font has no glyph for — checked before every export."""
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(font_path, fontNumber=0, lazy=True)
        cmap = font.getBestCmap()
        font.close()
    except Exception:  # noqa: BLE001
        return set()
    return {ch for ch in set(text) if not ch.isspace() and ord(ch) not in cmap}


def subset_font(font_path: Path, text: str, output_path: Path) -> Path:
    """Write a subsetted copy containing only the glyphs ``text`` needs.

    A full Ethiopic face runs to hundreds of kilobytes; a book typically uses a few hundred
    distinct characters, so subsetting keeps EPUB files small enough to be pleasant.
    """
    from fontTools import subset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.recalc_bounds = True
    options.drop_tables = ["FFTM"]

    font = subset.load_font(str(font_path), options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=text + COVERAGE_SAMPLE + " \n")
    subsetter.subset(font)
    subset.save_font(font, str(output_path), options)
    font.close()
    return output_path


def register_with_reportlab(font: FontInfo, alias: str = "Ethiopic") -> str:
    """Register a font with ReportLab and return the name to use in exports."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont as RLFont

    if alias in pdfmetrics.getRegisteredFontNames():
        return alias

    pdfmetrics.registerFont(RLFont(alias, str(font.path)))
    bold_alias = f"{alias}-Bold"
    if font.bold_path and font.bold_path.exists():
        pdfmetrics.registerFont(RLFont(bold_alias, str(font.bold_path)))
    else:
        # No bold face available: map bold onto regular so styled text still renders
        # rather than falling back to Helvetica, which has no fidel at all.
        pdfmetrics.registerFont(RLFont(bold_alias, str(font.path)))

    from reportlab.lib.fonts import addMapping

    addMapping(alias, 0, 0, alias)
    addMapping(alias, 1, 0, bold_alias)
    addMapping(alias, 0, 1, alias)
    addMapping(alias, 1, 1, bold_alias)
    return alias


class FontError(RuntimeError):
    """Raised when no font capable of rendering Ethiopic can be found."""


def require_font() -> FontInfo:
    font = default_font()
    if font is None:
        raise FontError(
            "No Ethiopic font found. Install Abyssinica SIL or Noto Serif Ethiopic, "
            f"or drop a .ttf into {bundled_fonts_dir()}"
        )
    return font
