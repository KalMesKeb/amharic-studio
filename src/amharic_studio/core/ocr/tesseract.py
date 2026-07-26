"""Tesseract backend.

Driven through the command line rather than a Python wrapper: it avoids a dependency, it
works with whatever Tesseract build the user already has, and — the reason that actually
matters — the same executable is what performs in-book LSTM fine-tuning later, so the
recognizer and the trainer stay in step.

Word geometry comes from TSV output, which is the only Tesseract format that gives
per-word boxes and confidences together.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from .base import (
    OcrBackend,
    OcrResult,
    OcrWord,
    PageSegMode,
    assemble_text,
    lines_from_words,
    split_wordspace_boxes,
)

#: Where Tesseract usually lands on Windows, checked when it is not on PATH.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
)

#: Amharic-capable traineddata, best first.
AMHARIC_LANGS = ("amh", "amh_old", "amh-layer", "Amharic")


def find_tesseract() -> str | None:
    override = os.environ.get("TESSERACT_CMD")
    if override and Path(override).exists():
        return override
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _WINDOWS_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


class TesseractBackend(OcrBackend):
    name = "tesseract"
    display_name = "Tesseract"

    def __init__(self, executable: str | None = None, tessdata_dir: str | Path | None = None) -> None:
        self.executable = executable or find_tesseract()
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else os.environ.get("TESSDATA_PREFIX")
        self._languages: list[str] | None = None

    # -- capability ------------------------------------------------------------------------

    def available(self) -> bool:
        return bool(self.executable)

    def unavailable_reason(self) -> str:
        if self.executable:
            return ""
        return (
            "Tesseract was not found. Install it and make sure 'tesseract' is on PATH, "
            "or set the TESSERACT_CMD environment variable."
        )

    def version(self) -> str:
        if not self.executable:
            return ""
        try:
            out = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True, timeout=20
            )
            return out.stdout.splitlines()[0].strip() if out.stdout else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def languages(self) -> list[str]:
        if self._languages is not None:
            return self._languages
        if not self.executable:
            self._languages = []
            return self._languages
        try:
            out = subprocess.run(
                [self.executable, "--list-langs", *self._tessdata_args()],
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = (out.stdout or "").splitlines()
            self._languages = sorted(ln.strip() for ln in lines[1:] if ln.strip())
        except (OSError, subprocess.SubprocessError):
            self._languages = []
        return self._languages

    def has_amharic(self) -> bool:
        return any(lang in self.languages() for lang in AMHARIC_LANGS)

    def best_amharic_language(self) -> str:
        installed = self.languages()
        for lang in AMHARIC_LANGS:
            if lang in installed:
                return lang
        return "amh"

    def _tessdata_args(self) -> list[str]:
        return ["--tessdata-dir", self.tessdata_dir] if self.tessdata_dir else []

    # -- recognition -------------------------------------------------------------------------

    def recognize(
        self,
        image: Image.Image,
        language: str = "amh",
        psm: PageSegMode = PageSegMode.SINGLE_BLOCK,
        **options: object,
    ) -> OcrResult:
        if not self.executable:
            return OcrResult(engine=self.name, error=self.unavailable_reason())

        started = time.perf_counter()
        oem = int(options.get("oem", 1))  # 1 = LSTM only
        timeout = float(options.get("timeout", 300))
        extra: list[str] = list(options.get("extra_args", []))  # type: ignore[arg-type]
        # Ethiopic has no case and no Latin-style word-joining rules; leaving dictionary
        # correction on lets Tesseract's Latin-centric heuristics mangle fidel sequences.
        config = [
            "-c", "preserve_interword_spaces=1",
            "-c", "load_system_dawg=0",
            "-c", "load_freq_dawg=0",
        ]

        with tempfile.TemporaryDirectory(prefix="amharic-studio-ocr-") as tmp:
            image_path = Path(tmp) / "page.png"
            image.save(image_path, format="PNG", dpi=(300, 300))
            command = [
                self.executable,
                str(image_path),
                "stdout",
                "-l", language,
                "--psm", str(int(psm)),
                "--oem", str(oem),
                *self._tessdata_args(),
                *config,
                *extra,
                "tsv",
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, timeout=timeout, check=False
                )
            except subprocess.TimeoutExpired:
                return OcrResult(engine=self.name, language=language, error="Tesseract timed out")
            except OSError as exc:
                return OcrResult(engine=self.name, language=language, error=str(exc))

        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            return OcrResult(engine=self.name, language=language, error=message or "Tesseract failed")

        words = parse_tsv(completed.stdout.decode("utf-8", errors="replace"))
        if options.get("split_wordspace", True):
            words = split_wordspace_boxes(words)
        text = assemble_text(words)

        return OcrResult(
            text=text,
            words=words,
            lines=lines_from_words(words),
            engine=self.name,
            language=language,
            duration_s=time.perf_counter() - started,
        )


def parse_tsv(payload: str) -> list[OcrWord]:
    """Parse Tesseract TSV into word boxes with sequential line indices."""
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t", quoting=csv.QUOTE_NONE)
    words: list[OcrWord] = []
    line_keys: dict[tuple[int, int, int, int], int] = {}
    per_line_counter: dict[int, int] = {}

    for row in reader:
        try:
            level = int(row.get("level") or 0)
        except ValueError:
            continue
        if level != 5:  # 5 = word
            continue

        text = (row.get("text") or "").strip()
        if not text:
            continue

        try:
            key = (
                int(row["block_num"]),
                int(row["par_num"]),
                int(row["line_num"]),
                int(row.get("page_num") or 1),
            )
            left, top = int(row["left"]), int(row["top"])
            width, height = int(row["width"]), int(row["height"])
            conf = float(row.get("conf") or -1)
        except (KeyError, TypeError, ValueError):
            continue

        line_idx = line_keys.setdefault(key, len(line_keys))
        word_idx = per_line_counter.get(line_idx, 0)
        per_line_counter[line_idx] = word_idx + 1

        words.append(
            OcrWord(
                text=text,
                x=left,
                y=top,
                w=width,
                h=height,
                conf=max(0.0, conf) / 100.0,
                line_idx=line_idx,
                word_idx=word_idx,
            )
        )

    words.sort(key=lambda w: (w.line_idx, w.word_idx))
    return words
