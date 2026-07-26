"""Orchestration: recognition, correction, learning and training-data harvest.

This layer owns the loop that makes the application improve as it is used:

1. Recognize a page, optionally with more than one engine, and reconcile the readings.
2. Normalize away unambiguous damage, then raise everything else as reviewable issues.
3. When the editor accepts a correction, feed it back into the confusion model, offer to
   propagate it across the whole book, and add the word to the user glossary.
4. When a page is marked verified, crop its line images and store them as ground truth for
   fine-tuning a recognizer on this specific book.

Everything is synchronous and cancellable, so the UI can run it on a worker thread without
this module knowing that Qt exists.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import confusion, imaging, normalize, suggest
from .confusion import ConfusionModel
from .lexicon import Lexicon
from .normalize import Normalizer, Policy
from .ocr import registry
from .ocr.base import OcrResult, PageSegMode
from .ocr.voting import VoteOutcome, vote
from .project import LineBox, PageStatus, Project, WordBox
from .suggest import Issue, Suggester

ProgressFn = Callable[[int, int, str], None]


@dataclass
class EngineSpec:
    """One recognizer run: a backend and the model it should use."""

    backend: str = "tesseract"
    language: str = "amh"

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.language}"


@dataclass
class RecognizeOptions:
    engines: list[EngineSpec] = field(default_factory=lambda: [EngineSpec()])
    psm: PageSegMode = PageSegMode.SINGLE_BLOCK
    preprocess: bool = False  # images are usually preprocessed at import time
    preprocess_options: imaging.PreprocessOptions | None = None
    normalize_text: bool = True
    policy: Policy = Policy.FAITHFUL
    auto_correct: bool = False
    analyze: bool = True
    overwrite_existing: bool = False


@dataclass
class PageOutcome:
    page_id: int
    label: str = ""
    text: str = ""
    issues: list[Issue] = field(default_factory=list)
    auto_applied: int = 0
    normalization_changes: int = 0
    agreement_rate: float = 1.0
    mean_confidence: float = 0.0
    engines: list[str] = field(default_factory=list)
    error: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.error


class Pipeline:
    def __init__(
        self,
        project: Project,
        lexicon: Lexicon,
        model: ConfusionModel | None = None,
        suggester: Suggester | None = None,
    ) -> None:
        self.project = project
        self.lexicon = lexicon
        self.model = model or ConfusionModel()
        self.suggester = suggester or Suggester(lexicon, self.model)

    # -- recognition -----------------------------------------------------------------------

    def recognize_page(self, page_id: int, options: RecognizeOptions | None = None) -> PageOutcome:
        options = options or RecognizeOptions()
        page = self.project.get_page(page_id)
        if page is None:
            return PageOutcome(page_id, error="page not found")

        outcome = PageOutcome(page_id, page.label)
        if page.status is PageStatus.VERIFIED and not options.overwrite_existing:
            outcome.skipped = True
            outcome.text = self.project.get_edited(page_id)
            return outcome

        image_path = self.project.image_path(page)
        if image_path is None or not image_path.exists():
            outcome.error = "no page image; nothing to recognize"
            return outcome

        image = Image.open(image_path)
        if options.preprocess:
            image, _report = imaging.preprocess(image, options.preprocess_options)

        results: list[OcrResult] = []
        for spec in options.engines:
            backend = registry.get_backend(spec.backend)
            if backend is None or not backend.available():
                continue
            results.append(backend.recognize(image, language=spec.language, psm=options.psm))

        if not results:
            outcome.error = "no OCR backend available"
            return outcome

        voted = vote(results, self.lexicon, self.model)
        if not voted.result.ok:
            outcome.error = voted.result.error
            return outcome

        return self._store_result(page_id, page.label, voted, options, outcome)

    def _store_result(
        self,
        page_id: int,
        label: str,
        voted: VoteOutcome,
        options: RecognizeOptions,
        outcome: PageOutcome,
    ) -> PageOutcome:
        result = voted.result
        text = result.text

        if options.normalize_text:
            normalized = Normalizer(options.policy).normalize(text)
            outcome.normalization_changes = normalized.total_changes
            # Normalization can change length, which would invalidate word offsets, so the
            # geometry is only trusted when the text is byte-identical afterwards.
            if len(normalized.text) == len(text):
                text = normalized.text
            elif normalized.total_changes:
                text = normalized.text
                for word in result.words:
                    word.start = word.end = -1

        self.project.set_raw_text(page_id, text)
        self.project.set_words(
            page_id,
            [
                WordBox(w.text, w.x, w.y, w.w, w.h, w.conf, w.line_idx, w.word_idx, w.start, w.end)
                for w in result.words
            ],
            engine=result.engine,
        )
        self.project.set_lines(
            page_id,
            [
                LineBox(ln.line_idx, ln.x, ln.y, ln.w, ln.h, ln.text, ln.conf, ln.baseline)
                for ln in result.lines
            ],
        )
        self.project.set_page_status(page_id, PageStatus.RECOGNIZED)

        if options.analyze:
            outcome.issues = self.suggester.analyze(
                text, result.confidence_spans(), voted.disagreements
            )
            if options.auto_correct:
                corrected, applied = suggest.auto_apply(text, outcome.issues)
                if corrected != text:
                    self.project.set_edited_text(page_id, corrected)
                    for issue in applied:
                        best = issue.best
                        if best:
                            self.project.record_edit(
                                page_id, issue.text, best.text, "auto", best.source
                            )
                    text = corrected
                outcome.auto_applied = len(applied)

        outcome.text = text
        outcome.label = label
        outcome.agreement_rate = voted.agreement_rate
        outcome.mean_confidence = result.mean_confidence
        outcome.engines = voted.engines
        return outcome

    def recognize_pages(
        self,
        page_ids: Sequence[int],
        options: RecognizeOptions | None = None,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> list[PageOutcome]:
        outcomes: list[PageOutcome] = []
        total = len(page_ids)
        for index, page_id in enumerate(page_ids):
            if cancel is not None and cancel.is_set():
                break
            page = self.project.get_page(page_id)
            if progress:
                progress(index, total, page.label if page else str(page_id))
            outcomes.append(self.recognize_page(page_id, options))
        if progress:
            progress(total, total, "done")
        return outcomes

    # -- corrections -------------------------------------------------------------------------

    def accept_correction(
        self, page_id: int | None, before: str, after: str, source: str = "manual"
    ) -> None:
        """Record an accepted correction and learn from it.

        Three things happen: the ledger records it for later propagation, the confusion
        model shifts its costs toward the substitutions this book actually exhibits, and
        the corrected form is added to the user glossary so it stops being flagged.
        """
        if before == after:
            return
        self.project.record_edit(page_id, before, after, "accepted", source)
        self.model.observe_alignment(before, after)
        for word in after.split():
            if word and not self.lexicon.contains(word):
                self.lexicon.add(word, freq=1, source="user")

    def find_occurrences(self, needle: str) -> list[tuple[int, int, int]]:
        return self.project.occurrences(needle)

    def propagate(
        self,
        before: str,
        after: str,
        page_ids: Sequence[int] | None = None,
        whole_word: bool = True,
        dry_run: bool = False,
    ) -> list[tuple[int, int]]:
        """Apply one correction across the book.

        Returns ``(page_id, replacements)`` per affected page. ``dry_run`` reports what
        would change without touching anything, which is what the review dialog shows
        before the editor commits to a document-wide edit.
        """
        import re

        pattern = re.compile(
            (r"(?<![^\s\u1361])" + re.escape(before) + r"(?![^\s\u1361\u1362-\u1368])")
            if whole_word
            else re.escape(before)
        )

        targets = page_ids if page_ids is not None else [p.id for p in self.project.pages()]
        changed: list[tuple[int, int]] = []
        for page_id in targets:
            text = self.project.get_edited(page_id)
            new_text, count = pattern.subn(after, text)
            if not count:
                continue
            changed.append((page_id, count))
            if not dry_run:
                self.project.set_edited_text(page_id, new_text)
                self.project.record_edit(page_id, before, after, "propagated", "ledger")
        if not dry_run and changed:
            self.model.observe_alignment(before, after)
        return changed

    def apply_ledger(self, min_count: int = 2, dry_run: bool = True) -> dict[str, list[tuple[int, int]]]:
        """Replay every correction seen at least ``min_count`` times over the whole book.

        The usual workflow after correcting the first few dozen pages: the same misreading
        recurs hundreds of times, and this fixes all of them in one reviewable pass.
        """
        out: dict[str, list[tuple[int, int]]] = {}
        for before, (after, _count) in self.project.known_corrections(min_count).items():
            changed = self.propagate(before, after, dry_run=dry_run)
            if changed:
                out[before] = changed
        return out

    # -- normalization ---------------------------------------------------------------------------

    def normalize_page(self, page_id: int, policy: Policy | None = None) -> normalize.NormalizationResult:
        text = self.project.get_edited(page_id)
        result = Normalizer(policy or self.project.policy).normalize(text)
        if result.text != text:
            self.project.set_edited_text(page_id, result.text)
        return result

    def analyze_page(self, page_id: int) -> list[Issue]:
        return self.suggester.analyze(self.project.get_edited(page_id))

    # -- ground truth ------------------------------------------------------------------------------

    def harvest_ground_truth(self, page_id: int) -> int:
        """Crop verified line images and store them as recognizer training data.

        This is what turns human review into recognition accuracy. Old printing is
        typographically consistent within a book, so a few dozen verified pages are enough
        to fine-tune a recognizer that dramatically outperforms the generic model on the
        remaining hundreds.
        """
        page = self.project.get_page(page_id)
        if page is None:
            return 0
        image_path = self.project.image_path(page)
        if image_path is None or not image_path.exists():
            return 0

        text = self.project.get_edited(page_id)
        lines = self.project.get_lines(page_id)
        if not lines:
            return 0

        text_lines = text.split("\n")
        image = Image.open(image_path)
        directory = self.project.path / "lines"
        directory.mkdir(exist_ok=True)

        harvested = 0
        for line in lines:
            if line.line_idx >= len(text_lines):
                continue
            line_text = text_lines[line.line_idx].strip()
            if not line_text or line.w < 8 or line.h < 6:
                continue
            crop = imaging.crop_box(image, (line.x, line.y, line.w, line.h), padding=4)
            filename = f"{page_id:05d}_{line.line_idx:03d}.png"
            crop.save(directory / filename)
            self.project.add_ground_truth(page_id, line.id, f"lines/{filename}", line_text)
            harvested += 1
        return harvested

    def mark_verified(self, page_id: int, harvest: bool = True) -> int:
        self.project.set_page_status(page_id, PageStatus.VERIFIED)
        return self.harvest_ground_truth(page_id) if harvest else 0

    def training_readiness(self) -> tuple[int, int, str]:
        """``(verified pages, ground-truth lines, advice)`` for the retraining prompt."""
        stats = self.project.stats()
        lines = stats.ground_truth_lines
        if lines < 150:
            advice = f"Verify more pages — about {max(0, 150 - lines)} more lines needed before fine-tuning pays off."
        elif lines < 600:
            advice = "Enough to fine-tune. Expect a solid gain on this book's typeface."
        else:
            advice = "Plenty of training data. Fine-tuning should give a large gain."
        return stats.verified, lines, advice

    # -- persistence ------------------------------------------------------------------------------

    def save_model(self, path: Path | None = None) -> Path:
        target = path or (self.project.path / "models" / "confusion.json")
        self.model.save(target)
        return target

    def load_model(self, path: Path | None = None) -> ConfusionModel:
        target = path or (self.project.path / "models" / "confusion.json")
        self.model = confusion.ConfusionModel.load(target)
        self.suggester.model = self.model
        return self.model
