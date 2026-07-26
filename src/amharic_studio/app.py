"""Entry point.

``amharic-studio`` with no arguments opens the window. A few subcommands exist so the
engine can be driven from a script or a batch job without the GUI, which is useful for
processing a shelf of books unattended.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import APP_NAME, __version__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in {"ocr", "export", "check", "info", "lexicon"}:
        return _run_cli(argv)

    from .ui.mainwindow import run

    return run([sys.argv[0], *argv])


# --------------------------------------------------------------------------------------
# Headless commands
# --------------------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="amharic-studio", description=f"{APP_NAME} {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr = subparsers.add_parser("ocr", help="Recognize every unrecognized page in a project")
    ocr.add_argument("project", type=Path)
    ocr.add_argument("--engine", action="append", default=None, help="backend:language, repeatable")
    ocr.add_argument("--all", action="store_true", help="re-recognize every page")
    ocr.add_argument("--auto-correct", action="store_true")

    export = subparsers.add_parser("export", help="Export a project")
    export.add_argument("project", type=Path)
    export.add_argument("--out", type=Path, default=None)
    export.add_argument(
        "--format", action="append",
        choices=["epub", "pdf", "searchable", "txt", "md", "alto"],
        default=None,
    )

    check = subparsers.add_parser("check", help="Report quality metrics")
    check.add_argument("project", type=Path)
    check.add_argument("--limit", type=int, default=15)

    subparsers.add_parser("info", help="Show what is installed and usable")

    lexicon = subparsers.add_parser("lexicon", help="Add a corpus to the lexicon")
    lexicon.add_argument("files", type=Path, nargs="+")

    args = parser.parse_args(argv)
    return {
        "ocr": _cmd_ocr,
        "export": _cmd_export,
        "check": _cmd_check,
        "info": _cmd_info,
        "lexicon": _cmd_lexicon,
    }[args.command](args)


def _cmd_ocr(args) -> int:
    from .core.lexicon import load_or_create
    from .core.pipeline import EngineSpec, Pipeline, RecognizeOptions
    from .core.project import PageStatus, Project

    project = Project(args.project)
    lexicon = load_or_create()
    pipeline = Pipeline(project, lexicon)
    pipeline.load_model()

    specs = []
    for raw in args.engine or ["tesseract:amh"]:
        backend, _, language = raw.partition(":")
        specs.append(EngineSpec(backend or "tesseract", language or "amh"))

    options = RecognizeOptions(engines=specs, auto_correct=args.auto_correct)
    pages = project.pages() if args.all else [p for p in project.pages() if p.status is PageStatus.NEW]
    if not pages:
        print("Nothing to recognize.")
        return 0

    def progress(done: int, total: int, message: str) -> None:
        print(f"\r[{done}/{total}] {message:<40}", end="", flush=True)

    outcomes = pipeline.recognize_pages([p.id for p in pages], options, progress)
    print()

    failures = [o for o in outcomes if not o.ok]
    for outcome in outcomes:
        if outcome.ok:
            print(
                f"  {outcome.label}: {len(outcome.issues)} issues, "
                f"confidence {outcome.mean_confidence:.0%}, agreement {outcome.agreement_rate:.0%}"
            )
        else:
            print(f"  {outcome.label}: FAILED — {outcome.error}")

    pipeline.save_model()
    project.close()
    print(f"\n{len(outcomes) - len(failures)} pages recognized, {len(failures)} failed.")
    return 1 if failures else 0


def _cmd_export(args) -> int:
    from .core.export import build_document
    from .core.export.epub import export_epub
    from .core.export.pdf_clean import export_pdf
    from .core.export.pdf_searchable import export_searchable_pdf
    from .core.export.text import export_alto, export_hocr, export_markdown, export_text
    from .core.project import Project

    project = Project(args.project)
    out = args.out or (project.path / "exports")
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in (project.title or project.path.stem) if c not in '\\/:*?"<>|').strip()

    wanted = args.format or ["epub", "pdf", "searchable"]
    document = build_document(project)

    for fmt in wanted:
        if fmt == "epub":
            print("EPUB      ", export_epub(document, out / f"{stem}.epub"))
        elif fmt == "pdf":
            path, report = export_pdf(document, out / f"{stem} (typeset).pdf")
            print("PDF       ", path, "—", report.describe())
        elif fmt == "searchable":
            path, report = export_searchable_pdf(project, out / f"{stem} (searchable).pdf")
            print("Searchable", path, "—", report.describe())
        elif fmt == "txt":
            print("Text      ", export_text(document, out / f"{stem}.txt"))
        elif fmt == "md":
            print("Markdown  ", export_markdown(document, out / f"{stem}.md"))
        elif fmt == "alto":
            print("ALTO      ", len(export_alto(project, out / "alto")), "files")
            print("hOCR      ", export_hocr(project, out / f"{stem}.hocr"))

    project.close()
    return 0


def _cmd_check(args) -> int:
    from .core.lexicon import load_or_create
    from .core.project import Project
    from .core.qa import assess_book
    from .core.suggest import Suggester

    project = Project(args.project)
    lexicon = load_or_create()
    book = assess_book(project, lexicon, Suggester(lexicon))

    print(book.summary())
    print(f"\nWorst {args.limit} pages:")
    for quality in book.worst_pages(args.limit):
        print("  " + quality.describe())

    impossible = book.impossible_characters()
    if impossible:
        print("\nCharacters outside the Amharic range (near-certain misreadings):")
        for ch, count in impossible[:15]:
            print(f"  {ch}  ×{count}")

    rare = book.rare_characters()
    if rare:
        print("\nVery rare characters (worth checking for systematic errors):")
        for ch, count in rare[:15]:
            print(f"  {ch}  ×{count}")

    project.close()
    return 0


def _cmd_info(_args) -> int:
    from .core import fonts
    from .core.ocr import registry
    from .paths import models_dir, user_data_dir

    print(f"{APP_NAME} {__version__}\n")

    print("OCR engines")
    backends = registry.all_backends()
    if not backends:
        print("  none registered")
    for backend in backends:
        if backend.available():
            languages = backend.languages()
            amharic = [x for x in languages if "amh" in x.lower() or "ethiopic" in x.lower()]
            version = getattr(backend, "version", lambda: "")()
            print(f"  [ok] {backend.display_name} {version}")
            print(f"       Amharic models: {', '.join(amharic) if amharic else 'NONE — install amh traineddata'}")
        else:
            print(f"  [--] {backend.display_name}: {backend.unavailable_reason()}")

    print("\nEthiopic fonts")
    found = fonts.find_ethiopic_fonts()
    if not found:
        print("  none found — exports will not render fidel. Install Abyssinica SIL.")
    for info in found:
        print(f"  [ok] {info.family:<24} coverage {fonts.coverage(info.path):.0%}  {info.path}")

    print("\nOptional components")
    for module, purpose in (
        ("cv2", "faster preprocessing"),
        ("scipy", "connected components for despeckling"),
        ("kenlm", "n-gram language model scoring"),
        ("onnxruntime", "neural post-OCR correction"),
    ):
        try:
            __import__(module)
            print(f"  [ok] {module:<12} {purpose}")
        except ImportError:
            print(f"  [--] {module:<12} {purpose} (not installed)")

    print(f"\nData directory   {user_data_dir()}")
    print(f"Models directory {models_dir()}")
    return 0


def _cmd_lexicon(args) -> int:
    from .core.lexicon import load_or_create

    lexicon = load_or_create()
    before = len(lexicon)
    total_tokens = 0

    for path in args.files:
        if not path.exists():
            print(f"  skipped (missing): {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = lexicon.ingest_text(text, source=path.name)
        total_tokens += tokens
        print(f"  {path.name}: {tokens:,} tokens")

    lexicon.rebuild_deletes()
    print(f"\nLexicon grew from {before:,} to {len(lexicon):,} words ({total_tokens:,} tokens read).")
    lexicon.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
