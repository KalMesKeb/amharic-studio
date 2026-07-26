"""Small ReportLab helpers shared by the two PDF exporters."""

from __future__ import annotations

from reportlab.pdfgen import canvas as rl_canvas


def set_pdf_language(pdf: rl_canvas.Canvas, language: str = "am") -> bool:
    """Tag the document with a natural language.

    Worth doing: it tells PDF readers and assistive technology that the content is
    Amharic, which affects font substitution and text extraction. ReportLab exposes this
    inconsistently across versions, so the catalog is written directly as a fallback.
    """
    setter = getattr(pdf, "setLang", None)
    if callable(setter):
        setter(language)
        return True

    try:
        from reportlab.pdfbase.pdfdoc import PDFString

        pdf._doc.Catalog.Lang = PDFString(language)  # noqa: SLF001 - no public API for this
        return True
    except Exception:  # noqa: BLE001 - language tagging is cosmetic, never fatal
        return False
