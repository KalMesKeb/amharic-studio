"""Pluggable OCR backends."""

from .base import OcrBackend, OcrLine, OcrResult, OcrWord, PageSegMode, assemble_text
from .registry import available_backends, get_backend, register_backend

__all__ = [
    "OcrBackend",
    "OcrLine",
    "OcrResult",
    "OcrWord",
    "PageSegMode",
    "assemble_text",
    "available_backends",
    "get_backend",
    "register_backend",
]
