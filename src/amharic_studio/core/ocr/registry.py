"""Backend registry.

Backends register themselves here so the UI can list what is actually usable on this
machine, and so adding Kraken or Calamari later touches nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import OcrBackend

_FACTORIES: dict[str, Callable[[], OcrBackend]] = {}
_INSTANCES: dict[str, OcrBackend] = {}


def register_backend(name: str, factory: Callable[[], OcrBackend]) -> None:
    _FACTORIES[name] = factory
    _INSTANCES.pop(name, None)


def get_backend(name: str) -> OcrBackend | None:
    if name in _INSTANCES:
        return _INSTANCES[name]
    factory = _FACTORIES.get(name)
    if factory is None:
        return None
    instance = factory()
    _INSTANCES[name] = instance
    return instance


def all_backends() -> list[OcrBackend]:
    return [b for b in (get_backend(name) for name in sorted(_FACTORIES)) if b is not None]


def available_backends() -> list[OcrBackend]:
    return [b for b in all_backends() if b.available()]


def default_backend() -> OcrBackend | None:
    backends = available_backends()
    return backends[0] if backends else None


def _register_builtins() -> None:
    from .tesseract import TesseractBackend

    register_backend("tesseract", TesseractBackend)


_register_builtins()
