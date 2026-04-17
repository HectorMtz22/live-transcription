"""Common Translator protocol shared by all translation backends."""
from __future__ import annotations

import threading
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    """Structural interface all translator backends satisfy.

    `context` is a list of (original, translation) tuples for prior lines.
    Backends that don't use context should accept and ignore it.
    """

    target_lang: str

    def translate(
        self, text: str, source_lang: str, context: Optional[list] = None
    ) -> Optional[str]: ...


def supports_gpu_lock(translator: object) -> bool:
    return hasattr(translator, "set_gpu_lock")


def set_gpu_lock(translator: object, lock: threading.Lock) -> None:
    """Inject a shared GPU lock into translators that need Metal coordination (Qwen)."""
    if supports_gpu_lock(translator):
        translator.set_gpu_lock(lock)
