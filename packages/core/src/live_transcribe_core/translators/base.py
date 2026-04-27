"""Common Translator protocol shared by all translation backends."""
from __future__ import annotations

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
