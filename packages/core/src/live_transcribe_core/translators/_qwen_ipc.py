"""IPC message types exchanged between the parent and the Qwen child process.

Frozen dataclasses are sent through `multiprocessing.Queue` (pickled).
Keep fields primitive / picklable — no model handles, no locks, no
file-like objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TranslateRequest:
    request_id: str
    text: str
    source_lang: str
    target_lang: str
    context: Optional[list]  # list of (orig, trans|None) tuples, or None


@dataclass(frozen=True)
class RetranslateBatchRequest:
    request_id: str
    items: list  # list of (text, source_lang) tuples
    target_lang: str
    context: Optional[list]


@dataclass(frozen=True)
class TranslateReply:
    request_id: str
    result: Optional[str]


@dataclass(frozen=True)
class RetranslateBatchReply:
    request_id: str
    results: list  # list of Optional[str], same length as request.items
