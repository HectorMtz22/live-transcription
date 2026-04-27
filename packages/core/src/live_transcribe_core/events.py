"""Event dataclasses emitted by TranscriptionEngine to its listener.

All events are frozen dataclasses with primitive fields so
`dataclasses.asdict(event)` yields a JSON-serializable dict for a
future WebSocket/HTTP transport.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol


@dataclass(frozen=True)
class SegmentEvent:
    id: str            # UUID (uuid.uuid4().hex)
    timestamp: str     # "HH:MM:SS"
    speaker: str       # "Speaker 1", etc. Always "Speaker" when diarize=off.
    text: str
    language: str      # One of SUPPORTED_LANGUAGES


@dataclass(frozen=True)
class TranslationEvent:
    segment_id: str    # References SegmentEvent.id
    text: str
    is_update: bool = False  # True for Qwen retranslate revisions


@dataclass(frozen=True)
class SummaryEvent:
    index: int         # 1-based chunk number
    timestamp: str     # "HH:MM:SS" when the chunk fired
    text: str
    is_final: bool = False   # True for the one emitted at engine shutdown


@dataclass(frozen=True)
class StatusEvent:
    state: Literal["starting", "ready", "stopping", "stopped", "error", "warning", "info"]
    message: Optional[str] = None


class EngineListener(Protocol):
    """Callbacks invoked synchronously from engine worker threads.

    Implementations MUST be thread-safe — multiple workers may emit concurrently.
    Use a lock (or equivalent) when touching shared state in the implementation.
    """

    def on_segment(self, event: SegmentEvent) -> None: ...
    def on_translation(self, event: TranslationEvent) -> None: ...
    def on_summary(self, event: SummaryEvent) -> None: ...
    def on_status(self, event: StatusEvent) -> None: ...
