"""Shared EngineListener base for Rich-based displays.

Tracks the per-segment speaker-change + translation-update state so each
display subclass can focus on rendering.
"""
from __future__ import annotations

import threading
from typing import Optional

from live_transcribe_core import (
    EngineListener,
    SegmentEvent,
    StatusEvent,
    SummaryEvent,
    TranslationEvent,
)


class BaseDisplay(EngineListener):
    def __init__(self, has_translator: bool):
        self._has_translator = has_translator
        self._last_speaker: Optional[str] = None
        # Map segment_id -> SegmentEvent, needed so late-arriving TranslationEvents
        # can re-render a previously printed line.
        self._segments: dict[str, SegmentEvent] = {}
        self.translations: dict[str, str] = {}
        self.summaries: list[SummaryEvent] = []
        self._lock = threading.Lock()

    # EngineListener --------------------------------------------------------

    def on_segment(self, event: SegmentEvent) -> None:
        with self._lock:
            self._segments[event.id] = event
            new_speaker = event.speaker != self._last_speaker
            self._last_speaker = event.speaker
        self._render_segment_header_if_needed(event, new_speaker)
        if not self._has_translator:
            self._render_segment_without_translation(event)

    def on_translation(self, event: TranslationEvent) -> None:
        with self._lock:
            self.translations[event.segment_id] = event.text
            seg = self._segments.get(event.segment_id)
        if seg is None:
            return
        if event.is_update:
            self._render_translation_update(seg, event.text)
        elif event.text:
            self._render_translation(seg, event.text)
        else:
            # No translation available for this segment — render as plain text.
            self._render_segment_without_translation(seg)

    def on_summary(self, event: SummaryEvent) -> None:
        with self._lock:
            self.summaries.append(event)
        self._render_summary(event)

    def on_status(self, event: StatusEvent) -> None:
        if event.state in ("error", "warning") and event.message:
            self._render_error(event.message)

    # Lifecycle (display-specific) ------------------------------------------

    def start(self) -> None: ...
    def stop(self) -> None: ...

    # Render hooks implemented by subclasses --------------------------------

    def _render_segment_header_if_needed(
        self, event: SegmentEvent, new_speaker: bool
    ) -> None:
        raise NotImplementedError

    def _render_segment_without_translation(self, event: SegmentEvent) -> None:
        raise NotImplementedError

    def _render_translation(self, segment: SegmentEvent, translation: str) -> None:
        raise NotImplementedError

    def _render_translation_update(
        self, segment: SegmentEvent, translation: str
    ) -> None:
        raise NotImplementedError

    def _render_summary(self, event: SummaryEvent) -> None:
        tag = f"FINAL SUMMARY #{event.index}" if event.is_final else f"SUMMARY #{event.index}"
        header = f"{tag} · {event.timestamp}"
        print(f"\n\033[1;35m{'─' * 40}")
        print(f"  {header}")
        print(f"{'─' * 40}\033[0m")
        print(f"\033[0;35m  {event.text}\033[0m")
        print(f"\033[1;35m{'─' * 40}\033[0m\n")

    def _render_error(self, message: str) -> None:
        print(f"\033[0;31m[Error] {message}\033[0m")
