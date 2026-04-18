"""Chat bubble display — subscribes to EngineListener events.

Messages appear as aligned chat bubbles per speaker, with per-speaker color
and alternating left/right alignment.
"""
from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from live_transcribe_cli.displays.base import BaseDisplay
from live_transcribe_core import SegmentEvent

console = Console()

SPEAKER_COLORS = ["cyan", "green", "magenta", "yellow", "blue"]


class ChatDisplay(BaseDisplay):
    """Chat-style layout with Rich Live for in-place translation updates."""

    def __init__(self, has_translator: bool):
        super().__init__(has_translator=has_translator)
        self._side_map: dict[str, str] = {}
        self._next_side = 0
        self._color_map: dict[str, str] = {}
        self._color_idx = 0
        self._entries: list = []
        self._entry_map: dict[str, int] = {}
        self._live = None
        # Chat bubbles are taller, so fewer fit in the live area
        self._max_entries = max(3, (console.height - 5) // 5)

    def start(self) -> None:
        self._live = Live(
            Group(*self._entries) if self._entries else Text(""),
            console=console,
            auto_refresh=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    # Render hooks ----------------------------------------------------------

    def _render_segment_header_if_needed(
        self, event: SegmentEvent, new_speaker: bool
    ) -> None:
        # No-op for chat mode — speaker info is shown in the bubble title.
        return

    def _render_segment_without_translation(self, event: SegmentEvent) -> None:
        self._append(
            self._render_bubble(
                event.speaker, event.text, lang_tag=event.language,
                timestamp=event.timestamp,
            ),
            entry_key=event.id,
        )

    def _render_translation(self, segment: SegmentEvent, translation: str) -> None:
        t = f"→ {translation}" if translation else None
        self._append(
            self._render_bubble(
                segment.speaker, segment.text, translation=t,
                lang_tag=segment.language, timestamp=segment.timestamp,
            ),
            entry_key=segment.id,
        )

    def _render_translation_update(
        self, segment: SegmentEvent, translation: str
    ) -> None:
        idx = self._entry_map.get(segment.id)
        if idx is None:
            return
        t = f"→ {translation}" if translation else None
        self._entries[idx] = self._render_bubble(
            segment.speaker, segment.text, translation=t,
            lang_tag=segment.language, timestamp=segment.timestamp, updated=True,
        )
        self._refresh()

    # Internal --------------------------------------------------------------

    def _append(self, renderable, entry_key=None):
        if self._live is None:
            console.print(renderable)
            return
        self._entries.append(renderable)
        if entry_key is not None:
            self._entry_map[entry_key] = len(self._entries) - 1
        while len(self._entries) > self._max_entries:
            old = self._entries.pop(0)
            self._live.console.print(old)
            self._entry_map = {k: v - 1 for k, v in self._entry_map.items() if v > 0}
        self._refresh()

    def _refresh(self):
        if self._live and self._entries:
            self._live.update(Group(*self._entries))
            self._live.refresh()

    def _speaker_side(self, speaker):
        if speaker not in self._side_map:
            self._side_map[speaker] = "left" if self._next_side % 2 == 0 else "right"
            self._next_side += 1
        return self._side_map[speaker]

    def _speaker_color(self, speaker):
        if speaker not in self._color_map:
            self._color_map[speaker] = SPEAKER_COLORS[self._color_idx % len(SPEAKER_COLORS)]
            self._color_idx += 1
        return self._color_map[speaker]

    def _render_bubble(self, speaker, text, translation=None, lang_tag=None,
                       timestamp=None, updated=False):
        side = self._speaker_side(speaker)
        color = self._speaker_color(speaker)
        bubble_width = min(console.width * 2 // 3, 60)

        body = Text()
        body.append(text, style="white")
        if lang_tag:
            body.append(f"  [{lang_tag}]", style="dim")
        if translation:
            body.append(f"\n{translation}", style="bold green" if updated else "yellow")

        title = f"{speaker}" + (f"  {timestamp}" if timestamp else "")
        panel = Panel(
            body,
            title=title,
            title_align="left",
            border_style=color,
            width=bubble_width,
            padding=(0, 1),
        )
        return Align(panel, align=side)
