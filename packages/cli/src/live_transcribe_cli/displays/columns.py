"""Column-based display — subscribes to EngineListener events."""
from __future__ import annotations

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from live_transcribe_cli.displays.base import BaseDisplay
from live_transcribe_core import SegmentEvent

console = Console()


class ColumnsDisplay(BaseDisplay):
    def __init__(self, has_translator: bool):
        super().__init__(has_translator=has_translator)
        self._entries: list = []
        self._entry_map: dict[str, int] = {}   # segment_id -> index in self._entries
        self._live = None
        self._max_entries = max(5, (console.height - 5) // 3)

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
        if not new_speaker:
            return
        tw, indent, left_w, _ = self._get_col_widths()
        parts = [Text(f"\n{'─' * tw}", style="dim")]
        parts.append(Text(f"[{event.timestamp}] {event.speaker}:", style="bold cyan"))
        if self._has_translator:
            header = Text()
            header.append(" " * indent)
            header.append(self._pad_display("TRANSCRIPTION", left_w), style="dim bold")
            header.append(" │ ", style="dim")
            header.append("TRANSLATION", style="dim bold")
            parts.append(header)
        self._append(Group(*parts))

    def _render_segment_without_translation(self, event: SegmentEvent) -> None:
        content = Text()
        content.append("  ")
        content.append(event.text, style="white")
        content.append(f"  [{event.language}]", style="dim")
        self._append(content, entry_key=event.id)

    def _render_translation(self, segment: SegmentEvent, translation: str) -> None:
        final_left = f"{segment.text} [{segment.language}]"
        right = f"→ {translation}" if translation else ""
        self._append(self._render_columns(final_left, right), entry_key=segment.id)

    def _render_translation_update(
        self, segment: SegmentEvent, translation: str
    ) -> None:
        idx = self._entry_map.get(segment.id)
        if idx is None:
            return
        final_left = f"{segment.text} [{segment.language}]"
        right = f"→ {translation}" if translation else ""
        self._entries[idx] = self._render_columns(
            final_left, right, right_style="bold green"
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

    def _get_col_widths(self):
        tw = console.width
        sep_len = 3
        indent = 2
        usable = tw - indent - sep_len
        left_w = usable // 2
        right_w = usable - left_w
        return tw, indent, left_w, right_w

    @staticmethod
    def _wrap_display(text, width):
        if not text:
            return [""]
        words = text.split()
        lines, line, line_w = [], "", 0
        for word in words:
            w = cell_len(word)
            if line:
                if line_w + 1 + w <= width:
                    line += " " + word
                    line_w += 1 + w
                else:
                    lines.append(line)
                    line = word
                    line_w = w
            else:
                line = word
                line_w = w
        if line:
            lines.append(line)
        return lines or [""]

    @staticmethod
    def _pad_display(text, width):
        pad = width - cell_len(text)
        return text + " " * max(pad, 0)

    def _render_columns(self, left_str, right_str, left_style="white", right_style="yellow"):
        _, indent, left_w, right_w = self._get_col_widths()
        left_lines = self._wrap_display(left_str, left_w)
        right_lines = self._wrap_display(right_str, right_w)
        n = max(len(left_lines), len(right_lines))
        left_lines += [""] * (n - len(left_lines))
        right_lines += [""] * (n - len(right_lines))
        result = Text()
        for i, (left, r) in enumerate(zip(left_lines, right_lines)):
            result.append(" " * indent)
            result.append(self._pad_display(left, left_w), style=left_style)
            result.append(" │ ", style="dim")
            result.append(r, style=right_style)
            if i < n - 1:
                result.append("\n")
        return result
