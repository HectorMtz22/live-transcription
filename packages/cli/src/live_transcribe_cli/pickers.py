"""Interactive picker wrappers (questionary).

Each `pick_*` function returns the primitive type the engine needs
(`int`, `str`, `set[str]`, `bool`) or the `BACK` sentinel when the user
selects "← Back". All pickers share `STYLE` and obey KeyboardInterrupt.
"""
from __future__ import annotations

from typing import Any

import questionary
from questionary import Choice

from live_transcribe_core.config import LANG_NAMES

BACK: Any = object()

_BACK_CHOICE = Choice(title=[("class:back", "← Back")], value=BACK)

STYLE = questionary.Style([
    ("qmark", "fg:#00afff bold"),       # cyan
    ("question", "fg:#00afff bold"),    # cyan
    ("answer", "fg:#00ff87 bold"),      # green
    ("pointer", "fg:#00ff87 bold"),     # green
    ("highlighted", "fg:#00ff87 bold"), # green
    ("selected", "fg:#00ff87"),         # green
    ("instruction", "fg:#808080"),      # gray
    ("separator", "fg:#808080"),        # gray
    ("back", "fg:#d7005f"),             # red-ish for ← Back
    ("locked", "fg:#d7af00"),           # yellow-dim
    ("hint", "fg:#d75fff"),             # magenta
])


_TRANSLATORS = [
    ("google", "Google Translate (cloud)"),
    ("deepl", "DeepL (cloud)"),
    ("qwen", "Qwen (local LLM, offline)"),
    ("nllb", "NLLB-200 (local, offline)"),
    ("none", "None (transcription only)"),
]

_DISPLAYS = [
    ("columns", "Columns (side-by-side transcription/translation)"),
    ("chat", "Chat (bubble UI per speaker)"),
]


def pick_device(devices: list[tuple[int, str]], default_idx: int) -> int:
    """Arrow-key list of (idx, name) input devices. No ← Back (first picker).

    Returns the selected device index.
    """
    choices = [
        Choice(title=f"[{idx}] {name}", value=idx) for idx, name in devices
    ]
    result = questionary.select(
        "Input device:",
        choices=choices,
        default=next((c for c in choices if c.value == default_idx), None),
        style=STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()
    return result


def pick_translator(default: str, show_back: bool) -> str | Any:
    choices = [Choice(title=label, value=value) for value, label in _TRANSLATORS]
    if show_back:
        choices.append(_BACK_CHOICE)
    return questionary.select(
        "Translation service:",
        choices=choices,
        default=next((c for c in choices if c.value == default), None),
        style=STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()


def pick_translate_from(default: set[str], show_back: bool) -> set[str] | Any:
    lang_choices = [
        Choice(
            title=f"{name} ({code})",
            value=code,
            checked=code in default,
        )
        for code, name in LANG_NAMES.items()
    ]
    result = questionary.checkbox(
        "Translate FROM (space toggles, enter confirms):",
        choices=lang_choices,
        style=STYLE,
        instruction="(↑↓ navigate · space toggle · enter confirm)",
    ).unsafe_ask()
    # questionary.checkbox cannot embed a ← Back entry; offer it as a follow-up
    # only when the user selected nothing AND show_back is True.
    if not result and show_back:
        go_back = questionary.confirm(
            "No languages selected. Go back?",
            default=True,
            style=STYLE,
        ).unsafe_ask()
        if go_back:
            return BACK
    return set(result) if result else {"ko"}


def pick_translate_to(default: str, show_back: bool) -> str | Any:
    choices = [
        Choice(title=f"{name} ({code})", value=code)
        for code, name in LANG_NAMES.items()
    ]
    if show_back:
        choices.append(_BACK_CHOICE)
    return questionary.select(
        "Translate TO:",
        choices=choices,
        default=next((c for c in choices if c.value == default), None),
        style=STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()


def pick_display(default: str, show_back: bool) -> str | Any:
    choices = [Choice(title=label, value=value) for value, label in _DISPLAYS]
    if show_back:
        choices.append(_BACK_CHOICE)
    return questionary.select(
        "Display mode:",
        choices=choices,
        default=next((c for c in choices if c.value == default), None),
        style=STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()


def pick_summary(default: bool, show_back: bool) -> bool | Any:
    choices = [
        Choice(title="Off", value=False),
        Choice(title="On (rolling summary via Qwen3-8B)", value=True),
    ]
    if show_back:
        choices.append(_BACK_CHOICE)
    return questionary.select(
        "Live summary (local LLM):",
        choices=choices,
        default=next((c for c in choices if c.value == default), None),
        style=STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()
