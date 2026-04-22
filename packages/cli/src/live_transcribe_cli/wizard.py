"""Interactive wizard: runs pickers in order, then a review/edit loop.

Returns a Choices dataclass; main.py calls settings_store.save_last_run()
only after the user confirms Start.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import sounddevice as sd
from questionary import Choice
from questionary import select as q_select

from live_transcribe_core.config import LANG_NAMES

from . import pickers
from .audio import find_blackhole_device

HARD_DEFAULTS = {
    "translator": "google",
    "translate_from": {"ko"},
    "translate_to": "en",
    "display": "columns",
    "summary": False,
}


@dataclass
class Choices:
    device_idx: int
    translator: str
    translate_from: set[str]
    translate_to: str
    display: str
    summary: bool

    def to_persistable(self, device_name: str) -> dict[str, Any]:
        return {
            "device_name": device_name,
            "translator": self.translator,
            "translate_from": sorted(self.translate_from),
            "translate_to": self.translate_to,
            "display": self.display,
            "summary": self.summary,
        }


def _input_devices() -> list[tuple[int, str]]:
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d.get("max_input_channels", 0) > 0
    ]


def _resolve_device_default(last_run: dict | None, devices: list[tuple[int, str]]) -> int:
    """Last-run name → index; else BlackHole; else sd default input."""
    if last_run and isinstance(last_run.get("device_name"), str):
        for idx, name in devices:
            if name == last_run["device_name"]:
                return idx
    bh_idx, _ = find_blackhole_device()
    if bh_idx is not None:
        return bh_idx
    default = sd.default.device[0]
    return default if any(i == default for i, _ in devices) else devices[0][0]


def _seed_defaults(args, last_run: dict | None) -> dict[str, Any]:
    """Merge hard defaults ← last_run ← CLI args into a defaults dict for pickers."""
    out: dict[str, Any] = dict(HARD_DEFAULTS)
    if last_run:
        if last_run.get("translator") in {"google", "deepl", "qwen", "nllb", "none"}:
            out["translator"] = last_run["translator"]
        tf = last_run.get("translate_from")
        if isinstance(tf, list):
            valid = {c for c in tf if c in LANG_NAMES}
            if valid:
                out["translate_from"] = valid
        if last_run.get("translate_to") in LANG_NAMES:
            out["translate_to"] = last_run["translate_to"]
        if last_run.get("display") in {"columns", "chat"}:
            out["display"] = last_run["display"]
        if isinstance(last_run.get("summary"), bool):
            out["summary"] = last_run["summary"]

    if args.translator is not None:
        out["translator"] = args.translator
    if args.translate_from is not None:
        out["translate_from"] = (
            set(LANG_NAMES) if args.translate_from == "all"
            else {c for c in args.translate_from.split(",") if c in LANG_NAMES}
        ) or {"ko"}
    if args.translate_to is not None:
        out["translate_to"] = args.translate_to
    if args.display is not None:
        out["display"] = args.display
    if args.summary is not None:
        out["summary"] = args.summary == "on"
    return out


def _locked_fields(args) -> set[str]:
    locked: set[str] = set()
    if args.device is not None:
        locked.add("device")
    if args.translator is not None:
        locked.add("translator")
    if args.translate_from is not None:
        locked.add("translate_from")
    if args.translate_to is not None:
        locked.add("translate_to")
    if args.display is not None:
        locked.add("display")
    if args.summary is not None:
        locked.add("summary")
    return locked


_ORDER = ["device", "translator", "translate_from", "translate_to", "display", "summary"]


def _run_linear(defaults: dict, locked: set[str], devices: list[tuple[int, str]]) -> dict:
    """Run pickers in order with a back-stack. Returns a values dict."""
    values: dict[str, Any] = dict(defaults)
    visible = [f for f in _ORDER if f not in locked]
    if values["translator"] == "none":
        visible = [f for f in visible if f not in {"translate_from", "translate_to"}]

    i = 0
    while i < len(visible):
        field = visible[i]
        show_back = i > 0
        if field == "device":
            values["device_idx"] = pickers.pick_device(
                devices, values.get("device_idx", devices[0][0])
            )
            i += 1
            continue
        if field == "translator":
            result = pickers.pick_translator(values["translator"], show_back)
            if result is pickers.BACK:
                i -= 1
                continue
            values["translator"] = result
            visible = [f for f in _ORDER if f not in locked]
            if values["translator"] == "none":
                visible = [f for f in visible if f not in {"translate_from", "translate_to"}]
            i = visible.index("translator") + 1
            continue
        if field == "translate_from":
            result = pickers.pick_translate_from(values["translate_from"], show_back)
            if result is pickers.BACK:
                i -= 1
                continue
            values["translate_from"] = result
        elif field == "translate_to":
            result = pickers.pick_translate_to(values["translate_to"], show_back)
            if result is pickers.BACK:
                i -= 1
                continue
            values["translate_to"] = result
        elif field == "display":
            result = pickers.pick_display(values["display"], show_back)
            if result is pickers.BACK:
                i -= 1
                continue
            values["display"] = result
        elif field == "summary":
            result = pickers.pick_summary(values["summary"], show_back)
            if result is pickers.BACK:
                i -= 1
                continue
            values["summary"] = result
        i += 1

    if isinstance(values.get("translate_from"), set) and isinstance(values.get("translate_to"), str):
        values["translate_from"] = values["translate_from"] - {values["translate_to"]}
    return values


def _device_name(devices: list[tuple[int, str]], idx: int) -> str:
    for i, name in devices:
        if i == idx:
            return name
    return f"#{idx}"


def _render_review(values: dict, devices: list[tuple[int, str]], locked: set[str],
                   model_repo: str, diarize: bool) -> None:
    """Print the review summary via Rich panel."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()

    def lock_tag(field: str, flag: str) -> str:
        return f"  [yellow dim](locked by {flag})[/]" if field in locked else ""

    table.add_row(
        "Device:",
        f"[{values['device_idx']}] {_device_name(devices, values['device_idx'])}"
        + lock_tag("device", "--device"),
    )
    table.add_row("Whisper model:", f"{model_repo}  [yellow dim](locked by --model)[/]")
    table.add_row(
        "Speaker diarization:",
        ("ON" if diarize else "OFF") + "  [yellow dim](locked by --diarize)[/]",
    )
    t = values["translator"]
    translator_label = {
        "google": "Google",
        "deepl": "DeepL",
        "qwen": "Qwen (local)",
        "nllb": "NLLB-200 (local)",
        "none": "None",
    }[t]
    table.add_row("Translator:", translator_label + lock_tag("translator", "--translator"))
    if t != "none":
        from_names = ", ".join(sorted(
            f"{LANG_NAMES.get(c, c)} ({c})" for c in values["translate_from"]
        )) or "(none)"
        to_name = (
            f"{LANG_NAMES.get(values['translate_to'], values['translate_to'])} "
            f"({values['translate_to']})"
        )
        table.add_row("Translate from:", from_names + lock_tag("translate_from", "--translate-from"))
        table.add_row("Translate to:", to_name + lock_tag("translate_to", "--translate-to"))
    table.add_row("Display:", values["display"].title() + lock_tag("display", "--display"))
    table.add_row("Summary:", ("ON" if values["summary"] else "OFF") + lock_tag("summary", "--summary"))

    Console().print(
        Panel(
            table,
            title="[bold cyan]Ready to start[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _review_action(values: dict, locked: set[str]) -> str:
    """Return one of: 'start', 'quit', 'edit:<field>'."""
    choices: list[Choice] = [Choice(title="▶ Start", value="start")]
    edit_items = [
        ("device", "✎ Edit device"),
        ("translator", "✎ Edit translator"),
        ("translate_from", "✎ Edit translate from"),
        ("translate_to", "✎ Edit translate to"),
        ("display", "✎ Edit display"),
        ("summary", "✎ Edit summary"),
    ]
    for field, label in edit_items:
        if field in locked:
            continue
        if values["translator"] == "none" and field in {"translate_from", "translate_to"}:
            continue
        choices.append(Choice(title=label, value=f"edit:{field}"))
    choices.append(Choice(title="✖ Quit", value="quit"))
    return q_select(
        "What next?",
        choices=choices,
        style=pickers.STYLE,
        instruction="(↑↓ navigate · enter confirm)",
    ).unsafe_ask()


def _edit_single(field: str, values: dict, devices: list[tuple[int, str]]) -> None:
    """Mutate values in place. Back cancels (no change)."""
    if field == "device":
        values["device_idx"] = pickers.pick_device(devices, values["device_idx"])
    elif field == "translator":
        r = pickers.pick_translator(values["translator"], show_back=True)
        if r is not pickers.BACK:
            values["translator"] = r
    elif field == "translate_from":
        r = pickers.pick_translate_from(values["translate_from"], show_back=True)
        if r is not pickers.BACK:
            values["translate_from"] = r
    elif field == "translate_to":
        r = pickers.pick_translate_to(values["translate_to"], show_back=True)
        if r is not pickers.BACK:
            values["translate_to"] = r
    elif field == "display":
        r = pickers.pick_display(values["display"], show_back=True)
        if r is not pickers.BACK:
            values["display"] = r
    elif field == "summary":
        r = pickers.pick_summary(values["summary"], show_back=True)
        if r is not pickers.BACK:
            values["summary"] = r


def run(args, last_run: dict | None, *, model_repo: str, diarize: bool) -> Choices | None:
    """Drive the wizard. Returns Choices on Start, None on Quit or Ctrl+C."""
    devices = _input_devices()
    if not devices:
        print("error: no input devices found", file=sys.stderr)
        sys.exit(1)

    defaults = _seed_defaults(args, last_run)
    defaults["device_idx"] = _resolve_device_default(last_run, devices)
    if args.device is not None:
        defaults["device_idx"] = args.device

    locked = _locked_fields(args)

    try:
        if len(locked) < len(_ORDER):
            values = _run_linear(defaults, locked, devices)
        else:
            values = dict(defaults)

        while True:
            _render_review(values, devices, locked, model_repo=model_repo, diarize=diarize)
            action = _review_action(values, locked)
            if action == "start":
                break
            if action == "quit":
                return None
            if action.startswith("edit:"):
                _edit_single(action.split(":", 1)[1], values, devices)
                if isinstance(values.get("translate_from"), set):
                    values["translate_from"] = values["translate_from"] - {values["translate_to"]}
    except KeyboardInterrupt:
        return None

    return Choices(
        device_idx=values["device_idx"],
        translator=values["translator"],
        translate_from=values["translate_from"] if values["translator"] != "none" else set(),
        translate_to=values["translate_to"],
        display=values["display"],
        summary=values["summary"],
    )
