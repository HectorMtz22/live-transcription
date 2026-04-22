"""Last-run persistence for the CLI wizard.

Stores user choices as JSON at `$XDG_CONFIG_HOME/live_transcribe/last.json`
(falls back to `~/.config/live_transcribe/last.json`). All I/O failures are
non-fatal: load returns None, save logs a warning.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1
_FILENAME = "last.json"


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "live_transcribe"


def _config_path() -> Path:
    return _config_dir() / _FILENAME


def load_last_run() -> dict[str, Any] | None:
    """Return the saved last-run dict, or None if missing/corrupt/wrong version.

    Never raises. Prints a one-line stderr warning on parse error.
    """
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read last-run config: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or data.get("version") != CONFIG_VERSION:
        return None
    return data


def save_last_run(choices: dict[str, Any]) -> None:
    """Atomically write the last-run dict. Swallows OSError with a stderr warning."""
    record = {"version": CONFIG_VERSION, **choices}
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(path.parent),
            prefix=".last.",
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(record, f, indent=2)
            tmp_path = f.name
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"warning: could not save last-run config: {exc}", file=sys.stderr)
