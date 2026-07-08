"""whisper.transcribe forwards decode params — notably the `task` kwarg used
for Whisper-native English translation (task="translate")."""

from __future__ import annotations

import numpy as np

import live_transcribe_core.whisper as whisper_mod


def _patch_mlx(monkeypatch):
    captured = {}

    def fake_mlx(audio, **kwargs):
        captured.update(kwargs)
        return {"language": "en", "segments": []}

    monkeypatch.setattr(whisper_mod.mlx_whisper, "transcribe", fake_mlx)
    return captured


def test_task_defaults_to_transcribe(monkeypatch):
    captured = _patch_mlx(monkeypatch)
    whisper_mod.transcribe(np.zeros(16, dtype=np.float32), "repo", None)
    assert captured["task"] == "transcribe"


def test_task_translate_is_forwarded(monkeypatch):
    captured = _patch_mlx(monkeypatch)
    whisper_mod.transcribe(
        np.zeros(16, dtype=np.float32), "repo", None, task="translate"
    )
    assert captured["task"] == "translate"
