"""build_from_last_run: non-interactive Choices construction for --continue."""

from __future__ import annotations

import argparse

import pytest

from live_transcribe_cli import wizard


def _args(**overrides):
    """argparse.Namespace with all wizard-read fields defaulting to None."""
    return argparse.Namespace(
        device=overrides.get("device"),
        translator=overrides.get("translator"),
        translate_from=overrides.get("translate_from"),
        translate_to=overrides.get("translate_to"),
        display=overrides.get("display"),
        summary=overrides.get("summary"),
        whisper_mode=overrides.get("whisper_mode"),
    )


@pytest.fixture
def devices():
    return [(0, "MacBook Pro Microphone"), (1, "BlackHole 2ch"), (2, "Pro Tools Aggregate")]


@pytest.fixture(autouse=True)
def stub_blackhole(monkeypatch):
    """Make _resolve_device_default deterministic without real audio hw."""
    monkeypatch.setattr(wizard, "find_blackhole_device", lambda: (1, "BlackHole 2ch"))

    class _DefaultDevice:
        device = (0, 0)

    monkeypatch.setattr(wizard.sd, "default", _DefaultDevice())


def test_returns_none_when_last_run_is_none(devices):
    assert wizard.build_from_last_run(_args(), None, devices) is None


def test_uses_persisted_record_when_present(devices):
    last_run = {
        "device_name": "MacBook Pro Microphone",
        "translator": "deepl",
        "translate_from": ["ko", "es"],
        "translate_to": "en",
        "display": "chat",
        "summary": True,
    }
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices is not None
    assert choices.device_idx == 0
    assert choices.translator == "deepl"
    assert choices.translate_from == {"ko", "es"}
    assert choices.translate_to == "en"
    assert choices.display == "chat"
    assert choices.summary is True


def test_cli_overrides_win_over_last_run(devices):
    last_run = {
        "device_name": "MacBook Pro Microphone",
        "translator": "google",
        "translate_from": ["ko"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    choices = wizard.build_from_last_run(
        _args(display="chat", summary="on"), last_run, devices
    )
    assert choices.display == "chat"
    assert choices.summary is True


def test_cli_device_override_wins(devices):
    last_run = {"device_name": "MacBook Pro Microphone"}
    choices = wizard.build_from_last_run(_args(device=2), last_run, devices)
    assert choices.device_idx == 2


def test_falls_back_when_persisted_device_name_missing(devices):
    """If saved device name isn't present, _resolve_device_default picks BlackHole."""
    last_run = {"device_name": "Unplugged Mic", "translator": "google"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.device_idx == 1  # BlackHole 2ch from fixture


def test_translator_none_clears_translate_from(devices):
    last_run = {
        "device_name": "MacBook Pro Microphone",
        "translator": "none",
        "translate_from": ["ko"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.translator == "none"
    assert choices.translate_from == set()


def test_translate_to_is_dropped_from_translate_from(devices):
    last_run = {
        "device_name": "MacBook Pro Microphone",
        "translator": "google",
        "translate_from": ["ko", "en"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert "en" not in choices.translate_from
    assert "ko" in choices.translate_from
