"""Whisper-native translator option in the wizard: locks target to English,
carries a dual/single mode, and hides translate_from in single mode."""

from __future__ import annotations

import argparse

import pytest

from live_transcribe_cli import wizard


def _args(**overrides):
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
    return [(0, "Mic"), (1, "BlackHole 2ch")]


@pytest.fixture(autouse=True)
def stub_blackhole(monkeypatch):
    monkeypatch.setattr(wizard, "find_blackhole_device", lambda: (1, "BlackHole 2ch"))

    class _DefaultDevice:
        device = (0, 0)

    monkeypatch.setattr(wizard.sd, "default", _DefaultDevice())


def test_whisper_locks_target_to_english(devices):
    last_run = {
        "device_name": "Mic",
        "translator": "whisper",
        "whisper_mode": "dual",
        "translate_from": ["ko"],
        "translate_to": "es",  # should be overridden to en
        "display": "columns",
        "summary": False,
    }
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.translator == "whisper"
    assert choices.whisper_mode == "dual"
    assert choices.translate_to == "en"


def test_whisper_mode_defaults_to_dual_when_absent(devices):
    last_run = {"device_name": "Mic", "translator": "whisper"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.whisper_mode == "dual"


def test_whisper_mode_cli_override(devices):
    last_run = {"device_name": "Mic", "translator": "whisper", "whisper_mode": "dual"}
    choices = wizard.build_from_last_run(_args(whisper_mode="single"), last_run, devices)
    assert choices.whisper_mode == "single"


def test_to_persistable_round_trips_whisper_mode(devices):
    last_run = {"device_name": "Mic", "translator": "whisper", "whisper_mode": "single"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    record = choices.to_persistable("Mic")
    assert record["translator"] == "whisper"
    assert record["whisper_mode"] == "single"


def test_visible_fields_whisper_dual_shows_from_hides_to():
    values = {"translator": "whisper", "whisper_mode": "dual"}
    visible = wizard._visible_fields(values, locked=set())
    assert "whisper_mode" in visible
    assert "translate_from" in visible
    assert "translate_to" not in visible


def test_visible_fields_whisper_single_hides_from_and_to():
    values = {"translator": "whisper", "whisper_mode": "single"}
    visible = wizard._visible_fields(values, locked=set())
    assert "whisper_mode" in visible
    assert "translate_from" not in visible
    assert "translate_to" not in visible


def test_visible_fields_non_whisper_hides_whisper_mode():
    values = {"translator": "google", "whisper_mode": "dual"}
    visible = wizard._visible_fields(values, locked=set())
    assert "whisper_mode" not in visible
    assert "translate_from" in visible
    assert "translate_to" in visible


def test_render_review_shows_whisper_mode_and_locked_english(devices, capsys):
    values = {
        "device_idx": 0,
        "translator": "whisper",
        "translate_from": {"ko"},
        "translate_to": "en",
        "display": "columns",
        "summary": False,
        "whisper_mode": "dual",
    }
    wizard._render_review(values, devices, locked=set(), model_repo="repo", diarize=False)
    out = capsys.readouterr().out
    assert "Whisper mode" in out
    assert "English" in out


def test_render_review_single_mode_omits_translate_from(devices, capsys):
    values = {
        "device_idx": 0,
        "translator": "whisper",
        "translate_from": {"ko"},
        "translate_to": "en",
        "display": "columns",
        "summary": False,
        "whisper_mode": "single",
    }
    wizard._render_review(values, devices, locked=set(), model_repo="repo", diarize=False)
    out = capsys.readouterr().out
    assert "Translate from" not in out
