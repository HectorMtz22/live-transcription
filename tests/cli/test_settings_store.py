"""settings_store: save/load roundtrip, plus tolerant handling of bad input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from live_transcribe_cli import settings_store


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "live_transcribe" / "last.json"


def test_load_returns_none_when_missing(cfg_home):
    assert settings_store.load_last_run() is None


def test_save_then_load_roundtrip(cfg_home):
    choices = {
        "device_name": "MacBook Pro Microphone",
        "translator": "google",
        "translate_from": ["ko"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    settings_store.save_last_run(choices)

    loaded = settings_store.load_last_run()
    assert loaded is not None
    assert loaded["version"] == settings_store.CONFIG_VERSION
    for k, v in choices.items():
        assert loaded[k] == v


def test_load_returns_none_for_malformed_json(cfg_home):
    cfg_home.parent.mkdir(parents=True, exist_ok=True)
    cfg_home.write_text("{not json")
    assert settings_store.load_last_run() is None


def test_load_returns_none_for_wrong_version(cfg_home):
    cfg_home.parent.mkdir(parents=True, exist_ok=True)
    cfg_home.write_text(json.dumps({"version": 999, "translator": "google"}))
    assert settings_store.load_last_run() is None


def test_load_returns_none_when_top_level_not_dict(cfg_home):
    cfg_home.parent.mkdir(parents=True, exist_ok=True)
    cfg_home.write_text(json.dumps(["not", "a", "dict"]))
    assert settings_store.load_last_run() is None


def test_save_is_atomic_no_tmp_file_left_behind(cfg_home):
    settings_store.save_last_run({"translator": "google"})
    leftovers = list(Path(cfg_home.parent).glob(".last.*.tmp"))
    assert leftovers == []
