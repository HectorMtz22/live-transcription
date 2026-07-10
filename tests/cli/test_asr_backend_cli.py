"""CLI wiring for the opt-in Qwen3-ASR backend.

Covers argparse flag parsing, wizard visibility/persistence, and the guard that
rejects `--asr-backend qwen` combined with the Whisper-native translator.
"""

from __future__ import annotations

import argparse

import pytest

from live_transcribe_cli import main, wizard


def _args(**overrides):
    return argparse.Namespace(
        device=overrides.get("device"),
        model=overrides.get("model", "full"),
        translator=overrides.get("translator"),
        whisper_mode=overrides.get("whisper_mode"),
        asr_backend=overrides.get("asr_backend"),
        translate_from=overrides.get("translate_from"),
        translate_to=overrides.get("translate_to"),
        display=overrides.get("display"),
        summary=overrides.get("summary"),
        diarize=overrides.get("diarize", "off"),
        continue_=overrides.get("continue_", False),
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


# --- argparse ---------------------------------------------------------------


def test_parser_defaults_asr_backend_to_none():
    parser = main.build_parser()
    ns = parser.parse_args([])
    assert ns.asr_backend is None


def test_parser_accepts_asr_backend_qwen():
    parser = main.build_parser()
    ns = parser.parse_args(["--asr-backend", "qwen"])
    assert ns.asr_backend == "qwen"


def test_parser_rejects_unknown_asr_backend():
    parser = main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--asr-backend", "bogus"])


# --- guard ------------------------------------------------------------------


def test_guard_rejects_qwen_with_whisper_translator():
    err = main._asr_translator_conflict("qwen", "whisper")
    assert err is not None
    assert "qwen" in err.lower()


def test_guard_allows_qwen_with_text_translator():
    assert main._asr_translator_conflict("qwen", "google") is None


def test_guard_allows_whisper_backend_with_whisper_translator():
    assert main._asr_translator_conflict("whisper", "whisper") is None


# --- wizard defaults / persistence ------------------------------------------


def test_choices_default_asr_backend_is_whisper(devices):
    choices = wizard.build_from_last_run(_args(), {"device_name": "Mic"}, devices)
    assert choices.asr_backend == "whisper"


def test_last_run_asr_backend_is_restored(devices):
    last_run = {"device_name": "Mic", "asr_backend": "qwen"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.asr_backend == "qwen"


def test_cli_override_asr_backend(devices):
    last_run = {"device_name": "Mic", "asr_backend": "whisper"}
    choices = wizard.build_from_last_run(_args(asr_backend="qwen"), last_run, devices)
    assert choices.asr_backend == "qwen"


def test_invalid_last_run_asr_backend_falls_back_to_default(devices):
    last_run = {"device_name": "Mic", "asr_backend": "nonsense"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    assert choices.asr_backend == "whisper"


def test_to_persistable_round_trips_asr_backend(devices):
    last_run = {"device_name": "Mic", "asr_backend": "qwen"}
    choices = wizard.build_from_last_run(_args(), last_run, devices)
    record = choices.to_persistable("Mic")
    assert record["asr_backend"] == "qwen"


# --- wizard visibility / locking --------------------------------------------


def test_asr_backend_is_a_visible_wizard_step():
    values = {"translator": "google", "whisper_mode": "dual"}
    visible = wizard._visible_fields(values, locked=set())
    assert "asr_backend" in visible


def test_asr_backend_locked_by_flag():
    locked = wizard._locked_fields(_args(asr_backend="qwen"))
    assert "asr_backend" in locked


def test_render_review_shows_asr_backend(devices, capsys):
    values = {
        "device_idx": 0,
        "asr_backend": "qwen",
        "translator": "google",
        "translate_from": {"ko"},
        "translate_to": "en",
        "display": "columns",
        "summary": False,
        "whisper_mode": "dual",
    }
    wizard._render_review(values, devices, locked=set(), model_repo="repo", diarize=False)
    out = capsys.readouterr().out
    assert "ASR" in out
    assert "Qwen" in out
