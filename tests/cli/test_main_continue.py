"""main._continue_or_none: the --continue branching helper in main.py.

These tests cover the main-level wiring (notice text, fallback signalling)
without invoking argparse or the engine.
"""

from __future__ import annotations

import argparse

import pytest

from live_transcribe_cli import main


def _args(continue_=False, **overrides):
    """argparse.Namespace with all wizard-read fields defaulting to None."""
    return argparse.Namespace(
        device=overrides.get("device"),
        translator=overrides.get("translator"),
        translate_from=overrides.get("translate_from"),
        translate_to=overrides.get("translate_to"),
        display=overrides.get("display"),
        summary=overrides.get("summary"),
        continue_=continue_,
    )


@pytest.fixture
def devices():
    return [(0, "Mic"), (1, "BlackHole 2ch")]


@pytest.fixture(autouse=True)
def stub_wizard_deps(monkeypatch):
    """Keep _continue_or_none from triggering real audio/Rich I/O."""
    monkeypatch.setattr(
        main.wizard, "find_blackhole_device", lambda: (1, "BlackHole 2ch")
    )

    class _DefaultDevice:
        device = (0, 0)

    monkeypatch.setattr(main.wizard.sd, "default", _DefaultDevice())
    # render_summary touches Rich — stub it. We test it elsewhere (none here
    # exercises its rendering); _continue_or_none only needs to know it's called.
    calls = []

    def _stub_render(choices, *, devices, model_repo, diarize, title="Continuing with last session"):
        calls.append(
            {"choices": choices, "devices": devices, "model_repo": model_repo,
             "diarize": diarize, "title": title}
        )

    monkeypatch.setattr(main.wizard, "render_summary", _stub_render)
    return calls


def test_returns_none_when_continue_flag_unset(devices, capsys):
    """When --continue is not set, the helper is a no-op (caller runs wizard)."""
    result = main._continue_or_none(
        _args(continue_=False),
        {"device_name": "Mic", "translator": "google"},
        devices,
        model_repo="repo",
        diarize=False,
    )
    assert result is None
    captured = capsys.readouterr()
    assert "No previous session" not in captured.out


def test_prints_notice_and_returns_none_when_no_last_run(devices, capsys):
    result = main._continue_or_none(
        _args(continue_=True),
        None,
        devices,
        model_repo="repo",
        diarize=False,
    )
    assert result is None
    captured = capsys.readouterr()
    assert "No previous session" in captured.out
    assert "wizard" in captured.out


def test_returns_choices_and_renders_summary_with_devices(
    devices, capsys, stub_wizard_deps
):
    last_run = {
        "device_name": "Mic",
        "translator": "google",
        "translate_from": ["ko"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    result = main._continue_or_none(
        _args(continue_=True),
        last_run,
        devices,
        model_repo="repo-x",
        diarize=True,
    )
    assert result is not None
    assert result.device_idx == 0
    assert result.translator == "google"

    # render_summary was called with the SAME devices list — not re-queried.
    assert len(stub_wizard_deps) == 1
    call = stub_wizard_deps[0]
    assert call["devices"] is devices
    assert call["model_repo"] == "repo-x"
    assert call["diarize"] is True


def test_cli_override_applied_and_persists_through_helper(devices, stub_wizard_deps):
    """A CLI override (--display chat) on top of --continue is respected."""
    last_run = {
        "device_name": "Mic",
        "translator": "google",
        "translate_from": ["ko"],
        "translate_to": "en",
        "display": "columns",
        "summary": False,
    }
    result = main._continue_or_none(
        _args(continue_=True, display="chat"),
        last_run,
        devices,
        model_repo="repo",
        diarize=False,
    )
    assert result is not None
    assert result.display == "chat"
