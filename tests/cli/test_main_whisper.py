"""main.py wiring for the Whisper-native translator: no Translator object,
engine gets whisper_translate mode, and has_translator is true only in dual."""

from __future__ import annotations

from live_transcribe_cli import main
from live_transcribe_cli.wizard import Choices


def _choices(translator, whisper_mode="dual"):
    return Choices(
        device_idx=0,
        translator=translator,
        translate_from={"ko"},
        translate_to="en",
        display="columns",
        summary=False,
        whisper_mode=whisper_mode,
    )


def test_build_translator_whisper_returns_none():
    assert main._build_translator("whisper", "en") is None


def test_whisper_translate_mode_from_choices():
    assert main._whisper_translate_mode(_choices("whisper", "dual")) == "dual"
    assert main._whisper_translate_mode(_choices("whisper", "single")) == "single"
    # Non-whisper translators never enable whisper-native translation.
    assert main._whisper_translate_mode(_choices("google")) is None


def test_has_translator_dual_true_single_false():
    # Dual pass produces translations → display should show the translation lane.
    assert main._has_translator(None, _choices("whisper", "dual")) is True
    # Single pass emits no TranslationEvents → no translation lane.
    assert main._has_translator(None, _choices("whisper", "single")) is False
    # A real text translator is always shown.
    assert main._has_translator(object(), _choices("google")) is True
