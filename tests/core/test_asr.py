"""ASR backend unit tests.

Covers:
- WhisperAsr delegates verbatim to the injected transcribe fn (no lang_hint leak).
- QwenAsr maps the model output → the single-segment dict the engine expects.
- QwenAsr raises a clear, extra-naming error when qwen3_asr_mlx is absent.
- build_asr selects the backend by name and rejects unknown ones.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from live_transcribe_core.asr import QwenAsr, WhisperAsr, build_asr
from live_transcribe_core.config import QWEN_ASR_MODEL, SAMPLE_RATE
from live_transcribe_core.engine import EngineConfig


# --- WhisperAsr -------------------------------------------------------------


def test_whisper_asr_delegates_with_exact_args():
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured["audio"] = audio
        captured.update(kwargs)
        return {"language": "en", "segments": []}

    asr = WhisperAsr("some-repo", transcribe_fn=fake_transcribe)
    audio = np.zeros(16, dtype=np.float32)
    result = asr.transcribe(audio, initial_prompt="hi", task="transcribe")

    assert result == {"language": "en", "segments": []}
    assert captured["audio"] is audio
    assert captured["model_repo"] == "some-repo"
    assert captured["initial_prompt"] == "hi"
    assert captured["task"] == "transcribe"
    # lang_hint is Whisper-irrelevant and must NOT be forwarded to transcribe().
    assert "lang_hint" not in captured


def test_whisper_asr_forwards_translate_task():
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        return {"language": "ko", "segments": []}

    asr = WhisperAsr("repo", transcribe_fn=fake_transcribe)
    asr.transcribe(np.zeros(4, dtype=np.float32), initial_prompt=None, task="translate")
    assert captured["task"] == "translate"


def test_whisper_asr_warmup_calls_transcribe_with_dummy():
    calls = []

    def fake_transcribe(audio, **kwargs):
        calls.append((audio, kwargs))
        return {"language": "en", "segments": []}

    asr = WhisperAsr("warm-repo", transcribe_fn=fake_transcribe)
    asr.warmup()

    assert len(calls) == 1
    audio, kwargs = calls[0]
    assert audio.shape == (SAMPLE_RATE,)
    assert kwargs["model_repo"] == "warm-repo"
    assert kwargs["initial_prompt"] is None


# --- QwenAsr ----------------------------------------------------------------


class _FakeQwenResult:
    def __init__(self, text, language=None):
        self.text = text
        self.language = language


class _FakeQwenModel:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def transcribe(self, audio, language=None, temperature=0.0):
        self.calls.append({"n": len(audio), "language": language, "temperature": temperature})
        return self._result


def _install_fake_qwen(monkeypatch, result):
    model = _FakeQwenModel(result)
    seen = {}

    class _FakeQwen3ASR:
        @classmethod
        def from_pretrained(cls, repo):
            seen["repo"] = repo
            return model

    mod = types.ModuleType("qwen3_asr_mlx")
    mod.Qwen3ASR = _FakeQwen3ASR
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", mod)
    return model, seen


def test_qwen_asr_synthesizes_single_segment(monkeypatch):
    model, seen = _install_fake_qwen(
        monkeypatch, _FakeQwenResult("전사된 텍스트", language="ko")
    )
    asr = QwenAsr()
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1.0s
    result = asr.transcribe(audio, initial_prompt="ignored", task="transcribe", lang_hint="ko")

    assert result["language"] == "ko"
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "전사된 텍스트"
    assert seg["start"] == 0.0
    assert seg["end"] == pytest.approx(1.0)
    # Confidence gate fields the engine reads must pass its filter.
    assert seg["avg_logprob"] >= -1.0
    assert seg["no_speech_prob"] <= 0.6
    # Model was loaded from the configured repo and given the language hint.
    assert seen["repo"] == QWEN_ASR_MODEL
    assert model.calls[0]["language"] == "ko"
    assert model.calls[0]["temperature"] == 0.0


def test_qwen_asr_falls_back_to_lang_hint_when_model_omits_language(monkeypatch):
    _install_fake_qwen(monkeypatch, _FakeQwenResult("hello", language=None))
    asr = QwenAsr()
    result = asr.transcribe(np.zeros(8000, dtype=np.float32), lang_hint="en")
    assert result["language"] == "en"
    assert result["segments"][0]["end"] == pytest.approx(0.5)


def test_qwen_asr_maps_full_language_name_to_iso_code(monkeypatch):
    # C1: qwen3_asr_mlx returns a full language name ("Korean"), but the
    # engine's SUPPORTED_LANGUAGES gate expects an ISO code ("ko"). Without
    # normalization every Qwen segment is silently dropped.
    _install_fake_qwen(monkeypatch, _FakeQwenResult("전사된 텍스트", language="Korean"))
    asr = QwenAsr()
    result = asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), lang_hint="ko")
    assert result["language"] == "ko"


def test_qwen_asr_maps_english_language_name_to_iso_code(monkeypatch):
    _install_fake_qwen(monkeypatch, _FakeQwenResult("hello", language="English"))
    asr = QwenAsr()
    result = asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), lang_hint="en")
    assert result["language"] == "en"


def test_qwen_asr_passes_through_already_iso_language(monkeypatch):
    _install_fake_qwen(monkeypatch, _FakeQwenResult("hello", language="ko"))
    asr = QwenAsr()
    result = asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), lang_hint="en")
    assert result["language"] == "ko"


def test_qwen_asr_empty_text_yields_no_segments(monkeypatch):
    _install_fake_qwen(monkeypatch, _FakeQwenResult("   ", language="ko"))
    asr = QwenAsr()
    result = asr.transcribe(np.zeros(1600, dtype=np.float32), lang_hint="ko")
    assert result["segments"] == []


def test_qwen_asr_missing_package_raises_actionable_error(monkeypatch):
    # Force the lazy import to fail even if the package is somehow installed.
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", None)
    asr = QwenAsr()
    with pytest.raises(RuntimeError) as exc:
        asr.transcribe(np.zeros(16, dtype=np.float32), lang_hint="ko")
    msg = str(exc.value)
    assert "qwen-asr" in msg


# --- build_asr --------------------------------------------------------------


def test_build_asr_whisper_default():
    cfg = EngineConfig(whisper_model="repo-x")
    backend = build_asr(cfg, transcribe_fn=lambda *a, **k: None)
    assert isinstance(backend, WhisperAsr)


def test_build_asr_qwen():
    cfg = EngineConfig(asr_backend="qwen")
    backend = build_asr(cfg)
    assert isinstance(backend, QwenAsr)


def test_engine_config_rejects_unknown_backend():
    with pytest.raises(ValueError):
        EngineConfig(asr_backend="bogus")
