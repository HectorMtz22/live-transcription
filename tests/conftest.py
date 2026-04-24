"""Shared fakes and fixtures.

CRITICAL MONKEYPATCH RULE:
    Patches for `transcribe`, `load_vad_model`, `SummarizerProcess` MUST
    target `live_transcribe_core.engine.<symbol>`. `from X import Y` in
    engine.py creates a new binding in engine's namespace; patching
    `live_transcribe_core.whisper.transcribe` is silently ineffective.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import numpy as np
import pytest

from live_transcribe_core import EngineConfig, TranscriptionEngine
from live_transcribe_core.config import SAMPLE_RATE
from live_transcribe_core.translators import QwenTranslator


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Stub of a torch 0-d tensor — only needs .item()."""

    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val


class FakeVAD:
    """Scripted stub matching Silero's (frame_tensor, sample_rate) -> tensor.

    When the script runs out, the last value is repeated indefinitely.
    """

    def __init__(self, script=None):
        self._script = list(script) if script else [0.0]
        self._idx = 0

    def __call__(self, frame_tensor, sample_rate):
        if self._idx < len(self._script):
            val = self._script[self._idx]
            self._idx += 1
        else:
            val = self._script[-1]
        return _FakeTensor(val)

    def reset_states(self):
        pass


class FakeTranslator:
    """Records calls, returns scripted strings. NOT a Qwen subclass."""

    target_lang = "en"

    def __init__(self, responses=None, default=None):
        self.calls = []
        self._responses = list(responses) if responses else []
        self._default = default

    def translate(self, text, source_lang, context=None):
        self.calls.append((text, source_lang, list(context) if context else None))
        if self._responses:
            return self._responses.pop(0)
        if self._default is not None:
            return self._default
        return f"<<{text}>>"


class FakeQwenTranslator(QwenTranslator):
    """Subclass so isinstance(t, QwenTranslator) passes, without loading MLX."""

    def __init__(self, target_lang="en", responses=None, default=None):
        # Skip super().__init__() — it loads the MLX model.
        self.target_lang = target_lang
        self._gpu_lock = None
        self._available = True
        self.calls = []
        self._responses = list(responses) if responses else []
        self._default = default
        self.set_gpu_lock_calls = []

    def set_gpu_lock(self, lock):
        self._gpu_lock = lock
        self.set_gpu_lock_calls.append(lock)

    def translate(self, text, source_lang, context=None):
        self.calls.append((text, source_lang, list(context) if context else None))
        if self._responses:
            return self._responses.pop(0)
        if self._default is not None:
            return self._default
        return f"<<{text}>>"


class StubSummarizerProcess:
    """Drop-in replacement — never spawns a real subprocess."""

    def __init__(self, *args, **kwargs):
        self.lines = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def add_line(self, speaker, text, language):
        self.lines.append((speaker, text, language))

    def stop(self):
        self.stopped = True
        return ""


class RecordingListener:
    """Thread-safe EngineListener that buffers events per kind."""

    def __init__(self):
        self._events = defaultdict(list)
        self._cond = threading.Condition()

    def _push(self, kind, event):
        with self._cond:
            self._events[kind].append(event)
            self._cond.notify_all()

    def on_segment(self, event):
        self._push("segment", event)

    def on_translation(self, event):
        self._push("translation", event)

    def on_summary(self, event):
        self._push("summary", event)

    def on_status(self, event):
        self._push("status", event)

    def events(self, kind):
        with self._cond:
            return list(self._events[kind])

    def wait_for(self, kind, timeout=2.0, predicate=None):
        """Block until one matching event arrives. Raises on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for evt in self._events[kind]:
                    if predicate is None or predicate(evt):
                        return evt
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"Timeout waiting for '{kind}' event "
                        f"(have: {[(k, len(v)) for k, v in self._events.items()]})"
                    )
                self._cond.wait(timeout=remaining)

    def wait_for_count(self, kind, n, timeout=2.0):
        """Block until at least n events of kind arrive. Returns them."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._events[kind]) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"Timeout waiting for {n} '{kind}' events; "
                        f"got {len(self._events[kind])}"
                    )
                self._cond.wait(timeout=remaining)
            return list(self._events[kind][:n])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silence_chunk():
    def _make(samples=512):
        return np.zeros(samples, dtype=np.float32)

    return _make


@pytest.fixture
def speech_chunk():
    """Sine wave with RMS well above ENERGY_THRESHOLD (0.002)."""

    def _make(samples=512, freq=440):
        t = np.arange(samples) / SAMPLE_RATE
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return _make


@pytest.fixture
def fake_whisper_result():
    def _make(
        text="hello", lang="en", avg_logprob=-0.3, no_speech_prob=0.1, segments=None
    ):
        if segments is None:
            segments = [
                {
                    "text": text,
                    "start": 0.0,
                    "end": 1.0,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob,
                }
            ]
        return {"language": lang, "segments": segments}

    return _make


@pytest.fixture
def fake_translator():
    """Factory → FakeTranslator(**kwargs).

    Exposed as a fixture (rather than a top-level class import) so tests don't
    need `from conftest import ...`, which is fragile under pytest's collection
    order when subdirectory conftest.py files exist.
    """

    def _make(**kwargs):
        return FakeTranslator(**kwargs)

    return _make


@pytest.fixture
def fake_qwen_translator():
    """Factory → FakeQwenTranslator(**kwargs). See `fake_translator` for why."""

    def _make(**kwargs):
        return FakeQwenTranslator(**kwargs)

    return _make


@pytest.fixture
def patched_engine(monkeypatch):
    """Factory → (engine, listener) with Whisper/VAD/summarizer patched.

    kwargs:
      whisper_result    — single dict (used for all transcribe calls)
      whisper_results   — list of dicts; popped successively, last value repeats
      vad_script        — list of float speech-probabilities (last value repeats)
      translator        — translator instance (or None)
      translate_langs   — iterable of source-lang codes eligible for translation
      enable_summary    — if True, StubSummarizerProcess is installed and started
      diarize           — forwarded to SpeakerTracker(enabled=...); always False in
                          practice because diarize=True needs resemblyzer
      target_lang       — config target language

    The engine is NOT started — tests call engine.start() / engine.stop()
    themselves and are responsible for cleanup.
    """
    import live_transcribe_core.engine as engine_mod

    def _make(
        whisper_result=None,
        whisper_results=None,
        vad_script=None,
        translator=None,
        translate_langs=None,
        enable_summary=False,
        diarize=False,
        target_lang="en",
    ):
        results = (
            list(whisper_results)
            if whisper_results
            else (
                [whisper_result]
                if whisper_result is not None
                else [{"language": "en", "segments": []}]
            )
        )

        def fake_transcribe(audio, model_repo, initial_prompt, gpu_lock):
            with gpu_lock:
                if len(results) > 1:
                    return results.pop(0)
                return results[0]

        def fake_load_vad():
            return FakeVAD(vad_script)

        monkeypatch.setattr(engine_mod, "transcribe", fake_transcribe)
        monkeypatch.setattr(engine_mod, "load_vad_model", fake_load_vad)
        monkeypatch.setattr(engine_mod, "SummarizerProcess", StubSummarizerProcess)

        listener = RecordingListener()
        config = EngineConfig(
            translator=translator,
            translate_langs=set(translate_langs or []),
            target_lang=target_lang,
            enable_summary=enable_summary,
            diarize=diarize,
        )
        engine = TranscriptionEngine(config, listener)
        return engine, listener

    return _make
