"""Engine ↔ ASR backend seam.

The engine builds its backend from config via `build_asr` and drives every
main decode through `self._asr.transcribe(...)`. These tests inject a fake
backend (patching `engine_mod.build_asr`) so a non-default backend can be
exercised without installing qwen3_asr_mlx.
"""

import time

import live_transcribe_core.engine as engine_mod
from live_transcribe_core.config import VAD_FRAME_SAMPLES


def _drain(engine, timeout=2.0):
    deadline = time.monotonic() + timeout
    while engine._audio_queue:
        if time.monotonic() > deadline:
            raise AssertionError("audio queue did not drain")
        time.sleep(0.005)


def _feed(engine, speech_chunk, silence_chunk, speech_frames=30, silence_frames=60):
    for _ in range(speech_frames):
        engine.push_audio(speech_chunk(samples=VAD_FRAME_SAMPLES))
    for _ in range(silence_frames):
        engine.push_audio(silence_chunk(samples=VAD_FRAME_SAMPLES))
    _drain(engine)


class _FakeBackend:
    """Records warmup + transcribe calls; returns a scripted result."""

    def __init__(self, result):
        self._result = result
        self.warmed = False
        self.transcribe_calls = []

    def warmup(self):
        self.warmed = True

    def transcribe(self, audio, *, initial_prompt=None, task="transcribe", lang_hint=None):
        self.transcribe_calls.append(
            {"n": len(audio), "initial_prompt": initial_prompt, "task": task, "lang_hint": lang_hint}
        )
        return self._result


def test_engine_uses_backend_from_build_asr(
    patched_engine, monkeypatch, speech_chunk, silence_chunk
):
    backend = _FakeBackend(
        {
            "language": "en",
            "segments": [
                {
                    "text": "from the qwen backend",
                    "start": 0.0,
                    "end": 1.0,
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.05,
                }
            ],
        }
    )
    captured = {}

    def fake_build_asr(config, *, transcribe_fn=None):
        captured["asr_backend"] = config.asr_backend
        return backend

    monkeypatch.setattr(engine_mod, "build_asr", fake_build_asr)

    engine, listener = patched_engine(
        vad_script=[0.9] * 30 + [0.1] * 60,
        asr_backend="qwen",
    )
    engine.start()
    _feed(engine, speech_chunk, silence_chunk)
    engine.stop()

    seg = listener.wait_for("segment", timeout=2.0)
    assert seg.text == "from the qwen backend"
    assert captured["asr_backend"] == "qwen"
    assert backend.warmed is True
    assert backend.transcribe_calls  # the main decode ran through the backend
    # First decode has no detected language yet → lang_hint is None.
    assert backend.transcribe_calls[0]["lang_hint"] is None
    assert backend.transcribe_calls[0]["task"] == "transcribe"
