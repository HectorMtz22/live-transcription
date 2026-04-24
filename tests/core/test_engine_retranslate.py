"""Qwen-specific retranslation path: after a new segment, the last 3 prior
translations are re-run with the expanded context. When a re-translation
differs from the stored value, the engine emits TranslationEvent(is_update=True).

White-box approach: drive _transcribe_segment directly rather than through
the VAD loop, so each call produces one distinct segment.
"""
import time

import numpy as np

from live_transcribe_core.config import SAMPLE_RATE

from conftest import FakeQwenTranslator


def _drive_n(engine, n):
    """Call _transcribe_segment n times with a dummy 1-second buffer.

    Each call uses the next queued whisper_result (see whisper_results in
    patched_engine). The whisper_results list is popped front-to-back.
    """
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    for _ in range(n):
        engine._transcribe_segment(dummy)


def test_qwen_emits_is_update_true_when_retranslation_differs(
    patched_engine, fake_whisper_result,
):
    """Drive 4 segments. Initial translations: 'tr1'..'tr4'. Retranslations
    return different strings ('retr0'..'retr29'), so is_update=True events
    should follow.
    """
    # 4 initial + enough retranslations for multiple rounds.
    responses = ["tr1", "tr2", "tr3", "tr4"] + [f"retr{i}" for i in range(30)]
    translator = FakeQwenTranslator(responses=responses)

    # Prepend a warmup result: engine.start() calls transcribe() once with a
    # silent dummy to warm up Whisper; that call consumes the first list entry.
    whisper_results = [
        fake_whisper_result(text="warmup", lang="en"),  # absorbed by start()
    ] + [
        fake_whisper_result(text=f"sentence {i}", lang="en")
        for i in range(1, 5)
    ]

    engine, listener = patched_engine(
        whisper_results=whisper_results,
        translator=translator,
        translate_langs=["en"],
        target_lang="es",
    )
    engine.start()
    try:
        _drive_n(engine, 4)
        # Wait for 4 initial translations.
        listener.wait_for_count("translation", 4, timeout=3.0)
        # Give the retranslate pool a beat.
        time.sleep(0.5)
    finally:
        engine.stop()

    updates = [e for e in listener.events("translation") if e.is_update]
    assert updates, "expected at least one TranslationEvent with is_update=True"


def test_qwen_retranslate_skipped_when_translation_unchanged(
    patched_engine, fake_whisper_result,
):
    """If FakeQwen returns the SAME translation on the retranslate call, no
    is_update event should be emitted for that entry.
    """
    translator = FakeQwenTranslator(default="same-translation")
    # Prepend a warmup result for the engine.start() warmup transcribe call.
    whisper_results = [
        fake_whisper_result(text="warmup", lang="en"),
    ] + [
        fake_whisper_result(text=f"line {i}", lang="en") for i in range(1, 5)
    ]

    engine, listener = patched_engine(
        whisper_results=whisper_results,
        translator=translator,
        translate_langs=["en"],
        target_lang="es",
    )
    engine.start()
    try:
        _drive_n(engine, 4)
        listener.wait_for_count("translation", 4, timeout=3.0)
        time.sleep(0.5)
    finally:
        engine.stop()

    updates = [e for e in listener.events("translation") if e.is_update]
    assert updates == []


def test_qwen_gpu_lock_injected_at_start(patched_engine, fake_whisper_result):
    """Engine.start() calls translators.set_gpu_lock(translator, self._gpu_lock)
    when the translator exposes set_gpu_lock. FakeQwenTranslator records it.
    """
    translator = FakeQwenTranslator()
    engine, _ = patched_engine(
        whisper_result=fake_whisper_result("hi", lang="en"),
        translator=translator,
        translate_langs=["en"],
    )
    engine.start()
    try:
        assert len(translator.set_gpu_lock_calls) == 1
        assert translator.set_gpu_lock_calls[0] is engine._gpu_lock
    finally:
        engine.stop()
