"""End-to-end real-thread engine tests.

Feed synthesized chunks through push_audio; assert SegmentEvents and
TranslationEvents arrive via the listener within a 2s timeout.

Note on flush timing: the VAD silence timer (SILENCE_AFTER_SPEECH = 0.5s)
never elapses during a tight CPU loop, so the speech buffer is flushed by
the _process_audio finally-clause when engine.stop() sets _running=False.
engine.stop() then calls transcription_pool.shutdown(wait=True), which blocks
until _transcribe_segment completes and on_segment() has been called.
Tests therefore call engine.stop() before asserting events.

Drain race: the _process_audio thread has a 10ms idle sleep; stop() must not
be called before the thread has had a chance to drain the audio queue, or the
speech buffer stays empty and no segment is emitted.  _drain_audio_queue()
spins (with short sleeps) until engine._audio_queue is empty.
"""

import time

from live_transcribe_core.config import VAD_FRAME_SAMPLES


def _drain_audio_queue(engine, timeout=2.0):
    """Block until the engine's internal audio queue is empty."""
    deadline = time.monotonic() + timeout
    while engine._audio_queue:
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for audio queue to drain")
        time.sleep(0.005)


def _feed_speech_then_silence(
    engine, speech_chunk, silence_chunk, speech_frames=30, silence_frames=60
):
    """Push enough VAD-frame-aligned audio to trigger one flush, then
    wait for the VAD thread to drain the queue before returning."""
    for _ in range(speech_frames):
        engine.push_audio(speech_chunk(samples=VAD_FRAME_SAMPLES))
    for _ in range(silence_frames):
        engine.push_audio(silence_chunk(samples=VAD_FRAME_SAMPLES))
    _drain_audio_queue(engine)


def test_push_audio_emits_segment_event(
    patched_engine,
    fake_whisper_result,
    speech_chunk,
    silence_chunk,
):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello world", lang="en"),
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed_speech_then_silence(engine, speech_chunk, silence_chunk)
    engine.stop()
    evt = listener.wait_for("segment", timeout=2.0)
    assert evt.text == "hello world"
    assert evt.language == "en"


def test_translation_event_follows_segment_when_lang_enabled(
    patched_engine,
    fake_whisper_result,
    fake_translator,
    speech_chunk,
    silence_chunk,
):
    translator = fake_translator(responses=["hola mundo"])
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello world", lang="en"),
        vad_script=[0.9] * 30 + [0.1] * 60,
        translator=translator,
        translate_langs=["en"],
        target_lang="es",
    )
    engine.start()
    _feed_speech_then_silence(engine, speech_chunk, silence_chunk)
    engine.stop()
    seg = listener.wait_for("segment", timeout=2.0)
    tevt = listener.wait_for(
        "translation",
        timeout=2.0,
        predicate=lambda e: e.segment_id == seg.id,
    )
    assert tevt.text == "hola mundo"
    assert tevt.is_update is False


def test_translator_configured_but_lang_not_eligible_emits_empty_translation(
    patched_engine,
    fake_whisper_result,
    fake_translator,
    speech_chunk,
    silence_chunk,
):
    """When translator is set but source_lang is not in translate_langs,
    the engine emits TranslationEvent(text="") so displays render the
    segment without waiting for a translation that will never come.
    """
    translator = fake_translator()
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello world", lang="en"),
        vad_script=[0.9] * 30 + [0.1] * 60,
        translator=translator,
        translate_langs=["ko"],  # NOT en
        target_lang="es",
    )
    engine.start()
    _feed_speech_then_silence(engine, speech_chunk, silence_chunk)
    engine.stop()
    seg = listener.wait_for("segment", timeout=2.0)
    tevt = listener.wait_for(
        "translation",
        timeout=2.0,
        predicate=lambda e: e.segment_id == seg.id,
    )
    assert tevt.text == ""
    assert tevt.is_update is False
    # Translator was not called
    assert translator.calls == []


def test_get_transcript_returns_emitted_segments(
    patched_engine,
    fake_whisper_result,
    speech_chunk,
    silence_chunk,
):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hi there", lang="en"),
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed_speech_then_silence(engine, speech_chunk, silence_chunk)
    engine.stop()
    listener.wait_for("segment", timeout=2.0)
    snapshot = engine.get_transcript()
    assert len(snapshot) == 1
    assert snapshot[0].text == "hi there"
