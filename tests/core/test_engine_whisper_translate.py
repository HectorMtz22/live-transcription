"""Whisper-native English translation (engine).

Dual mode: pass 1 transcribes the source language, pass 2 (task="translate")
produces the English translation → SegmentEvent (source) + TranslationEvent.
Single mode: one task="translate" decode → SegmentEvent already in English,
no TranslationEvent.
"""

import time

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


def _seg(text, lang):
    return {
        "language": lang,
        "segments": [
            {
                "text": text,
                "start": 0.0,
                "end": 1.0,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.1,
            }
        ],
    }


def test_dual_pass_emits_source_segment_and_english_translation(
    patched_engine, speech_chunk, silence_chunk
):
    engine, listener = patched_engine(
        whisper_result=_seg("안녕하세요", "ko"),
        translate_result=_seg("Hello", "ko"),
        whisper_translate="dual",
        translate_langs=["ko"],
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed(engine, speech_chunk, silence_chunk)
    engine.stop()

    seg = listener.wait_for("segment", timeout=2.0)
    assert seg.text == "안녕하세요"
    assert seg.language == "ko"
    tevt = listener.wait_for(
        "translation", timeout=2.0, predicate=lambda e: e.segment_id == seg.id
    )
    assert tevt.text == "Hello"
    assert tevt.is_update is False


def test_dual_pass_english_source_skips_second_pass(
    patched_engine, speech_chunk, silence_chunk
):
    engine, listener = patched_engine(
        whisper_result=_seg("hello world", "en"),
        translate_result=_seg("SHOULD-NOT-APPEAR", "en"),
        whisper_translate="dual",
        translate_langs=["ko", "en"],
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed(engine, speech_chunk, silence_chunk)
    engine.stop()

    seg = listener.wait_for("segment", timeout=2.0)
    tevt = listener.wait_for(
        "translation", timeout=2.0, predicate=lambda e: e.segment_id == seg.id
    )
    # English source: no second (translate) pass, empty translation so the row
    # still renders.
    assert tevt.text == ""


def test_dual_pass_lang_not_eligible_emits_empty_translation(
    patched_engine, speech_chunk, silence_chunk
):
    engine, listener = patched_engine(
        whisper_result=_seg("hola mundo", "es"),
        translate_result=_seg("SHOULD-NOT-APPEAR", "es"),
        whisper_translate="dual",
        translate_langs=["ko"],  # es NOT eligible
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed(engine, speech_chunk, silence_chunk)
    engine.stop()

    seg = listener.wait_for("segment", timeout=2.0)
    assert seg.text == "hola mundo"
    tevt = listener.wait_for(
        "translation", timeout=2.0, predicate=lambda e: e.segment_id == seg.id
    )
    assert tevt.text == ""


def test_single_pass_segment_is_english_and_no_translation(
    patched_engine, speech_chunk, silence_chunk
):
    engine, listener = patched_engine(
        # In single mode pass 1 runs task="translate", so the translate_result
        # is what comes back and becomes the segment text.
        translate_result=_seg("Hello there", "ko"),
        whisper_translate="single",
        vad_script=[0.9] * 30 + [0.1] * 60,
    )
    engine.start()
    _feed(engine, speech_chunk, silence_chunk)
    engine.stop()

    seg = listener.wait_for("segment", timeout=2.0)
    assert seg.text == "Hello there"
    assert listener.events("translation") == []
