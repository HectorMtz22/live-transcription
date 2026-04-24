"""White-box tests: call _transcribe_segment directly and assert the
filter branches don't emit a SegmentEvent.

The engine is started (so _speaker_tracker, _translation_pool, VAD model
are all set up), but we bypass the VAD loop entirely by invoking
_transcribe_segment with a synthesized array.
"""

import numpy as np

from live_transcribe_core.config import SAMPLE_RATE


def _drive_segment(engine):
    """Call _transcribe_segment with a dummy 1-second buffer."""
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    engine._transcribe_segment(dummy)


def test_empty_segments_list_emits_nothing(patched_engine):
    engine, listener = patched_engine(
        whisper_result={"language": "en", "segments": []},
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
    finally:
        engine.stop()


def test_unsupported_language_is_skipped(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("bonjour", lang="fr"),
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
        # And _detected_lang MUST NOT be updated to an unsupported code.
        assert engine._detected_lang != "fr"
    finally:
        engine.stop()


def test_low_avg_logprob_segments_are_dropped(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello", avg_logprob=-1.5),  # < -1.0
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
    finally:
        engine.stop()


def test_high_no_speech_prob_segments_are_dropped(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello", no_speech_prob=0.8),  # > 0.6
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
    finally:
        engine.stop()


def test_hallucinated_text_is_dropped(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("thank you for watching", lang="en"),
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
    finally:
        engine.stop()


def test_duplicate_text_is_dropped_second_time(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello there", lang="en"),
    )
    engine.start()
    try:
        _drive_segment(engine)  # first: emitted
        _drive_segment(engine)  # second: dropped as duplicate
        events = listener.events("segment")
        assert len(events) == 1
        assert events[0].text == "hello there"
    finally:
        engine.stop()


def test_very_short_segments_are_dropped(patched_engine, fake_whisper_result):
    # segment text of length < 2 after strip is filtered out before scoring.
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("x", lang="en"),
    )
    engine.start()
    try:
        _drive_segment(engine)
        assert listener.events("segment") == []
    finally:
        engine.stop()


def test_happy_path_segment_emits_event(patched_engine, fake_whisper_result):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hello there", lang="en"),
    )
    engine.start()
    try:
        _drive_segment(engine)
        events = listener.events("segment")
        assert len(events) == 1
        assert events[0].text == "hello there"
        assert events[0].language == "en"
        assert events[0].speaker == "Speaker"  # diarize=False default
    finally:
        engine.stop()
