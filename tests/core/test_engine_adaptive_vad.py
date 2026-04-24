"""Direct-call tests for TranscriptionEngine._adaptive_thresholds.

Formula: silence = min(SILENCE_AFTER_SPEECH + 0.5 * pending, 2.0)
         max_speech = min(MAX_SPEECH_DURATION + 3.0 * pending, 15.0)
"""
from live_transcribe_core.config import MAX_SPEECH_DURATION, SILENCE_AFTER_SPEECH


def test_base_case_no_pending(patched_engine):
    engine, _ = patched_engine()
    engine._pending_count = 0
    silence, max_speech = engine._adaptive_thresholds()
    assert silence == SILENCE_AFTER_SPEECH
    assert max_speech == MAX_SPEECH_DURATION


def test_growth_linear_before_cap(patched_engine):
    engine, _ = patched_engine()
    engine._pending_count = 2
    silence, max_speech = engine._adaptive_thresholds()
    assert silence == SILENCE_AFTER_SPEECH + 1.0    # 0.5 * 2
    assert max_speech == MAX_SPEECH_DURATION + 6.0  # 3.0 * 2


def test_silence_capped_at_two_seconds(patched_engine):
    engine, _ = patched_engine()
    engine._pending_count = 100
    silence, _ = engine._adaptive_thresholds()
    assert silence == 2.0


def test_max_speech_capped_at_fifteen_seconds(patched_engine):
    engine, _ = patched_engine()
    engine._pending_count = 100
    _, max_speech = engine._adaptive_thresholds()
    assert max_speech == 15.0
