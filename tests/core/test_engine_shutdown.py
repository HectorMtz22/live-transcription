"""Shutdown must flush any in-progress speech buffer.

The engine's _process_audio `finally:` block calls _flush_speech_buffer;
stop() then waits on the pools. After stop(), the final segment MUST
have been emitted.
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


def test_stop_flushes_in_progress_speech_buffer(
    patched_engine,
    fake_whisper_result,
    speech_chunk,
):
    """Push speech but NEVER enough silence to trigger a flush organically.
    stop() must still cause the final segment to be emitted.
    """
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("final words", lang="en"),
        vad_script=[0.9] * 200,  # endless speech, no silence
    )
    engine.start()
    try:
        for _ in range(80):
            engine.push_audio(speech_chunk(samples=VAD_FRAME_SAMPLES))
        # Spin-wait until the VAD thread has drained the audio queue.
        _drain_audio_queue(engine)
    finally:
        engine.stop()

    events = listener.events("segment")
    assert len(events) >= 1
    assert any(e.text == "final words" for e in events)


def test_stop_joins_within_timeout_under_normal_load(
    patched_engine,
    fake_whisper_result,
    speech_chunk,
    silence_chunk,
):
    engine, _ = patched_engine(
        whisper_result=fake_whisper_result("ok", lang="en"),
        vad_script=[0.9] * 10 + [0.1] * 30,
    )
    engine.start()
    for _ in range(10):
        engine.push_audio(speech_chunk(samples=VAD_FRAME_SAMPLES))
    for _ in range(30):
        engine.push_audio(silence_chunk(samples=VAD_FRAME_SAMPLES))

    t0 = time.monotonic()
    engine.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0  # stop() joins with timeout=5, so this must hold


def test_stop_emits_stopping_and_stopped_status(
    patched_engine,
    fake_whisper_result,
):
    engine, listener = patched_engine(
        whisper_result=fake_whisper_result("hi", lang="en"),
    )
    engine.start()
    engine.stop()

    states = [e.state for e in listener.events("status")]
    assert "stopping" in states
    assert "stopped" in states
    # Order: starting → ready → stopping → stopped
    start_idx = states.index("starting")
    assert states.index("stopping") > start_idx
    assert states.index("stopped") > states.index("stopping")
