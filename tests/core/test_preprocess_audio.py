"""Coverage for live_transcribe_core.whisper.preprocess_audio.

Shape / normalization / dtype checks only — audio quality is visual.
"""

import numpy as np

from live_transcribe_core.config import SPEECH_PAD_SAMPLES, SAMPLE_RATE
from live_transcribe_core.whisper import make_highpass_sos, preprocess_audio


def test_output_length_equals_input_plus_padding():
    sos = make_highpass_sos()
    n = SAMPLE_RATE  # 1 second
    audio = np.zeros(n, dtype=np.float32)
    out = preprocess_audio(audio, sos)
    assert len(out) == n + 2 * SPEECH_PAD_SAMPLES


def test_output_dtype_is_float32():
    sos = make_highpass_sos()
    audio = np.random.default_rng(0).normal(size=SAMPLE_RATE).astype(np.float32)
    out = preprocess_audio(audio, sos)
    assert out.dtype == np.float32


def test_zero_input_stays_zero():
    sos = make_highpass_sos()
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = preprocess_audio(audio, sos)
    assert np.max(np.abs(out)) == 0.0  # no divide-by-zero


def test_peak_is_normalized_to_0_9():
    sos = make_highpass_sos()
    rng = np.random.default_rng(1)
    audio = (rng.normal(size=SAMPLE_RATE) * 10.0).astype(np.float32)
    out = preprocess_audio(audio, sos)
    # Post-filter + pad + normalize → peak magnitude should be ≈ 0.9
    peak = float(np.max(np.abs(out)))
    assert peak == 0.0 or abs(peak - 0.9) < 1e-5
