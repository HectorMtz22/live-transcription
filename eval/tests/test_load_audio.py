"""Tests for bench_korean_asr._decode_audio — soundfile-based audio decoding
that replaces datasets' torchcodec-dependent auto-decoder (see eval/README.md).

Fully offline: writes tiny synthetic WAVs with `soundfile` to a temp path (or
an in-memory buffer) and decodes them back. No network, no models, no
`datasets` import.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from bench_korean_asr import SAMPLE_RATE, _decode_audio


class TestDecodeAudio:
    def test_mono_path_returns_1d_float32_array_of_expected_length(self, tmp_path):
        n_samples = 1600  # 0.1s @ 16kHz
        audio = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
        wav_path = tmp_path / "mono.wav"
        sf.write(wav_path, audio, SAMPLE_RATE)

        decoded = _decode_audio({"path": str(wav_path), "bytes": None})

        assert decoded.dtype == np.float32
        assert decoded.ndim == 1
        assert len(decoded) == n_samples

    def test_bytes_case_decodes_from_in_memory_buffer(self):
        n_samples = 1600
        audio = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        raw_bytes = buf.getvalue()

        decoded = _decode_audio({"path": None, "bytes": raw_bytes})

        assert decoded.dtype == np.float32
        assert decoded.ndim == 1
        assert len(decoded) == n_samples

    def test_stereo_input_is_downmixed_to_mono(self, tmp_path):
        n_samples = 800
        left = np.full(n_samples, 0.2, dtype=np.float32)
        right = np.full(n_samples, -0.2, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        wav_path = tmp_path / "stereo.wav"
        # subtype="FLOAT" avoids the PCM_16 default's quantization error,
        # which would otherwise make the mean-of-channels check inexact.
        sf.write(wav_path, stereo, SAMPLE_RATE, subtype="FLOAT")

        decoded = _decode_audio({"path": str(wav_path), "bytes": None})

        assert decoded.ndim == 1
        assert len(decoded) == n_samples
        assert decoded.dtype == np.float32
        # mean of +0.2 and -0.2 is ~0.0
        np.testing.assert_allclose(decoded, np.zeros(n_samples), atol=1e-6)

    def test_wrong_sample_rate_raises_value_error(self, tmp_path):
        wrong_sr = 8000
        audio = np.zeros(800, dtype=np.float32)
        wav_path = tmp_path / "wrong_sr.wav"
        sf.write(wav_path, audio, wrong_sr)

        with pytest.raises(ValueError, match=r"expected 16000 Hz, got 8000"):
            _decode_audio({"path": str(wav_path), "bytes": None})
