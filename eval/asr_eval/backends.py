"""ASR backends for the Korean A/B benchmark: Whisper (current runtime model)
vs Qwen3-ASR-1.7B (candidate).

Params mirror packages/core/src/live_transcribe_core/whisper.py (transcribe +
preprocess_audio). Kept in sync manually so the eval venv stays isolated from
the runtime dependency tree.

Heavy ML deps (mlx_whisper, qwen3_asr_mlx, scipy) are imported LAZILY, inside
methods, so this module can be imported (e.g. for tests or by the CLI's
argparse-only paths) without them installed.
"""

from __future__ import annotations

import time

import numpy as np

SAMPLE_RATE = 16000
SPEECH_PAD_SAMPLES = int(SAMPLE_RATE * 0.15)  # 2400
INITIAL_PROMPT_KO = (
    "안녕하세요. 네, 알겠습니다. 그래서 이제 어떻게 할까요? "
    "아, 그렇구나. 잠깐만요, 다시 한번 말씀해 주세요. 좋습니다, 진행하겠습니다."
)

DEFAULT_WHISPER_MODEL_REPO = "mlx-community/whisper-large-v3-mlx-4bit"
# bf16, not 4-bit: qwen3-asr-mlx 0.1.1 can't load the 4-bit quantized weights
# (it rejects the .biases/.scales params the quantized checkpoint carries).
# bf16 is the only repo that actually loads with this library version.
# ~3 GB RAM at load. Confirm the exact repo id on Hugging Face on first
# download — mlx-community's Qwen3-ASR quantization naming may shift.
DEFAULT_QWEN_MODEL_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"


def _clear_mlx_cache() -> None:
    """Best-effort release of MLX's cached GPU memory. Defensive: MLX may
    not be installed (light test venv) or its cache API may have moved
    between versions, so every failure mode is swallowed silently."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass


def _preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """High-pass filter, pad, and peak-normalize — replicates
    live_transcribe_core.whisper.preprocess_audio exactly."""
    from scipy.signal import butter, sosfilt

    sos = butter(5, 80, btype="high", fs=SAMPLE_RATE, output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)
    pad = np.zeros(SPEECH_PAD_SAMPLES, dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)
    return audio


class WhisperBackend:
    """Wraps mlx_whisper with the app's exact runtime transcription params."""

    name = "Whisper"

    def __init__(self, model_repo: str = DEFAULT_WHISPER_MODEL_REPO):
        self.model_repo = model_repo

    def load(self) -> float:
        # mlx_whisper loads the model lazily on first transcribe() call, so
        # force a warm-up here to move that load/compile cost out of the
        # per-utterance RTF loop (symmetry with QwenBackend.load() below,
        # which eagerly loads via from_pretrained()). The warm-up's output
        # is discarded — it only needs to populate mlx_whisper's model cache.
        import mlx_whisper

        start = time.perf_counter()
        warmup = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)  # 0.1s silence
        mlx_whisper.transcribe(warmup, path_or_hf_repo=self.model_repo, task="transcribe")
        return time.perf_counter() - start

    def unload(self) -> None:
        # mlx_whisper caches the loaded model in a module-level ModelHolder;
        # `del backend` does NOT free it. Clear it explicitly so it doesn't
        # contaminate the next backend's peak-RAM reading.
        try:
            from mlx_whisper import transcribe as _mw

            _mw.ModelHolder.model = None
        except Exception:
            pass
        _clear_mlx_cache()

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        audio = _preprocess_audio(audio)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_repo,
            initial_prompt=INITIAL_PROMPT_KO,
            task="transcribe",
            temperature=(0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            compression_ratio_threshold=1.8,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        return result["text"]


class QwenBackend:
    """Wraps qwen3_asr_mlx's Qwen3ASR model."""

    name = "Qwen3-ASR"

    def __init__(self, model_repo: str = DEFAULT_QWEN_MODEL_REPO):
        self.model_repo = model_repo
        self.model = None

    def load(self) -> float:
        from qwen3_asr_mlx import Qwen3ASR

        start = time.perf_counter()
        self.model = Qwen3ASR.from_pretrained(self.model_repo)
        return time.perf_counter() - start

    def unload(self) -> None:
        self.model = None
        _clear_mlx_cache()

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, language="ko", temperature=0.0)
        return result.text
