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
DEFAULT_QWEN_MODEL_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"


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
        # mlx_whisper loads the model lazily on first transcribe() call —
        # there's no separate load step worth timing here.
        return 0.0

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

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, language="ko", temperature=0.0)
        return result.text
