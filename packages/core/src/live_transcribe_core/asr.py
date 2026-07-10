"""ASR backends — pluggable speech-to-text engines behind a common interface.

The engine owns exactly one `AsrBackend` (built from `EngineConfig.asr_backend`)
and calls `backend.transcribe(...)` for the main decode pass. Both backends
return the same result shape mlx-whisper produces — a dict with a "language"
key and a "segments" list — so `TranscriptionEngine._transcribe_segment` can
consume either without branching.

- `WhisperAsr` is a thin wrapper over `live_transcribe_core.whisper.transcribe`.
  The concrete transcribe callable is injected by the engine so the existing
  monkeypatch seam (`live_transcribe_core.engine.transcribe`) keeps working and
  the Whisper decode stays byte-for-byte identical to the pre-backend code.
- `QwenAsr` drives the optional `qwen3_asr_mlx` model. It is imported lazily so
  the package (and the `qwen-asr` extra) is only required when the backend is
  actually selected.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from live_transcribe_core.config import QWEN_ASR_MODEL, SAMPLE_RATE
from live_transcribe_core.whisper import transcribe as _whisper_transcribe


@runtime_checkable
class AsrBackend(Protocol):
    """Speech-to-text backend consumed by the engine's transcription worker."""

    def warmup(self) -> None:
        """Load/prime the model so the first real decode isn't cold."""
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
        lang_hint: Optional[str] = None,
    ) -> dict:
        """Decode `audio` (float32 mono @ 16 kHz) into an mlx-whisper-shaped dict."""
        ...


class WhisperAsr:
    """mlx-whisper backend. Delegates verbatim to the injected transcribe fn."""

    def __init__(self, model_repo: str, transcribe_fn=_whisper_transcribe):
        self._model_repo = model_repo
        self._transcribe_fn = transcribe_fn

    def warmup(self) -> None:
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._transcribe_fn(
            dummy,
            model_repo=self._model_repo,
            initial_prompt=None,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
        lang_hint: Optional[str] = None,  # noqa: ARG002 — Whisper self-detects.
    ) -> dict:
        return self._transcribe_fn(
            audio,
            model_repo=self._model_repo,
            initial_prompt=initial_prompt,
            task=task,
        )


class QwenAsr:
    """Qwen3-ASR (MLX) backend.

    Qwen decodes the whole VAD chunk as one utterance, so this synthesizes the
    single segment the engine's speaker-grouping expects (`start=0.0`,
    `end=chunk duration`). `avg_logprob`/`no_speech_prob` are set to values that
    pass the engine's confidence gate — Qwen doesn't expose per-token logprobs.
    """

    def __init__(self, model_repo: str = QWEN_ASR_MODEL):
        self._model_repo = model_repo
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from qwen3_asr_mlx import Qwen3ASR
            except ImportError as exc:
                raise RuntimeError(
                    "The Qwen3-ASR backend requires the optional 'qwen-asr' extra. "
                    "Install it with:  uv sync --extra qwen-asr  "
                    "(or:  pip install 'live-transcribe-core[qwen-asr]')."
                ) from exc
            self._model = Qwen3ASR.from_pretrained(self._model_repo)
        return self._model

    def warmup(self) -> None:
        # Force the (lazy) import + model load now so a missing extra fails at
        # engine start() rather than mid-stream, and the first decode is warm.
        self._load()

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: Optional[str] = None,  # noqa: ARG002 — Qwen uses lang_hint.
        task: str = "transcribe",  # noqa: ARG002 — translate stays Whisper-only.
        lang_hint: Optional[str] = None,
    ) -> dict:
        model = self._load()
        result = model.transcribe(audio, language=lang_hint, temperature=0.0)
        text = (getattr(result, "text", "") or "").strip()
        language = getattr(result, "language", None) or lang_hint

        segments = []
        if text:
            segments.append(
                {
                    "text": text,
                    "start": 0.0,
                    "end": len(audio) / SAMPLE_RATE,
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0,
                }
            )
        return {"language": language, "segments": segments}


def build_asr(config, *, transcribe_fn=_whisper_transcribe) -> AsrBackend:
    """Construct the ASR backend named by `config.asr_backend`.

    `transcribe_fn` lets the engine inject its own (monkeypatch-friendly)
    reference to `whisper.transcribe`; it is only used by `WhisperAsr`.
    """
    backend = config.asr_backend
    if backend == "whisper":
        return WhisperAsr(config.whisper_model, transcribe_fn=transcribe_fn)
    if backend == "qwen":
        return QwenAsr(QWEN_ASR_MODEL)
    raise ValueError(
        f"Unknown asr_backend {backend!r} (expected one of 'whisper', 'qwen')"
    )
