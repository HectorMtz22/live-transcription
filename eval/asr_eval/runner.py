"""Backend-agnostic eval loop: run a Backend over samples, time each call."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class Backend(Protocol):
    """Minimal interface both WhisperBackend and QwenBackend implement."""

    name: str

    def load(self) -> float:
        """Load the model; return elapsed load time in seconds."""
        ...

    def unload(self) -> None:
        """Release the loaded model so its memory doesn't linger and
        contaminate the next backend's peak-RAM reading."""
        ...

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 audio array; return the text."""
        ...


def run_backend(
    backend: Backend,
    samples: list[tuple[np.ndarray, str]],
    sample_rate: int = 16000,
) -> tuple[list[str], list[str], list[float], list[float]]:
    """Run `backend.transcribe` over every (audio, reference) sample.

    Returns (hypotheses, references, proc_times, durations) — all parallel
    lists, one entry per sample. Does not import any ML/GPU dependency; the
    backend is fully responsible for its own model.
    """
    hypotheses: list[str] = []
    references: list[str] = []
    proc_times: list[float] = []
    durations: list[float] = []

    for audio, reference in samples:
        start = time.perf_counter()
        hypothesis = backend.transcribe(audio)
        elapsed = time.perf_counter() - start

        hypotheses.append(hypothesis)
        references.append(reference)
        proc_times.append(elapsed)
        durations.append(len(audio) / sample_rate)

    return hypotheses, references, proc_times, durations
