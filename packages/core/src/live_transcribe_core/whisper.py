"""Whisper invocation and text-level post-processing helpers.

These are pure (or nearly-pure) utilities that don't hold engine state,
so they live at module level and can be reused independently (e.g., for
batch offline transcription later).
"""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Optional

import mlx_whisper
import numpy as np
from scipy.signal import butter, sosfilt

from live_transcribe_core.config import (
    HALLUCINATION_PHRASES,
    SAMPLE_RATE,
    SPEECH_PAD_SAMPLES,
)


def make_highpass_sos():
    """High-pass filter @ 80 Hz, 5th order. Call once, reuse for all segments."""
    return butter(5, 80, btype="high", fs=SAMPLE_RATE, output="sos")


def preprocess_audio(audio: np.ndarray, hp_sos) -> np.ndarray:
    """Apply high-pass filter, padding, and peak normalization.

    Identical to the old `LiveTranscriber._preprocess_audio`.
    """
    audio = sosfilt(hp_sos, audio).astype(np.float32)
    pad = np.zeros(SPEECH_PAD_SAMPLES, dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)
    return audio


def is_hallucination(text: str) -> bool:
    """Detect Whisper hallucinations (repetitive tokens or known phantom phrases)."""
    stripped = text.strip()
    if not stripped:
        return True

    normalized = stripped.lower().strip(" .!?,。？！")
    if normalized in HALLUCINATION_PHRASES:
        return True

    no_spaces = stripped.replace(" ", "")
    if len(no_spaces) >= 4:
        unique_chars = set(no_spaces)
        if len(unique_chars) <= 2:
            return True
        for n in range(1, min(len(no_spaces) // 3 + 1, 6)):
            pattern = no_spaces[:n]
            repetitions = no_spaces.count(pattern)
            if repetitions >= 3 and (repetitions * n) / len(no_spaces) > 0.6:
                return True

    tokens = stripped.split()
    if len(tokens) < 3:
        return False
    unique_tokens = set(tokens)
    if len(unique_tokens) <= 2 and len(tokens) >= 4:
        return True
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(tokens) > 0.7 and len(tokens) >= 4:
        return True
    for n in range(2, min(len(tokens) // 2 + 1, 8)):
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counts = Counter(ngrams)
        most_common_ngram, mc_count = ngram_counts.most_common(1)[0]
        if mc_count >= 3 and (mc_count * len(most_common_ngram)) / len(tokens) > 0.5:
            return True
    return False


class DuplicateFilter:
    """Rolling filter that rejects near-duplicate transcriptions."""

    def __init__(self, maxlen: int = 5):
        self._recent: deque[str] = deque(maxlen=maxlen)

    def is_duplicate(self, text: str) -> bool:
        normalized = text.strip().lower()
        for recent in self._recent:
            if not recent:
                continue
            if normalized == recent:
                return True
            shorter, longer = sorted([normalized, recent], key=len)
            if len(shorter) > 5 and shorter in longer:
                return True
        return False

    def remember(self, text: str) -> None:
        self._recent.append(text.strip().lower())


def chunk_for_translation(text: str, max_chunk_len: int = 120) -> list[str]:
    """Sentence-level chunking for translation (keeps delimiters attached)."""
    sentence_pattern = r"(?<=[.!?。？！\n])\s*"
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 1 and len(sentences[0]) > max_chunk_len:
        clause_pattern = r"(?<=[,;:，；、])\s*"
        sentences = re.split(clause_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

    merged: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 <= max_chunk_len:
            buf += " " + s
        else:
            if buf:
                merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)

    return merged if merged else [text]


def transcribe(
    audio: np.ndarray,
    model_repo: str,
    initial_prompt: Optional[str],
) -> dict:
    """Run mlx-whisper with the engine's standard parameters."""
    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model_repo,
        initial_prompt=initial_prompt,
        temperature=(0.0, 0.2, 0.4),
        condition_on_previous_text=False,
        compression_ratio_threshold=1.8,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
