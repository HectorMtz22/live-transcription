"""Pure metrics for the Korean ASR A/B benchmark: text normalization, CER/WER,
and real-time-factor statistics.

Depends only on jiwer, numpy, and the stdlib — no heavy ML deps.
"""

from __future__ import annotations

import re
import statistics
import string
import unicodedata

import jiwer
import numpy as np

# Punctuation to strip before scoring: common Latin punctuation plus the
# CJK/fullwidth punctuation and quote/bracket marks typically emitted by
# Korean ASR/reference transcripts.
_EXTRA_PUNCT = ".,!?;:\"'()[]{}…·~-—。，！？、；：“”‘’「」『』（）"
_PUNCT_CHARS = set(_EXTRA_PUNCT) | set(string.punctuation)
_PUNCT_TABLE = {ord(ch): " " for ch in _PUNCT_CHARS}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_korean(text: str) -> str:
    """Normalize text for CER/WER scoring.

    1. Unicode NFC normalize (so precomposed and decomposed Hangul compare
       equal).
    2. Replace punctuation with a space (Latin + CJK/fullwidth + stdlib
       `string.punctuation`) — not delete outright, so punctuation between
       two words (e.g. "안녕,하세요") becomes a separating space
       ("안녕 하세요") rather than gluing the words together.
    3. Collapse whitespace runs to a single space and strip ends.
    4. Lowercase (harmless for Hangul, normalizes any Latin).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()


def cer(references: list[str], hypotheses: list[str]) -> float:
    """Character error rate over normalized text, as a fraction (e.g. 0.083)."""
    refs = [normalize_korean(r) for r in references]
    hyps = [normalize_korean(h) for h in hypotheses]
    return float(jiwer.cer(refs, hyps))


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Word error rate over normalized text, as a fraction."""
    refs = [normalize_korean(r) for r in references]
    hyps = [normalize_korean(h) for h in hypotheses]
    return float(jiwer.wer(refs, hyps))


def rtf_stats(proc_times: list[float], durations: list[float]) -> dict:
    """Real-time-factor stats: rtf = proc_time / duration, per utterance.

    Returns {"mean": ..., "p95": ..., "total": sum(proc)/sum(dur)}.

    Zero-duration items are skipped (can't compute an rtf for them). Raises
    ValueError if there's nothing left to compute a stat from (empty input,
    or every duration is zero).
    """
    pairs = [
        (proc, dur) for proc, dur in zip(proc_times, durations) if dur > 0
    ]
    if not pairs:
        raise ValueError(
            "rtf_stats requires at least one item with a non-zero duration"
        )

    rtfs = [proc / dur for proc, dur in pairs]
    total_proc = sum(proc for proc, _ in pairs)
    total_dur = sum(dur for _, dur in pairs)

    return {
        "mean": statistics.mean(rtfs),
        "p95": float(np.percentile(rtfs, 95)),
        "total": total_proc / total_dur,
    }
