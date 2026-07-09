"""CLI orchestration for the Korean ASR A/B benchmark: Whisper vs Qwen3-ASR.

Throwaway offline tooling — NOT part of the live_transcribe runtime. See
eval/README.md for setup and usage. Heavy deps (datasets, mlx_whisper,
qwen3_asr_mlx, psutil) are imported lazily so `--help` and imports work
without them installed; all testable logic lives in the `asr_eval` package.
"""

from __future__ import annotations

import argparse
import gc
import io
import threading
from pathlib import Path

import numpy as np

from asr_eval.backends import (
    DEFAULT_QWEN_MODEL_REPO,
    DEFAULT_WHISPER_MODEL_REPO,
    QwenBackend,
    WhisperBackend,
)
from asr_eval.metrics import cer, rtf_stats, wer
from asr_eval.report import EvalResult, format_markdown
from asr_eval.runner import run_backend

SAMPLE_RATE = 16000
DEFAULT_LIMIT = 150
DEFAULT_DATASET = "Bingsu/zeroth-korean"
# Anchored to this script's location (not cwd) so the default always lands
# at eval/results.md on disk, whether invoked as `bench_korean_asr.py` from
# inside eval/ or as `eval/bench_korean_asr.py` from the repo root.
DEFAULT_OUT = str(Path(__file__).resolve().parent / "results.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Korean ASR A/B benchmark: Whisper vs Qwen3-ASR-1.7B (MLX)."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Number of test-split utterances to evaluate (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"HF dataset repo id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL_REPO,
        help=f"Whisper MLX model repo (default: {DEFAULT_WHISPER_MODEL_REPO}).",
    )
    parser.add_argument(
        "--qwen-model", default=DEFAULT_QWEN_MODEL_REPO,
        help=(
            f"Qwen3-ASR MLX model repo (default: {DEFAULT_QWEN_MODEL_REPO}, "
            "bf16 — qwen3-asr-mlx 0.1.1 can't load the 4-bit quantized "
            "weights). Confirm the exact repo id on first download."
        ),
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Path to write the markdown report (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


def _decode_audio(audio_field: dict) -> np.ndarray:
    """Decode a datasets `Audio(decode=False)` field (`{"path": str|None,
    "bytes": bytes|None}`) into a mono float32 array via `soundfile`.

    `datasets`' built-in Audio decoding now requires `torchcodec`, which
    isn't part of this eval venv; reading with `soundfile` (already a dep)
    avoids that. `soundfile` is imported lazily here to keep the module's
    existing lazy-import discipline (heavy deps stay out of module scope).
    """
    import soundfile as sf

    path = audio_field.get("path")
    if path:
        audio, sr = sf.read(path, dtype="float32")
    else:
        audio, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")

    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")

    return audio


def _load_samples(dataset: str, limit: int) -> list[tuple[object, str]]:
    """Lazy-load the dataset's test split, decode audio via `soundfile`
    (bypassing `datasets`' torchcodec-dependent auto-decoder), take `limit`
    utterances, and return [(float32 audio array, reference text)]."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select(range(min(limit, len(ds))))

    samples = []
    for row in ds:
        audio = _decode_audio(row["audio"])
        text = row["text"]
        samples.append((audio, text))
    return samples


class _PeakRSSSampler:
    """Samples process RSS on a background thread; reports the peak in MB.

    Used as a context manager around a backend's load()+run_backend() so we
    capture the model's peak memory footprint, not just a single snapshot.
    """

    def __init__(self, interval_secs: float = 0.2):
        self._interval_secs = interval_secs
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            rss = proc.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self._interval_secs)

    def __enter__(self) -> "_PeakRSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)


def _evaluate(backend, samples, sampler_factory=_PeakRSSSampler) -> EvalResult:
    with sampler_factory() as sampler:
        load_secs = backend.load()
        hyps, refs, proc_times, durations = run_backend(
            backend, samples, sample_rate=SAMPLE_RATE
        )
    rtf = rtf_stats(proc_times, durations)
    return EvalResult(
        name=backend.name,
        cer=cer(refs, hyps),
        wer=wer(refs, hyps),
        rtf_mean=rtf["mean"],
        rtf_p95=rtf["p95"],
        peak_ram_mb=sampler.peak_mb,
        load_secs=load_secs,
        n=len(samples),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    samples = _load_samples(args.dataset, args.limit)

    whisper_backend = WhisperBackend(model_repo=args.whisper_model)
    whisper_result = _evaluate(whisper_backend, samples)
    whisper_backend.unload()
    del whisper_backend
    gc.collect()

    qwen_backend = QwenBackend(model_repo=args.qwen_model)
    qwen_result = _evaluate(qwen_backend, samples)
    qwen_backend.unload()
    del qwen_backend
    gc.collect()

    markdown = format_markdown(
        whisper_result, qwen_result, dataset=args.dataset, limit=args.limit
    )
    print(markdown)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)


if __name__ == "__main__":
    main()
