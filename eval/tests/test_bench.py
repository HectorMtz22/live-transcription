"""Behavioral tests for bench_korean_asr.py's CLI orchestration: argparse
defaults (C1, C2) and _evaluate's wiring of backend + sampler + metrics (I4).

Light deps only (jiwer/numpy/pytest) — no psutil/mlx/datasets/scipy/qwen.
FakeBackend/FakeSampler below stand in for the real (heavy-dep) backends and
the psutil-based _PeakRSSSampler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import bench_korean_asr


class FakeBackend:
    """Canned-hypothesis backend — no ML deps, no unload() needed for
    _evaluate (unload() is only invoked by main(), not _evaluate)."""

    name = "fake-backend"

    def __init__(self, hypotheses: list[str]):
        self._hypotheses = list(hypotheses)
        self._calls = 0
        self.load_calls = 0

    def load(self) -> float:
        self.load_calls += 1
        return 0.0

    def transcribe(self, audio: np.ndarray) -> str:
        text = self._hypotheses[self._calls]
        self._calls += 1
        return text


class FakeSampler:
    """Context manager standing in for _PeakRSSSampler, with a fixed
    peak_mb — no psutil, no background thread."""

    def __init__(self, peak_mb: float = 777.0):
        self.peak_mb = peak_mb
        self.entered = False
        self.exited = False

    def __enter__(self) -> "FakeSampler":
        self.entered = True
        return self

    def __exit__(self, *exc_info) -> None:
        self.exited = True


def make_samples(n: int, samples_per_utterance: int = 16000):
    return [
        (np.zeros(samples_per_utterance, dtype=np.float32), f"reference {i}")
        for i in range(n)
    ]


class TestParseArgsDefaults:
    def test_limit_defaults_to_150(self):
        args = bench_korean_asr.parse_args([])
        assert args.limit == 150

    def test_dataset_defaults_to_zeroth_korean(self):
        args = bench_korean_asr.parse_args([])
        assert args.dataset == "Bingsu/zeroth-korean"

    def test_qwen_model_defaults_to_4bit_repo(self):
        # C2: like-for-like quantization with the app's 4-bit Whisper.
        args = bench_korean_asr.parse_args([])
        assert args.qwen_model == "mlx-community/Qwen3-ASR-1.7B-4bit"

    def test_out_default_resolves_under_eval_dir_regardless_of_cwd(self):
        # C1: the default must always land at eval/results.md on disk, even
        # though it's computed relative to this script's location rather
        # than argparse-time cwd.
        args = bench_korean_asr.parse_args([])
        assert Path(args.out).resolve().parent.name == "eval"
        assert Path(args.out).name == "results.md"


class TestEvaluate:
    def test_returns_eval_result_with_expected_name_and_n(self):
        backend = FakeBackend(["hyp a", "hyp b"])
        samples = make_samples(2)
        result = bench_korean_asr._evaluate(
            backend, samples, sampler_factory=FakeSampler
        )
        assert result.name == "fake-backend"
        assert result.n == 2

    def test_peak_ram_mb_comes_from_injected_sampler(self):
        backend = FakeBackend(["hyp a"])
        samples = make_samples(1)
        result = bench_korean_asr._evaluate(
            backend,
            samples,
            sampler_factory=lambda: FakeSampler(peak_mb=1234.5),
        )
        assert result.peak_ram_mb == 1234.5

    def test_cer_and_wer_are_floats(self):
        backend = FakeBackend(["reference 0", "reference 1"])
        samples = make_samples(2)
        result = bench_korean_asr._evaluate(
            backend, samples, sampler_factory=FakeSampler
        )
        assert isinstance(result.cer, float)
        assert isinstance(result.wer, float)

    def test_perfect_hypotheses_score_zero_error(self):
        backend = FakeBackend(["reference 0", "reference 1"])
        samples = make_samples(2)
        result = bench_korean_asr._evaluate(
            backend, samples, sampler_factory=FakeSampler
        )
        assert result.cer == 0.0
        assert result.wer == 0.0

    def test_default_sampler_factory_is_peak_rss_sampler(self):
        # Confirms the injectable seam defaults to the real sampler without
        # requiring psutil to be installed just to check the wiring.
        import inspect

        sig = inspect.signature(bench_korean_asr._evaluate)
        assert (
            sig.parameters["sampler_factory"].default
            is bench_korean_asr._PeakRSSSampler
        )

    def test_calls_backend_load_before_transcribing(self):
        backend = FakeBackend(["hyp a"])
        samples = make_samples(1)
        bench_korean_asr._evaluate(backend, samples, sampler_factory=FakeSampler)
        assert backend.load_calls == 1

    def test_sampler_is_entered_and_exited(self):
        backend = FakeBackend(["hyp a"])
        samples = make_samples(1)
        sampler_holder: list[FakeSampler] = []

        def factory() -> FakeSampler:
            s = FakeSampler()
            sampler_holder.append(s)
            return s

        bench_korean_asr._evaluate(backend, samples, sampler_factory=factory)
        assert sampler_holder[0].entered
        assert sampler_holder[0].exited


class TestHeavyModulesStayLazy:
    def test_module_imports_without_heavy_deps(self):
        # bench_korean_asr itself must stay importable in this light venv —
        # regression guard for the C1/I2 edits (Path/gc usage must not pull
        # in mlx_whisper/psutil/datasets at module scope).
        assert hasattr(bench_korean_asr, "main")
        assert hasattr(bench_korean_asr, "_evaluate")
