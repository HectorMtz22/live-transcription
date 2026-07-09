import numpy as np
import pytest

from asr_eval.runner import run_backend


class FakeBackend:
    """Canned-output backend for exercising run_backend without any ML deps."""

    name = "fake"

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self._calls = 0

    def load(self) -> float:
        return 0.0

    def transcribe(self, audio: np.ndarray) -> str:
        text = self._outputs[self._calls]
        self._calls += 1
        return text


class SlowFakeBackend(FakeBackend):
    """Like FakeBackend, but transcribe() takes measurable wall time."""

    def __init__(self, outputs: list[str], delay: float):
        super().__init__(outputs)
        self._delay = delay

    def transcribe(self, audio: np.ndarray) -> str:
        import time

        time.sleep(self._delay)
        return super().transcribe(audio)


def make_samples(n: int, samples_per_utterance: int = 16000):
    return [
        (np.zeros(samples_per_utterance, dtype=np.float32), f"reference {i}")
        for i in range(n)
    ]


class TestRunBackend:
    def test_returns_four_lists(self):
        backend = FakeBackend(["hyp a", "hyp b"])
        samples = make_samples(2)
        result = run_backend(backend, samples, sample_rate=16000)
        assert len(result) == 4
        hyps, refs, proc_times, durations = result
        assert isinstance(hyps, list)
        assert isinstance(refs, list)
        assert isinstance(proc_times, list)
        assert isinstance(durations, list)

    def test_hypotheses_match_fake_backend_outputs(self):
        backend = FakeBackend(["hyp a", "hyp b", "hyp c"])
        samples = make_samples(3)
        hyps, refs, proc_times, durations = run_backend(backend, samples)
        assert hyps == ["hyp a", "hyp b", "hyp c"]

    def test_references_pass_through_unchanged(self):
        backend = FakeBackend(["x", "y"])
        samples = make_samples(2)
        _, refs, _, _ = run_backend(backend, samples)
        assert refs == ["reference 0", "reference 1"]

    def test_lengths_match_input_sample_count(self):
        backend = FakeBackend(["a", "b", "c", "d"])
        samples = make_samples(4)
        hyps, refs, proc_times, durations = run_backend(backend, samples)
        assert len(hyps) == len(refs) == len(proc_times) == len(durations) == 4

    def test_durations_computed_from_audio_length_and_sample_rate(self):
        backend = FakeBackend(["a"])
        samples = [(np.zeros(32000, dtype=np.float32), "ref")]
        _, _, _, durations = run_backend(backend, samples, sample_rate=16000)
        assert durations == [2.0]

    def test_proc_times_are_nonnegative(self):
        backend = FakeBackend(["a", "b"])
        samples = make_samples(2)
        _, _, proc_times, _ = run_backend(backend, samples)
        assert all(t >= 0 for t in proc_times)

    def test_proc_times_reflect_actual_elapsed_time(self):
        backend = SlowFakeBackend(["a", "b"], delay=0.05)
        samples = make_samples(2)
        _, _, proc_times, _ = run_backend(backend, samples)
        assert all(t >= 0.05 for t in proc_times)

    def test_empty_samples_returns_empty_lists(self):
        backend = FakeBackend([])
        hyps, refs, proc_times, durations = run_backend(backend, [])
        assert hyps == []
        assert refs == []
        assert proc_times == []
        assert durations == []

    def test_default_sample_rate_is_16000(self):
        backend = FakeBackend(["a"])
        samples = [(np.zeros(16000, dtype=np.float32), "ref")]
        _, _, _, durations = run_backend(backend, samples)
        assert durations == [1.0]


class TestHeavyModulesAreImportableWithoutHeavyDeps:
    """asr_eval.backends and bench_korean_asr must be importable in an env
    that only has jiwer/numpy/pytest installed — heavy ML deps (mlx_whisper,
    qwen3_asr_mlx, scipy, datasets, psutil) must stay behind lazy imports
    inside methods/functions, not at module scope.

    These tests prove the modules IMPORT cleanly; they must NOT instantiate
    a real backend or call transcribe()/load() (that would trigger the lazy
    heavy imports and fail in this light test venv).
    """

    def test_backends_module_imports(self):
        import asr_eval.backends as backends_module

        assert hasattr(backends_module, "WhisperBackend")
        assert hasattr(backends_module, "QwenBackend")

    def test_backend_classes_are_constructible_without_heavy_deps(self):
        # __init__ just stashes config; it must not trigger any heavy import.
        from asr_eval.backends import QwenBackend, WhisperBackend

        whisper = WhisperBackend()
        qwen = QwenBackend()
        assert whisper.model_repo
        assert qwen.model_repo
        assert qwen.model is None

    def test_whisper_backend_load_warms_up_via_mlx_whisper(self):
        # load() now forces a warm-up transcribe so the model is loaded
        # before the timed loop (I1) — the warm-up itself needs mlx_whisper,
        # which isn't installed in this light test venv. Calling load() must
        # surface that as an ImportError (lazy import deferred to call time),
        # not raise it eagerly at module import time.
        from asr_eval.backends import WhisperBackend

        with pytest.raises(ModuleNotFoundError):
            WhisperBackend().load()

    def test_whisper_backend_unload_is_safe_without_heavy_deps(self):
        # unload() defensively swallows ImportError so it's safe to call
        # even when mlx_whisper/mlx aren't installed (I2).
        from asr_eval.backends import WhisperBackend

        WhisperBackend().unload()  # must not raise

    def test_qwen_backend_unload_is_safe_without_heavy_deps(self):
        from asr_eval.backends import QwenBackend

        QwenBackend().unload()  # must not raise

    def test_bench_cli_module_imports(self):
        import bench_korean_asr

        assert hasattr(bench_korean_asr, "main")
        assert hasattr(bench_korean_asr, "parse_args")

    def test_bench_cli_parses_args_without_heavy_deps(self):
        import bench_korean_asr

        args = bench_korean_asr.parse_args(["--limit", "10"])
        assert args.limit == 10
        assert args.dataset == bench_korean_asr.DEFAULT_DATASET
