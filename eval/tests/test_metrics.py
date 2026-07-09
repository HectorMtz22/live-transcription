import unicodedata

import pytest

from asr_eval.metrics import cer, normalize_korean, rtf_stats, wer


class TestNormalizeKorean:
    def test_nfc_normalizes_decomposed_hangul(self):
        # A precomposed Hangul syllable and its NFD-decomposed jamo sequence
        # must normalize to the same NFC text.
        precomposed = "가"  # "GA" (U+AC00), precomposed
        decomposed = unicodedata.normalize("NFD", precomposed)
        assert precomposed != decomposed  # sanity: they differ before normalizing
        assert normalize_korean(precomposed) == normalize_korean(decomposed)

    def test_strips_latin_punctuation(self):
        assert normalize_korean("hello, world!") == "hello world"

    def test_strips_cjk_punctuation(self):
        assert normalize_korean("안녕하세요。 반갑습니다！") == "안녕하세요 반갑습니다"

    def test_strips_curly_quotes_and_brackets(self):
        assert normalize_korean("“안녕”「하세요」") == "안녕하세요"

    def test_collapses_whitespace_runs(self):
        assert normalize_korean("hello   world\t\n foo") == "hello world foo"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_korean("  hello world  ") == "hello world"

    def test_lowercases_latin(self):
        assert normalize_korean("HELLO World") == "hello world"

    def test_empty_string(self):
        assert normalize_korean("") == ""

    def test_only_punctuation_collapses_to_empty(self):
        assert normalize_korean("...!!!???") == ""


class TestCer:
    def test_identical_strings_zero_cer(self):
        assert cer(["안녕하세요"], ["안녕하세요"]) == 0.0

    def test_completely_different_strings_nonzero_cer(self):
        result = cer(["안녕하세요"], ["반갑습니다"])
        assert result > 0.0

    def test_ignores_punctuation_differences(self):
        assert cer(["안녕하세요."], ["안녕하세요"]) == 0.0

    def test_returns_float(self):
        assert isinstance(cer(["hello"], ["hello"]), float)

    def test_multiple_utterances_averages(self):
        refs = ["안녕하세요", "반갑습니다"]
        hyps = ["안녕하세요", "반갑습니다"]
        assert cer(refs, hyps) == 0.0


class TestWer:
    def test_identical_strings_zero_wer(self):
        assert wer(["hello world"], ["hello world"]) == 0.0

    def test_completely_different_strings_nonzero_wer(self):
        result = wer(["hello world"], ["goodbye moon"])
        assert result > 0.0

    def test_ignores_punctuation_differences(self):
        assert wer(["hello, world!"], ["hello world"]) == 0.0

    def test_returns_float(self):
        assert isinstance(wer(["hello"], ["hello"]), float)


class TestRtfStats:
    def test_basic_stats(self):
        proc_times = [1.0, 2.0, 3.0]
        durations = [2.0, 2.0, 2.0]
        stats = rtf_stats(proc_times, durations)
        assert stats["mean"] == pytest.approx((0.5 + 1.0 + 1.5) / 3)
        assert stats["total"] == pytest.approx(sum(proc_times) / sum(durations))
        assert "p95" in stats

    def test_p95_is_high_percentile(self):
        proc_times = [1.0] * 19 + [100.0]
        durations = [1.0] * 20
        stats = rtf_stats(proc_times, durations)
        # p95 of mostly-1.0 rtfs with one large outlier should sit near/at
        # the outlier, well above the mean.
        assert stats["p95"] > stats["mean"]

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError):
            rtf_stats([], [])

    def test_skips_zero_duration_items(self):
        proc_times = [1.0, 5.0]
        durations = [2.0, 0.0]
        stats = rtf_stats(proc_times, durations)
        # Only the first (non-zero-duration) item should be counted.
        assert stats["mean"] == pytest.approx(0.5)

    def test_all_zero_duration_raises_value_error(self):
        with pytest.raises(ValueError):
            rtf_stats([1.0, 2.0], [0.0, 0.0])
