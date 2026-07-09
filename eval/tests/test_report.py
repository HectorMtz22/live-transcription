import pytest

from asr_eval.report import EvalResult, decide, format_markdown


def make_result(name="model", cer=0.1, wer=0.15, rtf_mean=0.5, rtf_p95=0.6,
                 peak_ram_mb=1500.0, load_secs=2.0, n=150) -> EvalResult:
    return EvalResult(
        name=name,
        cer=cer,
        wer=wer,
        rtf_mean=rtf_mean,
        rtf_p95=rtf_p95,
        peak_ram_mb=peak_ram_mb,
        load_secs=load_secs,
        n=n,
    )


class TestDecide:
    def test_go_when_reduction_and_rtf_both_pass(self):
        # 20% relative CER reduction, rtf 0.5 -> GO
        whisper = make_result(name="Whisper", cer=0.10)
        qwen = make_result(name="Qwen", cer=0.08, rtf_mean=0.5)  # 20% reduction
        verdict, reason = decide(whisper, qwen)
        assert verdict == "GO"
        assert reason  # non-empty human explanation

    def test_no_go_when_reduction_too_small(self):
        # 5% relative CER reduction -> NO-GO regardless of rtf
        whisper = make_result(name="Whisper", cer=0.20)
        qwen = make_result(name="Qwen", cer=0.19, rtf_mean=0.3)  # 5% reduction
        verdict, reason = decide(whisper, qwen)
        assert verdict == "NO-GO"

    def test_no_go_when_rtf_too_high_despite_good_reduction(self):
        # 15%+ reduction but rtf 0.7 -> NO-GO
        whisper = make_result(name="Whisper", cer=0.20)
        qwen = make_result(name="Qwen", cer=0.17, rtf_mean=0.7)  # 15% reduction
        verdict, reason = decide(whisper, qwen)
        assert verdict == "NO-GO"

    def test_go_at_exact_thresholds(self):
        # Exactly 15% reduction and exactly rtf 0.6 -> GO (>=, <=)
        whisper = make_result(name="Whisper", cer=0.20)
        qwen = make_result(name="Qwen", cer=0.17, rtf_mean=0.6)  # exactly 15%
        verdict, reason = decide(whisper, qwen)
        assert verdict == "GO"

    def test_reason_mentions_qwen_peak_ram(self):
        whisper = make_result(name="Whisper", cer=0.10)
        qwen = make_result(name="Qwen", cer=0.08, rtf_mean=0.5, peak_ram_mb=4321.0)
        _, reason = decide(whisper, qwen)
        assert "4321" in reason or "4,321" in reason

    def test_reason_cites_numbers(self):
        whisper = make_result(name="Whisper", cer=0.20)
        qwen = make_result(name="Qwen", cer=0.17, rtf_mean=0.7)
        _, reason = decide(whisper, qwen)
        # Should mention the rtf and cer figures somewhere in the reason.
        assert "0.7" in reason

    def test_handles_zero_whisper_cer_without_crashing(self):
        whisper = make_result(name="Whisper", cer=0.0)
        qwen = make_result(name="Qwen", cer=0.0, rtf_mean=0.5)
        verdict, reason = decide(whisper, qwen)
        assert verdict in {"GO", "NO-GO"}
        assert reason


class TestFormatMarkdown:
    def test_contains_both_model_names(self):
        whisper = make_result(name="Whisper")
        qwen = make_result(name="Qwen3-ASR", cer=0.08, rtf_mean=0.5)
        md = format_markdown(whisper, qwen, dataset="Bingsu/zeroth-korean", limit=150)
        assert "Whisper" in md
        assert "Qwen3-ASR" in md

    def test_contains_cer_values(self):
        whisper = make_result(name="Whisper", cer=0.123)
        qwen = make_result(name="Qwen3-ASR", cer=0.045, rtf_mean=0.5)
        md = format_markdown(whisper, qwen, dataset="Bingsu/zeroth-korean", limit=150)
        assert "0.123" in md
        assert "0.045" in md

    def test_contains_verdict_line(self):
        whisper = make_result(name="Whisper", cer=0.20)
        qwen = make_result(name="Qwen3-ASR", cer=0.17, rtf_mean=0.6)
        md = format_markdown(whisper, qwen, dataset="Bingsu/zeroth-korean", limit=150)
        assert "**Verdict:**" in md
        assert "GO" in md

    def test_is_a_markdown_table(self):
        whisper = make_result(name="Whisper")
        qwen = make_result(name="Qwen3-ASR", cer=0.08, rtf_mean=0.5)
        md = format_markdown(whisper, qwen, dataset="Bingsu/zeroth-korean", limit=150)
        assert "| Model |" in md or "|Model|" in md
        assert "---" in md
