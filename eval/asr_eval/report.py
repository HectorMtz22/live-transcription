"""Go/no-go verdict and markdown reporting for the Korean ASR A/B benchmark."""

from __future__ import annotations

from dataclasses import dataclass

CER_REDUCTION_THRESHOLD = 0.15
RTF_MEAN_THRESHOLD = 0.6


@dataclass
class EvalResult:
    name: str
    cer: float
    wer: float
    rtf_mean: float
    rtf_p95: float
    peak_ram_mb: float
    load_secs: float
    n: int


def _relative_cer_reduction(whisper: EvalResult, qwen: EvalResult) -> float:
    if whisper.cer == 0:
        # No errors to reduce; treat "no change" as no reduction rather than
        # dividing by zero. If qwen also has zero CER there's nothing to gain
        # from switching on accuracy grounds alone.
        return 0.0
    return (whisper.cer - qwen.cer) / whisper.cer


def decide(whisper: EvalResult, qwen: EvalResult) -> tuple[str, str]:
    """Go/no-go call for adopting Qwen3-ASR in place of Whisper.

    GO iff ALL of:
      - relative CER reduction (whisper.cer - qwen.cer) / whisper.cer >= 0.15
      - qwen.rtf_mean <= 0.6
    RAM is reported but not auto-gated (mentioned in the reason so a human
    can judge whether it fits alongside the rest of the pipeline).
    """
    reduction = _relative_cer_reduction(whisper, qwen)
    meets_reduction = reduction >= CER_REDUCTION_THRESHOLD
    meets_rtf = qwen.rtf_mean <= RTF_MEAN_THRESHOLD
    verdict = "GO" if (meets_reduction and meets_rtf) else "NO-GO"

    reason = (
        f"CER {whisper.cer:.3f} -> {qwen.cer:.3f} "
        f"({reduction:+.1%} relative, need >= {CER_REDUCTION_THRESHOLD:.0%}); "
        f"Qwen RTF mean {qwen.rtf_mean:.2f} (need <= {RTF_MEAN_THRESHOLD:.2f}); "
        f"Qwen peak RAM {qwen.peak_ram_mb:.0f} MB "
        "(not auto-gated — confirm it fits alongside the rest of the pipeline)."
    )
    return verdict, reason


def format_markdown(
    whisper: EvalResult, qwen: EvalResult, dataset: str, limit: int
) -> str:
    """Render a markdown report: one row per model plus a verdict line."""
    verdict, reason = decide(whisper, qwen)

    header = (
        "| Model | CER | WER | RTF mean | RTF p95 | Peak RAM (MB) | Load (s) | N |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = "\n".join(
        f"| {r.name} | {r.cer:.3f} | {r.wer:.3f} | {r.rtf_mean:.2f} | "
        f"{r.rtf_p95:.2f} | {r.peak_ram_mb:.0f} | {r.load_secs:.1f} | {r.n} |"
        for r in (whisper, qwen)
    )
    title = f"# Korean ASR A/B: {whisper.name} vs {qwen.name}\n\nDataset: `{dataset}` (limit={limit})\n"

    return f"{title}\n{header}\n{rows}\n\n**Verdict:** {verdict} — {reason}\n"
