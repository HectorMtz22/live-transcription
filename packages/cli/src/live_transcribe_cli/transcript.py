"""Persist transcripts to disk. CLI-only."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Optional

from live_transcribe_core import SegmentEvent, SummaryEvent
from live_transcribe_core.config import LANG_NAMES


def save_transcript(
    segments: Iterable[SegmentEvent],
    translations: dict[str, str],
    target_lang: str,
    transcript_dir: str,
    summaries: Iterable[SummaryEvent] = (),
) -> tuple[str, Optional[str], Optional[str]]:
    """Write original, translated, and summary transcript files.

    Returns (original_path, translated_path_or_None, summaries_path_or_None).
    If `segments` is empty, returns ("", None, None) — summaries alone don't
    trigger a write since they only make sense alongside the transcript.
    """
    segments = list(segments)
    summaries = list(summaries)
    if not segments:
        return ("", None, None)

    os.makedirs(transcript_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    header_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    original_path = os.path.join(transcript_dir, f"transcript_{timestamp}_original.txt")
    with open(original_path, "w") as f:
        f.write(f"Transcript (Original) - {header_time}\n")
        f.write("=" * 60 + "\n\n")
        current_speaker = None
        for seg in segments:
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                f.write(f"\n[{seg.timestamp}] {current_speaker}:\n")
            f.write(f"  {seg.text}\n")

    translated_path: Optional[str] = None
    if translations:
        target_name = LANG_NAMES.get(target_lang, target_lang).lower()
        target_label = LANG_NAMES.get(target_lang, target_lang)
        translated_path = os.path.join(
            transcript_dir, f"transcript_{timestamp}_{target_name}.txt"
        )
        with open(translated_path, "w") as f:
            f.write(f"Transcript ({target_label}) - {header_time}\n")
            f.write("=" * 60 + "\n\n")
            current_speaker = None
            for seg in segments:
                if seg.speaker != current_speaker:
                    current_speaker = seg.speaker
                    f.write(f"\n[{seg.timestamp}] {current_speaker}:\n")
                line = translations.get(seg.id) or seg.text
                f.write(f"  {line}\n")

    summaries_path: Optional[str] = None
    if summaries:
        summaries_path = os.path.join(
            transcript_dir, f"transcript_{timestamp}_summaries.txt"
        )
        with open(summaries_path, "w") as f:
            f.write(f"Summaries - {header_time}\n")
            f.write("=" * 60 + "\n\n")
            for summary in summaries:
                tag = f"FINAL SUMMARY #{summary.index}" if summary.is_final else f"#{summary.index}"
                f.write(f"[{tag} · {summary.timestamp}]\n")
                f.write(f"{summary.text}\n\n")

    return (original_path, translated_path, summaries_path)
