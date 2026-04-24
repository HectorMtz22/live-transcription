"""Coverage for live_transcribe_cli.transcript.save_transcript.

Uses pytest's tmp_path fixture — no real disk pollution.
"""
from pathlib import Path

from live_transcribe_core import SegmentEvent
from live_transcribe_cli.transcript import save_transcript


def _seg(sid, speaker, text, lang="en", ts="00:00:00"):
    return SegmentEvent(id=sid, timestamp=ts, speaker=speaker, text=text, language=lang)


def test_empty_segments_returns_empty_paths(tmp_path):
    orig, trans = save_transcript(
        segments=[], translations={}, target_lang="es", transcript_dir=str(tmp_path),
    )
    assert orig == ""
    assert trans is None


def test_original_file_is_always_written_when_segments_present(tmp_path):
    segs = [_seg("1", "Speaker 1", "hello")]
    orig, trans = save_transcript(
        segments=segs, translations={}, target_lang="es", transcript_dir=str(tmp_path),
    )
    assert orig
    assert trans is None
    content = Path(orig).read_text()
    assert "hello" in content
    assert "Speaker 1" in content


def test_translated_file_written_when_translations_non_empty(tmp_path):
    segs = [_seg("1", "Speaker 1", "hello")]
    orig, trans = save_transcript(
        segments=segs,
        translations={"1": "hola"},
        target_lang="es",
        transcript_dir=str(tmp_path),
    )
    assert orig and trans
    assert Path(trans).read_text().count("hola") == 1


def test_translated_file_suffix_matches_target_lang_name(tmp_path):
    segs = [_seg("1", "Speaker 1", "hello")]
    _, trans = save_transcript(
        segs, {"1": "안녕"}, target_lang="ko", transcript_dir=str(tmp_path),
    )
    assert trans.endswith("_korean.txt")


def test_missing_translation_falls_back_to_original_text(tmp_path):
    segs = [_seg("1", "Speaker 1", "hello"), _seg("2", "Speaker 1", "world")]
    _, trans = save_transcript(
        segs,
        {"1": "hola"},  # no translation for segment 2
        target_lang="es",
        transcript_dir=str(tmp_path),
    )
    content = Path(trans).read_text()
    assert "hola" in content
    assert "world" in content  # fallback to original for seg 2


def test_speaker_changes_produce_new_headers(tmp_path):
    segs = [
        _seg("1", "Speaker 1", "hello", ts="00:00:01"),
        _seg("2", "Speaker 2", "hi", ts="00:00:02"),
        _seg("3", "Speaker 2", "there", ts="00:00:03"),
    ]
    orig, _ = save_transcript(segs, {}, target_lang="en", transcript_dir=str(tmp_path))
    content = Path(orig).read_text()
    assert content.count("Speaker 1") == 1
    # Speaker 2 header appears exactly once (not re-emitted on the consecutive line)
    assert content.count("Speaker 2") == 1
