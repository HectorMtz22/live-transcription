"""State checks for ColumnsDisplay — no Rich Live asserts on output."""
from live_transcribe_core import SegmentEvent, TranslationEvent
from live_transcribe_cli.displays.columns import ColumnsDisplay


def _seg(sid, speaker="Speaker 1", text="hello"):
    return SegmentEvent(id=sid, timestamp="00:00:01", speaker=speaker, text=text, language="en")


def test_translations_dict_populated_without_live():
    d = ColumnsDisplay(has_translator=True)
    # d.start() not called — _live is None so _append prints, but the parent
    # BaseDisplay still records translations.
    d.on_segment(_seg("1"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola"))
    assert d.translations["1"] == "hola"


def test_entry_map_populated_with_live_active():
    d = ColumnsDisplay(has_translator=True)
    d.start()
    try:
        d.on_segment(_seg("1"))
        d.on_translation(TranslationEvent(segment_id="1", text="hola"))
        assert "1" in d._entry_map
    finally:
        d.stop()


def test_translation_update_replaces_entry_when_live():
    d = ColumnsDisplay(has_translator=True)
    d.start()
    try:
        d.on_segment(_seg("1"))
        d.on_translation(TranslationEvent(segment_id="1", text="hola"))
        idx_before = d._entry_map["1"]
        d.on_translation(TranslationEvent(segment_id="1", text="hola mundo", is_update=True))
        # Index remains valid; entry has been rebuilt in-place.
        assert d._entry_map["1"] == idx_before
    finally:
        d.stop()
