"""Coverage for BaseDisplay state reconciliation.

We test the state dicts populated by the Listener methods, not rendering.
Rendering is a visual concern and is exercised by manual `just run`.
"""
from live_transcribe_core import SegmentEvent, TranslationEvent
from live_transcribe_cli.displays.base import BaseDisplay


class _NullDisplay(BaseDisplay):
    """Concrete subclass with no-op render hooks so we can exercise the base."""
    def _render_segment_header_if_needed(self, event, new_speaker): pass
    def _render_segment_without_translation(self, event): pass
    def _render_translation(self, segment, translation): pass
    def _render_translation_update(self, segment, translation): pass


def _seg(sid, speaker="Speaker", text="hi"):
    return SegmentEvent(id=sid, timestamp="00:00:00", speaker=speaker, text=text, language="en")


def test_segment_is_stored_by_id():
    d = _NullDisplay(has_translator=False)
    d.on_segment(_seg("1"))
    assert "1" in d._segments
    assert d._segments["1"].text == "hi"


def test_translation_is_stored_by_segment_id():
    d = _NullDisplay(has_translator=True)
    d.on_segment(_seg("1"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola"))
    assert d.translations["1"] == "hola"


def test_translation_update_replaces_existing_value():
    d = _NullDisplay(has_translator=True)
    d.on_segment(_seg("1"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola mundo", is_update=True))
    assert d.translations["1"] == "hola mundo"


def test_translation_for_unknown_segment_is_tolerated():
    d = _NullDisplay(has_translator=True)
    # No segment registered for id="zzz" — must not crash.
    d.on_translation(TranslationEvent(segment_id="zzz", text="orphan"))
    # State still records it (no harm), but rendering was skipped because seg is None.
    assert d.translations["zzz"] == "orphan"


def test_new_speaker_flag_tracks_consecutive_speakers():
    flags = []

    class _Spy(BaseDisplay):
        def _render_segment_header_if_needed(self, event, new_speaker):
            flags.append((event.id, new_speaker))
        def _render_segment_without_translation(self, event): pass
        def _render_translation(self, segment, translation): pass
        def _render_translation_update(self, segment, translation): pass

    d = _Spy(has_translator=False)
    d.on_segment(_seg("1", speaker="A"))
    d.on_segment(_seg("2", speaker="A"))
    d.on_segment(_seg("3", speaker="B"))
    assert flags == [("1", True), ("2", False), ("3", True)]
