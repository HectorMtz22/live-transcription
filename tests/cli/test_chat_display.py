"""State checks for ChatDisplay — speaker side/color mapping + state dicts."""
from live_transcribe_core import SegmentEvent, TranslationEvent
from live_transcribe_cli.displays.chat import ChatDisplay


def _seg(sid, speaker="Speaker 1", text="hello"):
    return SegmentEvent(id=sid, timestamp="00:00:01", speaker=speaker, text=text, language="en")


def test_first_speaker_gets_left_side():
    d = ChatDisplay(has_translator=False)
    assert d._speaker_side("Alice") == "left"


def test_second_speaker_gets_right_side():
    d = ChatDisplay(has_translator=False)
    _ = d._speaker_side("Alice")
    assert d._speaker_side("Bob") == "right"


def test_same_speaker_keeps_same_side():
    d = ChatDisplay(has_translator=False)
    assert d._speaker_side("Alice") == d._speaker_side("Alice")


def test_each_speaker_gets_distinct_color_until_palette_exhausts():
    d = ChatDisplay(has_translator=False)
    c1 = d._speaker_color("A")
    c2 = d._speaker_color("B")
    assert c1 != c2


def test_translation_update_keeps_entry_index_when_live():
    d = ChatDisplay(has_translator=True)
    d.start()
    try:
        d.on_segment(_seg("1"))
        d.on_translation(TranslationEvent(segment_id="1", text="hola"))
        idx = d._entry_map["1"]
        d.on_translation(TranslationEvent(segment_id="1", text="hola mundo", is_update=True))
        assert d._entry_map["1"] == idx
    finally:
        d.stop()


def test_segment_recorded_in_translations_dict():
    d = ChatDisplay(has_translator=True)
    d.on_segment(_seg("1"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola"))
    assert d.translations["1"] == "hola"
