"""State checks for ChatDisplay — speaker side/color mapping + state dicts."""

from live_transcribe_core import SegmentEvent, TranslationEvent
from live_transcribe_cli.displays.chat import ChatDisplay


def _seg(sid, speaker="Speaker 1", text="hello"):
    return SegmentEvent(
        id=sid, timestamp="00:00:01", speaker=speaker, text=text, language="en"
    )


def test_first_speaker_gets_left_side():
    d = ChatDisplay(has_translator=False)
    assert d._speaker_side("Alice") == "left"


def test_second_speaker_gets_right_side():
    d = ChatDisplay(has_translator=False)
    _ = d._speaker_side("Alice")
    assert d._speaker_side("Bob") == "right"


def test_same_speaker_keeps_same_side():
    d = ChatDisplay(has_translator=False)
    # Alice is the first speaker → "left"
    assert d._speaker_side("Alice") == "left"
    # Another speaker consumes the next side slot
    d._speaker_side("Bob")
    # Alice's side mapping is stable across other speaker insertions
    assert d._speaker_side("Alice") == "left"


def test_each_speaker_gets_distinct_color_until_palette_exhausts():
    from live_transcribe_cli.displays.chat import SPEAKER_COLORS

    d = ChatDisplay(has_translator=False)
    palette_size = len(SPEAKER_COLORS)
    # First N speakers each get a distinct palette entry in order.
    colors = [d._speaker_color(f"Speaker {i}") for i in range(palette_size)]
    assert colors == list(SPEAKER_COLORS)
    # The (N+1)-th speaker wraps modulo the palette size.
    wraparound = d._speaker_color(f"Speaker {palette_size}")
    assert wraparound == SPEAKER_COLORS[0]


def test_translation_update_keeps_entry_index_when_live():
    d = ChatDisplay(has_translator=True)
    d.start()
    try:
        d.on_segment(_seg("1"))
        d.on_translation(TranslationEvent(segment_id="1", text="hola"))
        idx = d._entry_map["1"]
        d.on_translation(
            TranslationEvent(segment_id="1", text="hola mundo", is_update=True)
        )
        assert d._entry_map["1"] == idx
    finally:
        d.stop()


def test_segment_recorded_in_translations_dict():
    d = ChatDisplay(has_translator=True)
    d.on_segment(_seg("1"))
    d.on_translation(TranslationEvent(segment_id="1", text="hola"))
    assert d.translations["1"] == "hola"
