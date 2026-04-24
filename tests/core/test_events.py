"""Coverage for live_transcribe_core.events dataclasses.

Events MUST be frozen (safe to share across threads) and asdict() output
MUST be JSON-serializable (future-ready for WebSocket transport).
"""
import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from live_transcribe_core.events import (
    SegmentEvent,
    StatusEvent,
    SummaryEvent,
    TranslationEvent,
)


def test_segment_event_is_frozen():
    e = SegmentEvent(id="x", timestamp="00:00:00", speaker="Speaker", text="hi", language="en")
    with pytest.raises(FrozenInstanceError):
        e.text = "mutated"


def test_translation_event_default_is_update_is_false():
    e = TranslationEvent(segment_id="x", text="hola")
    assert e.is_update is False


def test_summary_event_default_is_final_is_false():
    e = SummaryEvent(text="...")
    assert e.is_final is False


def test_status_event_default_message_is_none():
    e = StatusEvent(state="ready")
    assert e.message is None


def test_events_asdict_is_json_serializable():
    evts = [
        SegmentEvent(id="1", timestamp="00:00:01", speaker="Speaker 1", text="hi", language="en"),
        TranslationEvent(segment_id="1", text="hola", is_update=True),
        SummaryEvent(text="...", is_final=True),
        StatusEvent(state="error", message="boom"),
    ]
    for evt in evts:
        payload = asdict(evt)
        # Must round-trip through JSON without custom encoder.
        s = json.dumps(payload)
        assert json.loads(s) == payload
