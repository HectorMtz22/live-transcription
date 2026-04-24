"""Coverage for live_transcribe_core.whisper.DuplicateFilter."""

from live_transcribe_core.whisper import DuplicateFilter


def test_exact_duplicate_is_detected():
    f = DuplicateFilter(maxlen=5)
    f.remember("hello world")
    assert f.is_duplicate("hello world") is True


def test_duplicate_detection_is_case_insensitive():
    f = DuplicateFilter(maxlen=5)
    f.remember("Hello World")
    assert f.is_duplicate("hello world") is True
    assert f.is_duplicate("HELLO WORLD") is True


def test_short_substring_is_not_a_duplicate():
    # Substring match only triggers when the shorter string is >5 chars.
    f = DuplicateFilter(maxlen=5)
    f.remember("short")
    assert f.is_duplicate("this is a short sentence") is False


def test_long_substring_is_a_duplicate():
    f = DuplicateFilter(maxlen=5)
    f.remember("good morning everyone welcome")
    # "good morning" is len > 5 and appears in the remembered text.
    assert f.is_duplicate("good morning") is True


def test_maxlen_evicts_oldest_entry():
    f = DuplicateFilter(maxlen=2)
    f.remember("one")
    f.remember("two")
    f.remember("three")  # should evict "one"
    assert f.is_duplicate("one") is False
    assert f.is_duplicate("two") is True
    assert f.is_duplicate("three") is True


def test_empty_remembered_entry_is_tolerated():
    f = DuplicateFilter(maxlen=5)
    f.remember("")
    assert f.is_duplicate("anything") is False


def test_fresh_filter_has_no_duplicates():
    f = DuplicateFilter(maxlen=5)
    assert f.is_duplicate("whatever") is False
