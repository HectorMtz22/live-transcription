"""Qwen-only behaviors testable without spawning the subprocess child:
LRU cache, status callback emission, retranslate_batch wiring.
"""

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

from live_transcribe_core.translators import qwen as qmod


def _inert_qwen(target_lang="es"):
    """Build a QwenTranslator skipping __init__ (which spawns a subprocess)."""
    t = qmod.QwenTranslator.__new__(qmod.QwenTranslator)
    t.target_lang = target_lang
    t._cache = OrderedDict()
    t._cache_lock = threading.Lock()
    t._available = True
    t._degraded = False
    t._on_status = None
    return t


def test_lru_cache_eviction():
    t = _inert_qwen()
    cache_size = qmod.TRANSLATION_CACHE_SIZE
    for i in range(cache_size + 1):
        t._cache_put(f"text_{i}", "en", f"trans_{i}")
    assert len(t._cache) == cache_size
    assert ("text_0", "en") not in t._cache


def test_context_free_translation_is_cached():
    t = _inert_qwen()
    fake_reply = MagicMock()
    fake_reply.result = "hola"
    t._send_and_wait = MagicMock(return_value=fake_reply)

    assert t.translate("hello", "en") == "hola"
    # Cache must be keyed on (text, source_lang); a wrong key shape would still
    # let the second call hit (because get/put share the bug), so we pin it.
    assert ("hello", "en") in t._cache
    assert t._cache[("hello", "en")] == "hola"

    assert t.translate("hello", "en") == "hola"
    # Second call hit the cache, not the subprocess.
    assert t._send_and_wait.call_count == 1


def test_context_translation_bypasses_cache():
    t = _inert_qwen()
    fake_reply = MagicMock()
    fake_reply.result = "hola"
    t._send_and_wait = MagicMock(return_value=fake_reply)

    t.translate("hello", "en", context=[("x", "y")])
    t.translate("hello", "en", context=[("x", "y")])
    # Context-bearing calls aren't cached — both hit the subprocess.
    assert t._send_and_wait.call_count == 2


def test_emit_invokes_on_status_callback():
    t = _inert_qwen()
    received = []
    t._on_status = lambda state, message: received.append((state, message))

    t._emit("warning", "child crashed")
    assert received == [("warning", "child crashed")]


def test_emit_swallows_callback_exceptions():
    t = _inert_qwen()

    def broken(state, message):
        raise RuntimeError("listener boom")

    t._on_status = broken
    t._emit("info", "no harm")  # must not raise


def test_retranslate_batch_returns_none_list_when_degraded():
    t = _inert_qwen()
    t._degraded = True
    items = [("a", "en"), ("b", "en")]
    assert t.retranslate_batch(items) == [None, None]


def test_retranslate_batch_uses_subprocess_results():
    t = _inert_qwen()
    fake_reply = MagicMock()
    fake_reply.results = ["alpha-es", "beta-es"]
    t._send_and_wait = MagicMock(return_value=fake_reply)

    items = [("alpha", "en"), ("beta", "en")]
    assert t.retranslate_batch(items) == ["alpha-es", "beta-es"]


def test_retranslate_batch_returns_none_list_when_subprocess_fails():
    """If the child times out or dies mid-call, _send_and_wait returns None;
    the caller must get a None-aligned list, not crash."""
    t = _inert_qwen()
    t._send_and_wait = MagicMock(return_value=None)
    items = [("a", "en"), ("b", "en")]
    assert t.retranslate_batch(items) == [None, None]


def test_retranslate_batch_empty_items_returns_empty_list():
    t = _inert_qwen()
    # _send_and_wait must NOT be called at all on an empty batch.
    t._send_and_wait = MagicMock(side_effect=AssertionError("should not be called"))
    assert t.retranslate_batch([]) == []
