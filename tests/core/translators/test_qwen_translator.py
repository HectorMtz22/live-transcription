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


# ---------------------------------------------------------------------------
# Shutdown-aware watchdog (Ctrl+C should not spawn a fresh child).
# ---------------------------------------------------------------------------


def _dead_child_qwen(stopping):
    """Inert translator wired for _on_failure: a dead (crashed) child, not
    degraded, not in a restart cooldown, with _spawn_child mocked out."""
    t = _inert_qwen()
    t._stopping = stopping
    t._watchdog_lock = threading.Lock()
    t._last_restart_at = None
    t._process = MagicMock()
    t._process.is_alive.return_value = False  # child has died
    t._spawn_child = MagicMock()
    t._emits = []
    t._on_status = lambda state, message: t._emits.append((state, message))
    return t


def test_on_failure_skips_restart_during_shutdown():
    """When _stopping is set (Ctrl+C / engine shutdown), a dead child must NOT
    be restarted — that spurious respawn is the bug this fix targets."""
    t = _dead_child_qwen(stopping=True)

    t._on_failure()

    t._spawn_child.assert_not_called()
    assert t._degraded is False
    assert t._emits == []


def test_on_failure_restarts_on_first_crash_when_not_stopping():
    """Regression guard: a genuine mid-session crash (not shutting down) still
    triggers exactly one restart, emitting warning then info."""
    t = _dead_child_qwen(stopping=False)

    t._on_failure()

    t._spawn_child.assert_called_once()
    assert t._degraded is False
    assert [state for state, _ in t._emits] == ["warning", "info"]
    assert t._last_restart_at is not None


def test_begin_shutdown_sets_flag_and_is_idempotent():
    t = _inert_qwen()
    t._stopping = False

    t.begin_shutdown()
    assert t._stopping is True

    # Calling twice must be harmless — no spawn/join/terminate, just the flag.
    t.begin_shutdown()
    assert t._stopping is True


def test_worker_installs_sigint_ignore_handler(monkeypatch):
    """The child worker must install SIG_IGN for SIGINT before loading the
    model, so a Ctrl+C delivered to the foreground process group does not kill
    it — it must exit only via the None sentinel."""
    import signal

    recorded = []

    def spy_signal(signum, handler):
        recorded.append((signum, handler))
        return signal.SIG_DFL

    # qmod.signal is the (shared) signal module; monkeypatch restores it.
    monkeypatch.setattr(qmod.signal, "signal", spy_signal)

    request_q = MagicMock()
    request_q.get.return_value = None  # sentinel → _worker returns immediately
    reply_q = MagicMock()

    # mlx_lm is faked by the autouse guard in this dir's conftest, so load()
    # returns instantly without touching Metal.
    qmod._worker(request_q, reply_q, "fake-model")

    assert (signal.SIGINT, signal.SIG_IGN) in recorded
