"""Qwen-only behaviors: context threading into the prompt, lock acquisition,
LRU cache eviction.
"""

import sys
import threading
import types
from collections import OrderedDict
from unittest.mock import MagicMock


from live_transcribe_core.translators import qwen as qmod


def _build_inert_qwen():
    t = qmod.QwenTranslator.__new__(qmod.QwenTranslator)
    t.target_lang = "es"
    t._cache = OrderedDict()
    t._lock = threading.Lock()
    t._gpu_lock = None
    t._available = True
    t._model = object()
    t._tokenizer = MagicMock()
    t._tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")
    return t


def _install_fake_generate(monkeypatch, return_value="translated", spy=None):
    fake_module = types.ModuleType("mlx_lm")
    if spy is None:
        spy = MagicMock(return_value=return_value)
    else:
        spy.return_value = return_value
    fake_module.generate = spy
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    return spy


def test_context_is_forwarded_into_chat_template(monkeypatch):
    t = _build_inert_qwen()
    _install_fake_generate(monkeypatch, return_value="hola mundo")

    context = [("hello", "hola"), ("world", "mundo")]
    t.translate("goodbye", "en", context=context)

    # The prompt that apply_chat_template saw must mention the context originals
    args, kwargs = t._tokenizer.apply_chat_template.call_args
    messages = args[0]
    prompt_content = messages[0]["content"]
    assert "hello" in prompt_content
    assert "world" in prompt_content
    assert "hola" in prompt_content
    assert "mundo" in prompt_content


def _spy_lock():
    """Returns (lock, acquire_spy, release_spy) for counting context-manager use."""
    acquire_spy = MagicMock()
    release_spy = MagicMock()

    class SpyLock:
        def __enter__(self):
            acquire_spy()
            return self

        def __exit__(self, *a):
            release_spy()
            return False

    return SpyLock(), acquire_spy, release_spy


def test_translate_acquires_gpu_lock_when_injected(monkeypatch):
    t = _build_inert_qwen()
    _install_fake_generate(monkeypatch, return_value="done")

    lock, acquire_spy, release_spy = _spy_lock()
    t._gpu_lock = lock
    t.translate("hi", "en")

    assert acquire_spy.call_count == 1
    assert release_spy.call_count == 1


def test_gpu_lock_released_when_generate_raises(monkeypatch):
    """If generate() raises, the GPU lock MUST still be released.

    An orphaned hold on the singleton Metal lock would deadlock the whole
    pipeline (Whisper would block forever in _transcribe_segment).
    """
    t = _build_inert_qwen()
    fake_module = types.ModuleType("mlx_lm")
    fake_module.generate = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    lock, acquire_spy, release_spy = _spy_lock()
    t._gpu_lock = lock

    assert t.translate("hi", "en") is None  # exception swallowed, returns None
    assert acquire_spy.call_count == 1
    assert release_spy.call_count == 1  # released, not orphaned


def test_translate_falls_back_to_internal_lock_without_gpu_lock(monkeypatch):
    t = _build_inert_qwen()
    t._gpu_lock = None
    _install_fake_generate(monkeypatch, return_value="done")
    # Should not raise; internal self._lock is used.
    assert t.translate("hi", "en") == "done"


def test_lru_cache_eviction(monkeypatch):
    t = _build_inert_qwen()
    _install_fake_generate(monkeypatch, return_value="x")
    t._cache = OrderedDict()

    cache_size = qmod.TRANSLATION_CACHE_SIZE
    # Fill the cache + 1 to trigger eviction.
    for i in range(cache_size + 1):
        t._cache_put(f"text_{i}", "en", f"trans_{i}")
    assert len(t._cache) == cache_size
    # The oldest ("text_0") must have been evicted.
    assert ("text_0", "en") not in t._cache


def test_context_free_translation_is_cached(monkeypatch):
    t = _build_inert_qwen()
    generate_spy = _install_fake_generate(monkeypatch, return_value="hola")

    assert t.translate("hello", "en") == "hola"
    assert t.translate("hello", "en") == "hola"
    # Second call hit the cache, not generate.
    assert generate_spy.call_count == 1


def test_context_translation_bypasses_cache(monkeypatch):
    t = _build_inert_qwen()
    generate_spy = _install_fake_generate(monkeypatch, return_value="hola")

    t.translate("hello", "en", context=[("x", "y")])
    t.translate("hello", "en", context=[("x", "y")])
    # Context-bearing calls are not cached — both should invoke generate.
    assert generate_spy.call_count == 2
