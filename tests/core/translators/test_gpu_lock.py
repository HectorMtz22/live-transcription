"""set_gpu_lock only takes effect on translators that expose the method.

Spec: only QwenTranslator defines .set_gpu_lock(). Google/DeepL/NLLB
must be safely ignored — no attribute added, no exception.
"""
import threading

from live_transcribe_core.translators import (
    DeepLTranslator,
    GoogleTranslator,
    NLLBTranslator,
    QwenTranslator,
    set_gpu_lock,
    supports_gpu_lock,
)


def test_qwen_supports_gpu_lock():
    t = QwenTranslator.__new__(QwenTranslator)
    assert supports_gpu_lock(t) is True


def test_other_backends_do_not_support_gpu_lock():
    for cls in (GoogleTranslator, DeepLTranslator, NLLBTranslator):
        t = cls.__new__(cls)
        assert supports_gpu_lock(t) is False, f"{cls.__name__} unexpectedly has set_gpu_lock"


def test_set_gpu_lock_on_qwen_installs_lock():
    t = QwenTranslator.__new__(QwenTranslator)
    t._gpu_lock = None
    lock = threading.Lock()
    set_gpu_lock(t, lock)
    assert t._gpu_lock is lock


def test_set_gpu_lock_on_other_backends_is_noop():
    lock = threading.Lock()
    for cls in (GoogleTranslator, DeepLTranslator, NLLBTranslator):
        t = cls.__new__(cls)
        # Must not raise; must not add a _gpu_lock attr.
        set_gpu_lock(t, lock)
        assert not hasattr(t, "_gpu_lock") or getattr(t, "_gpu_lock", None) is not lock
