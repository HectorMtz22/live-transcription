"""Shared contract: every backend's translate() returns Optional[str],
returns None when the same source/target language is requested, and
returns None when the underlying client raises.

All external clients are mocked; no network calls, no model loads.
"""

import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


def test_google_returns_string_from_underlying_client(monkeypatch):
    from live_transcribe_core.translators import google as gmod

    stub_client = MagicMock()
    stub_client.translate.return_value = "hola"
    factory = MagicMock(return_value=stub_client)
    monkeypatch.setattr(gmod, "_DeepGoogleTranslator", factory)
    monkeypatch.setattr(gmod, "TRANSLATION_AVAILABLE", True)

    t = gmod.GoogleTranslator(target_lang="es")
    assert t.translate("hello", "en") == "hola"


def test_google_returns_none_when_same_source_and_target(monkeypatch):
    from live_transcribe_core.translators import google as gmod

    monkeypatch.setattr(gmod, "TRANSLATION_AVAILABLE", True)
    t = gmod.GoogleTranslator(target_lang="en")
    assert t.translate("hello", "en") is None


def test_google_returns_none_on_persistent_failure(monkeypatch):
    from live_transcribe_core.translators import google as gmod

    failing = MagicMock()
    failing.translate.side_effect = RuntimeError("network down")
    factory = MagicMock(return_value=failing)
    monkeypatch.setattr(gmod, "_DeepGoogleTranslator", factory)
    monkeypatch.setattr(gmod, "TRANSLATION_AVAILABLE", True)

    t = gmod.GoogleTranslator(target_lang="es")
    # Both retries fail → None
    assert t.translate("hello", "en") is None


# ---------------------------------------------------------------------------
# DeepL
# ---------------------------------------------------------------------------


def test_deepl_returns_string_from_underlying_client(monkeypatch):
    from live_transcribe_core.translators import deepl as dmod

    stub_client = MagicMock()
    stub_client.translate_text.return_value = "hola"
    t = dmod.DeepLTranslator(target_lang="es")
    t.client = stub_client  # bypass env-var gated __init__
    assert t.translate("hello", "en") == "hola"


def test_deepl_returns_none_when_client_missing():
    from live_transcribe_core.translators import deepl as dmod

    t = dmod.DeepLTranslator(target_lang="es")
    t.client = None
    assert t.translate("hello", "en") is None


def test_deepl_returns_none_on_exception(monkeypatch):
    from live_transcribe_core.translators import deepl as dmod

    stub_client = MagicMock()
    stub_client.translate_text.side_effect = RuntimeError("api down")
    t = dmod.DeepLTranslator(target_lang="es")
    t.client = stub_client
    assert t.translate("hello", "en") is None


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------


def test_qwen_returns_string_from_generate(monkeypatch):
    from live_transcribe_core.translators import qwen as qmod

    # Avoid real MLX load by constructing the translator then patching state.
    t = qmod.QwenTranslator.__new__(qmod.QwenTranslator)
    t.target_lang = "es"
    t._cache = __import__("collections").OrderedDict()
    import threading

    t._lock = threading.Lock()
    t._gpu_lock = None
    t._model = object()  # truthy placeholder
    t._tokenizer = MagicMock()
    t._tokenizer.apply_chat_template = MagicMock(return_value="prompt")
    t._available = True

    # Patch generate() at the lazy-import site: qwen's translate() does
    # `from mlx_lm import generate` inside the function, so we intercept that.
    fake_module = types.ModuleType("mlx_lm")
    fake_module.generate = MagicMock(return_value="hola mundo")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    assert t.translate("hello world", "en") == "hola mundo"


def test_qwen_returns_none_when_unavailable():
    from live_transcribe_core.translators import qwen as qmod

    t = qmod.QwenTranslator.__new__(qmod.QwenTranslator)
    t._available = False
    t.target_lang = "es"
    assert t.translate("hello", "en") is None


def test_qwen_returns_none_on_exception(monkeypatch):
    from live_transcribe_core.translators import qwen as qmod

    t = qmod.QwenTranslator.__new__(qmod.QwenTranslator)
    t.target_lang = "es"
    t._cache = __import__("collections").OrderedDict()
    import threading

    t._lock = threading.Lock()
    t._gpu_lock = None
    t._model = object()
    t._tokenizer = MagicMock()
    t._tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("boom"))
    t._available = True

    # Prevent real mlx_lm import (via `from mlx_lm import generate` inside
    # translate()); real Metal init on top of torch's causes a shutdown segfault.
    fake_module = types.ModuleType("mlx_lm")
    fake_module.generate = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    assert t.translate("hello", "en") is None


# ---------------------------------------------------------------------------
# NLLB
# ---------------------------------------------------------------------------


def test_nllb_returns_string_from_model(monkeypatch):
    from live_transcribe_core.translators import nllb as nmod

    t = nmod.NLLBTranslator.__new__(nmod.NLLBTranslator)
    t.target_lang = "es"
    t._tgt_code = nmod.NLLB_LANG_CODES["es"]
    t._cache = __import__("collections").OrderedDict()
    import threading

    t._lock = threading.Lock()
    t._available = True

    t._tokenizer = MagicMock()
    t._tokenizer.return_value = {"input_ids": object(), "attention_mask": object()}
    t._tokenizer.convert_tokens_to_ids = MagicMock(return_value=123)
    t._tokenizer.decode = MagicMock(return_value="hola mundo")

    t._model = MagicMock()
    t._model.generate = MagicMock(return_value=[object()])

    assert t.translate("hello world", "en") == "hola mundo"


def test_nllb_returns_none_when_unavailable():
    from live_transcribe_core.translators import nllb as nmod

    t = nmod.NLLBTranslator.__new__(nmod.NLLBTranslator)
    t._available = False
    t.target_lang = "es"
    assert t.translate("hello", "en") is None


def test_nllb_returns_none_for_unsupported_source_lang():
    from live_transcribe_core.translators import nllb as nmod

    t = nmod.NLLBTranslator.__new__(nmod.NLLBTranslator)
    t._available = True
    t.target_lang = "es"
    t._tgt_code = nmod.NLLB_LANG_CODES["es"]
    t._cache = __import__("collections").OrderedDict()
    assert t.translate("hello", "fr") is None  # fr not in NLLB_LANG_CODES


# ---------------------------------------------------------------------------
# Shared contract: every backend is a runtime-checkable Translator
# ---------------------------------------------------------------------------


def test_all_backends_satisfy_translator_protocol():
    from live_transcribe_core.translators import (
        DeepLTranslator,
        GoogleTranslator,
        NLLBTranslator,
        QwenTranslator,
        Translator,
    )

    instances = [
        DeepLTranslator.__new__(DeepLTranslator),
        GoogleTranslator.__new__(GoogleTranslator),
        NLLBTranslator.__new__(NLLBTranslator),
        QwenTranslator.__new__(QwenTranslator),
    ]
    for inst in instances:
        inst.target_lang = "en"
        assert isinstance(inst, Translator)
