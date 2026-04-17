"""Translation backends for live-transcribe-core."""

from .base import Translator, set_gpu_lock, supports_gpu_lock
from .deepl import DeepLTranslator
from .google import GoogleTranslator
from .nllb import NLLBTranslator
from .qwen import QwenTranslator

__all__ = [
    "Translator",
    "GoogleTranslator",
    "DeepLTranslator",
    "QwenTranslator",
    "NLLBTranslator",
    "set_gpu_lock",
    "supports_gpu_lock",
]
