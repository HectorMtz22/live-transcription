"""Autouse fake-mlx_lm guard for translator tests.

QwenTranslator.translate() contains `from mlx_lm import generate` at the top
of its try-block. If that import resolves to the real mlx_lm, Metal context
initialization on top of torch's Metal (loaded via live_transcribe_core) can
SIGSEGV at process shutdown — all tests pass but pytest exits 139.

Installing a fake mlx_lm module BEFORE any test in this directory runs
prevents the real import from ever happening, so future tests that invoke
translate() don't need to remember the guard.

Individual tests that need a specific return value from `generate()` can
still monkeypatch.setitem(sys.modules, "mlx_lm", <their fake>) — that
overwrites this default for the duration of their test.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _block_real_mlx_lm_import(monkeypatch):
    fake = types.ModuleType("mlx_lm")
    fake.generate = MagicMock(return_value="")
    fake.load = MagicMock(return_value=(object(), MagicMock()))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
