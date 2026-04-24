"""Coverage for live_transcribe_core.config consistency invariants.

CLAUDE.md: "Changing [SUPPORTED_LANGUAGES] requires updating core.config
(SUPPORTED_LANGUAGES, LANG_NAMES, INITIAL_PROMPTS, HALLUCINATION_PHRASES)".
This test pins the first three; HALLUCINATION_PHRASES intentionally has
per-language entries rather than a per-language bucket, so it's not part
of the keyset comparison.
"""

from live_transcribe_core.config import (
    INITIAL_PROMPTS,
    LANG_NAMES,
    SUPPORTED_LANGUAGES,
)


def test_lang_names_keys_match_supported_languages():
    assert set(LANG_NAMES.keys()) == set(SUPPORTED_LANGUAGES)


def test_initial_prompts_keys_match_supported_languages():
    assert set(INITIAL_PROMPTS.keys()) == set(SUPPORTED_LANGUAGES)


def test_supported_languages_has_no_duplicates():
    assert len(SUPPORTED_LANGUAGES) == len(set(SUPPORTED_LANGUAGES))


def test_supported_languages_contains_current_trio():
    # If this changes, update the CLI language pickers and translator LANG_MAPs too.
    assert set(SUPPORTED_LANGUAGES) == {"ko", "en", "es"}
