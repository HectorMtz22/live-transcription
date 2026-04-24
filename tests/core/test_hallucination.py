"""Coverage for live_transcribe_core.whisper.is_hallucination.

Pinned behavior:
- empty / whitespace → hallucination
- known HALLUCINATION_PHRASES (case-insensitive, trailing punct stripped) → hallucination
- 2-character-alphabet strings of length ≥ 4 → hallucination
- dominant repeated n-gram pattern → hallucination
- genuine short phrases ≥ 2 tokens are NOT flagged
"""

import pytest

from live_transcribe_core.whisper import is_hallucination


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\n",
        "thank you",  # known phrase
        "Thanks for watching!",  # case + trailing punct
        "시청해 주셔서 감사합니다.",  # Korean known phrase, trailing period
        "aaaa",  # 1-char alphabet
        "ababab",  # 2-char alphabet
        "la la la la",  # unique_tokens <= 2, len >= 4
        "the the the the the",  # dominant-token ratio > 70%
        "hello world foo hello world foo hello world foo",  # 3-gram repetition
    ],
)
def test_hallucinated_texts_return_true(text):
    assert is_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "hi there",
        "This is a normal sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "안녕하세요 오늘 회의 시작합시다",
        "Hola, cómo estás?",
    ],
)
def test_genuine_texts_return_false(text):
    assert is_hallucination(text) is False


def test_short_text_under_three_tokens_is_not_hallucinated():
    # "hi there" is 2 tokens: the token-level checks require len(tokens) >= 3,
    # so short genuine phrases must pass.
    assert is_hallucination("hi there") is False
