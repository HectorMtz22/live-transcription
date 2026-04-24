"""Coverage for live_transcribe_core.whisper.chunk_for_translation."""
from live_transcribe_core.whisper import chunk_for_translation


def test_single_short_sentence_returns_single_chunk():
    result = chunk_for_translation("Hello world.")
    assert result == ["Hello world."]


def test_multiple_sentences_are_merged_under_limit():
    # All three fit comfortably under max_chunk_len=120
    text = "First sentence. Second sentence. Third sentence."
    result = chunk_for_translation(text)
    assert result == ["First sentence. Second sentence. Third sentence."]


def test_sentences_split_when_combined_exceeds_limit():
    long_tail = "x" * 80
    text = f"Short one. {long_tail}. Short three."
    result = chunk_for_translation(text, max_chunk_len=100)
    assert len(result) >= 2
    assert all(len(c) <= 100 or len(c.split()) == 1 for c in result)


def test_giant_single_sentence_falls_back_to_clause_split():
    # One sentence > max_chunk_len triggers clause-pattern re-split.
    text = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t"
    result = chunk_for_translation(text, max_chunk_len=20)
    assert len(result) > 1
    for chunk in result:
        assert chunk  # no empties


def test_cjk_sentence_delimiters_are_respected():
    text = "안녕하세요。오늘도 좋은 하루。감사합니다。"
    result = chunk_for_translation(text, max_chunk_len=10)
    assert len(result) >= 2


def test_empty_input_returns_list_with_empty_text():
    # Fallback branch: if the splitter produces no sentences, the original
    # input is returned wrapped in a list.
    result = chunk_for_translation("")
    assert result == [""]


def test_whitespace_is_stripped_from_chunks():
    result = chunk_for_translation("  hello.   world.  ")
    for chunk in result:
        assert chunk == chunk.strip()
