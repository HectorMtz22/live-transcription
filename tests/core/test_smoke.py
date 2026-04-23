"""Smoke test: the workspace is installed and pytest can import both packages."""


def test_can_import_core():
    from live_transcribe_core import TranscriptionEngine, EngineConfig
    assert TranscriptionEngine is not None
    assert EngineConfig is not None


def test_can_import_cli():
    from live_transcribe_cli.transcript import save_transcript
    assert callable(save_transcript)
