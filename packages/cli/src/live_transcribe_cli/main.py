"""Live transcribe CLI entry point."""
from __future__ import annotations

import argparse
import os

# Must run before heavy imports (numpy/torch/mlx) to suppress the
# multiprocessing.resource_tracker warning on shutdown.
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

import signal
import sys
import threading

import sounddevice as sd

from live_transcribe_core import EngineConfig, TranscriptionEngine
from live_transcribe_core.config import (
    WHISPER_MODEL,
    WHISPER_MODEL_FULL,
    WHISPER_MODEL_TURBO,
)
from live_transcribe_core.translators import (
    DeepLTranslator,
    GoogleTranslator,
    NLLBTranslator,
    QwenTranslator,
)
from live_transcribe_cli import settings_store, wizard
from live_transcribe_cli.audio import open_stream
from live_transcribe_cli.displays import ChatDisplay, ColumnsDisplay
from live_transcribe_cli.transcript import save_transcript


MODEL_MAP = {
    "medium": WHISPER_MODEL,
    "turbo": WHISPER_MODEL_TURBO,
    "full": WHISPER_MODEL_FULL,
}


def _build_translator(choice: str, target_lang: str):
    if choice == "deepl":
        return DeepLTranslator(target_lang=target_lang)
    if choice == "qwen":
        return QwenTranslator(target_lang=target_lang)
    if choice == "nllb":
        return NLLBTranslator(target_lang=target_lang)
    if choice == "none":
        return None
    return GoogleTranslator(target_lang=target_lang)


def _device_name(idx: int) -> str:
    try:
        return sd.query_devices(idx)["name"]
    except Exception:
        return f"#{idx}"


def _resolve_transcript_dir() -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
    )
    return os.path.join(repo_root, "transcripts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live system audio transcription with speaker diarization"
    )
    parser.add_argument("-d", "--device", type=int, default=None,
                        help="Input device index (skip interactive prompt)")
    parser.add_argument("-m", "--model", choices=["medium", "turbo", "full"], default="full",
                        help="Whisper model: medium (fast, real-time), turbo (4 decoder layers), "
                             "or full (large-v3, 32 layers)")
    parser.add_argument("-t", "--translator",
                        choices=["google", "deepl", "qwen", "nllb", "none"], default=None,
                        help="Translation service: google, deepl, qwen, nllb, or none")
    parser.add_argument("--translate-from", default=None,
                        help="Comma-separated source language codes to translate (default: ko)")
    parser.add_argument("--translate-to", default=None,
                        help="Target language code (default: en)")
    parser.add_argument("--display", choices=["columns", "chat"], default=None,
                        help="Display mode: columns or chat")
    parser.add_argument("--summary", choices=["on", "off"], default=None,
                        help="Enable live chunked summary via local LLM (every 5 transcript lines)")
    parser.add_argument("--diarize", choices=["on", "off"], default="off",
                        help="Speaker diarization (default: off)")
    args = parser.parse_args()

    print("\n\033[1mLive Transcribe - System Audio\033[0m\n")

    model_repo = MODEL_MAP[args.model]
    last_run = settings_store.load_last_run()
    choices = wizard.run(args, last_run,
                         model_repo=model_repo,
                         diarize=(args.diarize == "on"))
    if choices is None:
        sys.exit(0)

    # Persist choices (best-effort; never fatal)
    settings_store.save_last_run(choices.to_persistable(_device_name(choices.device_idx)))

    translator = _build_translator(choices.translator, choices.translate_to)
    if choices.display == "chat":
        display = ChatDisplay(has_translator=translator is not None)
    else:
        display = ColumnsDisplay(has_translator=translator is not None)

    engine = TranscriptionEngine(
        EngineConfig(
            whisper_model=model_repo,
            translator=translator,
            translate_langs=choices.translate_from,
            target_lang=choices.translate_to,
            enable_summary=choices.summary,
            diarize=(args.diarize == "on"),
        ),
        listener=display,
    )

    stream = open_stream(choices.device_idx, on_chunk=engine.push_audio)
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        stop_event.set()
        print("\n\nStopping...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        engine.start()
        stream.start()
        display.start()
        stop_event.wait()
    finally:
        display.stop()
        stream.stop()
        stream.close()
        engine.stop()
        transcript_dir = _resolve_transcript_dir()
        original, translated, summaries_path = save_transcript(
            engine.get_transcript(),
            translations=display.translations,
            target_lang=choices.translate_to,
            transcript_dir=transcript_dir,
            summaries=display.summaries,
        )
        if original:
            paths = [original]
            if translated:
                paths.append(translated)
            if summaries_path:
                paths.append(summaries_path)
            if len(paths) == 1:
                print(f"\n\033[1;32mTranscript saved to: {paths[0]}\033[0m")
            else:
                joined = "\n  ".join(paths)
                print(f"\n\033[1;32mTranscripts saved to:\n  {joined}\033[0m")
        print("Done.")


if __name__ == "__main__":
    main()
