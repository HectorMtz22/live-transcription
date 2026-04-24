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


<<<<<<< feat/cli-improve-selection
=======
def _pick_device(args) -> int:
    if args.device is not None:
        dev_info = sd.query_devices(args.device)
        print(f"Using device: [{args.device}] {dev_info['name']}")
        return args.device

    bh_idx, _ = find_blackhole_device()
    if bh_idx is not None:
        default_idx = bh_idx
    else:
        default_idx = sd.default.device[0]
    list_input_devices(default_idx)
    try:
        choice = input(f"Select device index [Enter={default_idx}]: ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        return int(choice) if choice else default_idx
    except (ValueError, EOFError):
        sys.exit(1)


def _pick_translator_choice(args) -> str:
    if args.translator is not None:
        return args.translator
    print("\nTranslation service:")
    print("-" * 40)
    print("  [1] Google Translate")
    print("  [2] DeepL")
    print("  [3] Qwen (local LLM, offline)")
    print("  [4] NLLB-200 (local, offline, specialized translation)")
    print("  [5] None (transcription only)")
    print()
    try:
        t_choice = input("Select translation service [Enter=1]: ").strip()
    except (ValueError, EOFError):
        return "google"
    return {
        "2": "deepl",
        "3": "qwen",
        "4": "nllb",
        "5": "none",
    }.get(t_choice, "google")


def _pick_translate_langs(args) -> set[str]:
    if args.translate_from is not None:
        if args.translate_from == "all":
            return set(LANG_NAMES.keys())
        return set(args.translate_from.split(","))

    lang_options = list(LANG_NAMES.items())
    print("\nTranslate FROM (comma-separated, Enter=1):")
    for i, (code, name) in enumerate(lang_options, 1):
        print(f"  [{i}] {name} ({code})")
    print("  [*] All")
    try:
        choice = input("Select: ").strip()
    except (ValueError, EOFError):
        return {"ko"}
    if choice == "*":
        return {code for code, _ in lang_options}
    if not choice:
        return {"ko"}
    result: set[str] = set()
    try:
        for idx_str in choice.split(","):
            idx = int(idx_str.strip()) - 1
            if 0 <= idx < len(lang_options):
                result.add(lang_options[idx][0])
    except ValueError:
        return {"ko"}
    return result or {"ko"}


def _pick_target_lang(args) -> str:
    if args.translate_to is not None:
        return args.translate_to
    lang_options = list(LANG_NAMES.items())
    print("\nTranslate TO:")
    for i, (code, name) in enumerate(lang_options, 1):
        print(f"  [{i}] {name} ({code})")
    try:
        choice = input("Select [Enter=1]: ").strip()
    except (ValueError, EOFError):
        return "en"
    if not choice:
        return "en"
    try:
        idx = int(choice) - 1
    except ValueError:
        return "en"
    if 0 <= idx < len(lang_options):
        return lang_options[idx][0]
    return "en"


def _pick_display_mode(args) -> str:
    if args.display is not None:
        return args.display
    print("\nDisplay mode:")
    print("-" * 40)
    print("  [1] Columns (side-by-side transcription/translation)")
    print("  [2] Chat (bubble UI per speaker)")
    print()
    try:
        d_choice = input("Select display mode [Enter=1]: ").strip()
    except (ValueError, EOFError):
        return "columns"
    return "chat" if d_choice == "2" else "columns"


def _pick_summary(args) -> bool:
    if args.summary is not None:
        return args.summary == "on"
    print("\nLive summary (local LLM):")
    print("-" * 40)
    print("  [1] Off")
    print("  [2] On (chunked summary via Qwen 7B, every 5 lines)")
    print()
    try:
        s_choice = input("Select [Enter=1]: ").strip()
    except (ValueError, EOFError):
        return False
    return s_choice == "2"


>>>>>>> main
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
