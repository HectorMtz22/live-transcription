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
    LANG_NAMES,
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
from live_transcribe_cli.audio import (
    find_blackhole_device,
    list_input_devices,
    open_stream,
)
from live_transcribe_cli.displays import ChatDisplay, ColumnsDisplay
from live_transcribe_cli.transcript import save_transcript


MODEL_MAP = {
    "medium": WHISPER_MODEL,
    "turbo": WHISPER_MODEL_TURBO,
    "full": WHISPER_MODEL_FULL,
}


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
    print("  [2] On (rolling summary via Qwen 7B)")
    print()
    try:
        s_choice = input("Select [Enter=1]: ").strip()
    except (ValueError, EOFError):
        return False
    return s_choice == "2"


def _build_translator(choice: str, target_lang: str):
    if choice == "deepl":
        print("Using translator: DeepL")
        return DeepLTranslator(target_lang=target_lang)
    if choice == "qwen":
        print("Using translator: Qwen (local LLM)")
        return QwenTranslator(target_lang=target_lang)
    if choice == "nllb":
        print("Using translator: NLLB-200 (local)")
        return NLLBTranslator(target_lang=target_lang)
    if choice == "none":
        print("Using translator: None (disabled)")
        return None
    print("Using translator: Google Translate")
    return GoogleTranslator(target_lang=target_lang)


def _print_startup_banner(
    *,
    device_idx: int,
    model_repo: str,
    diarize: bool,
    display_mode: str,
    enable_summary: bool,
    translator,
    translate_langs: set[str],
    target_lang: str,
) -> None:
    """Summarize the chosen configuration before the engine goes live."""
    dev_name = sd.query_devices(device_idx)["name"]
    print("=" * 60)
    print("  LIVE TRANSCRIPTION")
    print("=" * 60)
    print(f"  Audio device:       {dev_name}")
    print(f"  Whisper model:      {model_repo}")
    print(f"  Speaker diarization:{' ON' if diarize else ' OFF'}")
    print(f"  Display mode:       {display_mode}")
    print(f"  Live summary:       {'ON' if enable_summary else 'OFF'}")
    if translator is not None:
        from_list = ", ".join(
            f"{LANG_NAMES.get(l, l)} ({l})" for l in sorted(translate_langs)
        ) or "(none)"
        to_name = f"{LANG_NAMES.get(target_lang, target_lang)} ({target_lang})"
        print(f"  Translation:        ON")
        print(f"  Translate from:     {from_list}")
        print(f"  Translate to:       {to_name}")
    else:
        print(f"  Translation:        OFF")
    print("=" * 60)
    print("  Press Ctrl+C to stop and save transcript")
    print("=" * 60)
    print("\nListening...\n")


def _resolve_transcript_dir() -> str:
    """Return the transcripts/ directory at the repo root.

    Falls back to `$PWD/transcripts` if the resolved path isn't writable.
    """
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
                        help="Enable live rolling summary via local LLM")
    parser.add_argument("--diarize", choices=["on", "off"], default="off",
                        help="Speaker diarization (default: off)")
    args = parser.parse_args()

    print("\n\033[1mLive Transcribe - System Audio\033[0m\n")

    device_idx = _pick_device(args)
    translator_choice = _pick_translator_choice(args)

    translate_langs: set[str] = set()
    target_lang = "en"

    if translator_choice != "none":
        translate_langs = _pick_translate_langs(args)
        target_lang = _pick_target_lang(args)
        translate_langs.discard(target_lang)

    translator = _build_translator(translator_choice, target_lang)
    display_mode = _pick_display_mode(args)
    print(f"Using display: {display_mode}")
    enable_summary = _pick_summary(args)
    model_repo = MODEL_MAP[args.model]
    print(f"Using Whisper model: {model_repo}")

    if display_mode == "chat":
        display = ChatDisplay(has_translator=translator is not None)
    else:
        display = ColumnsDisplay(has_translator=translator is not None)

    engine = TranscriptionEngine(
        EngineConfig(
            whisper_model=model_repo,
            translator=translator,
            translate_langs=translate_langs,
            target_lang=target_lang,
            enable_summary=enable_summary,
            diarize=(args.diarize == "on"),
        ),
        listener=display,
    )

    _print_startup_banner(
        device_idx=device_idx,
        model_repo=model_repo,
        diarize=args.diarize == "on",
        display_mode=display_mode,
        enable_summary=enable_summary,
        translator=translator,
        translate_langs=translate_langs,
        target_lang=target_lang,
    )

    stream = open_stream(device_idx, on_chunk=engine.push_audio)
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
        original, translated = save_transcript(
            engine.get_transcript(),
            translations=display.translations,
            target_lang=target_lang,
            transcript_dir=transcript_dir,
        )
        if translated:
            print(f"\n\033[1;32mTranscripts saved to:\n  {original}\n  {translated}\033[0m")
        elif original:
            print(f"\n\033[1;32mTranscript saved to: {original}\033[0m")
        print("Done.")


if __name__ == "__main__":
    main()
