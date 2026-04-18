#!/usr/bin/env python3
"""
Live System Audio Transcription with Speaker Diarization

Captures system audio via BlackHole virtual audio device,
transcribes using mlx-whisper (GPU-accelerated on Apple Silicon),
and separates speakers using resemblyzer voice embeddings.

Uses Silero VAD for voice activity detection to trigger transcription
on natural speech boundaries instead of fixed time windows.

Setup:
  1. Install BlackHole 2ch: brew install --cask blackhole-2ch
  2. Reboot Mac
  3. Create Multi-Output Device in Audio MIDI Setup
     (combine your speakers + BlackHole 2ch)
  4. Set Multi-Output Device as system output
  5. Run: ./live_transcribe_env/bin/python live_transcribe.py
"""

import argparse
import os

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

import sys
import signal
import threading

import sounddevice as sd

from live_transcribe_cli.audio import (
    find_blackhole_device,
    list_input_devices,
    open_stream,
)
from live_transcribe_cli.displays import ChatDisplay, ColumnsDisplay
from live_transcribe_cli.transcript import save_transcript
from live_transcribe_core import (
    EngineConfig,
    TranscriptionEngine,
)
from live_transcribe_core.translators import (
    DeepLTranslator,
    GoogleTranslator as Translator,
    NLLBTranslator,
    QwenTranslator,
)

# ─── Configuration ───────────────────────────────────────────────────────────

WHISPER_MODEL = "mlx-community/whisper-medium-mlx-q4"  # Q4 quantized medium; fast enough for real-time, decent multilingual quality
WHISPER_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"  # Distilled: same encoder, 4 decoder layers (vs 32); higher quality, slower
WHISPER_MODEL_FULL = "mlx-community/whisper-large-v3-mlx-4bit"  # Q4 quantized full model; 32 decoder layers, slowest but max accuracy
LANG_NAMES = {"ko": "Korean", "en": "English", "es": "Spanish"}


def main():
    parser = argparse.ArgumentParser(description="Live system audio transcription with speaker diarization")
    parser.add_argument("-d", "--device", type=int, default=None,
                        help="Input device index (skip interactive prompt)")
    parser.add_argument("-m", "--model", choices=["medium", "turbo", "full"], default="full",
                        help="Whisper model: medium (fast, real-time), turbo (4 decoder layers), or full (large-v3, 32 layers)")
    parser.add_argument("-t", "--translator", choices=["google", "deepl", "qwen", "nllb", "none"], default=None,
                        help="Translation service: google, deepl, or none to disable")
    parser.add_argument("--translate-from", default=None,
                        help="Comma-separated source language codes to translate (default: ko)")
    parser.add_argument("--translate-to", default=None,
                        help="Target language code (default: en)")
    parser.add_argument("--display", choices=["columns", "chat"], default=None,
                        help="Display mode: columns (side-by-side) or chat (bubble UI)")
    parser.add_argument("--summary", choices=["on", "off"], default=None,
                        help="Enable live rolling summary via local LLM")
    parser.add_argument("--diarize", choices=["on", "off"], default="off",
                        help="Speaker diarization (default: off)")
    args = parser.parse_args()

    print("\n\033[1mLive Transcribe - System Audio\033[0m\n")

    if args.device is not None:
        # CLI flag provided — use directly
        device_idx = args.device
        dev_info = sd.query_devices(device_idx)
        print(f"Using device: [{device_idx}] {dev_info['name']}")
    else:
        # Determine default: prefer BlackHole, fall back to system default input
        bh_idx, bh_name = find_blackhole_device()
        if bh_idx is not None:
            default_idx = bh_idx
        else:
            default_idx = sd.default.device[0]  # system default input

        list_input_devices(default_idx)

        try:
            choice = input(f"Select device index [Enter={default_idx}]: ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            device_idx = int(choice) if choice else default_idx
        except (ValueError, EOFError):
            sys.exit(1)

    # Select translation service
    if args.translator is not None:
        translator_choice = args.translator
    else:
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
            if t_choice == "2":
                translator_choice = "deepl"
            elif t_choice == "3":
                translator_choice = "qwen"
            elif t_choice == "4":
                translator_choice = "nllb"
            elif t_choice == "5":
                translator_choice = "none"
            else:
                translator_choice = "google"
        except (ValueError, EOFError):
            translator_choice = "google"

    # Defaults for source/target languages
    translate_langs = set()
    target_lang = "en"

    if translator_choice == "none":
        translator = None
        print("Using translator: None (disabled)")
    else:
        # Determine source languages to translate
        if args.translate_from is not None:
            if args.translate_from == "all":
                translate_langs = set(LANG_NAMES.keys())
            else:
                translate_langs = set(args.translate_from.split(","))
        else:
            # Interactive multi-select
            lang_options = list(LANG_NAMES.items())
            print("\nTranslate FROM (comma-separated, Enter=1):")
            for i, (code, name) in enumerate(lang_options, 1):
                print(f"  [{i}] {name} ({code})")
            print(f"  [*] All")
            try:
                choice = input("Select: ").strip()
                if choice == "*":
                    translate_langs = {code for code, _ in lang_options}
                elif choice:
                    for idx_str in choice.split(","):
                        idx = int(idx_str.strip()) - 1
                        if 0 <= idx < len(lang_options):
                            translate_langs.add(lang_options[idx][0])
                else:
                    translate_langs = {"ko"}  # default
            except (ValueError, EOFError):
                translate_langs = {"ko"}

        # Determine target language
        if args.translate_to is not None:
            target_lang = args.translate_to
        else:
            # Interactive single-select
            lang_options = list(LANG_NAMES.items())
            print("\nTranslate TO:")
            for i, (code, name) in enumerate(lang_options, 1):
                print(f"  [{i}] {name} ({code})")
            try:
                choice = input("Select [Enter=1]: ").strip()
                if choice:
                    idx = int(choice) - 1
                    if 0 <= idx < len(lang_options):
                        target_lang = lang_options[idx][0]
                else:
                    target_lang = "en"  # default
            except (ValueError, EOFError):
                target_lang = "en"

        # Remove target language from source set (no point translating to itself)
        translate_langs.discard(target_lang)

        if translator_choice == "deepl":
            translator = DeepLTranslator(target_lang=target_lang)
            print(f"Using translator: DeepL")
        elif translator_choice == "qwen":
            translator = QwenTranslator(target_lang=target_lang)
            print(f"Using translator: Qwen (local LLM)")
        elif translator_choice == "nllb":
            translator = NLLBTranslator(target_lang=target_lang)
            print(f"Using translator: NLLB-200 (local)")
        else:
            translator = Translator(target_lang=target_lang)
            print(f"Using translator: Google Translate")

    # Select display mode
    if args.display is not None:
        display_mode = args.display
    else:
        print("\nDisplay mode:")
        print("-" * 40)
        print("  [1] Columns (side-by-side transcription/translation)")
        print("  [2] Chat (bubble UI per speaker)")
        print()
        try:
            d_choice = input("Select display mode [Enter=1]: ").strip()
            display_mode = "chat" if d_choice == "2" else "columns"
        except (ValueError, EOFError):
            display_mode = "columns"
    print(f"Using display: {display_mode}")

    # Select summary mode
    if args.summary is not None:
        enable_summary = args.summary == "on"
    else:
        print("\nLive summary (local LLM):")
        print("-" * 40)
        print("  [1] Off")
        print("  [2] On (rolling summary via Qwen 7B)")
        print()
        try:
            s_choice = input("Select [Enter=1]: ").strip()
            enable_summary = s_choice == "2"
        except (ValueError, EOFError):
            enable_summary = False

    # Select Whisper model
    model_map = {"medium": WHISPER_MODEL, "turbo": WHISPER_MODEL_TURBO, "full": WHISPER_MODEL_FULL}
    model_repo = model_map[args.model]
    print(f"Using Whisper model: {model_repo}")

    # Build the display (directly implements EngineListener).
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

    stream = open_stream(device_idx, on_chunk=engine.push_audio)

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        stop_event.set()
        print("\n\nStopping...")

    signal.signal(signal.SIGINT, signal_handler)

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
        transcript_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "transcripts"
        )
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
