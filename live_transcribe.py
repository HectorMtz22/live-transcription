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
import time
from datetime import datetime

import sounddevice as sd

from display_columns import ColumnsDisplay
from display_chat import ChatDisplay
from live_transcribe_core import (
    EngineConfig,
    EngineListener,
    SegmentEvent,
    StatusEvent,
    SummaryEvent,
    TranscriptionEngine,
    TranslationEvent,
)
from live_transcribe_core.translators import (
    DeepLTranslator,
    GoogleTranslator as Translator,
    NLLBTranslator,
    QwenTranslator,
)

# ─── Configuration ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000          # Whisper expects 16kHz
WHISPER_MODEL = "mlx-community/whisper-medium-mlx-q4"  # Q4 quantized medium; fast enough for real-time, decent multilingual quality
WHISPER_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"  # Distilled: same encoder, 4 decoder layers (vs 32); higher quality, slower
WHISPER_MODEL_FULL = "mlx-community/whisper-large-v3-mlx-4bit"  # Q4 quantized full model; 32 decoder layers, slowest but max accuracy
LANG_NAMES = {"ko": "Korean", "en": "English", "es": "Spanish"}

# ─── Global State ────────────────────────────────────────────────────────────

running = True


def find_blackhole_device():
    """Find the BlackHole 2ch input device index."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i, dev["name"]
    return None, None


def list_input_devices(default_idx=None):
    """List all available input devices, highlighting the default."""
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            markers = []
            if "blackhole" in dev["name"].lower():
                markers.append("BlackHole")
            if i == default_idx:
                markers.append("default")
            suffix = f"  <-- [{', '.join(markers)}]" if markers else ""
            print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch, "
                  f"{dev['default_samplerate']:.0f}Hz){suffix}")
    print()


class _DisplayAdapter:
    """Temporary bridge: forwards EngineListener events to the old print_* display API.

    Removed in Task 5 when ColumnsDisplay and ChatDisplay implement
    EngineListener directly.
    """

    def __init__(self, display, has_translator: bool):
        self._display = display
        self._has_translator = has_translator
        self._entry_lookup = {}  # segment_id -> (speaker, text, lang, timestamp)
        self._last_speaker = None
        self._print_lock = threading.Lock()

    def on_segment(self, event):
        with self._print_lock:
            if event.speaker != self._last_speaker:
                self._display.print_segment_header(
                    event.speaker, event.timestamp,
                    has_translator=self._has_translator,
                    entry_key=event.id,
                )
                self._last_speaker = event.speaker
            self._entry_lookup[event.id] = (
                event.speaker, event.text, event.language, event.timestamp,
            )
            if not self._has_translator:
                self._display.print_without_translation(
                    event.speaker, event.text, event.language,
                    timestamp=event.timestamp, entry_key=event.id,
                )

    def on_translation(self, event):
        meta = self._entry_lookup.get(event.segment_id)
        if meta is None:
            return
        speaker, text, lang, timestamp = meta
        with self._print_lock:
            if event.is_update:
                self._display.update_translation(
                    event.segment_id, speaker, text, event.text,
                    lang, timestamp=timestamp,
                )
            else:
                self._display.print_translated(
                    speaker, text, event.text or None, lang,
                    timestamp=timestamp, entry_key=event.segment_id,
                )

    def on_summary(self, event):
        if not event.text:
            return
        print(f"\n\033[1;35m{'─' * 40}")
        print(f"  {'FINAL SUMMARY' if event.is_final else 'SUMMARY'}")
        print(f"{'─' * 40}\033[0m")
        print(f"\033[0;35m  {event.text}\033[0m")
        print(f"\033[1;35m{'─' * 40}\033[0m\n")

    def on_status(self, event):
        if event.state == "error" and event.message:
            print(f"\033[0;31m[Error] {event.message}\033[0m")


def _run_with_audio(engine, display, device_idx, target_lang):
    """Drive the engine with a sounddevice InputStream.

    Temporary — Task 5 moves this into live_transcribe_cli.audio.
    """
    global running
    running = True

    stream = sd.InputStream(
        device=device_idx,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=lambda indata, frames, time_info, status: engine.push_audio(indata[:, 0].copy()),
    )

    def signal_handler(sig, frame):
        global running
        running = False
        print("\n\nStopping...")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        engine.start()
        stream.start()
        display.start()
        while running:
            time.sleep(0.1)
    finally:
        display.stop()
        stream.stop()
        stream.close()
        engine.stop()
        _save_transcript_inline(engine.get_transcript(), target_lang)
        print("Done.")


def _save_transcript_inline(segments, target_lang):
    """Temporary — Task 5 moves this into live_transcribe_cli.transcript.save_transcript."""
    if not segments:
        return
    transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    header_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    original_path = os.path.join(transcript_dir, f"transcript_{timestamp}_original.txt")
    with open(original_path, "w") as f:
        f.write(f"Transcript (Original) - {header_time}\n")
        f.write("=" * 60 + "\n\n")
        current_speaker = None
        for seg in segments:
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                f.write(f"\n[{seg.timestamp}] {current_speaker}:\n")
            f.write(f"  {seg.text}\n")
    print(f"\n\033[1;32mTranscript saved to: {original_path}\033[0m")
    # Translations: not tracked in the SegmentEvent snapshot. Accepted one-commit
    # regression — Task 5's save_transcript captures translations from the display
    # listener state and restores the translated file.


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

    # Build the display directly (no longer owned by LiveTranscriber)
    if display_mode == "chat":
        display = ChatDisplay()
    else:
        display = ColumnsDisplay()

    listener = _DisplayAdapter(display, has_translator=translator is not None)

    # Construct the engine. It handles GPU-lock sharing with Qwen internally
    # (see engine.start() → translators.set_gpu_lock).
    engine = TranscriptionEngine(
        EngineConfig(
            whisper_model=model_repo,
            translator=translator,
            translate_langs=translate_langs,
            target_lang=target_lang,
            enable_summary=enable_summary,
            diarize=(args.diarize == "on"),
        ),
        listener=listener,
    )

    _run_with_audio(engine, display, device_idx, target_lang)


if __name__ == "__main__":
    main()
