# live_transcribe

Live system-audio transcription with speaker diarization and optional
translation/summarization, optimized for Apple Silicon (MLX).

Captures system audio via the BlackHole virtual audio device, detects speech
with Silero VAD, transcribes with `mlx-whisper`, and separates speakers using
`resemblyzer` voice embeddings.

## Requirements

- macOS on Apple Silicon
- Python ≥ 3.10
- [BlackHole 2ch](https://existential.audio/blackhole/)

## Setup

```sh
brew install --cask blackhole-2ch
# reboot, then in Audio MIDI Setup create a Multi-Output Device combining
# your speakers + BlackHole 2ch, and set it as the system output.

python -m venv live_transcribe_env
./live_transcribe_env/bin/pip install -e .

cp .env.example .env   # only if using the DeepL translator
```

## Usage

```sh
./live_transcribe_env/bin/python live_transcribe.py
```

Useful flags:

| Flag | Values | Purpose |
| --- | --- | --- |
| `-d, --device` | int | Input device index (defaults to BlackHole) |
| `-m, --model` | `medium` / `turbo` / `full` | Whisper model size |
| `-t, --translator` | `google` / `deepl` / `qwen` / `nllb` / `none` | Translation backend |
| `--translate-from` / `--translate-to` | lang code | Override source/target language |
| `--display` | `columns` / `chat` | Terminal layout |
| `--summary` | `on` / `off` | Rolling LLM summary |

Supported languages: Korean (`ko`), English (`en`), Spanish (`es`).

## Components

- `live_transcribe.py` — main capture/VAD/transcription loop
- `display_columns.py`, `display_chat.py` — Rich-based UIs
- `translator.py`, `deepl_translator.py`, `qwen_translator.py`,
  `nllb_translator.py` — translation backends
- `summarizer.py` — rolling summary via local MLX LLM
- `finetune_whisper_ko.py`, `merge_and_convert.py` — Korean LoRA fine-tuning
  and conversion to MLX format

Transcripts are written to `transcripts/`.
