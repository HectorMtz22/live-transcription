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
- [uv](https://github.com/astral-sh/uv) (`brew install uv`)
- [just](https://github.com/casey/just) (`brew install just`)

## Setup

```sh
brew install --cask blackhole-2ch
# reboot, then in Audio MIDI Setup create a Multi-Output Device combining
# your speakers + BlackHole 2ch, and set it as the system output.

uv sync
cp .env.example .env   # only if using the DeepL translator
```

## Usage

```sh
just run
# or
uv run live-transcribe
# or
python -m live_transcribe_cli
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
| `--diarize` | `on` / `off` | Speaker diarization (default off) |

Supported languages: Korean (`ko`), English (`en`), Spanish (`es`).

## Repository layout

```
packages/
  core/            # live_transcribe_core — engine, VAD, Whisper, translators, summarizer
  cli/             # live_transcribe_cli — terminal UI, audio capture, transcript save
training/          # Optional fine-tuning scripts (Korean LoRA); separate venv
transcripts/       # Runtime output (gitignored)
```

Transcripts are written to `transcripts/` on shutdown.

## Training (optional)

See `training/README.md` for Korean LoRA fine-tuning.
