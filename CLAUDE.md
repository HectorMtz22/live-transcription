# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

The project is a uv-managed monorepo with two workspace packages (`live-transcribe-core`, `live-transcribe-cli`) under `packages/`. Bootstrap and run via:

```sh
uv sync                                  # install both packages + shared .venv
just run                                 # run the CLI (== uv run live-transcribe)
uv run live-transcribe --help            # argparse help without engine load
python -m live_transcribe_cli            # alternate invocation

just fmt                                 # ruff format packages/
just lint                                # ruff check packages/
just test                                # run the pytest suite (== uv run pytest)
just clean                               # wipe .venv and build artifacts
```

Training scripts live in `training/` with their own `requirements.txt` — they're NOT part of the runtime and should run in a separate venv.

The pytest suite lives under `tests/` (mirrors `packages/{core,cli}/`); run it with `just test`. End-to-end verification is still a manual smoke test against a mic (BlackHole).

## Architecture

### Real-time pipeline (`live_transcribe_core`)

The engine is driven by a three-stage producer/consumer orchestrated by `TranscriptionEngine` (`packages/core/src/live_transcribe_core/engine.py`). Audio chunks are **pushed** into the engine from outside (the CLI wraps a `sounddevice.InputStream` callback); the engine runs Silero VAD + mlx-whisper + translator + (optional) summarizer, and emits events to a single `EngineListener`.

1. **`engine.push_audio(chunk)`** (any thread) — enqueues a `float32` mono @ 16 kHz chunk into `_audio_queue` under `_vad_lock`.
2. **`_process_audio`** (engine-owned thread) — drains `_audio_queue`, runs Silero VAD per 512-sample frame, accumulates `_speech_buffer`. On end-of-speech (silence timeout) or max-duration cap, it submits the segment to `_transcription_pool` (1 worker).
3. **`_transcribe_segment`** (transcription worker) — calls the configured `AsrBackend.transcribe(...)` (see "ASR backends" below), filters via `is_hallucination` + `DuplicateFilter`, groups segments by speaker, then for each group emits a `SegmentEvent` and submits translation work to `_translation_pool` (1 worker for Qwen, 4 otherwise).

Key invariants to preserve when editing:

- **The single-worker `_transcription_pool` serializes all ASR decode calls** (Whisper or Qwen — whichever backend is configured); there is no separate GPU lock. The Qwen *translator* runs independently on its own `_translation_pool`. The Summarizer sidesteps both pools by running in a **separate process** (`SummarizerProcess`, spawn context) with its own Metal context.
- **Adaptive VAD** (`_adaptive_thresholds`): `silence_cap` and `max_speech_cap` scale linearly with `_pending_count`. When Whisper is backed up, VAD produces fewer, bigger chunks.
- **Shutdown flushes the in-progress speech buffer.** `_process_audio`'s `finally` calls `_flush_speech_buffer()`; `stop()` then shuts the pools with `wait=True` so the final submission completes.
- **Speech duration is measured from samples, not wall clock.** Wall clock drifts when the VAD thread catches up on queued audio.
- **Listener callbacks are emitted synchronously from worker threads.** Implementations MUST be thread-safe. `_recent_context` and `_detected_lang` are safe without locks only because the single-worker invariant on the transcription pool serializes writers (documented in code comments).

### Event contract (core → CLI)

Core emits four **frozen dataclass** events via an `EngineListener` Protocol (`packages/core/src/live_transcribe_core/events.py`):

- `SegmentEvent(id, timestamp, speaker, text, language)` — UUID-keyed.
- `TranslationEvent(segment_id, text, is_update)` — `is_update=True` for Qwen retranslate revisions.
- `SummaryEvent(text, is_final)` — `is_final=True` for the one at shutdown.
- `StatusEvent(state, message)` — lifecycle + errors.

All fields are primitives, so `dataclasses.asdict(event)` yields a JSON-serializable dict — future-ready for a WebSocket/HTTP transport to a Flutter/Swift frontend.

Event IDs are `uuid.uuid4().hex` — stable across process boundaries.

### Translator interface

All backends (`live_transcribe_core.translators.{google,deepl,qwen,nllb}`) share:

```python
translate(text: str, source_lang: str, context=None) -> Optional[str]
```

- `context` is a list of `(original, translation)` tuples (last ~5 used). Google/DeepL/Qwen use it; NLLB ignores it.
- Returning `None` means "skipped/failed" — callers fall back gracefully.
- Only Qwen triggers `_retranslate_recent` to revise prior translations as context grows.

Chunking: Qwen translates the full text as one unit; the others get `whisper.chunk_for_translation` output translated in parallel and joined.

### ASR backends

The engine decodes audio through a pluggable `AsrBackend` (`packages/core/src/live_transcribe_core/asr.py`), built once at `start()` by `build_asr(config)` from `EngineConfig.asr_backend` (`"whisper"` default, or `"qwen"`):

- `WhisperAsr` wraps `live_transcribe_core.whisper.transcribe` (mlx-whisper) — the default, always available.
- `QwenAsr` drives the optional local Qwen3-ASR MLX model (`qwen3_asr_mlx`), lazily imported so it's only required when selected. Install via the `qwen-asr` extra (`uv sync --all-packages --extra qwen-asr`). It returns a full language name (e.g. `"Korean"`) rather than an ISO code, so `QwenAsr` normalizes it before returning.

Both backends return the same mlx-whisper-shaped `{"language": ..., "segments": [...]}` dict so `_transcribe_segment` doesn't need to branch. Selectable from the CLI via `--asr-backend {whisper,qwen}`.

### CLI (`live_transcribe_cli`)

- `live_transcribe_cli.main:main` — argparse + interactive pickers + engine wiring.
- `live_transcribe_cli.audio` — BlackHole discovery, `open_stream(device_idx, on_chunk)` wraps sounddevice.
- `live_transcribe_cli.transcript` — `save_transcript(segments, translations, target_lang, transcript_dir)` writes both `_original.txt` and (if translations are present) `_<lang>.txt`.
- `live_transcribe_cli.displays` — `BaseDisplay(EngineListener)` tracks segment/translation state (`translations: dict[str, str]`); `ColumnsDisplay` and `ChatDisplay` implement the `_render_*` hooks for Rich.

The root `live_transcribe.py` is a 5-line shim for backward-compatible `python live_transcribe.py` invocation.

### Supported languages

Only `ko`, `en`, `es` are supported. `_transcribe_segment` drops any other detected language early. Changing this requires updating `core.config` (`SUPPORTED_LANGUAGES`, `LANG_NAMES`, `INITIAL_PROMPTS`, `HALLUCINATION_PHRASES`) and each translator's per-backend code map.

### Korean fine-tuning pipeline

`training/finetune_whisper_ko.py` trains a LoRA adapter on Korean Common Voice (`openai/whisper-large-v3` base, PEFT, MPS backend) → outputs `./whisper-ko-lora/final`. `training/merge_and_convert.py` merges it and converts to quantized MLX. Independent of the live runtime; the live app currently loads public MLX Whisper models.

## Conventions

- Output artifacts (`transcripts/`, `whisper-ko-lora/`, `whisper-large-v3-ko-mlx/`) and the venv are gitignored.
- `docs/superpowers/` is gitignored — local-only plans and specs; do not commit.
- Tests live under `tests/` (mirrors `packages/{core,cli}/`); run with `just test`.
- Commit messages do not include Co-Authored-By trailers.

## Workflow & agents

Feature work follows a repeatable harness (brainstorm → spec → issue → worktree →
TDD → verify → review → PR). The *how* lives in [`HARNESS.md`](HARNESS.md) and
the *why* in [`AGENTS.md`](AGENTS.md), wrapped by two slash commands:

- **`/task-init <idea>`** — `superpowers:brainstorming` → local spec under
  `docs/superpowers/specs/` (gitignored) → file issue(s) in the tracker.
- **`/task-implement <LT-12 …>`** — worktree under `.worktrees/` (gitignored) →
  TDD (`just test`) → verify → `superpowers:requesting-code-review` → PR.

One-time setup: **`/harness-setup`** (choose tracker → writes
[`.claude/tracker.md`](.claude/tracker.md)), then **`/harness-bootstrap`**
(create project, states, labels, cycles). Key rules:

- **Always use superpowers** — invoke the named skill at each stage; report
  review findings by severity and **do not auto-fix** (the user picks scope).
- **Specs/plans are local-only** (`docs/superpowers/`); **issues live in the
  tracker**; **implementation always happens in a worktree**, never the main
  checkout. Conventional commits scoped per package (`core` / `cli`).
