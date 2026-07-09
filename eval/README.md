# Korean ASR A/B benchmark: Whisper vs Qwen3-ASR

Throwaway offline benchmark comparing the app's current Whisper model against
[Qwen3-ASR-1.7B](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
(MLX) on Korean, to decide go/no-go on adding a Qwen ASR backend to the live
runtime.

This is **separate tooling**, like `training/` — it is NOT part of the
`live_transcribe_core` / `live_transcribe_cli` runtime, has its own venv, and
is NOT run by `just test` (the workspace's root `pytest.ini`/`pyproject.toml`
scopes to `tests/`, which this directory is not part of). Nothing under
`packages/` depends on anything here.

## Setup

Unit tests only need light deps (`jiwer`, `numpy`, `pytest`) and run in
seconds with no model downloads:

```sh
cd eval
uv venv
uv pip install jiwer numpy pytest
.venv/bin/python -m pytest -q
```

The real benchmark run needs the full dependency list in `requirements.txt`
(`mlx-whisper`, `qwen3-asr-mlx`, `datasets`, `soundfile`, `scipy`, `psutil` —
Apple Silicon + Python 3.10-3.13 + MLX 0.31+ required; models download on
first run, several GB):

```sh
cd eval
uv venv
uv pip install -r requirements.txt
.venv/bin/python bench_korean_asr.py --limit 150
```

Flags: `--limit` (default 150), `--dataset` (default `Bingsu/zeroth-korean`),
`--whisper-model` (default `mlx-community/whisper-large-v3-mlx-4bit`),
`--qwen-model` (default `mlx-community/Qwen3-ASR-1.7B-4bit`, 4-bit — see
below), `--out` (default `eval/results.md`, always resolved relative to this
script's own location, so it lands there **regardless of cwd** — whether you
run `bench_korean_asr.py` from inside `eval/` as shown above, or
`eval/bench_korean_asr.py` from the repo root).

Qwen model default: `mlx-community/Qwen3-ASR-1.7B-4bit`, chosen for
like-for-like quantization with the app's 4-bit Whisper. Pass
`--qwen-model mlx-community/Qwen3-ASR-1.7B-bf16` to override for max
accuracy. Confirm the exact repo id on Hugging Face on first download —
mlx-community's Qwen3-ASR quantization naming may shift.

## What it measures

For each backend, over the same set of dataset utterances:

- **CER / WER** — character/word error rate against the dataset reference
  text, computed with `jiwer` after normalizing both sides (`asr_eval.metrics.normalize_korean`:
  NFC-normalize, replace Latin/CJK punctuation with a space, collapse
  whitespace, lowercase). Note: this makes CER **space-sensitive** — Korean
  spacing conventions (띄어쓰기) can shift CER slightly depending on where
  punctuation sat in the source text, since it now leaves a separating
  space rather than gluing adjacent words together.
- **RTF (real-time factor)** — `processing_time / audio_duration` per
  utterance; reports mean, p95, and an aggregate `total` (sum of processing
  time / sum of duration).
- **Peak RAM (MB)** — sampled via a background `psutil` thread polling RSS
  during `load()` + transcription, so it captures the model's peak memory
  footprint, not a single snapshot.
- **Load time (s)** — how long the backend's model load takes.

## Go/no-go thresholds

`asr_eval.report.decide(whisper, qwen)` returns `(verdict, reason)`. Verdict
is **GO** iff **all** of:

- relative CER reduction `(whisper.cer - qwen.cer) / whisper.cer >= 0.15`
  (Qwen must be at least 15% relatively more accurate), **and**
- `qwen.rtf_mean <= 0.6` (Qwen must transcribe faster than 0.6x real time on
  average).

Peak RAM is reported in the reason but **not auto-gated** — whether Qwen's
memory footprint fits alongside the rest of the live pipeline (Whisper +
Silero VAD + translator, all sharing the same Metal/MLX GPU context) is a
judgment call for a human, not a hard threshold here.

## Preprocessing

Whisper is fed the app's exact frontend (high-pass filter + pad +
peak-normalize, mirroring `whisper.py`'s `preprocess_audio` — see below),
while Qwen3-ASR is fed raw 16 kHz mono audio, as its package expects. This
asymmetry is **intentional**: each model is measured in its realistic
deployment configuration, not on artificially identical inputs. A possible
follow-up, if a purer model-quality signal is wanted instead of a deployment
comparison, would be to also run both backends on symmetric (identically
preprocessed, or identically raw) input.

## Whisper parameter replication

`asr_eval/backends.py::WhisperBackend` replicates the exact preprocessing and
`mlx_whisper.transcribe(...)` parameters used by the live runtime
(`packages/core/src/live_transcribe_core/whisper.py`'s `preprocess_audio` and
`transcribe`), so the comparison reflects how Whisper actually runs in
production rather than its out-of-the-box defaults. This is a **manually
kept-in-sync copy**, not a shared import — the eval venv is intentionally
isolated from the runtime's dependency tree (see requirements.txt vs. the
workspace's `pyproject.toml`). If `whisper.py`'s preprocessing or transcribe
params change, update `backends.py` to match.

## Caveat: the dataset is an accuracy floor, not a ceiling

`Bingsu/zeroth-korean` (the default `--dataset`) is clean, read Korean
speech recorded close-mic in a quiet room. Both models will score better here
than they will on the live app's actual input (far-field audio through
BlackHole, natural conversational speech, background noise, code-switching).
Treat these numbers as a **best-case accuracy floor** — a useful A/B signal,
not a prediction of real-world word error rate.
