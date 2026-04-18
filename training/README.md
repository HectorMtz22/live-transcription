# Training scripts

Standalone scripts for fine-tuning Whisper on Korean Common Voice. These are
NOT part of the live-transcribe runtime — they produce a fine-tuned model
artifact that can optionally be loaded by the core engine later.

Run them in a separate venv (kept isolated from the main `.venv` so the
training dependencies don't pollute the runtime):

```sh
python -m venv .venv-training
.venv-training/bin/pip install -r training/requirements.txt
.venv-training/bin/python training/finetune_whisper_ko.py --help
.venv-training/bin/python training/merge_and_convert.py --help
```

`finetune_whisper_ko.py` requires `MDC_API_KEY` in the environment for the
datacollective dataset download.
