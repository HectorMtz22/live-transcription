"""Live chunked summarizer using a local MLX LLM.

Every `interval` transcript lines, a chunk summary is produced over just
those lines plus the previous chunk's summary as context. Summaries are
emitted as dicts on the summary queue:

    {"index": int, "timestamp": "HH:MM:SS", "text": str, "is_final": bool}
"""

import multiprocessing as mp
import threading
import time
from datetime import datetime

# Model to use for summarization (small, fast, multilingual)
SUMMARIZER_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# How many new transcript lines before triggering a new chunk summary
SUMMARY_INTERVAL = 5


def _build_prompt(lines, target_lang, previous_summary):
    """Build the summarization prompt for one chunk."""
    transcript = "\n".join(
        f"{line['speaker']}: {line['text']}" for line in lines
    )
    prev = previous_summary if previous_summary else "(none — this is the first chunk)"
    return (
        f"You are a summarizer. Below is a chunk of a live conversation transcript "
        f"(may contain Korean, English, or Spanish).\n\n"
        f"Previous summary (reference only — do not restate or paraphrase it):\n{prev}\n\n"
        f"New transcript chunk:\n{transcript}\n\n"
        f"Write a concise summary in {target_lang} of ONLY the new chunk above. "
        f"Use the previous summary only to resolve implied references (pronouns, "
        f"unresolved topics) — do not restate or paraphrase it. "
        f"Focus on key topics, decisions, and important points. Keep it under 120 words.\n\n"
        f"Summary:"
    )


def _generate_with_model(model, tokenizer, lines, target_lang, previous_summary):
    """Generate a chunk summary using the loaded model."""
    from mlx_lm import generate

    prompt = _build_prompt(lines, target_lang, previous_summary)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    summary = generate(
        model, tokenizer,
        prompt=formatted,
        max_tokens=300,
        verbose=False,
    )
    return summary.strip()


def _now_hms():
    return datetime.now().strftime("%H:%M:%S")


class Summarizer:
    """In-process chunked summarizer. Not used by the engine (kept for parity).

    Produces one summary per `interval` transcript lines. Each summary covers
    only the chunk's lines, conditioned on the previous chunk's summary.
    """

    def __init__(self, target_lang="en", model_repo=SUMMARIZER_MODEL,
                 interval=SUMMARY_INTERVAL, on_summary=None):
        from mlx_lm import load

        self.target_lang = target_lang
        self.interval = interval
        self.on_summary = on_summary  # callback(item_dict)

        self._buffer: list[dict] = []
        self._previous_summary = ""
        self._chunk_index = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = None

        print(f"Loading summarizer model '{model_repo}'...")
        self._model, self._tokenizer = load(model_repo)
        print("Summarizer model ready.")

    def add_line(self, speaker, text, language):
        """Add a transcript line. Thread-safe."""
        with self._lock:
            self._buffer.append({"speaker": speaker, "text": text, "language": language})

    def start(self):
        """Start the background summarization thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal stop. Flushes any tail lines as a final chunk."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        self._flush_final()

    def _run(self):
        while self._running:
            time.sleep(1)
            self._maybe_fire_chunk()

    def _maybe_fire_chunk(self):
        with self._lock:
            if len(self._buffer) < self.interval:
                return
            chunk_lines = self._buffer[:self.interval]
            self._buffer = self._buffer[self.interval:]

        self._fire(chunk_lines, is_final=False)

    def _flush_final(self):
        with self._lock:
            tail = list(self._buffer)
            self._buffer.clear()
        if tail:
            self._fire(tail, is_final=True)

    def _fire(self, lines, is_final):
        self._chunk_index += 1
        summary = _generate_with_model(
            self._model, self._tokenizer, lines, self.target_lang, self._previous_summary
        )
        self._previous_summary = summary
        item = {
            "index": self._chunk_index,
            "timestamp": _now_hms(),
            "text": summary,
            "is_final": is_final,
        }
        if self.on_summary:
            self.on_summary(item)


class SummarizerProcess:
    """Runs chunked summarization in a separate process for GPU independence.

    Uses multiprocessing with 'spawn' context to safely create a second
    MLX/Metal context. Whisper transcription in the main process is never
    blocked by summary generation.
    """

    def __init__(self, target_lang="en", model_repo=SUMMARIZER_MODEL,
                 interval=SUMMARY_INTERVAL, on_summary=None):
        self.target_lang = target_lang
        self.model_repo = model_repo
        self.interval = interval
        self.on_summary = on_summary  # callback(item_dict)

        ctx = mp.get_context("spawn")
        self._line_queue = ctx.Queue()
        self._summary_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        self._process = ctx.Process(
            target=SummarizerProcess._worker,
            args=(self._line_queue, self._summary_queue, self._stop_event,
                  model_repo, target_lang, interval),
            daemon=True,
        )
        self._poll_thread = None

    def add_line(self, speaker, text, language):
        """Add a transcript line. Thread/process-safe via queue."""
        self._line_queue.put({"speaker": speaker, "text": text, "language": language})

    def start(self):
        """Start the child process and the polling thread."""
        self._process.start()
        self._poll_thread = threading.Thread(target=self._poll_summaries, daemon=True)
        self._poll_thread.start()

    def stop(self):
        """Signal the child to stop, drain the final-chunk event, and clean up.

        Joins _poll_thread so the `on_summary` callback has fired for every
        queued item before this returns — the engine relies on this ordering
        to keep `StatusEvent("stopped")` the terminal listener event.
        """
        # Sentinel triggers the final-chunk path inside the worker.
        self._line_queue.put(None)
        self._process.join(timeout=45)
        if self._process.is_alive():
            self._process.terminate()
            # Child was killed before signaling — unblock the poll thread so it
            # doesn't spin on `not self._stop_event.is_set()` forever.
            self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=30)

    def _poll_summaries(self):
        """Poll the summary queue and invoke on_summary(item_dict)."""
        while not self._stop_event.is_set() or not self._summary_queue.empty():
            try:
                item = self._summary_queue.get(timeout=1.0)
                if item and self.on_summary:
                    self.on_summary(item)
            except Exception:
                continue

    @staticmethod
    def _worker(line_queue, summary_queue, stop_event, model_repo, target_lang, interval):
        """Child process: loads model, produces chunk summaries."""
        from mlx_lm import load

        print(f"[SummarizerProcess] Loading model '{model_repo}'...")
        model, tokenizer = load(model_repo)
        print("[SummarizerProcess] Model ready.")

        buffer: list[dict] = []
        previous_summary = ""
        chunk_index = 0

        while True:
            # Drain all available lines from the queue (non-blocking).
            got_sentinel = False
            while True:
                try:
                    item = line_queue.get_nowait()
                    if item is None:
                        got_sentinel = True
                        break
                    buffer.append(item)
                except Exception:
                    break

            # Normal chunk: fire as many full chunks as the buffer allows.
            while len(buffer) >= interval and not got_sentinel:
                chunk_lines = buffer[:interval]
                buffer = buffer[interval:]
                chunk_index += 1
                summary = _generate_with_model(
                    model, tokenizer, chunk_lines, target_lang, previous_summary
                )
                previous_summary = summary
                summary_queue.put({
                    "index": chunk_index,
                    "timestamp": _now_hms(),
                    "text": summary,
                    "is_final": False,
                })

            if got_sentinel:
                # Final chunk: whatever tail is left (may be empty).
                if buffer:
                    chunk_index += 1
                    summary = _generate_with_model(
                        model, tokenizer, buffer, target_lang, previous_summary
                    )
                    summary_queue.put({
                        "index": chunk_index,
                        "timestamp": _now_hms(),
                        "text": summary,
                        "is_final": True,
                    })
                stop_event.set()
                break

            time.sleep(1)
