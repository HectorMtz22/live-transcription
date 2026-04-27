"""Translation support using a local Qwen MLX model in a child process.

QwenTranslator is a parent-process facade. It owns a `multiprocessing.Process`
that loads the model once and serves translation requests over a pair of
queues. Whisper transcription in the main process is never blocked by Qwen
generation, because they run in separate Metal contexts.

Crash handling:
- A reply timeout where the child is *still alive* is treated as transient
  (returns None, no restart).
- A reply timeout where the child has died triggers an auto-restart, unless
  another restart already happened in the last 60 s — in that case the
  translator enters degraded mode and all calls return None for the rest of
  the session.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import OrderedDict
from typing import Callable, Optional

from ._qwen_ipc import (
    RetranslateBatchReply,
    RetranslateBatchRequest,
    TranslateReply,
    TranslateRequest,
)

TRANSLATION_CACHE_SIZE = 256
QWEN_MODEL = "Qwen/Qwen3-8B-MLX-4bit"

LANG_NAMES = {
    "ko": "Korean",
    "en": "English",
    "es": "Spanish",
}

TRANSLATE_TIMEOUT = 30.0           # seconds, per single translate call
RETRANSLATE_BATCH_TIMEOUT = 90.0   # seconds, per batch call
RESTART_COOLDOWN = 60.0            # seconds; second crash inside this window → degraded


# ---------------------------------------------------------------------------
# Shared prompt builder (runs in the child).
# ---------------------------------------------------------------------------

def _build_translate_prompt(text: str, source_lang: str, target_lang: str, context):
    """Build the chat-template-formatted prompt for a single translation."""
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    prompt_parts = [
        f"Translate the following {src_name} text to {tgt_name}.",
        "Use the conversation context below to maintain consistency in terminology, "
        "names, and meaning. If previous translations have errors based on new context, "
        "translate the current text correctly using the improved understanding.",
        "Output ONLY the translation, nothing else.",
    ]

    if context:
        ctx_lines = []
        for item in context:
            if isinstance(item, tuple):
                orig, trans = item
                if trans:
                    ctx_lines.append(f"  {src_name}: {orig}\n  {tgt_name}: {trans}")
                else:
                    ctx_lines.append(f"  {src_name}: {orig}")
            else:
                ctx_lines.append(f"  {src_name}: {item}")
        prompt_parts.append(
            "\nConversation so far (for reference, do NOT translate these):\n"
            + "\n".join(ctx_lines)
        )

    prompt_parts.append(f"\nText to translate:\n{text}")
    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Child-process worker.
# ---------------------------------------------------------------------------

def _worker(request_q: "mp.Queue", reply_q: "mp.Queue", model_repo: str):
    """Child process entry point. Loads the model and serves requests until None."""
    from mlx_lm import generate, load

    print(f"[QwenProcess] Loading model '{model_repo}'...", flush=True)
    model, tokenizer = load(model_repo)
    print("[QwenProcess] Model ready.", flush=True)

    def _generate_one(text: str, source_lang: str, target_lang: str, context) -> Optional[str]:
        try:
            prompt = _build_translate_prompt(text, source_lang, target_lang, context)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            result = generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=512,
                verbose=False,
            )
            return result.strip() if result else None
        except Exception as e:
            print(f"[QwenProcess] generate error: {e}", flush=True)
            return None

    while True:
        try:
            item = request_q.get()
        except (EOFError, OSError):
            break
        if item is None:
            break

        if isinstance(item, TranslateRequest):
            if item.source_lang == item.target_lang:
                reply_q.put(TranslateReply(request_id=item.request_id, result=None))
                continue
            result = _generate_one(item.text, item.source_lang, item.target_lang, item.context)
            reply_q.put(TranslateReply(request_id=item.request_id, result=result))
        elif isinstance(item, RetranslateBatchRequest):
            results = []
            for text, source_lang in item.items:
                if source_lang == item.target_lang:
                    results.append(None)
                    continue
                results.append(_generate_one(text, source_lang, item.target_lang, item.context))
            reply_q.put(RetranslateBatchReply(request_id=item.request_id, results=results))
        # Unknown message types are silently dropped.


# ---------------------------------------------------------------------------
# Parent-process facade.
# ---------------------------------------------------------------------------

class QwenTranslator:
    """Subprocess-backed Qwen translator with LRU cache and auto-restart."""

    def __init__(
        self,
        target_lang: str = "en",
        model_repo: str = QWEN_MODEL,
        on_status: Optional[Callable[[str, str], None]] = None,
    ):
        """on_status, if provided, is called as on_status(state, message) for
        warning/info/error transitions. The engine wires this to its listener."""
        self.target_lang = target_lang
        self._model_repo = model_repo
        self._on_status = on_status

        self._cache: "OrderedDict[tuple[str, str], str]" = OrderedDict()
        self._cache_lock = threading.Lock()

        self._watchdog_lock = threading.Lock()
        self._last_restart_at: Optional[float] = None
        self._degraded = False

        self._ctx = mp.get_context("spawn")
        self._request_queue = self._ctx.Queue()
        self._reply_queue = self._ctx.Queue()
        self._process: Optional[mp.Process] = None

        # Match the legacy `_available` flag the engine checks at construction.
        # If the model can't be loaded the child will crash; we report it via
        # the watchdog. Construction always "succeeds" — there's no work to do
        # synchronously here.
        self._available = True
        self._spawn_child()

    # -- public API ---------------------------------------------------------

    def translate(self, text: str, source_lang: str, context=None) -> Optional[str]:
        if self._degraded or source_lang == self.target_lang:
            return None

        if not context:
            cached = self._cache_get(text, source_lang)
            if cached is not None:
                return cached

        request = TranslateRequest(
            request_id=uuid.uuid4().hex,
            text=text,
            source_lang=source_lang,
            target_lang=self.target_lang,
            context=context,
        )
        reply = self._send_and_wait(request, TRANSLATE_TIMEOUT)
        if reply is None:
            return None
        if reply.result:
            self._cache_put(text, source_lang, reply.result)
        return reply.result

    def retranslate_batch(self, items, context=None) -> list:
        """Translate N (text, source_lang) pairs in one IPC roundtrip.

        Returns a list of Optional[str] aligned with `items`. Cache writes
        happen for any non-None result.
        """
        if self._degraded or not items:
            return [None] * len(items)

        request = RetranslateBatchRequest(
            request_id=uuid.uuid4().hex,
            items=list(items),
            target_lang=self.target_lang,
            context=context,
        )
        reply = self._send_and_wait(request, RETRANSLATE_BATCH_TIMEOUT)
        if reply is None:
            return [None] * len(items)

        for (text, source_lang), result in zip(items, reply.results):
            if result:
                self._cache_put(text, source_lang, result)
        return list(reply.results)

    def stop(self) -> None:
        if self._degraded or self._process is None:
            return
        try:
            self._request_queue.put(None)
        except Exception:
            pass
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)

    # -- internals ----------------------------------------------------------

    def _spawn_child(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._reply_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker,
            args=(self._request_queue, self._reply_queue, self._model_repo),
            daemon=True,
        )
        self._process.start()

    def _send_and_wait(self, request, timeout: float):
        # Drain stale replies left over from a previous timed-out call where
        # the child was alive-but-slow; otherwise the late reply collides with
        # the next request's id.
        while True:
            try:
                self._reply_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self._request_queue.put(request)
        except Exception:
            self._on_failure()
            return None

        try:
            reply = self._reply_queue.get(timeout=timeout)
        except queue.Empty:
            self._on_failure()
            return None

        if reply.request_id != request.request_id:
            # Defensive: should not happen given the single-worker invariant
            # in the parent + the drain above.
            return None
        return reply

    def _on_failure(self) -> None:
        with self._watchdog_lock:
            if self._degraded:
                return
            if self._process is None or self._process.is_alive():
                # Alive-but-slow: transient, no action.
                return

            now = time.monotonic()
            if self._last_restart_at is not None and (now - self._last_restart_at) < RESTART_COOLDOWN:
                self._degraded = True
                self._emit("error", "Qwen translator crashed twice; disabled for this session")
                return

            self._emit("warning", "Qwen translator crashed, restarting…")
            self._spawn_child()
            self._last_restart_at = now
            self._emit("info", "Qwen translator restarted")

    def _emit(self, state: str, message: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(state, message)
            except Exception:
                pass

    # -- cache --------------------------------------------------------------

    def _cache_get(self, text: str, source_lang: str) -> Optional[str]:
        key = (text.strip(), source_lang)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, text: str, source_lang: str, result: str) -> None:
        key = (text.strip(), source_lang)
        with self._cache_lock:
            self._cache[key] = result
            if len(self._cache) > TRANSLATION_CACHE_SIZE:
                self._cache.popitem(last=False)
