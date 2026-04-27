"""TranscriptionEngine — the real-time transcription orchestrator.

Accepts audio chunks pushed in via push_audio(), runs Silero VAD + Whisper +
translator + summarizer, and emits events through an EngineListener.

Architecture note: events are emitted synchronously from worker threads.
Listener implementations MUST be thread-safe.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from live_transcribe_core.config import (
    DEFAULT_WHISPER_MODEL,
    ENERGY_THRESHOLD,
    INITIAL_PROMPTS,
    MAX_PENDING_SEGMENTS,
    MAX_SPEECH_DURATION,
    MIN_SPEECH_DURATION,
    SAMPLE_RATE,
    SILENCE_AFTER_SPEECH,
    SUPPORTED_LANGUAGES,
    VAD_FRAME_SAMPLES,
    VAD_THRESHOLD,
)
from live_transcribe_core.events import (
    EngineListener,
    SegmentEvent,
    StatusEvent,
    SummaryEvent,
    TranslationEvent,
)
from live_transcribe_core.speaker import SpeakerTracker
from live_transcribe_core.summarizer import SummarizerProcess
from live_transcribe_core.translators import (
    QwenTranslator,
    set_gpu_lock as _inject_gpu_lock,
)
from live_transcribe_core.vad import load_vad_model
from live_transcribe_core.whisper import (
    DuplicateFilter,
    chunk_for_translation,
    is_hallucination,
    make_highpass_sos,
    preprocess_audio,
    transcribe,
)


@dataclass
class EngineConfig:
    whisper_model: str = DEFAULT_WHISPER_MODEL
    translator: Optional[object] = None  # duck-types as translators.base.Translator
    translate_langs: set[str] = field(default_factory=set)
    target_lang: str = "en"
    enable_summary: bool = False
    diarize: bool = False


class TranscriptionEngine:
    def __init__(self, config: EngineConfig, listener: EngineListener):
        self._config = config
        self._listener = listener

        self._speaker_tracker: Optional[SpeakerTracker] = None
        self._vad_model = None
        self._summarizer: Optional[SummarizerProcess] = None

        self._gpu_lock = threading.Lock()

        self._transcription_pool: Optional[ThreadPoolExecutor] = None
        self._translation_pool: Optional[ThreadPoolExecutor] = None

        self._audio_queue: deque = deque()
        self._vad_lock = threading.Lock()
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_start_time = 0.0

        self._pending_count = 0
        self._pending_lock = threading.Lock()

        self._duplicate_filter = DuplicateFilter(maxlen=5)
        # Protected by the single transcription-worker invariant (transcription_pool has max_workers=1).
        self._detected_lang: Optional[str] = None
        self._hp_sos = make_highpass_sos()

        # Transcript state — read by get_transcript() and _retranslate_recent.
        self._transcript: list[dict] = []
        self._recent_context: deque = deque(maxlen=10)
        self._transcript_lock = threading.Lock()
        self._last_speaker: Optional[str] = None

        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    # Lifecycle --------------------------------------------------------------

    def start(self) -> None:
        self._listener.on_status(StatusEvent("starting"))

        self._speaker_tracker = SpeakerTracker(enabled=self._config.diarize)
        self._vad_model = load_vad_model()

        # Inject the GPU lock into any translator that needs it (Qwen).
        if self._config.translator is not None:
            _inject_gpu_lock(self._config.translator, self._gpu_lock)

        # Thread pools: Qwen serializes on GPU so 1 translation worker is enough.
        self._transcription_pool = ThreadPoolExecutor(max_workers=1)
        translation_workers = 1 if isinstance(self._config.translator, QwenTranslator) else 4
        self._translation_pool = (
            ThreadPoolExecutor(max_workers=translation_workers)
            if self._config.translator is not None
            else None
        )

        # Summarizer runs in a separate process.
        if self._config.enable_summary:
            self._summarizer = SummarizerProcess(
                target_lang=self._config.target_lang,
                on_summary=lambda item: self._listener.on_summary(
                    SummaryEvent(
                        index=item["index"],
                        timestamp=item["timestamp"],
                        text=item["text"],
                        is_final=item["is_final"],
                    )
                ),
            )
            self._summarizer.start()

        # Warm up Whisper.
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        transcribe(
            dummy,
            model_repo=self._config.whisper_model,
            initial_prompt=None,
            gpu_lock=self._gpu_lock,
        )

        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_audio, daemon=True
        )
        self._process_thread.start()

        self._listener.on_status(StatusEvent("ready"))

    def push_audio(self, chunk: np.ndarray) -> None:
        """Feed a mono float32 chunk @ 16 kHz. Safe to call from any thread."""
        with self._vad_lock:
            self._audio_queue.append(chunk)

    def stop(self) -> None:
        self._listener.on_status(StatusEvent("stopping"))
        self._running = False
        if self._process_thread is not None:
            self._process_thread.join(timeout=5)
        if self._transcription_pool is not None:
            self._transcription_pool.shutdown(wait=True, cancel_futures=False)
        if self._translation_pool is not None:
            self._translation_pool.shutdown(wait=True, cancel_futures=False)
        if self._summarizer is not None:
            self._summarizer.stop()
        self._listener.on_status(StatusEvent("stopped"))

    def get_transcript(self) -> list[SegmentEvent]:
        """Snapshot of all emitted SegmentEvents. Safe to call after stop()."""
        with self._transcript_lock:
            return [
                SegmentEvent(
                    id=e["id"],
                    timestamp=e["time"],
                    speaker=e["speaker"],
                    text=e["text"],
                    language=e["language"],
                )
                for e in self._transcript
            ]

    # Threading helpers ------------------------------------------------------

    def _adaptive_thresholds(self):
        p = self._pending_count
        silence = min(SILENCE_AFTER_SPEECH + 0.5 * p, 2.0)
        max_speech = min(MAX_SPEECH_DURATION + 3.0 * p, 15.0)
        return silence, max_speech

    def _submit_transcription(self, audio_data):
        with self._pending_lock:
            if self._pending_count >= MAX_PENDING_SEGMENTS:
                self._listener.on_status(StatusEvent(
                    state="warning",
                    message=f"Audio dropped: transcription backlog ({self._pending_count} pending)",
                ))
                return
            self._pending_count += 1
        fut = self._transcription_pool.submit(self._transcribe_segment, audio_data)
        fut.add_done_callback(lambda _f: self._dec_pending())

    def _dec_pending(self):
        with self._pending_lock:
            self._pending_count = max(0, self._pending_count - 1)

    def _flush_speech_buffer(self):
        if not self._speech_buffer:
            self._is_speaking = False
            self._silence_start_time = 0.0
            return

        audio_data = np.concatenate(self._speech_buffer).astype(np.float32)
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_start_time = 0.0
        self._vad_model.reset_states()

        duration = len(audio_data) / SAMPLE_RATE
        if duration < MIN_SPEECH_DURATION:
            return

        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < ENERGY_THRESHOLD:
            return

        audio_data = preprocess_audio(audio_data, self._hp_sos)
        self._submit_transcription(audio_data)

    # VAD loop (ported from LiveTranscriber.process_audio) -------------------

    def _process_audio(self):
        try:
            while self._running:
                with self._vad_lock:
                    if not self._audio_queue:
                        time.sleep(0.01)
                        continue
                    chunks = list(self._audio_queue)
                    self._audio_queue.clear()

                raw_audio = np.concatenate(chunks)
                offset = 0
                while offset + VAD_FRAME_SAMPLES <= len(raw_audio):
                    frame = raw_audio[offset:offset + VAD_FRAME_SAMPLES]
                    offset += VAD_FRAME_SAMPLES

                    frame_tensor = torch.from_numpy(frame).float()
                    speech_prob = self._vad_model(frame_tensor, SAMPLE_RATE).item()
                    now = time.monotonic()
                    silence_cap, max_speech_cap = self._adaptive_thresholds()

                    if speech_prob >= VAD_THRESHOLD:
                        if not self._is_speaking:
                            self._is_speaking = True
                        self._silence_start_time = 0.0
                        self._speech_buffer.append(frame)
                        speech_samples = sum(len(f) for f in self._speech_buffer)
                        speech_duration = speech_samples / SAMPLE_RATE
                        if speech_duration >= max_speech_cap:
                            self._flush_speech_buffer()
                    else:
                        if self._is_speaking:
                            self._speech_buffer.append(frame)
                            if self._silence_start_time == 0.0:
                                self._silence_start_time = now
                            elif now - self._silence_start_time >= silence_cap:
                                self._flush_speech_buffer()

                remainder = len(raw_audio) - offset
                if remainder > 0:
                    with self._vad_lock:
                        self._audio_queue.appendleft(raw_audio[offset:])
        finally:
            if self._speech_buffer:
                self._flush_speech_buffer()

    # Transcription body (ported from _transcribe_segment) -------------------

    def _transcribe_segment(self, audio_data):
        try:
            initial_prompt = INITIAL_PROMPTS.get(self._detected_lang)

            result = transcribe(
                audio_data,
                model_repo=self._config.whisper_model,
                initial_prompt=initial_prompt,
                gpu_lock=self._gpu_lock,
            )

            lang = result.get("language", "??")
            if lang not in SUPPORTED_LANGUAGES:
                return
            self._detected_lang = lang

            groups = []
            for segment in result["segments"]:
                text = segment["text"].strip()
                if not text or len(text) < 2:
                    continue
                avg_logprob = segment.get("avg_logprob", 0)
                no_speech_prob = segment.get("no_speech_prob", 0)
                if avg_logprob < -1.0 or no_speech_prob > 0.6:
                    continue
                if is_hallucination(text):
                    continue

                start_sample = int(segment["start"] * SAMPLE_RATE)
                end_sample = min(int(segment["end"] * SAMPLE_RATE), len(audio_data))
                segment_audio = audio_data[start_sample:end_sample]
                speaker = self._speaker_tracker.identify_speaker(segment_audio)

                if groups and groups[-1][0] == speaker:
                    groups[-1][1].append(text)
                else:
                    groups.append((speaker, [text]))

            for speaker, texts in groups:
                full_text = " ".join(texts)

                if self._duplicate_filter.is_duplicate(full_text):
                    continue
                self._duplicate_filter.remember(full_text)

                timestamp = datetime.now().strftime("%H:%M:%S")
                entry_id = uuid.uuid4().hex

                entry = {
                    "id": entry_id,
                    "time": timestamp,
                    "speaker": speaker,
                    "text": full_text,
                    "language": lang,
                    "translation": None,
                }
                with self._transcript_lock:
                    self._transcript.append(entry)
                self._last_speaker = speaker

                self._listener.on_segment(SegmentEvent(
                    id=entry_id,
                    timestamp=timestamp,
                    speaker=speaker,
                    text=full_text,
                    language=lang,
                ))

                if self._summarizer is not None:
                    self._summarizer.add_line(speaker, full_text, lang)

                if self._config.translator is not None and lang in self._config.translate_langs:
                    context = list(self._recent_context) or None

                    if isinstance(self._config.translator, QwenTranslator):
                        full_translation = self._config.translator.translate(
                            full_text, lang, context=context
                        )
                    else:
                        chunks = chunk_for_translation(full_text)
                        futures = []
                        for i, chunk_text in enumerate(chunks):
                            chunk_ctx = (
                                context if i == 0
                                else (context or []) + [(c, None) for c in chunks[:i]]
                            )
                            futures.append(self._translation_pool.submit(
                                self._config.translator.translate,
                                chunk_text, lang, context=chunk_ctx,
                            ))
                        all_translations = []
                        for future in futures:
                            try:
                                t = future.result(timeout=10.0)
                                if t:
                                    all_translations.append(t)
                            except Exception:
                                pass
                        full_translation = " ".join(all_translations) if all_translations else None

                    entry["translation"] = full_translation
                    self._listener.on_translation(TranslationEvent(
                        segment_id=entry_id,
                        text=full_translation or "",
                        is_update=False,
                    ))
                    self._recent_context.append((full_text, full_translation))

                    if isinstance(self._config.translator, QwenTranslator) and self._translation_pool is not None:
                        self._translation_pool.submit(self._retranslate_recent, lang)
                else:
                    # Translator configured but this language isn't in translate_langs —
                    # emit an empty TranslationEvent so the display renders the segment
                    # without waiting for a translation that will never arrive.
                    if self._config.translator is not None:
                        self._listener.on_translation(TranslationEvent(
                            segment_id=entry_id,
                            text="",
                            is_update=False,
                        ))
                    self._recent_context.append((full_text, None))
        except Exception as e:
            self._listener.on_status(StatusEvent(
                state="error", message=f"Transcription failed: {e}"
            ))

    def _retranslate_recent(self, source_lang):
        with self._transcript_lock:
            candidates = []
            for i, entry in enumerate(self._transcript):
                if entry.get("translation") and entry["language"] == source_lang:
                    candidates.append((i, entry))
            candidates = candidates[-4:-1]
        if not candidates:
            return

        context = list(self._recent_context)
        for _idx, entry in candidates:
            try:
                new_translation = self._config.translator.translate(
                    entry["text"], entry["language"], context=context
                )
            except Exception:
                continue

            if new_translation and new_translation != entry["translation"]:
                old_translation = entry["translation"]
                entry["translation"] = new_translation

                # Safe without locking: transcription_pool (1 worker) and translation_pool (1 worker for Qwen) serialize writers.
                for j, (orig, trans) in enumerate(self._recent_context):
                    if orig == entry["text"] and trans == old_translation:
                        self._recent_context[j] = (orig, new_translation)
                        break

                self._listener.on_translation(TranslationEvent(
                    segment_id=entry["id"],
                    text=new_translation,
                    is_update=True,
                ))
