import numpy as np

SAMPLE_RATE = 16000  # Will be sourced from core.config in Task 3

# Try to import resemblyzer for speaker diarization
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("[WARN] resemblyzer not available - speaker separation disabled")

SPEAKER_SIMILARITY = 0.72    # Cosine similarity threshold for same speaker (lower = more lenient matching)
NUM_SPEAKERS = 2             # Expected number of speakers (once reached, assigns to closest match)
MAX_SPEAKERS = 3             # Maximum number of speakers to track
MIN_CHUNKS_NEW_SPEAKER = 4   # Require N consecutive unmatched chunks before creating a new speaker


class SpeakerTracker:
    """Track and identify speakers using voice embeddings."""

    def __init__(self, enabled=True):
        self.enabled = enabled and DIARIZATION_AVAILABLE
        if self.enabled:
            print("Loading speaker encoder model...")
            self.encoder = VoiceEncoder()
            print("Speaker encoder ready.")
        else:
            self.encoder = None
        self.speaker_embeddings = []  # List of (label, embedding) tuples
        self.speaker_count = 0
        self.unmatched_streak = 0      # Consecutive chunks that didn't match any speaker
        self.pending_embedding = None  # Embedding accumulator for potential new speaker

    def identify_speaker(self, audio_chunk):
        """Identify or register a speaker from an audio chunk."""
        if not self.enabled or self.encoder is None:
            return "Speaker"

        if len(audio_chunk) < SAMPLE_RATE * 0.5:  # Need at least 0.5s
            return self._last_speaker_label()

        try:
            # Preprocess and get embedding
            processed = preprocess_wav(audio_chunk, source_sr=SAMPLE_RATE)
            if len(processed) < SAMPLE_RATE * 0.3:
                return self._last_speaker_label()

            embedding = self.encoder.embed_utterance(processed)

            # Compare against known speakers
            best_match = None
            best_similarity = -1

            for label, known_emb in self.speaker_embeddings:
                similarity = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = label

            # If good match found, update the embedding (rolling average)
            if best_match and best_similarity >= SPEAKER_SIMILARITY:
                for i, (label, known_emb) in enumerate(self.speaker_embeddings):
                    if label == best_match:
                        # Exponential moving average of embeddings
                        self.speaker_embeddings[i] = (
                            label,
                            0.8 * known_emb + 0.2 * embedding,
                        )
                        break
                self.unmatched_streak = 0
                self.pending_embedding = None
                return best_match

            # No good match — decide whether to create a new speaker
            # If we've already reached the expected number of speakers, assign to closest
            if self.speaker_count >= NUM_SPEAKERS:
                return best_match if best_match else "Speaker ?"

            # Require multiple consecutive unmatched chunks before creating a new speaker
            self.unmatched_streak += 1
            if self.pending_embedding is None:
                self.pending_embedding = embedding
            else:
                self.pending_embedding = 0.5 * self.pending_embedding + 0.5 * embedding

            if self.unmatched_streak >= MIN_CHUNKS_NEW_SPEAKER:
                if self.speaker_count < MAX_SPEAKERS:
                    self.speaker_count += 1
                    label = f"Speaker {self.speaker_count}"
                    self.speaker_embeddings.append((label, self.pending_embedding))
                    self.unmatched_streak = 0
                    self.pending_embedding = None
                    return label

            # Not enough evidence yet for a new speaker — assign to closest existing
            if best_match:
                return best_match
            return self._last_speaker_label()

        except Exception as e:
            return self._last_speaker_label()

    def _last_speaker_label(self):
        if self.speaker_embeddings:
            return self.speaker_embeddings[-1][0]
        return "Speaker"
