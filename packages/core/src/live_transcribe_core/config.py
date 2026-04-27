"""Engine configuration constants shared across core modules.

These values are used by the VAD loop, Whisper transcription, speaker tracker,
and hallucination filters. Keep them here so they can be tuned in one place.
"""

# Sample rate and model repos
SAMPLE_RATE = 16000
WHISPER_MODEL = "mlx-community/whisper-medium-mlx-q4"
WHISPER_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"
WHISPER_MODEL_FULL = "mlx-community/whisper-large-v3-mlx-4bit"
DEFAULT_WHISPER_MODEL = WHISPER_MODEL_FULL

# Speaker tracking
SPEAKER_SIMILARITY = 0.72
NUM_SPEAKERS = 2
MAX_SPEAKERS = 3
MIN_CHUNKS_NEW_SPEAKER = 4

# Supported languages
SUPPORTED_LANGUAGES = ["ko", "en", "es"]
LANG_NAMES = {"ko": "Korean", "en": "English", "es": "Spanish"}

# VAD thresholds
VAD_THRESHOLD = 0.3
MIN_SPEECH_DURATION = 0.3
MAX_SPEECH_DURATION = 5.0
SILENCE_AFTER_SPEECH = 0.5
VAD_FRAME_SAMPLES = 512
ENERGY_THRESHOLD = 0.002
SPEECH_PAD_SAMPLES = int(SAMPLE_RATE * 0.15)

# Whisper prompts per language
INITIAL_PROMPTS = {
    "ko": "안녕하세요. 네, 알겠습니다. 그래서 이제 어떻게 할까요? 아, 그렇구나. 잠깐만요, 다시 한번 말씀해 주세요. 좋습니다, 진행하겠습니다.",
    "en": "Hello. Yes, I understand. Thank you.",
    "es": "Hola. Sí, entiendo. Gracias.",
}

# Known Whisper hallucination phrases (exact set carried over from live_transcribe.py)
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thanks for listening",
    "subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "thank you for watching",
    "please subscribe", "the end", "you",
    "시청해 주셔서 감사합니다", "구독", "좋아요",
    "감사합니다", "고마워요",
    "다음 시간에 만나요", "구독과 좋아요",
    "좋아요와 구독", "채널에 가입", "알림 설정",
    "영상 시청해 주셔서 감사합니다",
    "오늘도 시청해 주셔서 감사합니다",
    "끝까지 시청해 주셔서 감사합니다",
    "SBS 뉴스", "YTN 뉴스", "JTBC 뉴스", "채널A 뉴스",
    "gracias por ver", "suscríbete",
    "MBC 뉴스", "KBS 뉴스",
}

# Backpressure: maximum number of audio segments queued for transcription
# before _submit_transcription drops new segments and emits a warning.
MAX_PENDING_SEGMENTS = 4
