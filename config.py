import os
import json
import random
import threading
from datetime import datetime, timezone, timedelta

# =====================================================================
# 1. ENVIRONMENT SECRETS
# =====================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
CREATOMATE_API_KEY = os.environ.get("CREATOMATE_API_KEY", "")
APP_PASSCODE = os.environ.get("APP_PASSCODE", "")

# =====================================================================
# 2. MASTER MODULAR SWITCHBOARD
# =====================================================================
# IMAGE_ENGINE: "FLUX" (black-forest-labs/FLUX.1-schnell) | "Z_IMAGE" (Tongyi-MAI/Z-Image-Turbo)
IMAGE_ENGINE = os.environ.get("IMAGE_ENGINE", "FLUX").upper()

# RENDER_ENGINE: "LOCAL" (Free Python FFmpeg+Pillow) | "CREATOMATE" (Cloud API) | "OFF" (URL Test Mode)
RENDER_ENGINE = os.environ.get("RENDER_ENGINE", "LOCAL").upper()

MAX_DAILY_GREETINGS = int(os.environ.get("MAX_DAILY_GREETINGS", 3))
GEMINI_MODEL = "gemini-2.5-flash"
SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "bulbul:v3")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# =====================================================================
# 3. RESOLUTION, TIMING & EXPERT AUDIO MIXING
# =====================================================================
IMAGE_WIDTH = 1088
IMAGE_HEIGHT = 1920
FLUX_STEPS = 8
Z_IMAGE_STEPS = 9

VIDEO_WIDTH = 1088
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

TEXT_REVEAL_DELAY = 4.0        # First 4 seconds show 100% pure artwork without text overlay
AUDIO_PADDING_SECONDS = 2.0    # Tail buffer after voiceover completes
DEFAULT_NO_TTS_DURATION = 9.0  # Total duration when voiceover is muted

# Audio Mixing (Used by both Local FFmpeg and Creatomate)
BGM_INTRO_VOLUME = 0.28        # 28% BGM volume during the first 4.0s visual intro
BGM_DUCKED_VOLUME = 0.14       # 14% BGM volume once voiceover starts speaking
BGM_NO_TTS_VOLUME = 0.75       # 75% BGM volume when voiceover is muted
TTS_VOLUME_MULTIPLIER = 1.85   # 185% Voiceover volume boost (paired with broadcast limiter)

BGM_VOLUME_WITH_TTS = "14%"    # For Creatomate mode
BGM_VOLUME_NO_TTS = "75%"      # For Creatomate mode
TTS_VOLUME_BOOST = "165%"      # For Creatomate mode

# =====================================================================
# 4. 3-TIER VOICE WATERFALL (SARVAM -> AZURE NEURAL -> ELEVENLABS)
# =====================================================================
# Tier 1 (Indian Primary): Sarvam AI bulbul:v3 speakers (must be lowercase)
SARVAM_VOICES = {
    "female": ["ritu", "priya", "shruti"],
    "male": ["shubh", "aditya", "rahul"]
}

# Tier 2 (Fallback): Microsoft Azure Neural TTS (edge-tts)
AZURE_NEURAL_VOICES = {
    "female": "en-IN-NeerjaExpressiveNeural",
    "male": "en-IN-PrabhatNeural"
}

# Tier 3 (Global Primary & Final Fallback): ElevenLabs Free-Tier Premade Voices
ELEVENLABS_VOICES = {
    "female": "XrExE9yKIg1WjnnlVkGX",  # Matilda
    "male": "bIHbv24MWmeRgasZH58o"     # Will
}

def resolve_voice_plan(
    voice_option: str,
    ai_archetype: str,
    cultural_context: str = "indian"
) -> list[dict]:
    """
    Builds the ordered 3-tier TTS waterfall plan:
    - Indian context: 1. Sarvam AI -> 2. Azure Neural (Neerja/Prabhat) -> 3. ElevenLabs (Matilda/Will)
    - Global context: 1. ElevenLabs (Matilda/Will) -> 2. Azure Neural (Neerja/Prabhat) -> 3. Sarvam AI
    """
    choice = (voice_option or "Auto - AI Choice").strip().lower()
    culture = (cultural_context or "indian").strip().lower()
    archetype = (ai_archetype or "warm_female").strip().lower()

    if choice == "female":
        gender = "female"
    elif choice == "male":
        gender = "male"
    else:
        is_male_arch = any(k in archetype for k in ("male", "deep_celebratory"))
        gender = "male" if is_male_arch else "female"

    sarvam_step = {"provider": "sarvam", "voice": random.choice(SARVAM_VOICES[gender])}
    azure_step = {"provider": "azure", "voice": AZURE_NEURAL_VOICES[gender]}
    eleven_step = {"provider": "elevenlabs", "voice": ELEVENLABS_VOICES[gender]}

    if culture == "global":
        return [eleven_step, azure_step, sarvam_step]
    return [sarvam_step, azure_step, eleven_step]

# =====================================================================
# 5. EXPANDABLE BGM LIBRARY
# =====================================================================
BGM_BASE_URL = "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/"

BGM_LIBRARY = {
    "festive": [
        f"{BGM_BASE_URL}festive.mp3",
    ],
    "calm": [
        f"{BGM_BASE_URL}calm.mp3",
    ],
    "energetic": [
        f"{BGM_BASE_URL}energetic.mp3",
    ],
    "elegant": [
        f"{BGM_BASE_URL}elegant.mp3",
    ],
    "sad": [
        f"{BGM_BASE_URL}sad.mp3",
    ],
    "spiritual": [
        f"{BGM_BASE_URL}calm.mp3",
    ],
    "romantic": [
        f"{BGM_BASE_URL}elegant.mp3",
    ]
}

def pick_bgm_url(mood: str) -> str:
    clean_mood = (mood or "festive").lower().strip()
    track_list = BGM_LIBRARY.get(clean_mood, BGM_LIBRARY["festive"])
    return random.choice(track_list)

# =====================================================================
# 6. MOBILE-FRIENDLY VISUAL ART STYLE MAP
# =====================================================================
STYLE_MAP = {
    "35mm Analog": "35mm Analog Editorial Photography (Kodak Portra 400 grain, golden-hour rim light, shallow depth of field, rich environmental storytelling)",
    "Cinematic": "Cinematic National Geographic Expedition Photography (dramatic landscape vista, warm lantern or fire glow contrasting against twilight sky)",
    "Luminous Gouache": "Luminous Gouache & Botanical Illustration (hand-painted scenery, layered foreground flora, cold-press paper texture, warm metallic highlights)",
    "Magical Realism": "Atmospheric Magical Realism (bioluminescent floating embers and fireflies, volumetric god-rays, dreamy cinematic color grading)",
    "Luxury Still-Life": "Luxury Editorial Still-Life & Scenic Architecture (artisanal crafted objects set inside a breathtaking scenic veranda or mountain lodge)",
    "Oil Impasto": "Masterpiece Oil Impasto Painting (expressive textured brushwork, dramatic warm chiaroscuro lighting, museum-grade color harmony)"
}

CREATIVE_STYLE_LENSES = list(STYLE_MAP.values())

# =====================================================================
# 7. DAILY QUOTA TRACKER (IST)
# =====================================================================
QUOTA_FILE = "daily_usage.json"
_quota_lock = threading.Lock()
IST = timezone(timedelta(hours=5, minutes=30))

def get_quota_status() -> tuple[int, int]:
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _quota_lock:
        if not os.path.exists(QUOTA_FILE):
            return 0, MAX_DAILY_GREETINGS
        try:
            with open(QUOTA_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") != today:
                return 0, MAX_DAILY_GREETINGS
            used = int(data.get("count", 0))
            return used, max(0, MAX_DAILY_GREETINGS - used)
        except Exception:
            return 0, MAX_DAILY_GREETINGS

def increment_quota() -> int:
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _quota_lock:
        used = 0
        if os.path.exists(QUOTA_FILE):
            try:
                with open(QUOTA_FILE, "r") as f:
                    data = json.load(f)
                if data.get("date") == today:
                    used = int(data.get("count", 0))
            except Exception:
                used = 0
        used += 1
        with open(QUOTA_FILE, "w") as f:
            json.dump({"date": today, "count": used}, f)
        return used