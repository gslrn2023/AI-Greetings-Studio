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
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
CREATOMATE_API_KEY = os.environ.get("CREATOMATE_API_KEY", "")
APP_PASSCODE = os.environ.get("APP_PASSCODE", "")

# =====================================================================
# 2. MASTER MODULAR SWITCHBOARD
# =====================================================================
# IMAGE_ENGINE: "FLUX" (black-forest-labs/FLUX.1-schnell) | "Z_IMAGE" (Tongyi-MAI/Z-Image-Turbo)
# Only the selected engine is loaded into VRAM to protect free GPU memory.
IMAGE_ENGINE = os.environ.get("IMAGE_ENGINE", "FLUX").upper()

# RENDER_ENGINE: "LOCAL" (Free Python FFmpeg+Pillow) | "CREATOMATE" (Cloud API) | "OFF" (URL Test Mode)
RENDER_ENGINE = os.environ.get("RENDER_ENGINE", "LOCAL").upper()

MAX_DAILY_GREETINGS = int(os.environ.get("MAX_DAILY_GREETINGS", 3))
GEMINI_MODEL = "gemini-2.5-flash"
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# =====================================================================
# 3. RESOLUTION, TIMING & EXPERT AUDIO MIXING
# =====================================================================
IMAGE_WIDTH = 1088
IMAGE_HEIGHT = 1920
FLUX_STEPS = 8
Z_IMAGE_STEPS = 9  # 9 steps = 8 DiT forwards for Z-Image-Turbo

VIDEO_WIDTH = 1088
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

# Unobstructed image showcase time before text & voiceover appear
TEXT_REVEAL_DELAY = 4.0        # First 4 seconds show 100% pure artwork without text overlay
AUDIO_PADDING_SECONDS = 2.0    # Tail buffer after voiceover completes
DEFAULT_NO_TTS_DURATION = 9.0  # Total duration when voiceover is muted (4s pure art + 5s text)

# Audio Mixing
BGM_VOLUME_WITH_TTS = "8%"     # 0.08 amplitude in FFmpeg
BGM_VOLUME_NO_TTS = "75%"      # 0.75 amplitude in FFmpeg
TTS_VOLUME_BOOST = "150%"      # 1.50 amplitude in FFmpeg

# =====================================================================
# 4. VALIDATED ELEVENLABS FREE-TIER PREMADE VOICE CATALOG
# =====================================================================
VOICE_CATALOG = {
    "Matilda (Warm Female)": {"id": "XrExE9yKIg1WjnnlVkGX", "archetype": "warm_female"},
    "Sarah (Soft Expressive Female)": {"id": "EXAVITQu4vr4xnSDxMaL", "archetype": "soft_empathetic"},
    "Rachel (Calm Empathetic Female)": {"id": "21m00Tcm4TlvDq8ikWAM", "archetype": "soft_empathetic"},
    "Alice (Clear British Female)": {"id": "Xb7hH8MSUJpSbSDYk0k2", "archetype": "warm_female"},
    "Liam (Energetic Upbeat Male)": {"id": "TX3LPaxmHKxFdv7VOQHJ", "archetype": "energetic_male"},
    "Brian (Deep Resonant Male)": {"id": "nPczCjzI2devNBz1zQrb", "archetype": "deep_celebratory"},
    "Adam (Classic Cinematic Male)": {"id": "pNInz6obpgDQGcFmaJgB", "archetype": "deep_celebratory"},
    "George (Warm Storyteller Male)": {"id": "JBFqnCBsd6RMkjVDRZzb", "archetype": "warm_male"}
}

ARCHETYPE_TO_VOICE_ID = {
    "warm_female": "XrExE9yKIg1WjnnlVkGX",
    "soft_empathetic": "EXAVITQu4vr4xnSDxMaL",
    "energetic_male": "TX3LPaxmHKxFdv7VOQHJ",
    "deep_celebratory": "nPczCjzI2devNBz1zQrb",
    "warm_male": "JBFqnCBsd6RMkjVDRZzb"
}

def resolve_voice_id(voice_choice: str, ai_archetype: str) -> str:
    if voice_choice and voice_choice in VOICE_CATALOG:
        return VOICE_CATALOG[voice_choice]["id"]
    return ARCHETYPE_TO_VOICE_ID.get(ai_archetype, "XrExE9yKIg1WjnnlVkGX")

# =====================================================================
# 5. EXPANDABLE BGM LIBRARY (Add new filenames/URLs to any category list)
# =====================================================================
BGM_BASE_URL = "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/"

BGM_LIBRARY = {
    "festive": [
        f"{BGM_BASE_URL}festive.mp3",
        # Add more later, e.g.: f"{BGM_BASE_URL}festive_2.mp3",
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
# 6. CREATIVE STYLE LENSES FOR IMAGE DIVERSITY
# =====================================================================
CREATIVE_STYLE_LENSES = [
    "35mm Analog Editorial Photography (Kodak Portra 400 grain, golden-hour rim light, shallow depth of field, rich environmental storytelling)",
    "Cinematic National Geographic Expedition Photography (dramatic landscape vista, warm lantern or fire glow contrasting against twilight sky)",
    "Luminous Gouache & Botanical Illustration (hand-painted scenery, layered foreground flora, cold-press paper texture, warm metallic highlights)",
    "Atmospheric Magical Realism (bioluminescent floating embers and fireflies, volumetric god-rays, dreamy cinematic color grading)",
    "Luxury Editorial Still-Life & Scenic Architecture (artisanal crafted objects set inside a breathtaking scenic veranda or mountain lodge)",
    "Masterpiece Oil Impasto Painting (expressive textured brushwork, dramatic warm chiaroscuro lighting, museum-grade color harmony)"
]

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