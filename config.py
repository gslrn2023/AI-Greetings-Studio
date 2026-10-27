import os
import json
import threading
from datetime import datetime, timezone, timedelta

# =====================================================================
# 1. ENVIRONMENT SECRETS (Configure these in Hugging Face Space Secrets)
# =====================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
CREATOMATE_API_KEY = os.environ.get("CREATOMATE_API_KEY", "")
APP_PASSCODE = os.environ.get("APP_PASSCODE", "admin123")

# =====================================================================
# 2. GLOBAL TUNING & QUOTA VARIABLES
# =====================================================================
MAX_DAILY_GREETINGS = int(os.environ.get("MAX_DAILY_GREETINGS", 3))
GEMINI_MODEL = "gemini-2.5-flash"

# ElevenLabs Model:
# "eleven_flash_v2_5" uses 0.5 credits/char (doubles Free Tier quota)
# "eleven_multilingual_v2" uses 1.0 credit/char (richer emotional nuance)
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# Flux Image Resolution (Optimal 9:16 1.2MP sweet spot for FLUX.1-schnell sharpness)
IMAGE_WIDTH = 832
IMAGE_HEIGHT = 1472
FLUX_STEPS = 6

# Creatomate Render Output Resolution (9:16 Full HD Vertical)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
DEFAULT_NO_TTS_DURATION = 7.0  # Seconds when voiceover is muted (solemn/bereavement)
AUDIO_PADDING_SECONDS = 2.2    # Tail buffer after voiceover finishes

# Audio Mixing Controls
BGM_VOLUME_WITH_TTS = "8%"     # Subtle ambient bed when voiceover is speaking
BGM_VOLUME_NO_TTS = "75%"      # Full emotional presence when voiceover is omitted
TTS_VOLUME_BOOST = "150%"      # Amplifies ElevenLabs voiceover for crystal clarity

# =====================================================================
# 3. ELEVENLABS FREE-TIER VOICE PALETTE (100% Free-Tier API Compatible)
# =====================================================================
VOICE_MAP = {
    "warm_female": "XrExE9yKIg1WjnnlVkGX",      # Matilda - Warm, friendly, articulate
    "energetic_male": "TX3LPaxmHKxFdv7VOQHJ",   # Liam - Energetic, upbeat, celebratory
    "deep_celebratory": "pNInz6obpgDQGcFmaJgB", # Adam - Deep, cinematic, grand occasions
    "soft_empathetic": "21m00Tcm4TlvDq8ikWAM"   # Rachel - Calm, gentle, heartfelt
}

# =====================================================================
# 4. BACKGROUND MUSIC MAPPING (GitHub Raw CDN)
# =====================================================================
BGM_MAP = {
    "calm": "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/calm.mp3",
    "elegant": "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/elegant.mp3",
    "energetic": "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/energetic.mp3",
    "festive": "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/festive.mp3",
    "sad": "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/sad.mp3"
}

# =====================================================================
# 5. ANTI-REPETITION CREATIVE STYLE LENSES
# =====================================================================
CREATIVE_STYLE_LENSES = [
    "35mm Analog Editorial Photography (Kodak Portra 400 film, golden-hour rim light, shallow depth of field, rich environmental storytelling, natural lens flare)",
    "Cinematic National Geographic Expedition Photography (wide-angle dramatic landscape vista, warm campfire or lantern glow contrasting with deep twilight sky, razor-sharp foreground textures)",
    "Luminous Gouache & Botanical Illustration (richly detailed hand-painted scenery, layered foreground flora, textured cold-press paper grain, warm metallic gold-leaf highlights)",
    "Atmospheric Magical Realism (bioluminescent warm floating embers and fireflies, rich architectural or natural depth, volumetric god-rays, dreamy cinematic color grading)",
    "Luxury Editorial Still-Life & Environment (artisanal crafted objects set inside a breathtaking scenic veranda or mountain lodge, warm candlelit radiance, crisp optical focus)",
    "Vibrant Contemporary Stained-Glass & Light Art (refraction of warm sunbeams through intricate colored glass, rich environmental depth, glowing atmospheric dust motes)",
    "Masterpiece Oil Impasto Painting (expressive textured brushwork, dramatic warm chiaroscuro lighting, rich multi-layered background scenery, museum-grade color harmony)"
]

# =====================================================================
# 6. THREAD-SAFE DAILY RATE LIMITER (IST Timezone Aligned)
# =====================================================================
QUOTA_FILE = "daily_usage.json"
_quota_lock = threading.Lock()
IST = timezone(timedelta(hours=5, minutes=30))

def get_today_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")

def get_quota_status() -> tuple[int, int]:
    """Returns (used_today, remaining_today)."""
    today = get_today_str()
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
    """Increments today's usage count and returns the new used count."""
    today = get_today_str()
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