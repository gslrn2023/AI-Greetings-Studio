import os
import json
import random
import threading
import requests
from datetime import datetime, timezone, timedelta

# Anchor root directory so relative files work from any launch path (Kaggle, Azure, Docker, CLI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional .env loader for local / Azure / Docker deployments
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

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
IMAGE_ENGINE = os.environ.get("IMAGE_ENGINE", "FLUX").upper()
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

TEXT_REVEAL_DELAY = 4.0
AUDIO_PADDING_SECONDS = 2.0
DEFAULT_NO_TTS_DURATION = 9.0

BGM_INTRO_VOLUME = 0.28
BGM_DUCKED_VOLUME = 0.14
BGM_NO_TTS_VOLUME = 0.75
TTS_VOLUME_MULTIPLIER = 1.85

BGM_VOLUME_WITH_TTS = "14%"
BGM_VOLUME_NO_TTS = "75%"
TTS_VOLUME_BOOST = "165%"

# =====================================================================
# 4. MULTILINGUAL VOICE WATERFALL (SARVAM V4 -> SARVAM V3 -> AZURE -> ELEVENLABS)
# =====================================================================
# bulbul:v4-flash curated Indian English personas (24kHz)
SARVAM_V4_EN_VOICES = {
    "female": ["ritu_en_reels", "simran_en_narration"],
    "male": ["sunny_en_social", "shubh_en_sports"]
}

# bulbul:v4-flash curated Regional Indian personas (Tier 1 for regional scripts, 24kHz)
SARVAM_V4_REGIONAL_VOICES = {
    "female": ["tanya_hi_narration"],
    "male": ["tarun_te_narration"]
}

# bulbul:v3 regional speakers (Tier 2 fallback for North/East/West Indic languages, 48kHz)
SARVAM_VOICES_GENERAL = {
    "female": ["ritu", "priya", "shruti"],
    "male": ["shubh", "aditya", "rahul"]
}

# bulbul:v3 speakers tuned for South Indian / Dravidian cadences (ta-IN, kn-IN, te-IN, ml-IN, 48kHz)
SARVAM_VOICES_SOUTH = {
    "female": ["kavitha", "shruti"],
    "male": ["mani", "gokul", "vijay"]
}

SOUTH_LANG_CODES = {"ta-IN", "kn-IN", "te-IN", "ml-IN"}

# Regional Azure Neural TTS Voices (Cloud Fallback for all 11 Indian languages + en-US)
AZURE_REGIONAL_VOICES = {
    "en-IN": {"female": "en-IN-NeerjaExpressiveNeural", "male": "en-IN-PrabhatNeural"},
    "en-US": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    "hi-IN": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "ta-IN": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
    "te-IN": {"female": "te-IN-ShrutiNeural", "male": "te-IN-MohanNeural"},
    "kn-IN": {"female": "kn-IN-SapnaNeural", "male": "kn-IN-GaganNeural"},
    "ml-IN": {"female": "ml-IN-SobhanaNeural", "male": "ml-IN-MidhunNeural"},
    "bn-IN": {"female": "bn-IN-TanishaaNeural", "male": "bn-IN-BashkarNeural"},
    "mr-IN": {"female": "mr-IN-AarohiNeural", "male": "mr-IN-ManoharNeural"},
    "gu-IN": {"female": "gu-IN-DhwaniNeural", "male": "gu-IN-NiranjanNeural"},
    "pa-IN": {"female": "pa-IN-GurpreetNeural", "male": "pa-IN-OjasNeural"},
    "od-IN": {"female": "or-IN-SubhasiniNeural", "male": "or-IN-SukantNeural"}
}

ELEVENLABS_VOICES = {
    "female": "XrExE9yKIg1WjnnlVkGX",  # Matilda
    "male": "bIHbv24MWmeRgasZH58o"     # Will
}

def resolve_voice_plan(
    voice_option: str,
    ai_archetype: str,
    cultural_context: str = "indian",
    tts_language_code: str = "en-IN"
) -> list[dict]:
    """
    Builds the ordered TTS waterfall plan:
    - en-IN (Indian English): 1. Sarvam v4-flash (24kHz) -> 2. Azure Neural -> 3. ElevenLabs
    - Regional Indian Languages: 1. Sarvam v4-flash (tanya/tarun, 24kHz) -> 2. Sarvam v3 (48kHz) -> 3. Azure Neural -> 4. ElevenLabs
    - Universal / Global English: 1. ElevenLabs -> 2. Azure Neural -> 3. Sarvam v4-flash
    """
    choice = (voice_option or "Auto - AI Choice").strip().lower()
    culture = (cultural_context or "indian").strip().lower()
    archetype = (ai_archetype or "warm_female").strip().lower()
    lang_code = (tts_language_code or "en-IN").strip()
    if lang_code not in AZURE_REGIONAL_VOICES:
        lang_code = "en-IN"

    if choice == "female":
        gender = "female"
    elif choice == "male":
        gender = "male"
    else:
        is_male_arch = any(k in archetype for k in ("male", "deep_celebratory"))
        gender = "male" if is_male_arch else "female"

    azure_step = {
        "provider": "azure",
        "voice": AZURE_REGIONAL_VOICES[lang_code][gender],
        "lang_code": lang_code
    }
    eleven_step = {
        "provider": "elevenlabs",
        "voice": ELEVENLABS_VOICES[gender],
        "lang_code": lang_code
    }

    sarvam_lang = "en-IN" if lang_code in ("en-IN", "en-US") else lang_code

    # Case 1: Regional Indian Languages (Tier 1: Sarvam v4-flash -> Tier 2: Sarvam v3 -> Tier 3: Azure -> Tier 4: ElevenLabs)
    if sarvam_lang != "en-IN":
        sarvam_v4_step = {
            "provider": "sarvam",
            "voice": random.choice(SARVAM_V4_REGIONAL_VOICES[gender]),
            "lang_code": sarvam_lang,
            "model": "bulbul:v4-flash"
        }
        v3_pool = SARVAM_VOICES_SOUTH if sarvam_lang in SOUTH_LANG_CODES else SARVAM_VOICES_GENERAL
        sarvam_v3_step = {
            "provider": "sarvam",
            "voice": random.choice(v3_pool[gender]),
            "lang_code": sarvam_lang,
            "model": "bulbul:v3"
        }
        return [sarvam_v4_step, sarvam_v3_step, azure_step, eleven_step]

    # Case 2: English (Indian vs Universal Accent)
    sarvam_en_step = {
        "provider": "sarvam",
        "voice": random.choice(SARVAM_V4_EN_VOICES[gender]),
        "lang_code": "en-IN",
        "model": "bulbul:v4-flash"
    }
    if culture == "indian":
        return [sarvam_en_step, azure_step, eleven_step]
    return [eleven_step, azure_step, sarvam_en_step]
# =====================================================================
# 5. EXPANDABLE BGM LIBRARY (WITH ANTI-REPEAT & LIVE 404 SAFEGUARD)
# =====================================================================

BGM_BASE_URL = "https://raw.githubusercontent.com/gslrn2023/Greetings-Bgms/main/"

BGM_LIBRARY = {
    "festive": [
        f"{BGM_BASE_URL}festive1.mp3",
        f"{BGM_BASE_URL}festive2.mp3",
        f"{BGM_BASE_URL}festive3.mp3",
        f"{BGM_BASE_URL}festive.mp3",
    ],
    "calm": [
        f"{BGM_BASE_URL}calm1.mp3",
        f"{BGM_BASE_URL}calm2.mp3",
        f"{BGM_BASE_URL}calm3.mp3",
        f"{BGM_BASE_URL}calm.mp3",
    ],
    "energetic": [
        f"{BGM_BASE_URL}energetic1.mp3",
        f"{BGM_BASE_URL}energetic2.mp3",
        f"{BGM_BASE_URL}energetic3.mp3",
        f"{BGM_BASE_URL}energetic.mp3",
    ],
    "elegant": [
        f"{BGM_BASE_URL}elegant1.mp3",
        f"{BGM_BASE_URL}elegant2.mp3",
        f"{BGM_BASE_URL}elegant3.mp3",
        f"{BGM_BASE_URL}elegant.mp3",
    ],
    "spiritual": [
        f"{BGM_BASE_URL}spiritual1.mp3",
        f"{BGM_BASE_URL}spiritual2.mp3",
        f"{BGM_BASE_URL}spiritual3.mp3",
        f"{BGM_BASE_URL}calm1.mp3",
        f"{BGM_BASE_URL}calm.mp3",
    ],
    "romantic": [
        f"{BGM_BASE_URL}romantic1.mp3",
        f"{BGM_BASE_URL}romantic2.mp3",
        f"{BGM_BASE_URL}romantic3.mp3",
        f"{BGM_BASE_URL}elegant1.mp3",
        f"{BGM_BASE_URL}elegant.mp3",
    ],
    "sad": [
        f"{BGM_BASE_URL}sad1.mp3",
        f"{BGM_BASE_URL}sad2.mp3",
        f"{BGM_BASE_URL}sad.mp3",
    ]
}

_last_played_bgm = None
_bgm_lock = threading.Lock()

def pick_bgm_url(mood: str) -> str:
    """
    Randomly selects an available BGM track for the given mood, avoiding consecutive repeats
    and skipping any numbered track URLs that have not been uploaded to GitHub yet.
    """
    global _last_played_bgm
    clean_mood = (mood or "festive").lower().strip()
    candidates = list(BGM_LIBRARY.get(clean_mood, BGM_LIBRARY["festive"]))
    random.shuffle(candidates)

    # Move the last-played track to the end of the priority queue to avoid back-to-back repeats
    with _bgm_lock:
        if _last_played_bgm in candidates and len(candidates) > 1:
            candidates.remove(_last_played_bgm)
            candidates.append(_last_played_bgm)

    for url in candidates:
        try:
            resp = requests.head(url, timeout=3, allow_redirects=True)
            if resp.status_code == 200:
                with _bgm_lock:
                    _last_played_bgm = url
                print(f"[BGM Selector] Mood='{clean_mood}' -> Selected: {url.split('/')[-1]}")
                return url
        except Exception:
            continue

    # Final fallback if HEAD checks time out
    fallback_url = BGM_LIBRARY.get(clean_mood, BGM_LIBRARY["festive"])[-1]
    return fallback_url

# =====================================================================
# 6. MOBILE-FRIENDLY VISUAL ART STYLE MAP
# =====================================================================
STYLE_MAP = {
    "Cinematic": "Cinematic National Geographic Expedition Photography (dramatic wide landscape vista, warm golden-hour rim light and lantern glow contrasting against a deep twilight sky, high dynamic range)",
    "Movie Still": "35mm Anamorphic Hollywood Movie Still (IMAX theatrical color grading, dramatic key and volumetric rim lighting, shallow depth of field, subtle anamorphic lens flare, narrative cinematic staging)",
    "Bokeh & Light Trails": "Dreamy f/1.2 Optical Bokeh & Long-Exposure Photography (tack-sharp foreground subject, creamy circular background bokeh orbs, silky long-exposure light trails, glowing floating embers, rich night ambiance)",
    "Oil Painting": "Masterpiece Oil Impasto Painting (thick expressive palette-knife and bristle brushwork, rich canvas texture, dramatic warm chiaroscuro lighting, museum-grade fine art color harmony)",
    "Sci-Fi Futuristic": "Cinematic Sci-Fi & Futuristic Concept Art (sleek futuristic architecture, holographic light accents, bioluminescent neon-cyan and warm amber reflections, atmospheric volumetric fog, ultra-detailed cyber-organic textures)",
    "35mm Analog": "35mm Analog Editorial Photography (Kodak Portra 400 film grain, warm golden-hour sunlight, f/1.4 shallow depth of field, authentic tactile storytelling)",
    "Luminous Gouache": "Luminous Gouache & Botanical Illustration (hand-painted matte gouache layers, intricate foreground flora, cold-press watercolor paper grain, warm metallic gold-leaf highlights)",
    "Magical Realism": "Atmospheric Magical Realism (bioluminescent floating embers and fireflies, ethereal volumetric god-rays, enchanted natural scenery, dreamy cinematic color grading)",
    "Luxury Still-Life": "Luxury Editorial Still-Life & Scenic Architecture (artisanal crafted objects set on a textured stone or wood table inside a breathtaking open veranda overlooking a scenic vista)"
}

CREATIVE_STYLE_LENSES = list(STYLE_MAP.values())

# =====================================================================
# 7. DAILY QUOTA TRACKER (IST)
# =====================================================================
QUOTA_FILE = os.path.join(BASE_DIR, "daily_usage.json")
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