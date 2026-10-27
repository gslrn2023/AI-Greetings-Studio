import io
import requests
from mutagen.mp3 import MP3
from PIL import Image
import config

def upload_bytes_to_uguu(file_bytes: bytes, filename: str, mime_type: str) -> str:
    """Uploads an in-memory byte buffer to uguu.se and returns the public URL."""
    buf = io.BytesIO(file_bytes)
    files = {"files[]": (filename, buf, mime_type)}
    response = requests.post("https://uguu.se/upload", files=files, timeout=60)
    response.raise_for_status()
    res_json = response.json()

    if res_json.get("success") and res_json.get("files"):
        return res_json["files"][0]["url"]
    raise RuntimeError(f"Uguu.se upload failed for {filename}: {response.text}")

def upload_pil_image(image: Image.Image) -> str:
    """Converts a PIL Image to PNG bytes and uploads to uguu.se."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return upload_bytes_to_uguu(buf.getvalue(), "greeting_visual.png", "image/png")

def generate_tts_and_upload(tts_script: str, voice_archetype: str) -> tuple[str, float]:
    """
    Calls ElevenLabs API for voice synthesis, calculates exact audio duration in seconds,
    uploads to uguu.se, and returns (public_audio_url, duration_seconds).
    Returns ("", 0.0) if tts_script is empty (solemn occasions).
    """
    clean_script = (tts_script or "").strip()
    if not clean_script or voice_archetype == "none":
        return "", 0.0

    if not config.ELEVENLABS_API_KEY:
        raise ValueError("Missing ELEVENLABS_API_KEY in Hugging Face Space Secrets.")

    voice_id = config.VOICE_MAP.get(voice_archetype, config.VOICE_MAP["warm_female"])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "text": clean_script,
        "model_id": config.ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API Error ({response.status_code}): {response.text}")

    audio_bytes = response.content

    # Calculate exact MP3 duration using mutagen
    try:
        audio_info = MP3(io.BytesIO(audio_bytes))
        duration_sec = float(audio_info.info.length)
    except Exception:
        # Fallback estimate (~2.5 words per second) if header parsing fails
        word_count = len(clean_script.split())
        duration_sec = max(3.5, word_count / 2.5)

    public_audio_url = upload_bytes_to_uguu(audio_bytes, "greeting_voiceover.mp3", "audio/mpeg")
    return public_audio_url, round(duration_sec, 2)