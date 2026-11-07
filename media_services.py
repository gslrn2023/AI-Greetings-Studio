import io
import os
import tempfile
import requests
from mutagen.mp3 import MP3
from PIL import Image
import config

def upload_bytes_to_uguu(file_bytes: bytes, filename: str, mime_type: str) -> str:
    buf = io.BytesIO(file_bytes)
    files = {"files[]": (filename, buf, mime_type)}
    response = requests.post("https://uguu.se/upload", files=files, timeout=60)
    response.raise_for_status()
    res_json = response.json()
    if res_json.get("success") and res_json.get("files"):
        return res_json["files"][0]["url"]
    raise RuntimeError(f"Uguu.se upload failed: {response.text}")

def save_and_optionally_upload_image(image: Image.Image, need_remote_url: bool = True) -> tuple[str, str]:
    """Saves PIL image locally and uploads to uguu.se if needed. Returns (local_path, public_url)."""
    temp_dir = tempfile.mkdtemp()
    local_img_path = os.path.join(temp_dir, "artwork.png")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    with open(local_img_path, "wb") as f:
        f.write(png_bytes)

    public_url = upload_bytes_to_uguu(png_bytes, "artwork.png", "image/png") if need_remote_url else ""
    return local_img_path, public_url

def _request_elevenlabs_tts(tts_script: str, voice_id: str) -> requests.Response:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": tts_script,
        "model_id": config.ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.35,
            "use_speaker_boost": True
        }
    }
    return requests.post(url, json=payload, headers=headers, timeout=60)

def synthesize_voiceover(
    tts_script: str,
    voice_id: str,
    need_remote_url: bool = True
) -> tuple[str, str, float]:
    """
    Synthesizes speech via ElevenLabs. Automatically falls back to Global premade voice
    if a community voice ID hits a Free-Tier API restriction.
    Returns (local_mp3_path, public_audio_url, duration_seconds).
    """
    clean_script = (tts_script or "").strip()
    if not clean_script or not voice_id:
        return "", "", 0.0

    if not config.ELEVENLABS_API_KEY:
        raise ValueError("Missing ELEVENLABS_API_KEY in Hugging Face Space Secrets.")

    response = _request_elevenlabs_tts(clean_script, voice_id)

    # Automatic fallback if community voice (Vanishree/Sid) is restricted on Free-Tier API
    if response.status_code != 200:
        fallback_id = config.get_fallback_voice_id(voice_id)
        if fallback_id != voice_id:
            print(f"[ElevenLabs] Voice {voice_id} returned {response.status_code}. Falling back to {fallback_id}...")
            response = _request_elevenlabs_tts(clean_script, fallback_id)

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API Error ({response.status_code}): {response.text}")

    audio_bytes = response.content
    temp_dir = tempfile.mkdtemp()
    local_mp3_path = os.path.join(temp_dir, "voiceover.mp3")
    with open(local_mp3_path, "wb") as f:
        f.write(audio_bytes)

    try:
        audio_info = MP3(io.BytesIO(audio_bytes))
        duration_sec = float(audio_info.info.length)
    except Exception:
        duration_sec = max(3.5, len(clean_script.split()) / 2.5)

    public_url = upload_bytes_to_uguu(audio_bytes, "voiceover.mp3", "audio/mpeg") if need_remote_url else ""
    return local_mp3_path, public_url, round(duration_sec, 2)