import io
import os
import base64
import tempfile
import subprocess
import requests
from mutagen.mp3 import MP3
from PIL import Image
import config

# =====================================================================
# SECTION 1: UGUU.SE TEMPORARY CLOUD HOSTING & IMAGE SAVER
# =====================================================================
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


# =====================================================================
# SECTION 2: TIER 1 ENGINE - SARVAM AI (BULBUL:V3 INDIAN ACCENT TTS)
# =====================================================================
def _tts_sarvam(tts_script: str, speaker: str, out_mp3_path: str) -> bool:
    """Calls Sarvam AI bulbul:v3 REST API and writes MP3 bytes to out_mp3_path."""
    if not config.SARVAM_API_KEY:
        print("[TTS Tier-1: Sarvam] SARVAM_API_KEY not set. Skipping to Tier-2...")
        return False

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": tts_script,
        "target_language_code": "en-IN",
        "speaker": speaker.lower(),
        "model": config.SARVAM_MODEL,
        "pace": 0.96,
        "temperature": 0.65,
        "speech_sample_rate": 24000,
        "output_audio_codec": "mp3"
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=45)
        if resp.status_code != 200:
            print(f"[TTS Tier-1: Sarvam] HTTP {resp.status_code}: {resp.text}")
            return False

        audios = resp.json().get("audios", [])
        if not audios:
            print("[TTS Tier-1: Sarvam] Empty 'audios' array returned.")
            return False

        audio_bytes = base64.b64decode(audios[0])
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)
        print(f"[TTS Tier-1: Sarvam] Success with speaker='{speaker}'.")
        return True
    except Exception as e:
        print(f"[TTS Tier-1: Sarvam] Exception: {e}")
        return False


# =====================================================================
# SECTION 3: TIER 2 ENGINE - AZURE NEURAL TTS (EDGE-TTS FALLBACK)
# =====================================================================
def _tts_azure_edge(tts_script: str, edge_voice: str, out_mp3_path: str) -> bool:
    """Synthesizes speech using Microsoft Azure Neural TTS (edge-tts CLI)."""
    try:
        cmd = [
            "edge-tts",
            "--voice", edge_voice,
            "--text", tts_script,
            "--write-media", out_mp3_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=40)
        if proc.returncode == 0 and os.path.exists(out_mp3_path) and os.path.getsize(out_mp3_path) > 500:
            print(f"[TTS Tier-2: Azure Neural] Success with voice='{edge_voice}'.")
            return True
        print(f"[TTS Tier-2: Azure Neural] Failed: {proc.stderr}")
        return False
    except Exception as e:
        print(f"[TTS Tier-2: Azure Neural] Exception: {e}")
        return False


# =====================================================================
# SECTION 4: TIER 3 ENGINE - ELEVENLABS PREMADE VOICES (MATILDA / WILL)
# =====================================================================
def _tts_elevenlabs(tts_script: str, voice_id: str, out_mp3_path: str) -> bool:
    """Calls ElevenLabs TTS API with Free-Tier safe premade voices."""
    if not config.ELEVENLABS_API_KEY:
        print("[TTS Tier-3: ElevenLabs] ELEVENLABS_API_KEY not set.")
        return False

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

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            print(f"[TTS Tier-3: ElevenLabs] HTTP {resp.status_code}: {resp.text}")
            return False

        with open(out_mp3_path, "wb") as f:
            f.write(resp.content)
        print(f"[TTS Tier-3: ElevenLabs] Success with voice_id='{voice_id}'.")
        return True
    except Exception as e:
        print(f"[TTS Tier-3: ElevenLabs] Exception: {e}")
        return False


# =====================================================================
# SECTION 5: MASTER 3-TIER WATERFALL ORCHESTRATOR
# =====================================================================
def synthesize_voiceover(
    tts_script: str,
    voice_plan: list[dict],
    need_remote_url: bool = True
) -> tuple[str, str, float]:
    """
    Executes the ordered 3-tier voice_plan (Sarvam -> Azure Neural -> ElevenLabs).
    Returns (local_mp3_path, public_audio_url, duration_seconds).
    """
    clean_script = (tts_script or "").strip()
    if not clean_script or not voice_plan:
        return "", "", 0.0

    temp_dir = tempfile.mkdtemp()
    local_mp3_path = os.path.join(temp_dir, "voiceover.mp3")
    synthesized = False

    for step in voice_plan:
        provider = step.get("provider")
        voice = step.get("voice")

        if provider == "sarvam":
            synthesized = _tts_sarvam(clean_script, voice, local_mp3_path)
        elif provider == "azure":
            synthesized = _tts_azure_edge(clean_script, voice, local_mp3_path)
        elif provider == "elevenlabs":
            synthesized = _tts_elevenlabs(clean_script, voice, local_mp3_path)

        if synthesized:
            break

    if not synthesized:
        raise RuntimeError("All 3 TTS engines (Sarvam, Azure Neural, ElevenLabs) failed to synthesize audio.")

    with open(local_mp3_path, "rb") as f:
        audio_bytes = f.read()

    try:
        audio_info = MP3(io.BytesIO(audio_bytes))
        duration_sec = float(audio_info.info.length)
    except Exception:
        duration_sec = max(3.5, len(clean_script.split()) / 2.5)

    public_url = upload_bytes_to_uguu(audio_bytes, "voiceover.mp3", "audio/mpeg") if need_remote_url else ""
    return local_mp3_path, public_url, round(duration_sec, 2)