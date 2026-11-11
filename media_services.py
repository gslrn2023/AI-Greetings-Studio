import io
import os
import re
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
# SECTION 2: TIER 1 ENGINE - SARVAM AI (BULBUL:V4-FLASH & BULBUL:V3)
# =====================================================================
def _tts_sarvam(
    tts_script: str,
    speaker: str,
    lang_code: str,
    out_mp3_path: str,
    model_override: str = ""
) -> tuple[bool, str, str]:
    """
    Calls Sarvam AI REST API with pace=1.0 and sample_rate=24000.
    Uses bulbul:v4-flash for en-IN personas and bulbul:v3 for regional Indian scripts.
    Returns (success, active_model, error_message).
    """
    if not config.SARVAM_API_KEY:
        msg = "SARVAM_API_KEY not configured in environment."
        print(f"[TTS Tier-1: Sarvam] {msg}")
        return False, "", msg

    clean_lang = (lang_code or "en-IN").strip()
    clean_speaker = speaker.strip().lower()

    # Determine active model: bulbul:v3 for ml-IN/od-IN or short v3 speaker IDs; bulbul:v4-flash for v4 personas
    if clean_lang in ("ml-IN", "od-IN") or "_" not in clean_speaker:
        active_model = "bulbul:v3"
    else:
        active_model = model_override or "bulbul:v4-flash"

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    dynamic_temp = 0.45 if clean_lang == "en-IN" else 0.35
    payload = {
        "text": tts_script,
        "target_language_code": clean_lang,
        "speaker": clean_speaker,
        "model": active_model,
        "pace": 1.0,
        "temperature": dynamic_temp,
        "speech_sample_rate": 48000,
        "output_audio_codec": "mp3"
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=45)
        if resp.status_code != 200:
            err_msg = f"HTTP {resp.status_code} ({active_model}): {resp.text[:240]}"
            print(f"[TTS Tier-1: Sarvam] {err_msg}")
            return False, active_model, err_msg

        audios = resp.json().get("audios", [])
        if not audios:
            err_msg = f"Empty 'audios' array returned from Sarvam API ({active_model})."
            print(f"[TTS Tier-1: Sarvam] {err_msg}")
            return False, active_model, err_msg

        audio_bytes = base64.b64decode(audios[0])
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)
        print(
            f"[TTS Tier-1: Sarvam] Success: model='{active_model}', speaker='{clean_speaker}', "
            f"lang='{clean_lang}', temp={dynamic_temp}, pace=1.0, rate=48000."
        )
        return True, active_model, ""
    except Exception as e:
        err_msg = f"Exception ({active_model}): {str(e)}"
        print(f"[TTS Tier-1: Sarvam] {err_msg}")
        return False, active_model, err_msg

# =====================================================================
# SECTION 3: TIER 2 ENGINE - AZURE NEURAL TTS (EDGE-TTS REGIONAL FALLBACK)
# =====================================================================
def _tts_azure_edge(tts_script: str, edge_voice: str, out_mp3_path: str) -> tuple[bool, str]:
    """Synthesizes speech using Microsoft Azure Neural TTS (supports all Indian languages)."""
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
            return True, ""
        err_msg = f"edge-tts exit {proc.returncode}: {(proc.stderr or 'Empty audio output')[:240]}"
        print(f"[TTS Tier-2: Azure Neural] Failed: {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"Exception: {str(e)}"
        print(f"[TTS Tier-2: Azure Neural] {err_msg}")
        return False, err_msg

# =====================================================================
# SECTION 4: TIER 3 ENGINE - ELEVENLABS PREMADE VOICES (MATILDA / WILL)
# =====================================================================
def _tts_elevenlabs(tts_script: str, voice_id: str, out_mp3_path: str) -> tuple[bool, str]:
    """Calls ElevenLabs TTS API with Free-Tier safe premade voices."""
    if not config.ELEVENLABS_API_KEY:
        msg = "ELEVENLABS_API_KEY not configured in environment."
        print(f"[TTS Tier-3: ElevenLabs] {msg}")
        return False, msg

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
            err_msg = f"HTTP {resp.status_code}: {resp.text[:240]}"
            print(f"[TTS Tier-3: ElevenLabs] {err_msg}")
            return False, err_msg

        with open(out_mp3_path, "wb") as f:
            f.write(resp.content)
        print(f"[TTS Tier-3: ElevenLabs] Success with voice_id='{voice_id}'.")
        return True, ""
    except Exception as e:
        err_msg = f"Exception: {str(e)}"
        print(f"[TTS Tier-3: ElevenLabs] {err_msg}")
        return False, err_msg

# =====================================================================
# SECTION 5: MASTER 3-TIER WATERFALL ORCHESTRATOR WITH BLACK BOX TRACE
# =====================================================================
def synthesize_voiceover(
    tts_script: str,
    voice_plan: list[dict],
    need_remote_url: bool = True
) -> tuple[str, str, float, dict]:
    """
    Executes the ordered 3-tier voice_plan (Sarvam -> Azure Regional -> ElevenLabs).
    Returns (local_mp3_path, public_audio_url, duration_seconds, tts_trace).
    """
    clean_script = re.sub(r"[*_#`~<>]", "", (tts_script or "")).strip()
    tts_trace = {
        "status": "skipped",
        "used_provider": "None",
        "used_voice": "Muted",
        "language_code": "en-IN",
        "duration_sec": 0.0,
        "tts_script": clean_script,
        "waterfall_attempts": []
    }

    if not clean_script or not voice_plan:
        return "", "", 0.0, tts_trace

    temp_dir = tempfile.mkdtemp()
    local_mp3_path = os.path.join(temp_dir, "voiceover.mp3")
    synthesized = False

    PROVIDER_LABELS = {
        "sarvam": "Sarvam AI",
        "azure": "MS Azure",
        "elevenlabs": "ElevenLabs"
    }

    for idx, step in enumerate(voice_plan, start=1):
        provider = step.get("provider", "")
        voice = step.get("voice", "")
        lang_code = step.get("lang_code", "en-IN")
        step_model = step.get("model", "")
        err_detail = ""
        used_model = ""

        if provider == "sarvam":
            synthesized, used_model, err_detail = _tts_sarvam(
                clean_script, voice, lang_code, local_mp3_path, model_override=step_model
            )
        elif provider == "azure":
            synthesized, err_detail = _tts_azure_edge(clean_script, voice, local_mp3_path)
            used_model = "edge-tts-neural"
        elif provider == "elevenlabs":
            synthesized, err_detail = _tts_elevenlabs(clean_script, voice, local_mp3_path)
            used_model = config.ELEVENLABS_MODEL

        attempt_record = {
            "tier": idx,
            "provider": PROVIDER_LABELS.get(provider, provider),
            "model": used_model,
            "voice": voice,
            "lang_code": lang_code,
            "status": "success" if synthesized else "failed"
        }
        if err_detail:
            attempt_record["error"] = err_detail
        tts_trace["waterfall_attempts"].append(attempt_record)

        if synthesized:
            tts_trace["status"] = "success" if idx == 1 else f"fallback_tier_{idx}"
            tts_trace["used_provider"] = PROVIDER_LABELS.get(provider, provider)
            tts_trace["used_model"] = used_model
            tts_trace["used_voice"] = voice
            tts_trace["language_code"] = lang_code
            break

    if not synthesized:
        tts_trace["status"] = "all_failed"
        raise RuntimeError(f"All 3 TTS engines failed. Trace: {tts_trace['waterfall_attempts']}")

    with open(local_mp3_path, "rb") as f:
        audio_bytes = f.read()

    try:
        audio_info = MP3(io.BytesIO(audio_bytes))
        duration_sec = round(float(audio_info.info.length), 2)
    except Exception:
        duration_sec = round(max(3.5, len(clean_script.split()) / 2.5), 2)

    tts_trace["duration_sec"] = duration_sec
    public_url = upload_bytes_to_uguu(audio_bytes, "voiceover.mp3", "audio/mpeg") if need_remote_url else ""
    return local_mp3_path, public_url, duration_sec, tts_trace