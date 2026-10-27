import os
import time
import tempfile
import requests
import config

def build_renderscript(
    blueprint: dict,
    image_url: str,
    audio_url: str,
    audio_duration: float,
    bgm_url: str
) -> dict:
    """Builds a template-free 9:16 Creatomate RenderScript JSON payload."""
    comp_meta = blueprint.get("composition_metadata", {})
    overlay_title = comp_meta.get("overlay_title", "Warm Wishes")
    overlay_subtitle = comp_meta.get("overlay_subtitle", "")
    font_family = comp_meta.get("font_family", "Montserrat")
    primary_color = comp_meta.get("primary_color_hex", "#FFD700")
    accent_color = comp_meta.get("accent_color_hex", "#FFFFFF")
    anim_choice = comp_meta.get("animation_type", "slide")

    has_tts = bool(audio_url and audio_duration > 0)
    total_duration = round(max(6.0, audio_duration + config.AUDIO_PADDING_SECONDS), 2) if has_tts else config.DEFAULT_NO_TTS_DURATION

    # Duck BGM volume when voiceover is active so voice is crystal clear
    bgm_volume = "18%" if has_tts else "75%"

    # Map AI Director animation choice to Creatomate text animations
    if anim_choice == "scale":
        title_anim = {
            "time": 0.3,
            "duration": 1.0,
            "easing": "back-out",
            "type": "text-scale",
            "scope": "split-clip",
            "split": "word"
        }
    elif anim_choice == "fade":
        title_anim = {
            "time": 0.3,
            "duration": 1.2,
            "easing": "quadratic-out",
            "type": "fade"
        }
    else:
        title_anim = {
            "time": 0.3,
            "duration": 0.9,
            "easing": "quadratic-out",
            "type": "text-slide",
            "scope": "split-clip",
            "split": "line",
            "direction": "up"
        }

    elements = [
        # Track 1: Background Image with smooth Ken Burns slow zoom
        {
            "name": "Background-Visual",
            "type": "image",
            "track": 1,
            "time": 0,
            "duration": total_duration,
            "source": image_url,
            "width": "100%",
            "height": "100%",
            "fit": "cover",
            "animations": [
                {
                    "time": 0,
                    "duration": total_duration,
                    "easing": "linear",
                    "type": "scale",
                    "fade": False,
                    "start_scale": "100%",
                    "end_scale": "112%"
                }
            ]
        },
        # Track 2: Lower-Third Contrast Card Scrim for guaranteed typography legibility
        {
            "name": "Typography-Backdrop",
            "type": "shape",
            "track": 2,
            "time": 0.2,
            "duration": total_duration - 0.2,
            "x": "50%",
            "y": "81%",
            "width": "90%",
            "height": "24%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": "rgba(10, 12, 18, 0.68)",
            "stroke_color": primary_color,
            "stroke_width": "0.4 vmin",
            "border_radius": "4 vmin",
            "shadow_color": "rgba(0,0,0,0.65)",
            "shadow_blur": "3 vmin",
            "animations": [
                {
                    "time": 0,
                    "duration": 0.7,
                    "easing": "quadratic-out",
                    "type": "fade"
                }
            ]
        },
        # Track 3: Main Headline Text
        {
            "name": "Headline-Text",
            "type": "text",
            "track": 3,
            "time": 0.3,
            "duration": total_duration - 0.3,
            "x": "50%",
            "y": "76.5%",
            "width": "82%",
            "height": "10%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "text": overlay_title,
            "font_family": font_family,
            "font_weight": "700",
            "fill_color": primary_color,
            "shadow_color": "rgba(0,0,0,0.8)",
            "shadow_blur": "1.2 vmin",
            "animations": [title_anim]
        },
        # Track 4: Subtitle / Message Text
        {
            "name": "Subtitle-Text",
            "type": "text",
            "track": 4,
            "time": 0.6,
            "duration": total_duration - 0.6,
            "x": "50%",
            "y": "85.5%",
            "width": "80%",
            "height": "9%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "text": overlay_subtitle,
            "font_family": "Montserrat",
            "font_weight": "500",
            "fill_color": accent_color,
            "line_height": "135%",
            "animations": [
                {
                    "time": 0,
                    "duration": 0.9,
                    "easing": "quadratic-out",
                    "type": "fade"
                }
            ]
        },
        # Track 5: Background Music (auto-ducked and faded out at end)
        {
            "name": "BGM-Track",
            "type": "audio",
            "track": 5,
            "time": 0,
            "duration": total_duration,
            "source": bgm_url,
            "volume": bgm_volume,
            "loop": True,
            "audio_fade_out": 1.8
        }
    ]

    # Track 6: Voiceover Track (starts at 0.5s for natural breathing room)
    if has_tts:
        elements.append({
            "name": "Voiceover-Track",
            "type": "audio",
            "track": 6,
            "time": 0.5,
            "source": audio_url,
            "volume": "100%"
        })

    return {
        "output_format": "mp4",
        "width": config.VIDEO_WIDTH,
        "height": config.VIDEO_HEIGHT,
        "frame_rate": 30,
        "duration": total_duration,
        "elements": elements
    }

def render_greeting_video(
    blueprint: dict,
    image_url: str,
    audio_url: str,
    audio_duration: float
) -> tuple[str, str]:
    """
    Submits the template-free RenderScript to Creatomate v2 API, polls until finished,
    downloads the MP4 locally for Gradio sharing, and returns (remote_mp4_url, local_mp4_path).
    """
    if not config.CREATOMATE_API_KEY:
        raise ValueError("Missing CREATOMATE_API_KEY in Hugging Face Space Secrets.")

    raw_mood = blueprint.get("audio_prompt", {}).get("bgm_mood", "festive").lower().strip()
    bgm_url = config.BGM_MAP.get(raw_mood, config.BGM_MAP["festive"])

    render_payload = build_renderscript(
        blueprint=blueprint,
        image_url=image_url,
        audio_url=audio_url,
        audio_duration=audio_duration,
        bgm_url=bgm_url
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.CREATOMATE_API_KEY}"
    }

    post_url = "https://api.creatomate.com/v2/renders"
    response = requests.post(post_url, json=render_payload, headers=headers, timeout=60)
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(f"Creatomate Render Creation Error ({response.status_code}): {response.text}")

    render_data = response.json()
    # Creatomate v2 returns a single object or list depending on endpoint options
    if isinstance(render_data, list):
        render_data = render_data[0]

    render_id = render_data.get("id")
    status = render_data.get("status")
    video_url = render_data.get("url")

    # Poll status if render is queued/rendering
    poll_url = f"https://api.creatomate.com/v2/renders/{render_id}"
    max_wait_seconds = 120
    elapsed = 0

    while status not in ("succeeded", "failed") and elapsed < max_wait_seconds:
        time.sleep(2.5)
        elapsed += 2.5
        poll_resp = requests.get(poll_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        poll_json = poll_resp.json()
        status = poll_json.get("status")
        video_url = poll_json.get("url")
        if status == "failed":
            error_msg = poll_json.get("error_message", "Unknown Creatomate render failure.")
            raise RuntimeError(f"Creatomate render failed: {error_msg}")

    if status != "succeeded" or not video_url:
        raise RuntimeError(f"Creatomate render timed out (last status: {status}).")

    # Download MP4 to a local file so Gradio's DownloadButton works seamlessly on mobile & desktop
    vid_resp = requests.get(video_url, timeout=60)
    vid_resp.raise_for_status()

    recipient_slug = (
        blueprint.get("context_analysis", {})
        .get("recipient_name", "Greeting")
        .replace(" ", "_")
    )
    temp_dir = tempfile.mkdtemp()
    local_mp4_path = os.path.join(temp_dir, f"Greeting_{recipient_slug}.mp4")
    with open(local_mp4_path, "wb") as f:
        f.write(vid_resp.content)

    return video_url, local_mp4_path