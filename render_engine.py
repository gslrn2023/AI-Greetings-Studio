import os
import time
import random
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
    """Builds a high-motion, template-free 9:16 Creatomate RenderScript JSON payload."""
    comp_meta = blueprint.get("composition_metadata", {})
    overlay_title = comp_meta.get("overlay_title", "Warm Wishes")
    overlay_subtitle = comp_meta.get("overlay_subtitle", "")
    font_family = comp_meta.get("font_family", "Montserrat")
    primary_color = comp_meta.get("primary_color_hex", "#FFD700")
    accent_color = comp_meta.get("accent_color_hex", "#FFFFFF")
    anim_choice = comp_meta.get("animation_type", "scale")

    has_tts = bool(audio_url and audio_duration > 0)
    total_duration = (
        round(max(6.5, audio_duration + config.AUDIO_PADDING_SECONDS), 2)
        if has_tts
        else config.DEFAULT_NO_TTS_DURATION
    )

    # Audio mixing: Drop BGM to 8% when voiceover is active; boost voiceover to 150%
    bgm_volume = config.BGM_VOLUME_WITH_TTS if has_tts else config.BGM_VOLUME_NO_TTS

    # Randomize camera pan focal point so the 100% -> 126% zoom glides diagonally across the scene
    pan_anchors = [
        ("35%", "30%"),  # Glide toward upper-left subject
        ("65%", "30%"),  # Glide toward upper-right subject
        ("50%", "25%"),  # Glide upward toward sky/vista
        ("45%", "40%")   # Balanced cinematic push-in
    ]
    cam_x_anchor, cam_y_anchor = random.choice(pan_anchors)

    # Dynamic Headline word-by-word animation
    if anim_choice == "slide":
        title_anim = {
            "time": 0,
            "duration": 1.1,
            "easing": "back-out",
            "type": "text-slide",
            "scope": "split-clip",
            "split": "word",
            "distance": "120%",
            "direction": "up"
        }
    else:
        title_anim = {
            "time": 0,
            "duration": 1.1,
            "easing": "back-out",
            "type": "text-scale",
            "scope": "split-clip",
            "split": "word",
            "start_scale": "0%"
        }

    elements = [
        # Track 1: Background Image with strong Ken Burns Diagonal Pan + Zoom (100% -> 126%)
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
                    "duration": 0.9,
                    "easing": "quadratic-out",
                    "type": "fade"
                },
                {
                    "time": 0,
                    "duration": total_duration,
                    "easing": "sine-in-out",
                    "type": "scale",
                    "fade": False,
                    "start_scale": "100%",
                    "end_scale": "126%",
                    "x_anchor": cam_x_anchor,
                    "y_anchor": cam_y_anchor
                }
            ]
        },
        # Track 2: Full-Screen Decorative Luxury Border Frame (draws in with spring scale)
        {
            "name": "Luxury-Outer-Border",
            "type": "shape",
            "track": 2,
            "time": 0.15,
            "duration": total_duration - 0.15,
            "x": "50%",
            "y": "50%",
            "width": "93%",
            "height": "95%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": "rgba(0, 0, 0, 0)",
            "stroke_color": primary_color,
            "stroke_width": "0.35 vmin",
            "border_radius": "3.5 vmin",
            "opacity": "65%",
            "animations": [
                {
                    "time": 0,
                    "duration": 1.0,
                    "easing": "back-out",
                    "type": "scale",
                    "fade": True,
                    "start_scale": "92%",
                    "end_scale": "100%"
                }
            ]
        },
        # Track 3: Lower-Third Contrast Glass Card (slides up from bottom with spring bounce)
        {
            "name": "Typography-Backdrop",
            "type": "shape",
            "track": 3,
            "time": 0.25,
            "duration": total_duration - 0.25,
            "x": "50%",
            "y": "80.5%",
            "width": "88%",
            "height": "23%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": "rgba(10, 12, 20, 0.76)",
            "stroke_color": primary_color,
            "stroke_width": "0.45 vmin",
            "border_radius": "3.5 vmin",
            "shadow_color": "rgba(0,0,0,0.75)",
            "shadow_blur": "3.5 vmin",
            "animations": [
                {
                    "time": 0,
                    "duration": 0.85,
                    "easing": "back-out",
                    "type": "slide",
                    "direction": "up",
                    "distance": "45%",
                    "fade": True
                }
            ]
        },
        # Track 4: Main Headline Text (word-by-word 3D pop/slide)
        {
            "name": "Headline-Text",
            "type": "text",
            "track": 4,
            "time": 0.55,
            "duration": total_duration - 0.55,
            "x": "50%",
            "y": "75.0%",
            "width": "82%",
            "height": "9.5%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "text": overlay_title,
            "font_family": font_family,
            "font_weight": "700",
            "fill_color": primary_color,
            "shadow_color": "rgba(0,0,0,0.85)",
            "shadow_blur": "1.5 vmin",
            "animations": [title_anim]
        },
        # Track 5: Animated Golden Divider Line between Title and Subtitle
        {
            "name": "Accent-Divider-Line",
            "type": "shape",
            "track": 5,
            "time": 0.75,
            "duration": total_duration - 0.75,
            "x": "50%",
            "y": "80.3%",
            "width": "36%",
            "height": "0.35%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": primary_color,
            "border_radius": "1 vmin",
            "animations": [
                {
                    "time": 0,
                    "duration": 0.8,
                    "easing": "exponential-out",
                    "type": "scale",
                    "fade": True,
                    "start_scale": "0%",
                    "end_scale": "100%"
                }
            ]
        },
        # Track 6: Subtitle / Message Text (line-by-line upward reveal)
        {
            "name": "Subtitle-Text",
            "type": "text",
            "track": 6,
            "time": 0.9,
            "duration": total_duration - 0.9,
            "x": "50%",
            "y": "85.5%",
            "width": "80%",
            "height": "8.5%",
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
                    "type": "text-slide",
                    "scope": "split-clip",
                    "split": "line",
                    "direction": "up",
                    "distance": "100%",
                    "fade": True
                }
            ]
        },
        # Track 7: Background Music (lowered to 8% when TTS is active, faded out at end)
        {
            "name": "BGM-Track",
            "type": "audio",
            "track": 7,
            "time": 0,
            "duration": total_duration,
            "source": bgm_url,
            "volume": bgm_volume,
            "loop": True,
            "audio_fade_out": 1.8
        }
    ]

    # Track 8: Boosted Voiceover Track (150% volume, starts at 0.55s as title pops in)
    if has_tts:
        elements.append({
            "name": "Voiceover-Track",
            "type": "audio",
            "track": 8,
            "time": 0.55,
            "source": audio_url,
            "volume": config.TTS_VOLUME_BOOST
        })

    return {
        "output_format": "mp4",
        "width": config.VIDEO_WIDTH,
        "height": config.VIDEO_HEIGHT,
        "frame_rate": 30,
        "crf": 18,
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
    if isinstance(render_data, list):
        render_data = render_data[0]

    render_id = render_data.get("id")
    status = render_data.get("status")
    video_url = render_data.get("url")

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