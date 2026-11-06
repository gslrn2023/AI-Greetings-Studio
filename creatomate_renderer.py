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
    meta = blueprint.get("composition_metadata", {})
    overlay_title = meta.get("overlay_title", "Warm Wishes")
    overlay_subtitle = meta.get("overlay_subtitle", "")
    font_family = meta.get("font_family", "Cinzel")
    primary_color = meta.get("primary_color_hex", "#F4A261")
    accent_color = meta.get("accent_color_hex", "#FFFFFF")
    card_bg_hex = meta.get("card_bg_hex", "#0B132B")
    pos = meta.get("text_position", "bottom").lower()

    reveal_t = config.TEXT_REVEAL_DELAY  # 4.0s pure unobstructed image showcase
    has_tts = bool(audio_url and audio_duration > 0)
    total_duration = (
        round(reveal_t + audio_duration + config.AUDIO_PADDING_SECONDS, 2)
        if has_tts
        else config.DEFAULT_NO_TTS_DURATION
    )

    # Dynamic vertical positioning so subject is never concealed
    if pos == "top":
        card_y, title_y, div_y, sub_y = "16.5%", "12.5%", "16.3%", "20.2%"
    else:
        card_y, title_y, div_y, sub_y = "84.0%", "80.0%", "83.8%", "87.8%"

    cam_x, cam_y = random.choice([("35%", "30%"), ("65%", "30%"), ("50%", "25%"), ("45%", "40%")])
    bgm_volume = config.BGM_VOLUME_WITH_TTS if has_tts else config.BGM_VOLUME_NO_TTS

    elements = [
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
                {"time": 0, "duration": 0.9, "easing": "quadratic-out", "type": "fade"},
                {
                    "time": 0,
                    "duration": total_duration,
                    "easing": "sine-in-out",
                    "type": "scale",
                    "fade": False,
                    "start_scale": "100%",
                    "end_scale": "110%",
                    "x_anchor": cam_x,
                    "y_anchor": cam_y
                }
            ]
        },
        {
            "name": "Typography-Backdrop",
            "type": "shape",
            "track": 2,
            "time": reveal_t,
            "duration": total_duration - reveal_t,
            "x": "50%",
            "y": card_y,
            "width": "86%",
            "height": "17%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": card_bg_hex,
            "opacity": "76%",
            "stroke_color": primary_color,
            "stroke_width": "0.4 vmin",
            "border_radius": "3.5 vmin",
            "animations": [{"time": 0, "duration": 0.8, "easing": "back-out", "type": "scale", "start_scale": "85%", "fade": True}]
        },
        {
            "name": "Headline-Text",
            "type": "text",
            "track": 3,
            "time": reveal_t + 0.2,
            "duration": total_duration - (reveal_t + 0.2),
            "x": "50%",
            "y": title_y,
            "width": "80%",
            "height": "7.5%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "text": overlay_title,
            "font_family": font_family,
            "font_weight": "700",
            "fill_color": primary_color,
            "animations": [{"time": 0, "duration": 0.9, "easing": "back-out", "type": "text-scale", "scope": "split-clip", "split": "word", "start_scale": "0%"}]
        },
        {
            "name": "Accent-Divider",
            "type": "shape",
            "track": 4,
            "time": reveal_t + 0.35,
            "duration": total_duration - (reveal_t + 0.35),
            "x": "50%",
            "y": div_y,
            "width": "34%",
            "height": "0.3%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "path": "M 0 0 L 100 0 L 100 100 L 0 100 Z",
            "fill_color": primary_color,
            "animations": [{"time": 0, "duration": 0.7, "easing": "exponential-out", "type": "scale", "start_scale": "0%", "fade": True}]
        },
        {
            "name": "Subtitle-Text",
            "type": "text",
            "track": 5,
            "time": reveal_t + 0.45,
            "duration": total_duration - (reveal_t + 0.45),
            "x": "50%",
            "y": sub_y,
            "width": "78%",
            "height": "7%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "text": overlay_subtitle,
            "font_family": "Montserrat",
            "font_weight": "500",
            "fill_color": accent_color,
            "animations": [{"time": 0, "duration": 0.8, "easing": "quadratic-out", "type": "fade"}]
        },
        {
            "name": "BGM-Track",
            "type": "audio",
            "track": 6,
            "time": 0,
            "duration": total_duration,
            "source": bgm_url,
            "volume": bgm_volume,
            "loop": True,
            "audio_fade_out": 1.8
        }
    ]

    if has_tts:
        elements.append({
            "name": "Voiceover-Track",
            "type": "audio",
            "track": 7,
            "time": reveal_t + 0.15,
            "source": audio_url,
            "volume": config.TTS_VOLUME_BOOST
        })

    return {
        "output_format": "mp4",
        "width": config.VIDEO_WIDTH,
        "height": config.VIDEO_HEIGHT,
        "frame_rate": 60,
        "crf": 18,
        "duration": total_duration,
        "elements": elements
    }

def render_creatomate_video(
    blueprint: dict,
    image_url: str,
    audio_url: str,
    audio_duration: float,
    bgm_url: str
) -> str:
    if not config.CREATOMATE_API_KEY:
        raise ValueError("Missing CREATOMATE_API_KEY in Hugging Face Space Secrets.")

    payload = build_renderscript(blueprint, image_url, audio_url, audio_duration, bgm_url)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.CREATOMATE_API_KEY}"}
    resp = requests.post("https://api.creatomate.com/v2/renders", json=payload, headers=headers, timeout=60)
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(f"Creatomate Error ({resp.status_code}): {resp.text}")

    data = resp.json()
    if isinstance(data, list):
        data = data[0]
    render_id, status, video_url = data.get("id"), data.get("status"), data.get("url")

    elapsed = 0
    while status not in ("succeeded", "failed") and elapsed < 120:
        time.sleep(2.5)
        elapsed += 2.5
        poll = requests.get(f"https://api.creatomate.com/v2/renders/{render_id}", headers=headers, timeout=30).json()
        status, video_url = poll.get("status"), poll.get("url")
        if status == "failed":
            raise RuntimeError(f"Creatomate render failed: {poll.get('error_message')}")

    vid_resp = requests.get(video_url, timeout=60)
    vid_resp.raise_for_status()
    slug = blueprint.get("context_analysis", {}).get("recipient_name", "Greeting").replace(" ", "_")
    local_mp4 = os.path.join(tempfile.mkdtemp(), f"Greeting_{slug}.mp4")
    with open(local_mp4, "wb") as f:
        f.write(vid_resp.content)
    return local_mp4