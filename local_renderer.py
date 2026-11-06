import os
import textwrap
import tempfile
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFont
import config

FONT_URLS = {
    "Cinzel": "https://github.com/google/fonts/raw/main/ofl/cinzel/static/Cinzel-Bold.ttf",
    "Montserrat": "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf",
    "Montserrat-Medium": "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Medium.ttf",
    "Playfair Display": "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/static/PlayfairDisplay-Bold.ttf",
    "Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/static/Caveat-Bold.ttf",
    "Bebas Neue": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf"
}

def _get_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    cache_dir = "/tmp/greeting_fonts"
    os.makedirs(cache_dir, exist_ok=True)
    url = FONT_URLS.get(font_name, FONT_URLS["Montserrat"])
    local_ttf = os.path.join(cache_dir, f"{font_name.replace(' ', '_')}.ttf")
    if not os.path.exists(local_ttf):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(local_ttf, "wb") as f:
                f.write(r.content)
        except Exception:
            return ImageFont.load_default()
    return ImageFont.truetype(local_ttf, size=size)

def _hex_to_rgba(hex_code: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = (hex_code or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

def create_overlay_png(blueprint: dict, out_path: str):
    """Renders a high-res 1088x1920 transparent PNG containing the styled typography card."""
    W, H = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    meta = blueprint.get("composition_metadata", {})
    title = meta.get("overlay_title", "Warm Wishes")
    subtitle = meta.get("overlay_subtitle", "")
    font_family = meta.get("font_family", "Cinzel")
    primary_rgba = _hex_to_rgba(meta.get("primary_color_hex", "#F4A261"), 255)
    accent_rgba = _hex_to_rgba(meta.get("accent_color_hex", "#F8F9FA"), 255)
    card_bg_rgba = _hex_to_rgba(meta.get("card_bg_hex", "#0B132B"), 185)  # 72% opacity glass
    border_rgba = _hex_to_rgba(meta.get("primary_color_hex", "#F4A261"), 170)
    pos = meta.get("text_position", "bottom").lower()

    # Subtle outer frame
    draw.rounded_rectangle([38, 38, W - 38, H - 38], radius=32, outline=border_rgba, width=3)

    # Compact floating glass card (16% of screen height so image stays maximally visible)
    card_w, card_h = int(W * 0.86), int(H * 0.165)
    card_x1 = (W - card_w) // 2
    card_y1 = int(H * 0.07) if pos == "top" else int(H * 0.775)
    card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

    draw.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=36,
        fill=card_bg_rgba,
        outline=primary_rgba,
        width=4
    )

    title_font = _get_font(font_family, 58)
    sub_font = _get_font("Montserrat-Medium", 35)

    # Draw Title (centered in upper half of card)
    title_y = card_y1 + int(card_h * 0.30)
    draw.text((W // 2 + 2, title_y + 2), title, font=title_font, fill=(0, 0, 0, 200), anchor="mm")
    draw.text((W // 2, title_y), title, font=title_font, fill=primary_rgba, anchor="mm")

    # Draw Accent Divider Line
    div_y = card_y1 + int(card_h * 0.52)
    div_half = int(card_w * 0.22)
    draw.line([(W // 2 - div_half, div_y), (W // 2 + div_half, div_y)], fill=primary_rgba, width=3)

    # Draw Wrapped Subtitle
    wrapped_sub = "\n".join(textwrap.wrap(subtitle, width=36)[:2])
    sub_y = card_y1 + int(card_h * 0.75)
    draw.multiline_text((W // 2, sub_y), wrapped_sub, font=sub_font, fill=accent_rgba, anchor="mm", align="center", spacing=8)

    canvas.save(out_path, format="PNG")

def render_local_video(
    blueprint: dict,
    local_image_path: str,
    local_tts_path: str,
    audio_duration: float,
    bgm_url: str
) -> str:
    """Renders 1088x1920 MP4 locally via FFmpeg with 4s pure image showcase, Ken Burns zoom, and audio mixing."""
    temp_dir = tempfile.mkdtemp()
    overlay_png = os.path.join(temp_dir, "overlay.png")
    create_overlay_png(blueprint, overlay_png)

    bgm_path = os.path.join(temp_dir, "bgm.mp3")
    r = requests.get(bgm_url, timeout=45)
    r.raise_for_status()
    with open(bgm_path, "wb") as f:
        f.write(r.content)

    has_tts = bool(local_tts_path and audio_duration > 0)
    reveal_delay = config.TEXT_REVEAL_DELAY  # 4.0 seconds pure image first
    total_duration = (
        round(reveal_delay + audio_duration + config.AUDIO_PADDING_SECONDS, 2)
        if has_tts
        else config.DEFAULT_NO_TTS_DURATION
    )
    total_frames = int(total_duration * config.VIDEO_FPS)
    fade_out_start = max(1.0, total_duration - 1.8)

    recipient_slug = blueprint.get("context_analysis", {}).get("recipient_name", "Greeting").replace(" ", "_")
    out_mp4 = os.path.join(temp_dir, f"Greeting_{recipient_slug}.mp4")

    # Video Filter: Upscale 2x for jitter-free subpixel Ken Burns zoom (1.0 -> 1.10), then fade in overlay PNG at t=4.0s
    v_filter = (
        f"[0:v]scale=2176:3840,zoompan=z='min(zoom+0.00035,1.10)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1088x1920:fps={config.VIDEO_FPS},"
        f"fade=t=in:st=0:d=0.8[bg];"
        f"[1:v]format=rgba,fade=t=in:st={reveal_delay}:d=0.85:alpha=1[ov];"
        f"[bg][ov]overlay=0:0:format=auto,format=yuv420p[vout]"
    )

    if has_tts:
        delay_ms = int(reveal_delay * 1000)
        # BGM plays at 25% during the 4s visual intro, then ducks to 8% when voiceover starts at 4.0s
        a_filter = (
            f"[2:a]volume=0.10,afade=t=in:st=0:d=1.0,afade=t=out:st={fade_out_start}:d=1.8[bgm];"
            f"[3:a]volume=1.55,adelay={delay_ms}|{delay_ms}[tts];"
            f"[bgm][tts]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(total_duration), "-i", local_image_path,
            "-loop", "1", "-t", str(total_duration), "-i", overlay_png,
            "-stream_loop", "-1", "-t", str(total_duration), "-i", bgm_path,
            "-i", local_tts_path,
            "-filter_complex", f"{v_filter};{a_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_mp4
        ]
    else:
        a_filter = f"[2:a]volume=0.75,afade=t=in:st=0:d=1.0,afade=t=out:st={fade_out_start}:d=1.8[aout]"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(total_duration), "-i", local_image_path,
            "-loop", "1", "-t", str(total_duration), "-i", overlay_png,
            "-stream_loop", "-1", "-t", str(total_duration), "-i", bgm_path,
            "-filter_complex", f"{v_filter};{a_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_mp4
        ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg Local Render Error:\n{proc.stderr[-600:]}")
    return out_mp4