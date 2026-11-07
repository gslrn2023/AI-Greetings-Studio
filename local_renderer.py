import os
import textwrap
import tempfile
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFont
import config

# Reliable jsDelivr Fontsource CDN URLs for static TrueType (.ttf) fonts
FONT_URLS = {
    "Cinzel": "https://cdn.jsdelivr.net/fontsource/fonts/cinzel@latest/latin-700-normal.ttf",
    "Montserrat": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-700-normal.ttf",
    "Montserrat-Medium": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-500-normal.ttf",
    "Playfair Display": "https://cdn.jsdelivr.net/fontsource/fonts/playfair-display@latest/latin-700-normal.ttf",
    "Caveat": "https://cdn.jsdelivr.net/fontsource/fonts/caveat@latest/latin-700-normal.ttf",
    "Bebas Neue": "https://cdn.jsdelivr.net/fontsource/fonts/bebas-neue@latest/latin-400-normal.ttf"
}

SYSTEM_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
]

def _get_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    cache_dir = os.path.join(tempfile.gettempdir(), "greeting_fonts_v2")
    os.makedirs(cache_dir, exist_ok=True)
    url = FONT_URLS.get(font_name, FONT_URLS["Montserrat"])
    local_ttf = os.path.join(cache_dir, f"{font_name.replace(' ', '_')}.ttf")

    if not os.path.exists(local_ttf) or os.path.getsize(local_ttf) < 1000:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(local_ttf, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Font download warning for {font_name}: {e}")

    if os.path.exists(local_ttf) and os.path.getsize(local_ttf) >= 1000:
        try:
            return ImageFont.truetype(local_ttf, size=size)
        except Exception:
            pass

    # Fallback 1: Standard Linux/Docker pre-installed TrueType fonts
    for sys_font in SYSTEM_FALLBACK_FONTS:
        if os.path.exists(sys_font):
            try:
                return ImageFont.truetype(sys_font, size=size)
            except Exception:
                pass

    # Fallback 2: Pillow 10+ scalable default font (never shrinks to 10px)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

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
    card_bg_rgba = _hex_to_rgba(meta.get("card_bg_hex", "#0B132B"), 195)
    border_rgba = _hex_to_rgba(meta.get("primary_color_hex", "#F4A261"), 175)
    pos = meta.get("text_position", "bottom").lower()

    # Subtle outer frame
    draw.rounded_rectangle([38, 38, W - 38, H - 38], radius=32, outline=border_rgba, width=3)

    # Compact floating glass card (17% of screen height so image stays maximally visible)
    card_w, card_h = int(W * 0.88), int(H * 0.175)
    card_x1 = (W - card_w) // 2
    card_y1 = int(H * 0.065) if pos == "top" else int(H * 0.765)
    card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

    draw.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=36,
        fill=card_bg_rgba,
        outline=primary_rgba,
        width=4
    )

    # Auto-scale title font size if headline has many characters so it never overflows the card
    title_font_size = 56 if len(title) <= 22 else 46
    title_font = _get_font(font_family, title_font_size)
    sub_font = _get_font("Montserrat-Medium", 36)

    # Draw Title (centered in upper section of card)
    title_y = card_y1 + int(card_h * 0.30)
    draw.text((W // 2 + 3, title_y + 3), title, font=title_font, fill=(0, 0, 0, 210), anchor="mm")
    draw.text((W // 2, title_y), title, font=title_font, fill=primary_rgba, anchor="mm")

    # Draw Accent Divider Line
    div_y = card_y1 + int(card_h * 0.52)
    div_half = int(card_w * 0.24)
    draw.line([(W // 2 - div_half, div_y), (W // 2 + div_half, div_y)], fill=primary_rgba, width=3)

    # Draw Wrapped Subtitle
    wrapped_sub = "\n".join(textwrap.wrap(subtitle, width=34)[:2])
    sub_y = card_y1 + int(card_h * 0.76)
    draw.multiline_text(
        (W // 2, sub_y),
        wrapped_sub,
        font=sub_font,
        fill=accent_rgba,
        anchor="mm",
        align="center",
        spacing=10
    )

    canvas.save(out_path, format="PNG")

def render_local_video(
    blueprint: dict,
    local_image_path: str,
    local_tts_path: str,
    audio_duration: float,
    bgm_url: str
) -> str:
    """Fast 1088x1920 MP4 local render via FFmpeg (~12-18 seconds on CPU)."""
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

    # Fast Video Filter:
    # - Input 0 is a single static frame fed into zoompan (no slow -loop 1 needed!)
    # - Uses 1360x2400 intermediate buffer for fast, jitter-free 100% -> 110% Ken Burns zoom
    # - Input 1 (overlay_png) uses -loop 1 with fade-in at t=4.0s
    v_filter = (
        f"[0:v]scale=1360:2400,zoompan=z='min(zoom+0.00035,1.10)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s=1088x1920:fps={config.VIDEO_FPS},"
        f"fade=t=in:st=0:d=0.8[bg];"
        f"[1:v]format=rgba,fade=t=in:st={reveal_delay}:d=0.85:alpha=1[ov];"
        f"[bg][ov]overlay=0:0:shortest=1,format=yuv420p[vout]"
    )

    if has_tts:
        delay_ms = int(reveal_delay * 1000)
        a_filter = (
            f"[2:a]volume=0.08,afade=t=in:st=0:d=1.0,afade=t=out:st={fade_out_start}:d=1.8[bgm];"
            f"[3:a]volume=1.55,adelay={delay_ms}|{delay_ms}[tts];"
            f"[bgm][tts]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-i", local_image_path,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", overlay_png,
            "-stream_loop", "-1", "-t", str(total_duration), "-i", bgm_path,
            "-i", local_tts_path,
            "-filter_complex", f"{v_filter};{a_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_mp4
        ]
    else:
        a_filter = f"[2:a]volume=0.75,afade=t=in:st=0:d=1.0,afade=t=out:st={fade_out_start}:d=1.8[aout]"
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-i", local_image_path,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", overlay_png,
            "-stream_loop", "-1", "-t", str(total_duration), "-i", bgm_path,
            "-filter_complex", f"{v_filter};{a_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_mp4
        ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg Local Render Error:\n{proc.stderr[-600:]}")
    return out_mp4