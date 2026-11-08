import os
import re
import random
import textwrap
import tempfile
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFont
import config

# Western Latin TrueType (.ttf) fonts via jsDelivr Fontsource CDN
FONT_URLS = {
    "Cinzel": "https://cdn.jsdelivr.net/fontsource/fonts/cinzel@latest/latin-700-normal.ttf",
    "Montserrat": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-700-normal.ttf",
    "Montserrat-SemiBold": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-600-normal.ttf",
    "Playfair Display": "https://cdn.jsdelivr.net/fontsource/fonts/playfair-display@latest/latin-700-normal.ttf",
    "Caveat": "https://cdn.jsdelivr.net/fontsource/fonts/caveat@latest/latin-700-normal.ttf",
    "Bebas Neue": "https://cdn.jsdelivr.net/fontsource/fonts/bebas-neue@latest/latin-400-normal.ttf"
}

# Google Noto Sans Regional Indic TrueType (.ttf) Fonts (Supports all major Indian scripts)
INDIC_FONT_URLS = {
    "hi-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "mr-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "ta-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansTamil/NotoSansTamil-Bold.ttf",
    "te-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Bold.ttf",
    "kn-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansKannada/NotoSansKannada-Bold.ttf",
    "ml-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansMalayalam/NotoSansMalayalam-Bold.ttf",
    "bn-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansBengali/NotoSansBengali-Bold.ttf",
    "gu-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansGujarati/NotoSansGujarati-Bold.ttf",
    "pa-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansGurmukhi/NotoSansGurmukhi-Bold.ttf",
    "od-IN": "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSansOriya/NotoSansOriya-Bold.ttf"
}

SYSTEM_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
]

def _detect_script_lang_code(text: str, declared_lang: str = "en-IN") -> str:
    """Detects Indic Unicode block in text as a safeguard even if declared_lang is en-IN."""
    for ch in (text or ""):
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F: return "hi-IN"  # Devanagari (Hindi/Marathi)
        if 0x0980 <= cp <= 0x09FF: return "bn-IN"  # Bengali
        if 0x0A00 <= cp <= 0x0A7F: return "pa-IN"  # Gurmukhi (Punjabi)
        if 0x0A80 <= cp <= 0x0AFF: return "gu-IN"  # Gujarati
        if 0x0B00 <= cp <= 0x0B7F: return "od-IN"  # Odia
        if 0x0B80 <= cp <= 0x0BFF: return "ta-IN"  # Tamil
        if 0x0C00 <= cp <= 0x0C7F: return "te-IN"  # Telugu
        if 0x0C80 <= cp <= 0x0CFF: return "kn-IN"  # Kannada
        if 0x0D00 <= cp <= 0x0D7F: return "ml-IN"  # Malayalam
    return declared_lang if declared_lang in INDIC_FONT_URLS else "en-IN"

def _get_font(font_name: str, size: int, lang_code: str = "en-IN") -> ImageFont.FreeTypeFont:
    cache_dir = os.path.join(tempfile.gettempdir(), "greeting_fonts_v4")
    os.makedirs(cache_dir, exist_ok=True)

    if lang_code in INDIC_FONT_URLS:
        url = INDIC_FONT_URLS[lang_code]
        local_ttf = os.path.join(cache_dir, f"NotoIndic_{lang_code}.ttf")
    else:
        url = FONT_URLS.get(font_name, FONT_URLS["Montserrat"])
        local_ttf = os.path.join(cache_dir, f"{font_name.replace(' ', '_')}.ttf")

    if not os.path.exists(local_ttf) or os.path.getsize(local_ttf) < 1000:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(local_ttf, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Font download warning for {font_name} ({lang_code}): {e}")

    if os.path.exists(local_ttf) and os.path.getsize(local_ttf) >= 1000:
        try:
            return ImageFont.truetype(local_ttf, size=size)
        except Exception:
            pass

    for sys_font in SYSTEM_FALLBACK_FONTS:
        if os.path.exists(sys_font):
            try:
                return ImageFont.truetype(sys_font, size=size)
            except Exception:
                pass

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _hex_to_rgba(hex_code: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = (hex_code or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

def create_split_overlay_layers(blueprint: dict, card_png_path: str, text_png_path: str) -> str:
    """Renders split Card and Text PNG layers supporting both English and 10 Indian regional scripts."""
    W, H = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    card_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    draw_card = ImageDraw.Draw(card_canvas)
    draw_text = ImageDraw.Draw(text_canvas)

    meta = blueprint.get("composition_metadata", {})
    ctx = blueprint.get("context_analysis", {})
    title = meta.get("overlay_title", "Warm Wishes")
    subtitle = meta.get("overlay_subtitle", "")
    font_family = meta.get("font_family", "Cinzel")
    declared_overlay_lang = ctx.get("overlay_language_code", "en-IN")

    # Auto-detect script in case Indic characters are present
    active_lang = _detect_script_lang_code(f"{title} {subtitle}", declared_overlay_lang)
    is_indic = (active_lang in INDIC_FONT_URLS)

    primary_rgba = _hex_to_rgba(meta.get("primary_color_hex", "#F4A261"), 255)
    accent_rgba = _hex_to_rgba(meta.get("accent_color_hex", "#FFFFFF"), 255)
    card_bg_rgba = _hex_to_rgba(meta.get("card_bg_hex", "#0B132B"), 205)
    border_rgba = _hex_to_rgba(meta.get("primary_color_hex", "#F4A261"), 185)
    pos = meta.get("text_position", "bottom").lower()

    # 1. Outer luxury border frame on Card Layer
    draw_card.rounded_rectangle([36, 36, W - 36, H - 36], radius=32, outline=border_rgba, width=3)

    # 2. Floating Glass Card
    card_w, card_h = int(W * 0.89), int(H * 0.195)
    card_x1 = (W - card_w) // 2
    card_y1 = int(H * 0.06) if pos == "top" else int(H * 0.745)
    card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h

    draw_card.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=38,
        fill=card_bg_rgba,
        outline=primary_rgba,
        width=4
    )

    # 3. Accent Divider Line with central diamond glow on Card Layer
    div_y = card_y1 + int(card_h * 0.48)
    div_half = int(card_w * 0.28)
    draw_card.line([(W // 2 - div_half, div_y), (W // 2 + div_half, div_y)], fill=primary_rgba, width=3)
    draw_card.polygon(
        [(W // 2, div_y - 6), (W // 2 + 8, div_y), (W // 2, div_y + 6), (W // 2 - 8, div_y)],
        fill=primary_rgba
    )

    # 4. Load Fonts (Noto Sans Indic for regional scripts, Luxury Latin for English)
    if is_indic:
        title_font_size = 54 if len(title) <= 20 else 44
        sub_font_size = 40
        wrap_width = 24
    else:
        title_font_size = 66 if len(title) <= 18 else (56 if len(title) <= 24 else 48)
        sub_font_size = 45
        wrap_width = 28

    title_font = _get_font(font_family, title_font_size, active_lang)
    sub_font = _get_font("Montserrat-SemiBold", sub_font_size, active_lang)

    # Draw Headline
    title_y = card_y1 + int(card_h * 0.27)
    draw_text.text((W // 2 + 3, title_y + 4), title, font=title_font, fill=(0, 0, 0, 230), anchor="mm")
    draw_text.text((W // 2, title_y), title, font=title_font, fill=primary_rgba, anchor="mm")

    # Draw Wrapped Subtitle
    wrapped_sub = "\n".join(textwrap.wrap(subtitle, width=wrap_width)[:2])
    sub_y = card_y1 + int(card_h * 0.74)
    draw_text.multiline_text(
        (W // 2 + 2, sub_y + 3),
        wrapped_sub,
        font=sub_font,
        fill=(0, 0, 0, 220),
        anchor="mm",
        align="center",
        spacing=12
    )
    draw_text.multiline_text(
        (W // 2, sub_y),
        wrapped_sub,
        font=sub_font,
        fill=accent_rgba,
        anchor="mm",
        align="center",
        spacing=12
    )

    card_canvas.save(card_png_path, format="PNG")
    text_canvas.save(text_png_path, format="PNG")
    return pos

def render_local_video(
    blueprint: dict,
    local_image_path: str,
    local_tts_path: str,
    audio_duration: float,
    bgm_url: str
) -> str:
    """Renders 1088x1920 MP4 locally via FFmpeg with diagonal camera glide, slide-up card, and boosted audio."""
    temp_dir = tempfile.mkdtemp()
    card_png = os.path.join(temp_dir, "card_layer.png")
    text_png = os.path.join(temp_dir, "text_layer.png")
    pos = create_split_overlay_layers(blueprint, card_png, text_png)

    bgm_path = os.path.join(temp_dir, "bgm.mp3")
    r = requests.get(bgm_url, timeout=45)
    r.raise_for_status()
    with open(bgm_path, "wb") as f:
        f.write(r.content)

    has_tts = bool(local_tts_path and audio_duration > 0)
    reveal_t = config.TEXT_REVEAL_DELAY
    card_start_t = max(0.5, reveal_t - 0.45)

    total_duration = (
        round(reveal_t + audio_duration + config.AUDIO_PADDING_SECONDS, 2)
        if has_tts
        else config.DEFAULT_NO_TTS_DURATION
    )
    total_frames = int(total_duration * config.VIDEO_FPS)
    fade_out_start = max(1.0, total_duration - 1.8)

    raw_name = blueprint.get("context_analysis", {}).get("recipient_name", "Greeting")
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "", raw_name.replace(" ", "_")) or "Greeting"
    out_mp4 = os.path.join(temp_dir, f"Greeting_{safe_slug}.mp4")

    pan_directions = [
        ("iw/2-(iw/zoom/2)+((iw-iw/zoom)*0.18*(on/{F}))", "ih/2-(ih/zoom/2)-((ih-ih/zoom)*0.15*(on/{F}))"),
        ("iw/2-(iw/zoom/2)-((iw-iw/zoom)*0.18*(on/{F}))", "ih/2-(ih/zoom/2)-((ih-ih/zoom)*0.15*(on/{F}))"),
        ("iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)-((ih-ih/zoom)*0.22*(on/{F}))")
    ]
    x_expr_tpl, y_expr_tpl = random.choice(pan_directions)
    x_expr = x_expr_tpl.format(F=total_frames)
    y_expr = y_expr_tpl.format(F=total_frames)

    slide_sign = "-" if pos == "top" else "+"
    card_y_expr = f"if(lt(t,{card_start_t}),H,({slide_sign}120)*pow(max(0,1-(t-{card_start_t})/0.55),2))"
    text_y_expr = f"if(lt(t,{reveal_t}),H,({slide_sign}28)*pow(max(0,1-(t-{reveal_t})/0.50),2))"

    v_filter = (
        f"[0:v]scale=1360:2400,zoompan=z='min(zoom+0.00055,1.16)':"
        f"x='{x_expr}':y='{y_expr}':d={total_frames}:s=1088x1920:fps={config.VIDEO_FPS},"
        f"eq=saturation=1.06:contrast=1.03,fade=t=in:st=0:d=0.8[bg];"
        f"[1:v]format=rgba,fade=t=in:st={card_start_t}:d=0.45:alpha=1[card];"
        f"[2:v]format=rgba,fade=t=in:st={reveal_t}:d=0.45:alpha=1[txt];"
        f"[bg][card]overlay=x=0:y='{card_y_expr}':shortest=1[bg_card];"
        f"[bg_card][txt]overlay=x=0:y='{text_y_expr}':shortest=1,format=yuv420p[vout]"
    )

    bgm_intro_vol = getattr(config, "BGM_INTRO_VOLUME", 0.28)
    bgm_duck_vol = getattr(config, "BGM_DUCKED_VOLUME", 0.14)
    bgm_notts_vol = getattr(config, "BGM_NO_TTS_VOLUME", 0.75)
    tts_boost_vol = getattr(config, "TTS_VOLUME_MULTIPLIER", 1.85)

    if has_tts:
        delay_ms = int(reveal_t * 1000)
        a_filter = (
            f"[3:a]volume='if(lt(t,{card_start_t}),{bgm_intro_vol},{bgm_duck_vol})':eval=frame,"
            f"afade=t=in:st=0:d=1.0,afade=t=out:st={fade_out_start}:d=1.8[bgm];"
            f"[4:a]volume={tts_boost_vol},adelay={delay_ms}|{delay_ms}[tts];"
            f"[bgm][tts]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
            f"alimiter=limit=0.95:attack=5:release=50[aout]"
        )
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-i", local_image_path,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", card_png,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", text_png,
            "-stream_loop", "-1", "-t", str(total_duration), "-i", bgm_path,
            "-i", local_tts_path,
            "-filter_complex", f"{v_filter};{a_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_mp4
        ]
    else:
        a_filter = (
            f"[3:a]volume={bgm_notts_vol},afade=t=in:st=0:d=1.0,"
            f"afade=t=out:st={fade_out_start}:d=1.8[aout]"
        )
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-i", local_image_path,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", card_png,
            "-loop", "1", "-framerate", str(config.VIDEO_FPS), "-t", str(total_duration), "-i", text_png,
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