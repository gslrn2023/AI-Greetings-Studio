import json
import gradio as gr
import config
from ai_director import generate_creative_blueprint
from image_gen import generate_artwork
from media_services import save_and_optionally_upload_image, synthesize_voiceover
from local_renderer import render_local_video
from creatomate_renderer import render_creatomate_video

def format_status_banner() -> str:
    used, remaining = config.get_quota_status()
    return (
        f"📊 **Daily Quota:** `{remaining} / {config.MAX_DAILY_GREETINGS}` remaining today | "
        f"🖼️ **Active Image Engine:** `{config.IMAGE_ENGINE}` | "
        f"🎬 **Default Render Engine:** `{config.RENDER_ENGINE}`"
    )

def run_pipeline(
    passcode: str,
    user_prompt: str,
    style_override: str,
    voice_override: str,
    render_mode_ui: str,
    progress=gr.Progress()
):
    if not config.APP_PASSCODE or passcode.strip() != config.APP_PASSCODE:
        raise gr.Error("🔒 Invalid Access Passcode.")
    if not user_prompt or len(user_prompt.strip()) < 4:
        raise gr.Error("⚠️ Please enter a descriptive greeting request.")

    _, remaining = config.get_quota_status()
    if remaining <= 0:
        raise gr.Error(f"🛑 Daily limit of {config.MAX_DAILY_GREETINGS} greetings reached.")

    active_renderer = render_mode_ui.split(" ")[0].upper()  # "LOCAL", "CREATOMATE", or "OFF"
    need_uguu = active_renderer in ("CREATOMATE", "OFF")

    # Step 1: AI Creative Director
    progress(0.15, desc="🧠 Step 1/4: AI Creative Director designing blueprint & color harmony...")
    blueprint = generate_creative_blueprint(user_prompt.strip(), style_override)

    if not blueprint.get("is_safe", True):
        return None, gr.update(visible=False), "⚠️ Declined by Safety Check.", None, None, json.dumps(blueprint, indent=2), format_status_banner()

    img_prompt = blueprint["visual_prompt"]["image_gen_prompt"]
    tts_script = blueprint["audio_prompt"].get("tts_script", "")
    ai_archetype = blueprint["audio_prompt"].get("voice_archetype", "warm_female")
    bgm_mood = blueprint["audio_prompt"].get("bgm_mood", "festive")
    bgm_url = config.pick_bgm_url(bgm_mood)

    # Step 2: Image Generation (FLUX or Z-Image)
    progress(0.42, desc=f"🎨 Step 2/4: Generating 1088x1920 artwork with {config.IMAGE_ENGINE}...")
    pil_img = generate_artwork(img_prompt)
    local_img_path, image_url = save_and_optionally_upload_image(pil_img, need_remote_url=need_uguu)

    # Step 3: ElevenLabs TTS
    if tts_script and ai_archetype != "none":
        chosen_voice_id = config.resolve_voice_id(voice_override, ai_archetype)
        progress(0.68, desc="🎙️ Step 3/4: Synthesizing ElevenLabs voiceover...")
        local_tts_path, audio_url, audio_dur = synthesize_voiceover(tts_script, chosen_voice_id, need_remote_url=need_uguu)
    else:
        progress(0.68, desc="🕊️ Step 3/4: Solemn occasion — omitting voiceover...")
        local_tts_path, audio_url, audio_dur = "", "", 0.0

    # Step 4: Render Switchboard
    if active_renderer == "OFF":
        progress(1.0, desc="🧪 Test Mode Complete (Creatomate bypassed)!")
        test_links_md = (
            f"### 🧪 Test Mode Asset Links (Zero Render Credits Used)\n"
            f"- 🖼️ **Image URL:** [{image_url}]({image_url})\n"
            f"- 🎙️ **TTS Audio URL:** [{audio_url or 'Muted (Solemn)'}]({audio_url})\n"
            f"- 🎵 **Selected BGM URL ({bgm_mood}):** [{bgm_url}]({bgm_url})\n"
            f"- 🎨 **Color Scheme:** Title `{blueprint['composition_metadata'].get('primary_color_hex')}` | "
            f"Card `{blueprint['composition_metadata'].get('card_bg_hex')}` | "
            f"Placement `{blueprint['composition_metadata'].get('text_position')}`"
        )
        return (
            None,
            gr.update(visible=False),
            test_links_md,
            pil_img,
            local_tts_path if local_tts_path else None,
            json.dumps(blueprint, indent=2),
            format_status_banner()
        )

    elif active_renderer == "CREATOMATE":
        progress(0.85, desc="☁️ Step 4/4: Rendering 1088x1920 MP4 via Creatomate API...")
        local_mp4 = render_creatomate_video(blueprint, image_url, audio_url, audio_dur, bgm_url)
    else:
        progress(0.85, desc="⚡ Step 4/4: Rendering 1088x1920 MP4 locally via Python FFmpeg + Pillow...")
        local_mp4 = render_local_video(blueprint, local_img_path, local_tts_path, audio_dur, bgm_url)

    config.increment_quota()
    progress(1.0, desc="✅ Video Greeting Ready!")

    meta = blueprint.get("composition_metadata", {})
    summary_md = (
        f"✅ **Render Complete ({active_renderer})!** | **First 4.0s:** Pure Unobstructed Artwork | "
        f"**Card Position:** `{meta.get('text_position')}` | **Palette:** `{meta.get('primary_color_hex')}`"
    )

    return (
        local_mp4,
        gr.update(value=local_mp4, visible=True),
        summary_md,
        pil_img,
        local_tts_path if local_tts_path else None,
        json.dumps(blueprint, indent=2),
        format_status_banner()
    )

STYLE_CHOICES = ["Auto (AI Director's Choice)"] + config.CREATIVE_STYLE_LENSES
VOICE_CHOICES = ["Auto (AI Director's Choice)"] + list(config.VOICE_CATALOG.keys())
RENDER_CHOICES = [
    "LOCAL (Free Python FFmpeg - Unlimited)",
    "CREATOMATE (Cloud API)",
    "OFF (Test Mode - Output Image/TTS/BGM URLs Only)"
]

default_render_label = next((c for c in RENDER_CHOICES if c.startswith(config.RENDER_ENGINE)), RENDER_CHOICES[0])

with gr.Blocks(title="AI Greetings AV Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Modular AI Greetings AV Studio (9:16 Vertical)")
    banner = gr.Markdown(value=format_status_banner)

    with gr.Row():
        with gr.Column(scale=5):
            passcode_in = gr.Textbox(label="🔐 Access Passcode", type="password")
            prompt_in = gr.Textbox(
                label="✨ Occasion, Recipient & Personal Context",
                lines=3,
                placeholder="e.g., Wish my friend Vikram a Happy 40th Birthday! He loves acoustic guitars, stargazing, and Himalayan road trips."
            )
            with gr.Row():
                style_in = gr.Dropdown(label="🎨 Visual Art Style", choices=STYLE_CHOICES, value=STYLE_CHOICES[0])
                voice_in = gr.Dropdown(label="🎙️ ElevenLabs Voice (Free Tier)", choices=VOICE_CHOICES, value=VOICE_CHOICES[0])
            render_in = gr.Radio(label="⚙️ Render Engine Switch", choices=RENDER_CHOICES, value=default_render_label)
            gen_btn = gr.Button("🚀 Generate Greeting", variant="primary", size="lg")
            status_md = gr.Markdown("*Ready.*")

        with gr.Column(scale=5):
            video_out = gr.Video(label="📱 Rendered 9:16 Greeting Video (4s Pure Art Intro)", height=500, autoplay=True)
            dl_btn = gr.DownloadButton(label="📥 Download MP4 Video", visible=False, variant="secondary", size="lg")

    with gr.Accordion("🔍 Asset & Blueprint Inspector (Preview Artwork, Audio & JSON)", open=True):
        with gr.Row():
            img_out = gr.Image(label="Generated 1088x1920 Artwork", height=360)
            with gr.Column():
                audio_out = gr.Audio(label="Synthesized ElevenLabs Voiceover Preview", type="filepath")
                json_out = gr.Code(label="AI Director Blueprint (JSON)", language="json")

    gen_btn.click(
        fn=run_pipeline,
        inputs=[passcode_in, prompt_in, style_in, voice_in, render_in],
        outputs=[video_out, dl_btn, status_md, img_out, audio_out, json_out, banner]
    )

if __name__ == "__main__":
    demo.launch()