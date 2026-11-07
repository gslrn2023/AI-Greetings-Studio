import gradio as gr
import config
from ai_director import generate_creative_blueprint
from image_gen import generate_artwork
from media_services import save_and_optionally_upload_image, synthesize_voiceover
from local_renderer import render_local_video
from creatomate_renderer import render_creatomate_video

def format_status_banner() -> str:
    _, remaining = config.get_quota_status()
    return (
        f"📊 **Daily Limit:** `{remaining}/{config.MAX_DAILY_GREETINGS}` | "
        f"🖼️ **Active Image Engine:** `{config.IMAGE_ENGINE}` | "
        f"🎬 **Render Engine:** `{config.RENDER_ENGINE}`"
    )

def run_pipeline(
    passcode: str,
    user_prompt: str,
    style_override: str,
    voice_override: str,
    progress=gr.Progress()
):
    if not config.APP_PASSCODE or passcode.strip() != config.APP_PASSCODE:
        raise gr.Error("🔒 Invalid Access Passcode.")
    if not user_prompt or len(user_prompt.strip()) < 4:
        raise gr.Error("⚠️ Please enter a descriptive greeting request.")

    _, remaining = config.get_quota_status()
    if remaining <= 0:
        raise gr.Error(f"🛑 Daily limit of {config.MAX_DAILY_GREETINGS} greetings reached.")

    active_renderer = config.RENDER_ENGINE
    need_uguu = active_renderer in ("CREATOMATE", "OFF")

    try:
        # Step 1: AI Creative Director (with automatic 30s retry if Gemini is busy)
        progress(0.15, desc="Step 1/4: Designing blueprint & color harmony...")
        blueprint = generate_creative_blueprint(
            user_prompt=user_prompt.strip(),
            style_override=style_override,
            status_callback=lambda msg: progress(0.20, desc=msg)
        )

        if not blueprint.get("is_safe", True):
            return None, gr.update(visible=False), "⚠️ Request declined by safety check.", format_status_banner()

        img_prompt = blueprint["visual_prompt"]["image_gen_prompt"]
        tts_script = blueprint["audio_prompt"].get("tts_script", "")
        ai_archetype = blueprint["audio_prompt"].get("voice_archetype", "warm_female")
        bgm_mood = blueprint["audio_prompt"].get("bgm_mood", "festive")
        cultural_context = blueprint.get("context_analysis", {}).get("cultural_context", "global")
        bgm_url = config.pick_bgm_url(bgm_mood)

        # Step 2: Image Generation
        progress(0.42, desc="Step 2/4: Generating artwork ...")
        pil_img = generate_artwork(img_prompt)
        local_img_path, image_url = save_and_optionally_upload_image(pil_img, need_remote_url=need_uguu)

        # Step 3: Voiceover Synthesis (Routes Indian context to Vanishree/Sid, Global to Matilda/Will)
        if tts_script and ai_archetype != "none":
            voice_plan = config.resolve_voice_plan(voice_override, ai_archetype, cultural_context)
            progress(0.68, desc="Step 3/4: Synthesizing voiceover...")
            local_tts_path, audio_url, audio_dur = synthesize_voiceover(tts_script, voice_plan, need_remote_url=need_uguu)
        else:
            progress(0.68, desc="Step 3/4: Synthesizing voiceover...")
            local_tts_path, audio_url, audio_dur = "", "", 0.0

        # Step 4: Video Rendering
        if active_renderer == "OFF":
            progress(1.0, desc="Test Mode Complete!")
            test_links_md = (
                f"### 🧪 Test Mode Asset Links\n"
                f"- 🖼️ **Image URL:** [{image_url}]({image_url})\n"
                f"- 🎙️ **TTS Audio URL:** [{audio_url or 'Muted'}]({audio_url})\n"
                f"- 🎵 **BGM URL:** [{bgm_url}]({bgm_url})\n"
                f"- 🌏 **Cultural Context:** `{cultural_context}`"
            )
            return None, gr.update(visible=False), test_links_md, format_status_banner()

        elif active_renderer == "CREATOMATE":
            progress(0.85, desc="Step 4/4: Rendering Greetings MP4 via Creatomate..")
            local_mp4 = render_creatomate_video(blueprint, image_url, audio_url, audio_dur, bgm_url)
        else:
            progress(0.85, desc="Step 4/4: Rendering Greetings MP4 ..")
            local_mp4 = render_local_video(blueprint, local_img_path, local_tts_path, audio_dur, bgm_url)

        config.increment_quota()
        progress(1.0, desc="✅ Greeting Video Ready!")

        return (
            local_mp4,
            gr.update(value=local_mp4, visible=True),
            "✅ **Your personalized video greeting is ready!**",
            format_status_banner()
        )

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"⚠️ Pipeline Error: {str(e)}")

STYLE_CHOICES = ["Auto - AI Choice"] + list(config.STYLE_MAP.keys())
VOICE_CHOICES = ["Auto - AI Choice", "Female", "Male"]

with gr.Blocks(title="AI Greetings AV Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AI Greetings AV Studio")
    banner = gr.Markdown(value=format_status_banner)

    with gr.Row():
        with gr.Column(scale=5, min_width=320):
            passcode_in = gr.Textbox(label="🔐 Access Passcode", type="password")
            prompt_in = gr.Textbox(
                label="✨ Occasion, Recipient & Personal Context",
                lines=4,
                placeholder="e.g., Wish my friend Vikram a Happy 40th Birthday! He loves acoustic guitars, stargazing, and Himalayan road trips."
            )
            with gr.Row():
                style_in = gr.Dropdown(
                    label="🎨 Visual Art Style",
                    choices=STYLE_CHOICES,
                    value=STYLE_CHOICES[0],
                    min_width=150
                )
                voice_in = gr.Dropdown(
                    label="🎙️ Voice Option",
                    choices=VOICE_CHOICES,
                    value=VOICE_CHOICES[0],
                    min_width=150
                )
            gen_btn = gr.Button("🚀 Generate Greeting", variant="primary", size="lg")
            status_md = gr.Markdown("*Ready.*")

        with gr.Column(scale=5, min_width=320):
            video_out = gr.Video(label="📱 Video Greeting", height=480, autoplay=True)
            dl_btn = gr.DownloadButton(label="📥 Download MP4 Video", visible=False, variant="secondary", size="lg")

    gen_btn.click(
        fn=run_pipeline,
        inputs=[passcode_in, prompt_in, style_in, voice_in],
        outputs=[video_out, dl_btn, status_md, banner]
    )

if __name__ == "__main__":
    demo.launch()