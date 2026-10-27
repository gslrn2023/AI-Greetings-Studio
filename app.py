import json
import spaces
import torch
from diffusers import FluxPipeline
import gradio as gr

import config
from ai_director import generate_creative_blueprint
from media_services import upload_pil_image, generate_tts_and_upload
from render_engine import render_greeting_video

# =====================================================================
# 1. LOAD FLUX.1-SCHNELL MODEL FOR ZEROGPU
# =====================================================================
print("Loading FLUX.1-schnell model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    token=config.HF_TOKEN
)
pipe.to("cuda")
print("FLUX.1-schnell loaded successfully!")

@spaces.GPU(duration=60)
def generate_flux_image(prompt: str):
    """Generates a 9:16 vertical image on ZeroGPU and returns a PIL Image."""
    image = pipe(
        prompt=prompt,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        num_inference_steps=config.FLUX_STEPS,
        guidance_scale=0.0
    ).images[0]
    return image

# =====================================================================
# 2. MAIN END-TO-END ORCHESTRATION PIPELINE
# =====================================================================
def format_quota_badge() -> str:
    used, remaining = config.get_quota_status()
    return f"📊 **Daily Quota:** `{remaining} / {config.MAX_DAILY_GREETINGS}` greetings remaining today (Used: {used})"

def create_greeting_av(
    passcode: str,
    user_prompt: str,
    style_override: str,
    progress=gr.Progress()
):
    # 1. Passcode Verification (Zero API/GPU cost if unauthorized)
    if not passcode or passcode.strip() != config.APP_PASSCODE:
        raise gr.Error("🔒 Invalid Access Passcode. Please enter the valid passcode to generate a greeting.")

    # 2. Input Validation
    if not user_prompt or len(user_prompt.strip()) < 4:
        raise gr.Error("⚠️ Please enter a descriptive greeting request (e.g., 'Happy Birthday Rahul, loves cycling and mountains').")

    # 3. Daily Rate Limit Check
    used, remaining = config.get_quota_status()
    if remaining <= 0:
        raise gr.Error(f"🛑 Daily limit of {config.MAX_DAILY_GREETINGS} greetings reached. Quota resets at midnight IST.")

    # 4. Step 1: AI Creative Director (Gemini)
    progress(0.15, desc="🧠 Step 1/4: AI Creative Director designing visual & audio blueprint...")
    blueprint = generate_creative_blueprint(user_prompt.strip(), style_override)

    # Safety Gate Check
    if not blueprint.get("is_safe", True):
        return (
            None,
            gr.update(value=None, visible=False),
            "⚠️ Request declined by Safety Check. Please keep prompts respectful.",
            None,
            json.dumps(blueprint, indent=2),
            format_quota_badge()
        )

    image_prompt = blueprint["visual_prompt"]["image_gen_prompt"]
    tts_script = blueprint["audio_prompt"].get("tts_script", "")
    voice_archetype = blueprint["audio_prompt"].get("voice_archetype", "warm_female")

    # 5. Step 2: FLUX.1-schnell 9:16 Image Generation on ZeroGPU
    progress(0.40, desc="🎨 Step 2/4: Generating 9:16 vertical artwork with FLUX.1-schnell...")
    pil_image = generate_flux_image(image_prompt)
    image_url = upload_pil_image(pil_image)

    # 6. Step 3: ElevenLabs Voiceover Synthesis (Skipped automatically for solemn occasions)
    if tts_script and voice_archetype != "none":
        progress(0.65, desc=f"🎙️ Step 3/4: Synthesizing '{voice_archetype}' voiceover via ElevenLabs...")
        audio_url, audio_duration = generate_tts_and_upload(tts_script, voice_archetype)
    else:
        progress(0.65, desc="🕊️ Step 3/4: Solemn occasion detected — omitting voiceover and using gentle BGM...")
        audio_url, audio_duration = "", 0.0

    # 7. Step 4: Creatomate Template-Free MP4 Render
    progress(0.85, desc="🎬 Step 4/4: Rendering 1080x1920 MP4 video in Creatomate...")
    remote_video_url, local_mp4_path = render_greeting_video(
        blueprint=blueprint,
        image_url=image_url,
        audio_url=audio_url,
        audio_duration=audio_duration
    )

    # Increment daily usage quota only after successful generation
    config.increment_quota()
    progress(1.0, desc="✅ Greeting Video Ready!")

    status_summary = (
        f"✅ **Render Complete!** | **Style:** {blueprint['visual_prompt'].get('selected_art_style')} | "
        f"**Voice:** {voice_archetype} | **BGM:** {blueprint['audio_prompt'].get('bgm_mood')}"
    )

    return (
        local_mp4_path,
        gr.update(value=local_mp4_path, visible=True),
        status_summary,
        pil_image,
        json.dumps(blueprint, indent=2),
        format_quota_badge()
    )

# =====================================================================
# 3. GRADIO UI LAYOUT
# =====================================================================
STYLE_CHOICES = [
    "Auto (AI Director's Choice)",
    "35mm Analog Editorial Photography",
    "Sculptural 3D Papercraft & Origami Art",
    "Luminous Gouache & Watercolor Illustration",
    "Atmospheric Magical Realism",
    "Luxury Still-Life Studio Photography",
    "Vibrant Stained-Glass & Light Art",
    "Fine-Art Oil Impasto Painting"
]

with gr.Blocks(title="Greetings AV Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 AI Greetings AV Studio (9:16 Vertical Shorts)
        Transform any occasion into a bespoke vertical video greeting powered by **Gemini Creative Director**, **FLUX.1-schnell**, **ElevenLabs**, and **Creatomate RenderScript**.
        """
    )

    quota_banner = gr.Markdown(value=format_quota_badge)

    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=5):
            passcode_input = gr.Textbox(
                label="🔐 Access Passcode",
                type="password",
                placeholder="Enter your secret passcode..."
            )
            prompt_input = gr.Textbox(
                label="✨ Occasion, Recipient & Personal Details",
                lines=4,
                placeholder="Example: Wish my friend Vikram a Happy 40th Birthday! He loves acoustic guitars, stargazing, and Himalayan road trips."
            )
            style_dropdown = gr.Dropdown(
                label="🎨 Visual Art Style",
                choices=STYLE_CHOICES,
                value="Auto (AI Director's Choice)"
            )
            generate_btn = gr.Button("🚀 Generate Greeting Video", variant="primary", size="lg")
            status_box = gr.Markdown(value="*Ready to generate.*")

        # Right Column: Video Output & Download
        with gr.Column(scale=5):
            video_output = gr.Video(
                label="📱 Rendered 9:16 Greeting Video",
                height=520,
                autoplay=True
            )
            download_btn = gr.DownloadButton(
                label="📥 Download MP4 Video (Share to WhatsApp / Instagram)",
                visible=False,
                variant="secondary",
                size="lg"
            )

    with gr.Accordion("🔍 Creative Director Inspector (Preview Image & AI JSON Blueprint)", open=False):
        with gr.Row():
            image_preview = gr.Image(label="Generated 9:16 FLUX Artwork", height=400)
            json_inspector = gr.Code(label="AI Director Blueprint (JSON)", language="json")

    generate_btn.click(
        fn=create_greeting_av,
        inputs=[passcode_input, prompt_input, style_dropdown],
        outputs=[
            video_output,
            download_btn,
            status_box,
            image_preview,
            json_inspector,
            quota_banner
        ]
    )

if __name__ == "__main__":
    demo.launch()