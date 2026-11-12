import os
import json
import shutil
import threading
from datetime import datetime
import gradio as gr
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
import config
from ai_director import generate_creative_blueprint
from image_gen import generate_artwork
from media_services import save_and_optionally_upload_image, synthesize_voiceover
from local_renderer import render_local_video
from creatomate_renderer import render_creatomate_video

EXECUTION_LOG_FILE = os.path.join(config.BASE_DIR, "execution_logs.json")
_trace_lock = threading.Lock()

def _normalize_repo_id(raw_val: str) -> str:
    """Extracts 'owner/repo' even if a full Hugging Face URL was pasted."""
    val = (raw_val or "").strip().rstrip("/")
    for prefix in ("https://huggingface.co/datasets/", "https://huggingface.co/"):
        if val.startswith(prefix):
            val = val[len(prefix):]
    return val.strip("/")

TRACE_REPO_ID = _normalize_repo_id(os.environ.get("TRACE_REPO_ID", ""))

def _sync_trace_worker(trace_entry: dict) -> None:
    """
    Thread-safe worker: treats the Private HF Dataset as the source of truth.
    If execution_logs.json is manually deleted on HF Dataset, starts a fresh file automatically.
    """
    with _trace_lock:
        history = []
        remote_checked = False

        # 1. Check Private HF Dataset first (handles manual file deletion on HF gracefully)
        if TRACE_REPO_ID and config.HF_TOKEN:
            try:
                downloaded = hf_hub_download(
                    repo_id=TRACE_REPO_ID,
                    filename="execution_logs.json",
                    repo_type="dataset",
                    token=config.HF_TOKEN,
                    force_download=True
                )
                with open(downloaded, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        history = loaded
                remote_checked = True
            except (EntryNotFoundError, RepositoryNotFoundError):
                # File was manually deleted on HF Dataset (or first ever run) -> reset history cleanly
                history = []
                remote_checked = True
            except Exception as e:
                print(f"[Trace Logger] Remote fetch warning, falling back to local cache: {e}")

        # 2. Fallback to local file only if remote dataset wasn't configured or temporarily unreachable
        if not remote_checked and os.path.exists(EXECUTION_LOG_FILE):
            try:
                with open(EXECUTION_LOG_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        history = loaded
            except Exception:
                pass

        # 3. Prepend latest run (keep latest 50 runs) & save locally
        history.insert(0, trace_entry)
        history = history[:50]
        try:
            with open(EXECUTION_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Trace Logger] Local write warning: {e}")

        # 4. Push updated execution_logs.json to Private HF Dataset
        if TRACE_REPO_ID and config.HF_TOKEN:
            try:
                api = HfApi(token=config.HF_TOKEN)
                api.upload_file(
                    path_or_fileobj=EXECUTION_LOG_FILE,
                    path_in_repo="execution_logs.json",
                    repo_id=TRACE_REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Trace log: {trace_entry.get('status', 'run')} ({trace_entry.get('timestamp_ist', '')})"
                )
                print(f"[Trace Logger] Synced execution_logs.json to private dataset '{TRACE_REPO_ID}'.")
            except Exception as e:
                print(f"[Trace Logger] Remote dataset sync warning ({TRACE_REPO_ID}): {e}")

def save_execution_trace(trace_entry: dict) -> None:
    """Fires non-blocking background thread to save and upload trace log."""
    threading.Thread(target=_sync_trace_worker, args=(trace_entry,), daemon=True).start()

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
    accent_override: str,
    progress=gr.Progress()
):
    if not config.APP_PASSCODE or passcode.strip() != config.APP_PASSCODE:
        gr.Warning("🔒 Invalid Access Passcode.")
        return None, gr.update(visible=False), "🔒 **Invalid Access Passcode.**", format_status_banner()

    if not user_prompt or len(user_prompt.strip()) < 4:
        gr.Warning("⚠️ Please enter a descriptive greeting request.")
        return None, gr.update(visible=False), "⚠️ **Please enter a descriptive greeting request.**", format_status_banner()

    _, remaining = config.get_quota_status()
    if remaining <= 0:
        limit_msg = f"🛑 Daily limit of {config.MAX_DAILY_GREETINGS} greetings reached."
        gr.Warning(limit_msg)
        return None, gr.update(visible=False), f"**{limit_msg}**", format_status_banner()

    active_renderer = config.RENDER_ENGINE
    need_uguu = active_renderer in ("CREATOMATE", "OFF")

    local_img_path = ""
    local_tts_path = ""

    # Initialize Black Box Trace Record (includes user_prompt)
    trace_log = {
        "timestamp_ist": datetime.now(config.IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "status": "in_progress",
        "inputs": {
            "user_prompt": user_prompt.strip(),
            "style_override": style_override,
            "voice_override": voice_override,
            "accent_override": accent_override
        },
        "steps": {}
    }

    try:
        # Step 1: AI Creative Director
        progress(0.15, desc="Step 1/4: Designing blueprint & color harmony...")
        blueprint, director_trace = generate_creative_blueprint(
            user_prompt=user_prompt.strip(),
            style_override=style_override,
            status_callback=lambda msg: progress(0.20, desc=msg)
        )
        trace_log["steps"]["1_ai_director"] = director_trace

        if not blueprint.get("is_safe", True):
            trace_log["status"] = "declined_safety"
            save_execution_trace(trace_log)
            return None, gr.update(visible=False), "⚠️ Request declined by safety check.", format_status_banner()

        visual_block = blueprint.get("visual_prompt", {})
        audio_block = blueprint.get("audio_prompt", {})
        ctx_analysis = blueprint.get("context_analysis", {})
        comp_meta = blueprint.get("composition_metadata", {})

        overlay_title = comp_meta.get("overlay_title", "")
        overlay_subtitle = comp_meta.get("overlay_subtitle", "")

        img_prompt = visual_block.get("image_gen_prompt", "").strip()
        if not img_prompt:
            raise RuntimeError("AI Director returned an empty visual prompt. Please try again.")

        tts_script = audio_block.get("tts_script", "")
        ai_archetype = audio_block.get("voice_archetype", "warm_female")
        bgm_mood = audio_block.get("bgm_mood", "festive")

        # Deterministic Accent Routing from UI Dropdown ("Indian" -> Sarvam Tier 1, "Universal" -> ElevenLabs Tier 1)
        cultural_context = "indian" if accent_override == "Indian" else "global"
        tts_lang_code = ctx_analysis.get("tts_language_code", "en-IN")
        if cultural_context == "global" and tts_lang_code == "en-IN":
            tts_lang_code = "en-US"

        bgm_url = config.pick_bgm_url(bgm_mood)
        bgm_filename = bgm_url.split("/")[-1]

        trace_log["recipient"] = ctx_analysis.get("recipient_name", "Greeting")
        trace_log["resolved_context"] = {
            "cultural_context": cultural_context,
            "tts_language_code": tts_lang_code,
            "overlay_language_code": ctx_analysis.get("overlay_language_code", "en-IN"),
            "bgm_mood": bgm_mood,
            "bgm_track": bgm_filename
        }

        # Step 2: Image Generation (logs image_gen_prompt sent to FLUX / Z-Image)
        progress(0.42, desc="Step 2/4: Generating artwork ...")
        pil_img = generate_artwork(img_prompt)
        local_img_path, image_url = save_and_optionally_upload_image(pil_img, need_remote_url=need_uguu)
        trace_log["steps"]["2_image_gen"] = {
            "engine": config.IMAGE_ENGINE,
            "status": "success",
            "image_gen_prompt": img_prompt
        }

        # Step 3: Voiceover Synthesis (logs tts_script and active Sarvam model)
        if tts_script and ai_archetype != "none":
            voice_plan = config.resolve_voice_plan(voice_override, ai_archetype, cultural_context, tts_lang_code)
            progress(0.68, desc="Step 3/4: Synthesizing voiceover...")
            local_tts_path, audio_url, audio_dur, tts_trace = synthesize_voiceover(
                tts_script, voice_plan, need_remote_url=need_uguu
            )
        else:
            progress(0.68, desc="Step 3/4: Synthesizing voiceover...")
            local_tts_path, audio_url, audio_dur = "", "", 0.0
            tts_trace = {
                "status": "muted",
                "used_provider": "Muted",
                "used_voice": "none",
                "duration_sec": 0.0,
                "tts_script": "",
                "waterfall_attempts": []
            }
        trace_log["steps"]["3_voiceover_tts"] = tts_trace

        # Step 4: Video Rendering
        if active_renderer == "OFF":
            progress(1.0, desc="Test Mode Complete!")
            trace_log["steps"]["4_renderer"] = {
                "engine": "OFF",
                "status": "skipped_test_mode",
                "overlay_title": overlay_title,
                "overlay_subtitle": overlay_subtitle
            }
            trace_log["status"] = "success_test_mode"
            save_execution_trace(trace_log)
            test_links_md = (
                f"### 🧪 Test Mode Asset Links\n"
                f"- 🖼️ **Image URL:** [{image_url}]({image_url})\n"
                f"- 🎙️ **TTS:** `{tts_trace['used_provider']} ({tts_trace['used_voice']})`\n"
                f"- 🎵 **BGM:** `{bgm_filename}`"
            )
            return None, gr.update(visible=False), test_links_md, format_status_banner()

        elif active_renderer == "CREATOMATE":
            progress(0.85, desc="Step 4/4: Rendering Greetings MP4 via Creatomate..")
            local_mp4 = render_creatomate_video(blueprint, image_url, audio_url, audio_dur, bgm_url)
            render_trace = {
                "engine": "CREATOMATE",
                "status": "success",
                "overlay_title": overlay_title,
                "overlay_subtitle": overlay_subtitle,
                "bgm_track": bgm_filename
            }
        else:
            progress(0.85, desc="Step 4/4: Rendering Greetings MP4 ..")
            local_mp4, render_trace = render_local_video(blueprint, local_img_path, local_tts_path, audio_dur, bgm_url)

        trace_log["steps"]["4_renderer"] = render_trace
        trace_log["status"] = "success"
        save_execution_trace(trace_log)

        config.increment_quota()
        progress(1.0, desc="✅ Greeting Video Ready!")

        # Mobile-friendly concise icon-only telemetry line
        voice_badge = (
            f"{tts_trace['used_provider']} ({tts_trace['used_voice']})"
            if tts_trace["used_provider"] != "Muted"
            else "Muted"
        )
        summary_line = f"✅ **Video Ready!** | 🎙️ `{voice_badge}` | 🎵 `{bgm_filename}`"

        return (
            local_mp4,
            gr.update(value=local_mp4, visible=True),
            summary_line,
            format_status_banner()
        )

    except Exception as e:
        err_msg = f"⚠️ Pipeline Error: {str(e)}"
        trace_log["status"] = "error"
        trace_log["fatal_error"] = str(e)
        save_execution_trace(trace_log)
        gr.Warning(err_msg)
        return None, gr.update(visible=False), f"**{err_msg}**", format_status_banner()
    finally:
        for tmp_asset in (local_img_path, local_tts_path):
            if tmp_asset and os.path.exists(tmp_asset):
                try:
                    shutil.rmtree(os.path.dirname(tmp_asset), ignore_errors=True)
                except OSError:
                    pass

STYLE_CHOICES = ["Auto - AI Choice"] + list(config.STYLE_MAP.keys())
VOICE_CHOICES = ["Auto - AI Choice", "Female", "Male"]
ACCENT_CHOICES = ["Indian", "Universal"]

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
                    label="🎨 Art Style",
                    choices=STYLE_CHOICES,
                    value=STYLE_CHOICES[0],
                    min_width=130
                )
                voice_in = gr.Dropdown(
                    label="🎙️ Voice",
                    choices=VOICE_CHOICES,
                    value=VOICE_CHOICES[0],
                    min_width=110
                )
                accent_in = gr.Dropdown(
                    label="🌏 Accent",
                    choices=ACCENT_CHOICES,
                    value="Indian",
                    min_width=110
                )
            gen_btn = gr.Button("🚀 Generate Greeting", variant="primary", size="lg")
            status_md = gr.Markdown("*Ready.*")

        with gr.Column(scale=5, min_width=320):
            video_out = gr.Video(label="📱 Video Greeting", height=480, autoplay=True)
            dl_btn = gr.DownloadButton(label="📥 Download MP4 Video", visible=False, variant="secondary", size="lg")

    gen_btn.click(
        fn=run_pipeline,
        inputs=[passcode_in, prompt_in, style_in, voice_in, accent_in],
        outputs=[video_out, dl_btn, status_md, banner],
        show_progress_on=video_out
    )

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    share_flag = os.environ.get("GRADIO_SHARE", "false").strip().lower() in ("true", "1", "yes")
    demo.launch(server_name="0.0.0.0", server_port=server_port, share=share_flag)