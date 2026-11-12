import os
import json
import time
import random
from google import genai
from google.genai import types
import config

def load_system_prompt(filename: str = "AI_prompt.txt") -> str:
    filepath = os.path.join(config.BASE_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"System prompt file '{filepath}' is missing.")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def generate_creative_blueprint(
    user_prompt: str,
    style_override: str = "Auto - AI Choice",
    status_callback=None
) -> dict:
    if not config.GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment secrets.")

    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta")
    )
    base_system_prompt = load_system_prompt()

    sampled_lenses = random.sample(config.CREATIVE_STYLE_LENSES, k=min(3, len(config.CREATIVE_STYLE_LENSES)))
    lenses_text = "\n".join([f"- {lens}" for lens in sampled_lenses])

    if style_override and style_override in config.STYLE_MAP:
        full_style_desc = config.STYLE_MAP[style_override]
        style_directive = (
            f"MANDATORY VISUAL STYLE OVERRIDE: You MUST render this scene strictly in '{full_style_desc}'. "
            f"Begin 'image_gen_prompt' with the exact medium, camera/brush physics, and lighting tokens of this style, "
            f"and adapt all foreground, midground, and background elements to fit this aesthetic."
        )
    else:
        style_directive = f"CREATIVE STYLE SEED (Pick or blend the best fit for the occasion):\n{lenses_text}"

    model_hint = (
        "Optimize prompt for Tongyi-MAI/Z-Image-Turbo: emphasize rich spatial hierarchy, micro-textures, and atmospheric lighting."
        if config.IMAGE_ENGINE == "Z_IMAGE"
        else "Optimize prompt for FLUX.1-schnell: emphasize optical lens characteristics, physical material textures, and volumetric lighting."
    )

    user_message = (
        f"USER REQUEST: {user_prompt}\n"
        f"{style_directive}\n"
        f"MODEL TARGET: {model_hint}\n"
        f"TEXT & VOICEOVER RULES (Adapt tone strictly to occasion—celebratory, health recovery, comforting, or solemn):\n"
        f"1. 'overlay_title': Must be a complete, self-contained 2-to-4 word headline matching the occasion's mood (e.g., 'గృహ ప్రవేశ శుభాకాంక్షలు', 'Get Well Soon', or 'With Deepest Sympathy'). NEVER split a phrase halfway across the title and subtitle.\n"
        f"2. 'overlay_subtitle': Must be a complete, grammatically whole 4-to-6 word message or recipient dedication below the divider line matching the emotional tone (e.g., 'విజయ్ మరియు వనిత గారికి శుభమస్తు', 'Wishing you a speedy recovery', or 'Our thoughts are with you') so it fits cleanly in 1-2 lines.\n"
            f"3. 'tts_script': Naturally include the recipient's name(s) in the spoken voiceover script. Write smooth, flowing sentences for TTS synthesis: avoid comma-separated word lists (use conjunctions like 'మరియు'/'और'/'and' instead of serial commas), and NEVER use ellipses ('...'), semicolons (';'), or mid-sentence dashes ('—'/'-') which cause unnatural TTS pauses.\n"
        f"Return raw JSON blueprint now."
    )

    max_attempts = 2
    last_error = None
    director_trace = {
        "model": config.GEMINI_MODEL,
        "status": "failed",
        "attempts_used": 0,
        "errors": []
    }

    for attempt in range(1, max_attempts + 1):
        director_trace["attempts_used"] = attempt
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=base_system_prompt,
                    temperature=0.88,
                    top_p=0.92,
                    response_mime_type="application/json"
                )
            )
            raw_text = (response.text or "").strip().replace("```json", "").replace("```", "").strip()
            if not raw_text:
                raise RuntimeError("Empty response received from Gemini AI Director.")
            blueprint = json.loads(raw_text)
            director_trace["status"] = "success"
            return blueprint, director_trace

        except Exception as e:
            last_error = e
            err_msg = f"Attempt {attempt}: {str(e)}"
            director_trace["errors"].append(err_msg)
            print(f"[AI Director] {err_msg}")
            if attempt < max_attempts:
                if status_callback:
                    status_callback("⏳ AI Director is busy, retrying in 30 seconds (Attempt 2/2)...")
                time.sleep(30)

    raise RuntimeError(f"AI Director unavailable after 2 attempts: {last_error}")