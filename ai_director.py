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
        f"Return raw JSON blueprint now."
    )

    max_attempts = 2
    last_error = None

    for attempt in range(1, max_attempts + 1):
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
            return json.loads(raw_text)

        except Exception as e:
            last_error = e
            print(f"[AI Director] Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                if status_callback:
                    status_callback("⏳ AI Director is busy, retrying in 30 seconds (Attempt 2/2)...")
                time.sleep(30)

    raise RuntimeError(f"AI Director unavailable after 2 attempts: {last_error}")