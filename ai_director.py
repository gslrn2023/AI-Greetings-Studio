import json
import time
import random
from google import genai
from google.genai import types
import config

def load_system_prompt() -> str:
    with open("AI_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

def generate_creative_blueprint(
    user_prompt: str,
    style_override: str = "Auto - AI Choice",
    status_callback=None
) -> dict:
    if not config.GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in Hugging Face Space Secrets.")

    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta")
    )
    base_system_prompt = load_system_prompt()

    sampled_lenses = random.sample(config.CREATIVE_STYLE_LENSES, k=3)
    lenses_text = "\n".join([f"- {lens}" for lens in sampled_lenses])

    if style_override and style_override in config.STYLE_MAP:
        full_style_desc = config.STYLE_MAP[style_override]
        style_directive = f"USER STYLE OVERRIDE: Craft the visual_prompt strictly in the '{full_style_desc}' style."
    else:
        style_directive = f"CREATIVE STYLE SEED (Pick or blend the best fit):\n{lenses_text}"

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
            return json.loads(raw_text)

        except Exception as e:
            last_error = e
            print(f"[AI Director] Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                if status_callback:
                    status_callback("⏳ AI Director is busy, retrying in 30 seconds (Attempt 2/2)...")
                time.sleep(30)

    raise RuntimeError(f"AI Director unavailable after 2 attempts: {last_error}")