import json
import random
from google import genai
from google.genai import types
import config

def load_system_prompt() -> str:
    with open("AI_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

def generate_creative_blueprint(user_prompt: str, style_override: str = "Auto (AI Director's Choice)") -> dict:
    if not config.GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in Hugging Face Space Secrets.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    base_system_prompt = load_system_prompt()

    # Sample 3 random creative lenses to break LLM repetition
    sampled_lenses = random.sample(config.CREATIVE_STYLE_LENSES, k=3)
    lenses_text = "\n".join([f"- Option {i+1}: {lens}" for i, lens in enumerate(sampled_lenses)])

    if style_override and style_override != "Auto (AI Director's Choice)":
        style_directive = (
            f"\nUSER STYLE OVERRIDE: You MUST craft the visual_prompt using the '{style_override}' aesthetic."
        )
    else:
        style_directive = (
            f"\nCREATIVE DIVERSITY SEED: To ensure fresh, non-repetitive visual direction, select the most fitting "
            f"aesthetic for this occasion from these 3 curated visual lenses (or blend them creatively):\n{lenses_text}"
        )

    augmented_user_message = (
        f"USER REQUEST: {user_prompt}\n"
        f"{style_directive}\n"
        f"Generate the JSON blueprint now."
    )

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=augmented_user_message,
        config=types.GenerateContentConfig(
            system_instruction=base_system_prompt,
            temperature=0.85,
            top_p=0.92,
            response_mime_type="application/json"
        )
    )

    raw_text = (response.text or "").strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        blueprint = json.loads(raw_text)
        return blueprint
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse AI Director JSON output: {e}\nRaw Output: {raw_text}")