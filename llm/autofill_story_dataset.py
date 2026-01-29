import json
import os
from google import genai

# ======================================================
# CONFIGURATION
# ======================================================

INPUT_JSON = "story_director_dataset_600.json"
OUTPUT_JSON = "story_director_dataset_600_filled.json"

MODEL_NAME = "gemini-2.5-flash"   # ✅ correct model naming

MAX_NEW_TOKENS = 350
TEMPERATURE = 0.35
TOP_P = 0.9

# ======================================================
# API KEY (SAFE PRACTICE)
# ======================================================

# ⚠️ DO NOT hardcode API keys in real projects
# Set once in PowerShell:
# setx GEMINI_API_KEY "your_key_here"

GEMINI_API_KEY ="AIzaSyCOdCbxNjZVwpi4sKwZ9Hd041Dr0-5zB3w"

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================================================
# DIRECTOR PROMPT
# ======================================================

DIRECTOR_PROMPT = """
You are a Narrative Director AI.

All narrative decisions are already made.
You must NOT invent new plot logic.

STRICT RULES:
- Follow character actions EXACTLY
- Follow genre constraints STRICTLY
- Do NOT introduce new characters
- Do NOT resolve the story
- Do NOT add new events

OUTPUT FORMAT (MANDATORY):

[STORY_SCENE]
1–3 short paragraphs.

[DIRECTOR_NOTE]
One paragraph explaining how the scene follows
the actions, genre, and story phase.
"""

print("✅ Connected to Gemini.")

# ======================================================
# PROMPT BUILDER
# ======================================================

def build_prompt(sample):
    agents = sample["input"]["agents"]

    return f"""
{DIRECTOR_PROMPT}

GENRE: {sample['input']['genre']}
STORY PHASE: {sample['input']['phase']}
TENSION: {sample['input']['tension']}

SELECTED JOINT ACTIONS:
- Protagonist: {agents['protagonist']}
- Antagonist: {agents['antagonist']}
- Ally: {agents['ally']}
"""

# ======================================================
# GENERATION FUNCTION (NEW GEMINI SDK)
# ======================================================

def generate_scene(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_NEW_TOKENS,
        },
    )
    return response.text.strip()


# ======================================================
# MAIN
# ======================================================

def main():
    print("Loading dataset...")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    filled = []
    total = len(dataset)

    print(f"Generating story text for {total} samples...\n")

    for i, sample in enumerate(dataset, start=1):
        prompt = build_prompt(sample)
        generated = generate_scene(prompt)

        # Retry once if format breaks
        if "[STORY_SCENE]" not in generated or "[DIRECTOR_NOTE]" not in generated:
            print(f"Format issue at sample {i}, regenerating...")
            generated = generate_scene(prompt)

        try:
            scene = generated.split("[STORY_SCENE]")[1].split("[DIRECTOR_NOTE]")[0].strip()
            note = generated.split("[DIRECTOR_NOTE]")[1].strip()
        except Exception:
            scene = ""
            note = ""

        sample["output"]["story_scene"] = scene
        sample["output"]["director_note"] = note
        filled.append(sample)

        if i % 25 == 0:
            print(f"{i}/{total} samples generated")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(filled, f, indent=2)

    print("\n✅ DONE")
    print(f"Saved filled dataset to: {OUTPUT_JSON}")

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    main()
