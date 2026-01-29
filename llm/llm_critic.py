# llm_critic.py
import json
import google.generativeai as genai
from llm.critic_prompt import CRITIC_PROMPT

# --- Gemini client setup (reuse pattern from story_generator.py) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

# --- Critic function ---
def run_llm_critic(state, actions, story_text, model_name=MODEL_NAME, temperature=0.15):
    """
    Evaluate a generated story scene using Gemini LLM.
    Args:
        state: dict with keys 'genre', 'phase', 'tension'
        actions: dict with keys 'protagonist', 'antagonist', 'ally'
        story_text: str, the generated story scene
        model_name: str, Gemini model name
        temperature: float, LLM temperature (default 0.15)
    Returns:
        dict with evaluation fields (see critic_prompt.py)
    """
    prompt = build_critic_prompt(state, actions, story_text)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=temperature,
                top_p=1.0
            )
        )
        result = try_parse_json(response.text)
        if result is not None:
            return result
    except Exception as e:
        pass
    # Fallback: return default structure with error note
    return {
        "action_adherence": 0.0,
        "genre_consistency": 0.0,
        "coherence": 0.0,
        "violations_detected": True,
        "overall_quality": 0.0,
        "brief_explanation": "Evaluation failed or invalid JSON returned."
    }

def build_critic_prompt(state, actions, story_text):
    return f"""{CRITIC_PROMPT}\n\nGENRE: {state.get('genre','')}
PHASE: {state.get('phase','')}
TENSION: {state.get('tension','')}
AGENT ACTIONS:\n- Protagonist: {actions.get('protagonist','')}
- Antagonist: {actions.get('antagonist','')}
- Ally: {actions.get('ally','')}
\nSTORY SCENE TO EVALUATE:\n{story_text}\n"""

def try_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON substring
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None

# --- Minimal usage example ---
if __name__ == "__main__":
    # Example state, actions, and story_text
    state = {"genre": "mystery", "phase": "climax", "tension": 0.8}
    actions = {"protagonist": "investigate", "antagonist": "sabotage", "ally": "warn"}
    story_text = "[STORY_SCENE] The protagonist sneaks into the dark lab... [DIRECTOR_NOTE] This scene follows..."
    result = run_llm_critic(state, actions, story_text)
    print(json.dumps(result, indent=2))
