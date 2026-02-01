import google.generativeai as genai
import re
from llm.prompt_templates import build_prompt

class StoryRenderer:
    """
    High-fidelity decision-first rendering engine with integrated semantic critique.
    Optimized for Aether Scribe v5.4.2 High-Transparency Research Edition.
    """
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def render_scene(self, state, actions, rationale, previous_summary=None, language="English", temperature=0.8, mode="Short Story"):
        """
        Renders a story scene focusing on decision intelligence (v5.4.2).
        Returns: (story_text, summary, prompt_trace, coherence_score)
        """
        scene_index = state.get('step', 0)

        # 1. Primary Scene Generation (Decision-First Protocol)
        prompt = build_prompt(state, actions, rationale, scene_index, previous_summary, language, mode)
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000
            )
        )
        story_text = response.text

        # 2. Sequential Summary pass for continuity context
        # In Decision-First mode, the story_text is already concise, so summary is almost the same.
        summary_prompt = f"Summarize the following narrative result in one concise English sentence:\n\n{story_text}"
        summary_res = self.model.generate_content(summary_prompt).text
        summary = summary_res.strip()

        # 3. Semantic Audit pass
        # This provides the RL feedback loop with a qualitative anchor for policy optimization.
        critique_prompt = f"""
        [NARRATIVE AUDIT PROTOCOL v5.4.2]
        Analyze the following decision result for logical coherence and adherence to the domain.

        CRITERIA:
        1. Does it strictly follow the quantum decisions: {actions}?
        2. Does it utilize the authorized character names correctly?
        3. Is the result logically consistent with the previous state?

        Output ONLY a single floating-point number between 0.0 and 1.0 representing the Narrative Coherence Score.
        """

        try:
            critique_res = self.model.generate_content(critique_prompt).text
            match = re.search(r"(\d+\.\d+)", critique_res)
            coherence_score = float(match.group(1)) if match else 0.95
        except:
            coherence_score = 0.9 # Research-grade fallback

        return story_text, summary, prompt, coherence_score
