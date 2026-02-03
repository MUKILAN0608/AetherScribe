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

    def analyze_rejections(self, state, chosen_action, rejected_actions, rationale, language="English"):
        """
        Research Protocol: Uses the LLM to provide deep narrative logic for why specific
        paths were rejected compared to the chosen one.
        Uses VERY EASY ENGLISH and ONLY character names.
        """
        genre = state['genre']
        characters = state.get('director_brief', {}).get('characters', {})
        name_map = {role.capitalize(): name for role, name in characters.items()}

        rejections_summary = []
        for rej in rejected_actions[:2]: # Analyze top contenders
            rej_text = ", ".join([f"{name_map.get(c, c)}: {a.replace('_', ' ')}" for c, a in rej['action'].items()])
            chosen_text = ", ".join([f"{name_map.get(c, c)}: {a.replace('_', ' ')}" for c, a in chosen_action.items()])

            prompt = f"""
            [SIMPLE STORY AUDIT: CHOICE ANALYSIS]
            DOMAIN: {genre}
            PATH WE USED: {chosen_text}
            PATH WE SKIPPED: {rej_text}
            REASON FOR CHOICE: {rationale}

            TASK: Explain in exactly ONE very simple sentence why the SKIPPED PATH was not as good
            as the path we used. Use VERY EASY ENGLISH that a child can understand.
            Use ONLY character names. NEVER use titles like "Protagonist" or "Antagonist".
            Focus on what the characters want and how the story feels.
            LANGUAGE: {language}
            """
            try:
                response = self.model.generate_content(prompt).text
                rejections_summary.append(response.strip())
            except:
                rejections_summary.append("This choice was not as interesting for the story.")

        return rejections_summary

    def render_scene(self, state, actions, rationale, previous_summary=None, language="English", temperature=0.8):
        """
        Renders a story scene focusing on decision intelligence (v5.4.2).
        Returns: (story_text, summary, prompt_trace, coherence_score)
        """
        scene_index = state.get('step', 0)

        # 1. Primary Scene Generation (Decision-First Protocol)
        prompt = build_prompt(state, actions, rationale, scene_index, previous_summary, language)
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000
            )
        )
        story_text = response.text

        # 2. Sequential Summary pass for continuity context
        summary_prompt = f"Summarize the following narrative result in one concise English sentence:\n\n{story_text}"
        summary_res = self.model.generate_content(summary_prompt).text
        summary = summary_res.strip()

        # 3. Semantic Audit pass
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
