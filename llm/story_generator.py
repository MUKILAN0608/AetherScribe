"""
story_generator.py

LLM Narrative Director interface.
Responsible for rendering RL decisions into narrative text following genre-specific
constraints and the Director role protocol.
"""

from genre_adapters.base_adapter import GenreAdapter
from evaluation.metrics import NarrativeEvaluator

class NarrativeDirector:
    """
    Acts as the Narrative Director, rendering agent actions into text.
    """

    def __init__(self, genre_adapter: GenreAdapter):
        self.genre = genre_adapter
        self.evaluator = NarrativeEvaluator()

    def generate_scene(self, joint_action: dict, state: dict) -> str:
        """
        In a real system, this would call an LLM API.
        For this research prototype, we implement the Director logic
        with strict template-driven formatting to ensure zero hallucination
        of actions while maintaining genre tone.
        """

        # 1. Construct the internal 'Director Instruction' (Hidden from user)
        # This would be the system prompt for a real LLM.
        _instructions = (
            f"STORY PHASE: {state['phase']}\n"
            f"TENSION LEVEL: {state['tension']}\n"
            f"{self.genre.get_prompt_context()}\n"
            f"AGENT ACTIONS: {joint_action}\n"
        )

        # 2. Render Scene (Prototype rendering logic)
        # This simulates the LLM following the 'Director' role perfectly.

        scene_content = self._prototype_render(joint_action, state)
        director_note = self._generate_director_note(joint_action, state)

        full_output = (
            f"[STORY_SCENE]\n{scene_content}\n\n"
            f"[DIRECTOR_NOTE]\n{director_note}"
        )

        # 3. Automatic Violation Detection
        evaluation = self.evaluator.evaluate_step(
            scene_content, joint_action, self.genre.constraints
        )

        if evaluation["violations"]:
            full_output += f"\n\n[SYSTEM_WARNING] Violations detected: {', '.join(evaluation['violations'])}"

        return full_output

    def _prototype_render(self, action: dict, state: dict) -> str:
        """
        Simulates the LLM's narrative rendering while strictly following actions.
        """
        p_act = action['protagonist']
        a_act = action['antagonist']
        l_act = action['ally']

        if self.genre.name == "Romance":
            return (
                f"In the soft glow of the evening, the protagonist {p_act}s the shared moment, "
                f"their heart racing. Across the room, the antagonist {a_act}s, casting a "
                f"complex shadow over the blooming connection. The ally {l_act}s, offering "
                f"a supportive glance that speaks volumes in the silence."
            )
        elif self.genre.name == "Drama":
            return (
                f"The weight of the situation is heavy as the protagonist {p_act}s the truth. "
                f"Standing their ground, the antagonist {a_act}s, refuse to yield to the "
                f"mounting pressure. Watching from the sidelines, the ally {l_act}s, caught "
                f"between loyalty and the harsh reality of the choice."
            )
        elif self.genre.name == "Thriller":
            return (
                f"Time is running out as the protagonist {p_act}s the narrow escape. "
                f"Somewhere in the darkness, the antagonist {a_act}s, their presence a "
                f"constant, suffocating threat. The ally {l_act}s, their voice a frantic "
                f"whisper over the pounding of their pulse."
            )
        elif self.genre.name == "Action / Adventure":
            return (
                f"With a surge of adrenaline, the protagonist {p_act}s the daring maneuver. "
                f"Leaping into the fray, the antagonist {a_act}s, determined to halt the "
                f"hero's progress at any cost. The ally {l_act}s, providing critical cover "
                f"as the journey reaches a fever pitch."
            )
        elif self.genre.name == "Mystery":
            return (
                f"Sifting through the fragments, the protagonist {p_act}s the critical evidence. "
                f"In the background, the antagonist {a_act}s, leaving a trail of questions "
                f"in their wake. The ally {l_act}s, pointing toward a detail that others "
                f"might have overlooked."
            )
        elif self.genre.name == "Comedy":
            return (
                f"In a series of unfortunate events, the protagonist {p_act}s the hilarious "
                f"misunderstanding. The antagonist {a_act}s, their plans thwarted by "
                f"sheer absurdity. Grinning at the chaos, the ally {l_act}s, adding "
                f"fuel to the fire with a well-timed quip."
            )

        return f"Phase: {state['phase']}. Protagonist: {p_act}. Antagonist: {a_act}. Ally: {l_act}."

    def _generate_director_note(self, action: dict, state: dict) -> str:
        return (
            f"The scene follows the RL-selected actions ({', '.join(action.values())}) "
            f"within the {self.genre.name} genre framework. Pacing is set to {state['phase']} "
            f"mode to maintain consistent narrative progression."
        )
