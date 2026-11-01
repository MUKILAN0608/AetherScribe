from continuity import StoryContinuity, validate_scene_text
from llm.prompt_templates import build_prompt
from environment.story_environment import StoryEnvironment
from evaluation.episode_runner import EpisodeRunner
from explainability.explanation_engine import ExplanationEngine
import pytest
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy


class ContinuityRenderer:
    def __init__(self):
        self.states = []
        self.actions = []

    def render_scene(self, state, actions, rationale, story_continuity=None, **_kwargs):
        self.states.append(story_continuity)
        self.actions.append(dict(actions))
        step = state.get("step", 0)
        consequence = (
            "The alarm exposes a hidden route." if step == 0
            else "The hidden route forces Mira to abandon the station entrance. The confrontation changes their options, and Voss accepts the evidence as Nia secures it for the record."
        )
        opening = (
            "Mira enters the station while Voss blocks her route."
            if step == 0 else "Mira reaches the control room while Voss guards the exit."
        )
        return (
            f"[SCENE {step}]\nNARRATIVE:\n"
            f"{opening} "
            f"The conflict changes because {next(iter(actions.values())).replace('_', ' ')}. "
            f"{consequence}\n"
            "[QUANTUM_TRACE]\nThe locked action creates a clear consequence.",
            "Mira's immediate move creates a new obstacle.",
            "continuity prompt",
            0.9,
        )


def test_opening_validation_requires_named_conflict():
    result = validate_scene_text(
        "[SCENE 0]\nNARRATIVE:\nMira entered the station. The lights failed. She ran.\n"
        "[QUANTUM_TRACE]\nThe choice changed the scene.",
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert result.valid is False
    assert "opening_missing_named_conflict" in result.issues


def test_continuity_keeps_scene_summary_separate_from_facts_and_character_knowledge():
    initial = StoryContinuity(
        situation="Mira reaches the vault.",
        protagonist_goal="Mira must recover the archive.",
        unresolved_conflicts=("Voss is blocking the vault.",),
        next_scene_purpose="Establish the immediate conflict.",
    )
    next_state = initial.after_scene(
        "Mira opened the vault but triggered an alarm.",
        {"Protagonist": "Retrieve Artifact", "Antagonist": "Guard Vault"},
        "Advance the established conflict through a concrete consequence.",
    )
    prompt = build_prompt(
        {"genre": "Thriller", "director_brief": {"characters": {
            "protagonist": "Mira", "antagonist": "Voss", "ally": "Nia",
        }}},
        {"Protagonist": "Retrieve Artifact"},
        "The action follows the goal.",
        1,
        story_continuity=next_state,
    )
    assert "Mira opened the vault but triggered an alarm." in prompt
    assert "narrative context, not verified world fact" in prompt
    assert "Retrieve Artifact" in prompt
    assert "Advance the established conflict" in prompt


def test_runner_passes_continuity_between_scenes():
    renderer = ContinuityRenderer()
    brief = {
        "initial_story_premise": "Mira investigates a station while Voss obstructs her.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
    }
    runner = EpisodeRunner(
        StoryEnvironment("thriller", brief, max_steps=2),
        QuantumPolicy(n_qubits=4),
        QuantumActionSpace(),
        ExplanationEngine(),
        renderer,
        memory_enabled=False,
        planner_enabled=False,
        secret_enabled=False,
    )
    result = runner.run_episode(verbose=False, render_depth="fast", seed=2)
    assert len(result["scenes"]) == 2, result["scene_failures"]
    assert renderer.states[0].previous_outcome == ""
    assert renderer.states[1].previous_outcome
    assert renderer.states[1].character_knowledge == {}
    assert renderer.states[1].established_facts == ()
    assert result["scenes"][0]["action"] == renderer.actions[0]
    assert result["scenes"][0]["story"].startswith("[SCENE 1]")
    assert result["scenes"][1]["story"].startswith("[SCENE 2]")
    assert "QUANTUM_TRACE" not in result["scenes"][0]["story"]
    assert result["scenes"][0]["decision_trace"]
    selection = result["scenes"][0]["candidate_selection"]
    assert selection["method"].startswith("stochastic sampling")
    assert selection["valid_candidate_count"] == 16
    assert selection["selected_candidate_id"] == result["scenes"][0]["action_candidate"]["action_id"]
    assert sum(selection["valid_conditioned_posterior"]) == pytest.approx(1.0)
    assert result["narrative_state_version"] == 2
    assert result["world_state"]["version"] == 2
    assert len(result["state_updates"]) == 2
    assert result["world_usda"].startswith("#usda 1.0")
    assert all(scene["coordination"]["accepted"] for scene in result["scenes"])


def test_runner_seeds_authoritative_story_and_character_profiles():
    brief = {
        "initial_story_premise": "Mira searches the station while Voss blocks her.",
        "genre": "drama",
        "characters": {"protagonist": "Mira", "antagonist": "Voss"},
        "character_profiles": {
            "Mira": {"goal": "Find her missing sister", "knowledge": ["The station map is altered"],
                     "location": "Station", "condition": "uninjured"},
        },
        "canonical_narrative_state": {
            "facts": ["The station closes at midnight"],
            "locations": ["Station"],
            "objects": {"map": {"name": "Station map", "condition": "altered"}},
            "active_conflicts": [{"conflict_id": "access", "description": "Voss blocks the records room"}],
            "open_threads": ["Who changed the map?"],
        },
    }
    runner = EpisodeRunner(
        StoryEnvironment("drama", brief, max_steps=1), QuantumPolicy(4), QuantumActionSpace(),
        ExplanationEngine(), ContinuityRenderer(), memory_enabled=False, planner_enabled=False,
        secret_enabled=False,
    )
    state = runner.narrative_coordinator.state
    assert state.facts == ("The station closes at midnight",)
    assert state.character_states["Mira"].goal == "Find her missing sister"
    assert state.character_states["Mira"].knowledge == ("The station map is altered",)
    assert state.character_states["Mira"].location == "Station"
    assert "map" in state.objects
    assert runner.world_state.version == state.version == 0
    assert "Who changed the map?" in runner.world_state.environment["open_threads"]
