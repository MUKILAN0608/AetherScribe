import json

from evaluation.database import AetherDatabase
from evaluation.episode_runner import EpisodeRunner
from environment.story_environment import StoryEnvironment
from explainability.explanation_engine import ExplanationEngine
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy
from story_secrets import RevealConditions, SecretRecord


class LeakingRenderer:
    provider = "fixture"
    model = "leak-fixture"
    last_latency_ms = 0.1
    last_input_tokens = None
    last_output_tokens = None
    last_estimated_cost_usd = None
    last_used_fallback = False
    captured_action = None

    def render_scene(
        self, state, actions, rationale, previous_summary=None, language="English",
        temperature=0.8, mode="Short Story", lite_mode=False,
        memory_context=None, planning_guidance=None, secret_guidance=None,
    ):
        self.captured_action = dict(actions)
        return (
            f"[SCENE {state['step']}]\nNARRATIVE:\n"
            f"The protected coordinate is 7-14-21; actions remain {dict(actions)}.\n\n"
            "[QUANTUM_TRACE]\nFixture trace.",
            "Fixture summary.",
            "Fixture prompt.",
            0.9,
        )


def test_early_secret_leak_fails_after_retry_without_accepting_prose(tmp_path):
    database = AetherDatabase("sqlite:///" + str(tmp_path / "secret-pipeline.db"))
    database.add_secret(
        SecretRecord(
            "story-secret",
            "coordinate",
            "7-14-21",
            conditions=RevealConditions(min_scene=5),
        )
    )
    brief = {
        "initial_story_premise": "A sealed observatory protects a coordinate.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
    }
    renderer = LeakingRenderer()
    runner = EpisodeRunner(
        StoryEnvironment("thriller", brief, max_steps=1),
        QuantumPolicy(n_qubits=4),
        QuantumActionSpace(),
        ExplanationEngine(),
        renderer,
        memory_enabled=False,
        secret_repository=database.secrets,
        secret_enabled=True,
        story_id="story-secret",
    )
    result = runner.run_episode(verbose=False, render_depth="fast", seed=1)
    assert len(result["scenes"]) == 1
    assert result["scenes"][0]["generation_status"] == "generated_provisional"
    assert result["scenes"][0]["quality_gate"]["accepted"] is False
    assert result["status"] == "generated"
    assert "protected_secret_leak" in result["scene_failures"][0]["issues"]
    assert renderer.captured_action
    assert database.list_scene_history("story-secret") == []
    database.engine.dispose()


def test_secret_gate_disabled_still_fails_closed(tmp_path):
    database = AetherDatabase("sqlite:///" + str(tmp_path / "secret-disabled.db"))
    database.add_secret(
        SecretRecord("story-disabled", "key", "ALPHA-9")
    )
    assert database.get_secret("other-story", "key") is None
    database.engine.dispose()
