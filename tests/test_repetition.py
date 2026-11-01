from continuity.repetition import check_repetition
from evaluation.database import AetherDatabase
from environment.story_environment import StoryEnvironment
from evaluation.episode_runner import EpisodeRunner
from explainability.explanation_engine import ExplanationEngine
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy
from llm.story_renderer import StoryRenderer


def _scene(body):
    return f"[SCENE 0]\nNARRATIVE:\n{body}\n[QUANTUM_TRACE]\nTrace."


def test_duplicate_scene_is_detected():
    text = _scene("Mira opens the vault and sees the red archive.")
    result = check_repetition(text, [{"story": text, "opening": "", "beat": ""}])
    assert result.repeated
    assert "duplicate_scene" in result.reasons


def test_repeated_opening_is_detected_but_new_development_can_pass():
    first = _scene("Mira enters the station. She finds a locked door.")
    second = _scene("Mira enters the station. She disables the alarm and reaches the vault.")
    result = check_repetition(
        second,
        [{"story": first, "opening": "mira enters the station", "beat": "old"}],
        action_beat="new",
    )
    assert not result.repeated


def test_repeated_opening_with_same_beat_is_detected():
    first = _scene("Mira enters the station and searches the control room. She finds the same map.")
    second = _scene("Mira enters the station and searches the control room. She finds the same map.")
    result = check_repetition(
        second,
        [{"story": first, "opening": "mira enters the station and searches the control room",
          "beat": "old"}],
        action_beat="new",
    )
    assert "repeated_scene_opening" in result.reasons


def test_repeated_opening_with_changed_tail_is_still_detected():
    first = _scene("Mira reaches out, fingers trembling, and brushes aside the dusty cobweb to reveal a crumbling index card.")
    second = _scene("Mira reaches out, fingers trembling, and brushes aside the dusty cobweb to reveal a crumbling wall.")
    result = check_repetition(second, [{"story": first, "opening": "mira reaches out fingers trembling and brushes aside the dusty cobweb to reveal a crumbling index card"}])
    assert "repeated_scene_opening" in result.reasons


def test_repetition_reports_the_prior_opening_that_matched():
    previous = _scene(
        "Mira entered the archive and found the ledger while Voss watched from beside the door. "
        "Nia counted the pages."
    )
    current = _scene(
        "Mira entered the archive and found a ledger while Voss watched from beside the door. "
        "Nia counted every page twice."
    )
    result = check_repetition(
        current,
        [{"story": previous, "opening": "mira entered the archive and found the ledger while voss watched from beside the door"}],
    )
    assert "repeated_scene_opening" in result.reasons
    assert result.matched_opening.startswith("Mira entered the archive and found the ledger")


def test_repeated_opening_motif_with_rephrased_first_words_is_detected():
    first = _scene("Mira reached out, fingers trembling, and brushed aside a cobweb to reveal a row of dusty, forgotten index cards.")
    second = _scene("Mira's fingers trembled as she brushed aside the cobweb, revealing a row of dusty photographs.")
    result = check_repetition(second, [{"story": first, "opening": "mira reached out fingers trembling and brushed aside a cobweb to reveal a row of dusty forgotten index cards"}])
    assert "repeated_scene_opening" in result.reasons


def test_repeated_narrative_beat_is_detected():
    first = _scene("Mira enters the station and searches the control room.")
    second = _scene("Mira enters the station and searches the control room.")
    result = check_repetition(
        second,
        [{"story": first, "opening": "different", "beat": "search-control-room"}],
        action_beat="search-control-room",
    )
    assert result.repeated
    assert "duplicate_scene" in result.reasons
    assert "repeated_narrative_beat" in result.reasons


def test_repeated_long_sentence_is_detected_across_scenes():
    sentence = "Elias Vance investigates the concealed corporate vault while Director Kael obstructs the inquiry."
    previous = _scene(sentence + " Lyra sends a maintenance record.")
    current = _scene(sentence + " Elias follows the power line to another corridor.")
    result = check_repetition(current, [{"story": previous, "opening": "", "beat": "different"}])
    assert result.repeated
    assert "repeated_sentence" in result.reasons


def test_history_survives_database_restart_and_is_story_scoped(tmp_path):
    url = "sqlite:///" + str(tmp_path / "history.db")
    db = AetherDatabase(url)
    db.save_scene_history(
        story_id="story-a", episode_id="ep-1", scene_index=0,
        story="A", summary="A", outcome="A", opening="a", ending="a",
        beat="beat-a", objective="goal",
    )
    db.engine.dispose()
    restarted = AetherDatabase(url)
    assert len(restarted.list_scene_history("story-a")) == 1
    assert restarted.list_scene_history("story-b") == []
    restarted.engine.dispose()


class RepeatingRenderer:
    def __init__(self):
        self.calls = []

    def render_scene(self, state, actions, **kwargs):
        self.calls.append((dict(actions), kwargs.get("repetition_warning")))
        return (
            f"[SCENE {state['step']}]\nNARRATIVE:\n"
            "Mira enters the station and Voss blocks the door. "
            "The same alarm sounds again.\n[QUANTUM_TRACE]\nTrace.",
            "The same alarm sounds again.",
            "prompt",
            0.8,
        )


def test_runner_bounds_retry_preserves_action_and_rejects_duplicate_history(tmp_path):
    database = AetherDatabase("sqlite:///" + str(tmp_path / "retry.db"))
    prior_story = (
        "[SCENE 0]\nNARRATIVE:\n"
        "Mira enters the station and Voss blocks the door. "
        "The same alarm sounds again.\n[QUANTUM_TRACE]\nTrace."
    )
    database.save_scene_history(
        story_id="story-retry", episode_id="old", scene_index=0,
        story=prior_story, summary="old", outcome="old",
        opening="mira enters the station and voss blocks the door",
        ending="the same alarm sounds again", beat="old", objective="old",
    )
    renderer = RepeatingRenderer()
    brief = {
        "initial_story_premise": "Mira investigates a station while Voss blocks her.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
    }
    runner = EpisodeRunner(
        StoryEnvironment("thriller", brief, max_steps=1),
        QuantumPolicy(n_qubits=4),
        QuantumActionSpace(),
        ExplanationEngine(),
        renderer,
        story_id="story-retry",
        memory_enabled=False,
        planner_enabled=False,
        secret_enabled=False,
        scene_history_repository=database.scene_history,
        repetition_retry_limit=1,
    )
    result = runner.run_episode(verbose=False, render_depth="fast", seed=3)
    assert renderer.calls[0][0] == renderer.calls[1][0]
    assert renderer.calls[1][1]
    assert "REQUIRED: replace the opening sentence" in renderer.calls[1][1]
    assert "Mira enters the station and Voss blocks the door." in renderer.calls[1][1]
    assert result["scenes"] == []
    assert result["status"] == "failed"
    assert result["scene_failures"][0]["retry_count"] == 1
    assert len(database.list_scene_history("story-retry")) == 1
    database.engine.dispose()


def test_failed_scene_blocks_sequential_dependents():
    renderer = RepeatingRenderer()
    brief = {
        "initial_story_premise": "Mira investigates a station while Voss blocks her.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
    }
    result = EpisodeRunner(
        StoryEnvironment("thriller", brief, max_steps=3),
        QuantumPolicy(n_qubits=4), QuantumActionSpace(), ExplanationEngine(), renderer,
        memory_enabled=False, planner_enabled=False, secret_enabled=False,
        repetition_retry_limit=1,
    ).run_episode(verbose=False, render_depth="fast", seed=3)
    assert len(renderer.calls) == 2  # one attempt plus its bounded retry
    assert result["attempted_scene_indices"] == [0]
    assert result["blocked_scene_indices"] == [1, 2]
    assert result["accepted_scene_count"] == 0
    assert result["narrative_state_version"] == 0
    assert result["status"] == "failed"


def test_template_fallback_is_not_counted_as_an_accepted_scene(monkeypatch):
    monkeypatch.setenv("NARRATIVE_LLM_PROVIDER", "deterministic")
    brief = {
        "initial_story_premise": "Mira investigates a vault while Voss obstructs her.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
    }
    result = EpisodeRunner(
        StoryEnvironment("thriller", brief, max_steps=1),
        QuantumPolicy(n_qubits=4),
        QuantumActionSpace(),
        ExplanationEngine(),
        StoryRenderer(),
        memory_enabled=False,
        planner_enabled=False,
        secret_enabled=False,
    ).run_episode(verbose=False, render_depth="fast", seed=9)
    assert result["accepted_scene_count"] == 0
    assert result["story_complete"] is False
    assert result["status"] == "failed"
    assert "template_fallback_not_real_generation" in result["scene_failures"][0]["issues"]
