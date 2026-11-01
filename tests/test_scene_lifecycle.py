import pytest

from evaluation.scene_lifecycle import SceneLifecycle
from continuity.quality import validate_scene_output
from llm.story_renderer import StoryRenderer


def test_scene_lifecycle_enforces_retry_path_and_terminal_states():
    lifecycle = SceneLifecycle("scene-0001")
    with pytest.raises(ValueError):
        lifecycle.transition("accepted")
    lifecycle.transition("generating")
    lifecycle.transition("generated")
    lifecycle.transition("validating")
    lifecycle.transition("revision_required", reason="repeated opening")
    lifecycle.transition("retrying")
    lifecycle.transition("generating")
    lifecycle.transition("generated")
    lifecycle.transition("validating")
    lifecycle.transition("accepted")
    with pytest.raises(ValueError):
        lifecycle.transition("generating")


def test_validation_returns_actionable_structured_errors():
    result = validate_scene_output(
        "[SCENE 2]\nNARRATIVE:\n", 1,
        {"protagonist": "Mira"}, [],
    )
    error = next(error for error in result.errors if error.code == "missing_user_facing_content")
    assert error.category == "structure"
    assert error.scene_id == "scene-0002"
    assert error.expected and error.actual and error.repair_suggestion
    assert error.retry_permitted is True


def test_continuation_join_removes_repeated_boundary_words():
    result = StoryRenderer._join_continuation(
        "Mira found the altered ledger beside the door.",
        "beside the door. Voss reached for it before she could turn.",
    )
    assert result == "Mira found the altered ledger beside the door. Voss reached for it before she could turn."


def test_continuation_budget_reserves_room_for_a_bounded_ending():
    prefix = "Mira found the ledger. " * 20
    assert StoryRenderer._continuation_word_budget(prefix, 130) > 0
    assert StoryRenderer._continuation_word_budget(prefix + "word " * 100, 130) == 0


def test_exact_duplicate_sentence_repair_preserves_unique_sentences():
    repaired, removed = StoryRenderer._remove_exact_duplicate_sentences(
        "Voss reached the door. Mira checked the ledger. voss reached the door! Nia found the seal."
    )
    assert removed == 1
    assert repaired == "Voss reached the door. Mira checked the ledger. Nia found the seal."


def test_length_repair_trims_only_at_complete_sentence_boundaries():
    original = "Mira found the altered ledger. Nia confirmed the seal was false. Voss fled through the service exit."
    trimmed, changed = StoryRenderer._trim_to_complete_sentences(original, 14)
    assert changed is True
    assert trimmed == "Mira found the altered ledger. Nia confirmed the seal was false."
    assert trimmed.endswith(".")


def test_renderer_uses_focused_opening_repair_when_retry_opener_is_ignored(monkeypatch):
    from llm.providers import GenerationResult

    class Router:
        provider = type("Provider", (), {"provider_name": "local-endpoint"})()

        def __init__(self):
            self.configs = []
            self.responses = [
                "Mira entered the archive. Voss guarded the ledger. Nia found the loose page.",
                "Nia slid the loose page beneath the glass before Voss could reach it.",
            ]

        def generate(self, _prompt, _config):
            self.configs.append(_config)
            return GenerationResult(
                text=self.responses.pop(0), provider="local-endpoint", model="test",
                latency_ms=1, provider_metadata={"finish_reason": "stop"},
            )

    monkeypatch.setenv("NARRATIVE_FAST_MODE", "false")
    renderer = StoryRenderer.__new__(StoryRenderer)
    renderer.router = Router()
    renderer.scene_latency_ms = renderer.scene_input_tokens = renderer.scene_output_tokens = 0
    renderer.scene_estimated_cost_usd = 0
    state = {
        "step": 1, "genre": "thriller", "tension": 0.5,
        "director_brief": {
            "initial_story_premise": "Mira investigates an archive.",
            "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
        },
    }
    story, *_ = renderer.render_scene(
        state, {"Protagonist": "investigate"}, "", repetition_warning=(
            "RETRY 1 OF 3. REQUIRED: The FIRST WORD of the first sentence MUST be Nia. "
            "The rejected opening was: 'Mira opened the archive door.'"
        ),
    )
    assert story.split("NARRATIVE:\n", 1)[1].startswith("Nia slid the loose page")
    assert renderer.last_provider_metadata["focused_opening_rewrite"] is True
    assert renderer.router.configs[-1].max_output_tokens == 128


def test_renderer_compresses_overlong_retry_without_dropping_scene_contract(monkeypatch):
    from llm.providers import GenerationResult

    class Router:
        provider = type("Provider", (), {"provider_name": "local-endpoint"})()

        def __init__(self):
            self.configs = []
            self.prompts = []
            self.responses = [
                "Mira examines the ledger while Voss watches closely. " * 30,
                "Mira compares the altered ledger with Nia's verified access record. "
                "The mismatch proves Voss changed the index. Nia gives Mira a copy, making the evidence public.",
            ]

        def generate(self, prompt, config):
            self.prompts.append(prompt)
            self.configs.append(config)
            return GenerationResult(
                text=self.responses.pop(0), provider="local-endpoint", model="test",
                latency_ms=1, provider_metadata={"finish_reason": "stop"},
            )

    monkeypatch.setenv("NARRATIVE_FAST_MODE", "false")
    renderer = StoryRenderer.__new__(StoryRenderer)
    renderer.router = Router()
    renderer.scene_latency_ms = renderer.scene_input_tokens = renderer.scene_output_tokens = 0
    renderer.scene_estimated_cost_usd = 0
    state = {
        "step": 7, "genre": "thriller", "tension": 0.8,
        "director_brief": {
            "initial_story_premise": "Mira investigates an altered archive index.",
            "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
        },
    }
    story, *_ = renderer.render_scene(
        state, {"Protagonist": "investigate"}, "", repetition_warning=(
            "RETRY 2 OF 3. Cut the passage to 110 words maximum. Preserve the planned resolution."
        ),
    )
    assert renderer.last_provider_metadata["focused_length_compression"] is True
    assert "REWRITE" in renderer.router.prompts[-1] or "Rewrite the complete scene" in renderer.router.prompts[-1]
    assert renderer.router.configs[-1].max_output_tokens == 220
    assert "mismatch proves" in story
