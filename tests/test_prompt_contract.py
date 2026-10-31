from continuity import StoryContinuity
from llm.prompt_templates import build_compact_local_prompt, build_prompt


def _state():
    return {
        "genre": "Thriller",
        "phase": "rising",
        "tension": 0.62,
        "director_brief": {
            "initial_story_premise": "Mira investigates a sealed vault while Voss obstructs her.",
            "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
        },
    }


def test_short_prompt_contains_all_authoritative_context():
    continuity = StoryContinuity(
        situation="The alarm has trapped Mira inside the vault.",
        protagonist_goal="Mira must recover the archive.",
        established_facts=("The outer door is sealed.",),
        character_knowledge={"Mira": ("The alarm is active.",)},
        unresolved_conflicts=("Voss controls the exit.",),
        previous_outcome="Nia's warning gave Mira one route forward.",
        next_scene_purpose="Force a costly choice at the inner door.",
    )
    prompt = build_prompt(
        _state(),
        {"Protagonist": "investigate_technical_anomaly", "Antagonist": "obstruct_vital_information"},
        "The action follows Mira's goal and increases pressure.",
        3,
        language="English",
        mode="Short Story",
        planning_guidance=("Show the immediate consequence before a new reveal.",),
        secret_guidance=("The vault contents remain concealed.",),
        story_continuity=continuity,
        repetition_warning="Do not repeat the sealed-door opening.",
    )

    required = (
        "SCENE: 4", "Mira", "Voss", "Nia", "investigate technical anomaly",
        "obstruct vital information", "The alarm has trapped Mira inside the vault.",
        "Force a costly choice at the inner door.", "Show the immediate consequence",
        "The vault contents remain concealed.", "Do not repeat the sealed-door opening.",
        "<3–4 sentences; minimum 120 words total; no blank lines>",
    )
    assert all(value in prompt for value in required)
    assert "<5–8 sentences; one sentence per line; minimum 180 words total" not in prompt


def test_long_prompt_has_structured_output_sections():
    prompt = build_prompt(_state(), {"Ally": "provide_critical_intelligence"}, "Support opens a route.", 1, mode="Long Story")
    assert "[SCENE 1]" in prompt
    assert "NARRATIVE:" in prompt
    assert "[QUANTUM_TRACE]" in prompt
    assert "[COHERENCE]" in prompt
    assert "<5–8 sentences; one sentence per line; minimum 180 words total; no blank lines>" in prompt


def test_compact_local_prompt_tracks_scene_outline_and_one_based_number():
    first = build_compact_local_prompt(
        _state(),
        {"Protagonist": "investigate_technical_anomaly"},
        0,
        planning_guidance=("SCENE PURPOSE: Verify the vault using facility records", "NEW INFORMATION: A maintenance ledger shows power use"),
    )
    second = build_compact_local_prompt(
        _state(),
        {"Protagonist": "investigate_technical_anomaly"},
        1,
        planning_guidance=("SCENE PURPOSE: Reach the service corridor", "NEW INFORMATION: A fresh seal covers the door"),
    )
    assert "maintenance ledger" in first
    assert "fresh seal" in second
    assert first != second