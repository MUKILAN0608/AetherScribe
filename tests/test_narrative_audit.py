from evaluation.narrative_audit import audit_scene_narrative, audit_story
from continuity.quality import validate_story_ending


def _scene(step, prose, accepted=True):
    return {"step": step, "story": f"[SCENE {step + 1}]\nNARRATIVE:\n{prose}",
            "quality_gate": {"accepted": accepted}}


def test_whole_story_audit_rejects_missing_scene_and_unaccepted_scene():
    audit = audit_story([_scene(0, "Mira opens the sealed ledger.")], 2, {"protagonist": "Mira"})
    codes = {finding["code"] for finding in audit["findings"]}
    assert audit["passed"] is False
    assert "incomplete_story_sequence" in codes


def test_whole_story_audit_rejects_abstract_claim_without_dramatized_event():
    scenes = [
        _scene(0, "Mira enters the archive and opens the ledger."),
        _scene(1, "Their resolve was strengthened. The consequences unfolded."),
    ]
    audit = audit_story(scenes, 2, {"protagonist": "Mira", "antagonist": "Voss"})
    assert "abstract_development_without_event" in {finding["code"] for finding in audit["findings"]}


def test_scene_critic_returns_targeted_abstract_prose_feedback():
    findings = audit_scene_narrative(
        "[SCENE 2]\nNARRATIVE:\nTheir resolve was strengthened. The consequences unfolded.", 1
    )
    assert findings[0]["code"] == "abstract_development_without_event"
    assert "observable result" in findings[0]["repair_suggestion"]


def test_scene_critic_flags_unestablished_family_motivation():
    findings = audit_scene_narrative(
        "[SCENE 8]\nNARRATIVE:\nMira invokes her sister's name to change Voss's mind.",
        7, known_context="Mira investigates an altered archive with Voss and Nia.",
    )
    assert findings[0]["code"] == "unsupported_family_relationship"


def test_whole_story_audit_accepts_complete_concrete_story_structure():
    scenes = [
        _scene(0, "Mira finds the altered ledger and copies the entry."),
        _scene(1, "Mira confronts Voss with the copy. Voss admits changing the index, and Nia releases the record."),
    ]
    audit = audit_story(scenes, 2, {"protagonist": "Mira", "antagonist": "Voss"})
    assert audit["passed"]


def test_unresolved_rhetorical_finale_cannot_pass_ending_gate():
    ending = (
        "Mira and Voss faced the chaos. Voss said they must resolve this. "
        "The central question hangs: how will they proceed? The choice is hers alone to make."
    )
    valid, issues = validate_story_ending(
        f"[SCENE 8]\nNARRATIVE:\n{ending}",
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert valid is False
    assert "final_scene_explicitly_unresolved" in issues


def test_symbolic_decisive_move_without_outcome_fails_ending_gate():
    prose = (
        "Mira and Voss faced the archive. The decisive move symbolized a new chapter, "
        "changing their fates and marking the end of a difficult journey for them."
    )
    valid, issues = validate_story_ending(
        f"[SCENE 8]\nNARRATIVE:\n{prose}", {"protagonist": "Mira", "antagonist": "Voss"}
    )
    assert not valid
    assert "final_scene_lacks_concrete_outcome" in issues
    assert "final_scene_explicitly_unresolved" in issues


def test_final_scene_must_resolve_a_conflict_grounded_in_the_user_premise():
    text = (
        "Mira and Voss resolve their dispute. Mira secures a victory, and Voss can no longer obstruct her. "
        "The decision changes their future."
    )
    valid, issues = validate_story_ending(
        text,
        {"protagonist": "Mira", "antagonist": "Voss"},
        premise="Mira investigates a sealed archive where Voss altered the index.",
    )
    assert valid is False
    assert "final_scene_not_tied_to_premise" in issues


def test_scene_audit_detects_unseeded_plot_device_and_sudden_location_jump():
    prose = (
        "Mira found a hidden switch and pressed it, unlocking the archive's sealed vault. "
        "Voss arrived at the alley before Nia could follow."
    )
    findings = audit_scene_narrative(
        prose, 3, known_context="Mira investigates the archive with Voss and Nia. They remain inside the archive."
    )
    codes = {item["code"] for item in findings}
    assert "unseeded_plot_device" in codes
    assert "unexplained_location_transition" in codes


def test_scene_audit_detects_character_knowledge_without_a_prior_source():
    findings = audit_scene_narrative(
        "Nia already knew Voss had changed the access record before Mira reached the archive.",
        2,
        known_context="Mira and Voss met in the archive. Nia was not present.",
    )
    assert "unsupported_character_knowledge" in {item["code"] for item in findings}


def test_scene_audit_allows_a_plot_device_and_knowledge_with_established_setup():
    prior = (
        "Mira found the hidden switch during the opening scene. Nia learned from the access record that Voss changed it."
    )
    findings = audit_scene_narrative(
        "Nia already knew Voss changed the access record. Mira pressed the hidden switch and opened the door.",
        2,
        known_context=prior,
    )
    assert not {"unseeded_plot_device", "unsupported_character_knowledge"} & {
        item["code"] for item in findings
    }


def test_whole_story_audit_rejects_choice_deferred_to_a_future_scene():
    scenes = [
        _scene(0, "Mira finds the altered ledger and copies the entry."),
        _scene(1, "Mira and Voss face the archive. The central question hangs: how will they proceed? The choice is hers alone to make."),
    ]
    audit = audit_story(scenes, 2, {"protagonist": "Mira", "antagonist": "Voss"})
    codes = {finding["code"] for finding in audit["findings"]}
    assert "ending_leaves_central_question_unanswered" in codes
