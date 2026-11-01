from continuity import validate_scene_output, validate_story_ending


def _scene(body):
    return f"[SCENE 0]\nNARRATIVE:\n{body}\n[QUANTUM_TRACE]\nTrace."


def test_final_quality_gate_rejects_internal_metadata_in_narrative():
    result = validate_scene_output(
        _scene(
            "Mira confronts Voss. The internal prompt says to reveal the archive. "
            "Mira refuses and locks the door."
        ),
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
    )
    assert not result.accepted
    assert "internal_metadata_leak" in result.issues


def test_final_quality_gate_accepts_progressing_scene():
    result = validate_scene_output(
        _scene(
            "Mira confronts Voss. The alarm forces them toward the sealed archive. "
            "She takes the only key before the doors close."
        ),
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
    )
    assert result.accepted
    assert result.issues == ()


def test_quality_gate_reports_length_and_duplicate_sentence_corrections():
    sentence = "Mira confronts Voss at the sealed door."
    result = validate_scene_output(
        _scene(f"{sentence} {sentence}"),
        1,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
        min_words=20,
        max_words=12,
    )
    assert not result.accepted
    assert "duplicate_sentence" in result.issues
    assert "scene_below_minimum_length" in result.issues
    assert "scene_above_maximum_length" in result.issues
    assert result.retry_required is True
    assert result.correction_instruction


def test_ending_gate_rejects_unresolved_final_scene():
    valid, issues = validate_story_ending(
        _scene("Mira confronts Voss. The door remains closed."),
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert not valid
    assert "final_scene_lacks_resolution_signal" in issues


def test_ending_gate_rejects_open_question_despite_resolution_keywords():
    valid, issues = validate_story_ending(
        _scene("Mira faces Voss and their situation changed. The consequence was clear, but the central question loomed larger: what would they do next?"),
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert not valid
    assert "final_scene_explicitly_unresolved" in issues


def test_ending_gate_rejects_outcome_deferred_to_unknown_future():
    valid, issues = validate_story_ending(
        _scene("Mira confronts Voss and makes a choice that changed their lives. Nia says the answer lies within the record, its impact unknown, and the outcome hanging on what comes next."),
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert not valid
    assert "final_scene_explicitly_unresolved" in issues


def test_ending_gate_rejects_central_question_that_still_has_alternatives():
    valid, issues = validate_story_ending(
        _scene("Mira faced Voss and her courage changed their future. The central question now bore weight, promising either transformation or ruin. The decision that awaited would determine everything."),
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert not valid
    assert "final_scene_explicitly_unresolved" in issues


def test_ending_gate_rejects_final_outcome_presented_as_unresolved_alternatives():
    valid, issues = validate_story_ending(
        _scene("Mira made a choice that changed their future. The central question now held a meaningful outcome: trust or deception."),
        {"protagonist": "Mira", "antagonist": "Voss"},
    )
    assert not valid
    assert "final_scene_explicitly_unresolved" in issues


def test_quality_gate_rejects_truncated_final_sentence():
    result = validate_scene_output(
        "[SCENE 2]\nNARRATIVE:\nMira follows the ledger. Voss blocks the door. Lyra sends a key. Elias reaches the service",
        2,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
    )
    assert not result.accepted
    assert "truncated_output" in result.issues
    assert "finish the final sentence" in result.correction_instruction.lower()


def test_quality_gate_requires_scene_specific_evidence():
    result = validate_scene_output(
        "[SCENE 1]\nNARRATIVE:\nMira compares the approved records. Voss blocks her access. Lyra sends a clue.",
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
        required_terms=("maintenance ledger", "power allocation"),
    )
    assert not result.accepted
    assert "missing_planned_evidence:maintenance ledger / power allocation" in result.issues


def test_quality_gate_accepts_semantic_equivalent_evidence():
    result = validate_scene_output(
        _scene("Mira finds a maintenance log showing electricity allocated to the sealed level. Voss orders her away. She copies the dated entry before security can remove it."),
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
        required_terms=("maintenance ledger", "power allocation"),
    )
    assert result.accepted


def test_quality_gate_rejects_missing_evidence_fact():
    result = validate_scene_output(
        _scene("Mira compares the approved records while Voss blocks the office door."),
        0,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
        required_terms=("maintenance ledger", "power allocation"),
    )
    assert not result.accepted
    assert any(issue.startswith("missing_planned_evidence:") for issue in result.issues)


def test_quality_gate_rejects_prompt_state_labels_in_narrative():
    result = validate_scene_output(
        _scene("Mira opens the panel. Latest accepted change: the vault remains sealed. Voss blocks her."),
        1,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
    )
    assert not result.accepted
    assert "internal_metadata_leak" in result.issues


def test_quality_gate_rejects_role_placeholders_in_reader_prose():
    result = validate_scene_output(
        _scene("The protagonist follows the evidence while the antagonist blocks the exit."),
        1, {"protagonist": "Mira", "antagonist": "Voss"}, [],
    )
    assert "internal_metadata_leak" in result.issues


def test_action_label_leak_detection_is_dynamic_and_accepts_realized_action():
    leaked = validate_scene_output(
        _scene("Mira opens the panel. The action is decrypt_obelisk. Voss blocks her."),
        1, {"protagonist": "Mira", "antagonist": "Voss"}, [],
        forbidden_internal_terms=("decrypt_obelisk",),
    )
    realized = validate_scene_output(
        _scene("Mira studies the carved seal and turns its hidden mechanism. Voss blocks her."),
        1, {"protagonist": "Mira", "antagonist": "Voss"}, [],
        forbidden_internal_terms=("decrypt_obelisk",),
    )
    assert "internal_action_label_leak" in leaked.issues
    assert realized.accepted


def test_quality_gate_rejects_unestablished_city_in_narrative():
    result = validate_scene_output(
        _scene("Mira reaches the city of Eldoria. Voss blocks the records room."),
        1,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
    )
    assert not result.accepted
    assert "unsupported_location" in result.issues


def test_quality_gate_rejects_later_sentence_that_repeats_premise():
    result = validate_scene_output(
        _scene("Mira follows the ledger. Elias Vance investigates a concealed corporate vault while Voss obstructs the inquiry."),
        1,
        {"protagonist": "Mira", "antagonist": "Voss"},
        [],
        premise="Elias Vance investigates a concealed corporate vault while Voss obstructs the inquiry.",
    )
    assert not result.accepted
    assert "premise_restatement" in result.issues
