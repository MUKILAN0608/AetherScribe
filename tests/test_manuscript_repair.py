from evaluation.manuscript_repair import repair_manuscript


def _scene(step, accepted):
    return {
        "step": step,
        "story": f"Scene {step + 1} draft.",
        "quality_gate": {"accepted": accepted},
        "diagnostics": {"issues": ["needs revision"]} if not accepted else {},
    }


def test_whole_story_repair_preserves_scene_count_order_and_provisional_status():
    scenes = [_scene(0, False), _scene(1, True), _scene(2, False)]

    def audit(items, requested):
        findings = [
            {"scene_id": f"scene-{scene['step'] + 1:04d}", "code": "needs_revision"}
            for scene in items if not scene["quality_gate"]["accepted"]
        ]
        return {"passed": not findings, "findings": findings, "requested_scene_count": requested}

    def revise(scene, _manuscript, _findings):
        return {"story": f"Scene {scene['step'] + 1} revised with a concrete action."}

    revised, final_audit, attempts = repair_manuscript(
        scenes, 3, audit=audit, repair_scene=revise,
        validate_scene=lambda _scene, _step: (True, []),
    )
    assert [scene["step"] for scene in revised] == [0, 1, 2]
    assert len(revised) == 3
    assert [item["scene_index"] for item in attempts] == [0, 2]
    assert all(item["repaired"] for item in attempts)
    assert all(not revised[index]["quality_gate"]["accepted"] for index in (0, 2))
    assert all(revised[index]["manuscript_repair"]["canonical_recommit_required"] for index in (0, 2))
    assert final_audit["passed"] is False


def test_whole_story_repair_keeps_original_draft_when_candidate_fails_validation():
    scenes = [_scene(0, False)]
    result, _audit, attempts = repair_manuscript(
        scenes, 1,
        audit=lambda items, requested: {
            "passed": False,
            "findings": [{"scene_id": "scene-0001", "code": "needs_revision"}],
        },
        repair_scene=lambda *_args: {"story": "Bad alternate."},
        validate_scene=lambda *_args: (False, ["still_invalid"]),
    )
    assert len(result) == 1
    assert result[0]["story"] == "Scene 1 draft."
    assert result[0]["quality_gate"]["accepted"] is False
    assert attempts[0]["issues"] == ["still_invalid"]


def test_incomplete_manuscript_is_not_shrunk_or_falsely_repaired():
    scenes = [_scene(0, False)]
    calls = []
    result, _audit, attempts = repair_manuscript(
        scenes, 2,
        audit=lambda items, requested: {"passed": False, "findings": []},
        repair_scene=lambda *_args: calls.append(True),
        validate_scene=lambda *_args: (True, []),
    )
    assert len(result) == 1
    assert calls == []
    assert attempts == []
