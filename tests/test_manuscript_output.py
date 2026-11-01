from ui.manuscript import compile_full_manuscript, compile_plain_text_export
from ui.research_views import render_entities, render_dashboard_tab


def _episode():
    return {
        "id": "episode-1",
        "genre": "Thriller",
        "premise": "Mira investigates the vault.",
        "total_reward": 1.0,
        "scenes": [
            {
                "step": 0,
                "story": (
                    "[SCENE 0]\nNARRATIVE:\n"
                    "Mira enters the vault and sees the sealed archive.\n\n"
                    "[QUANTUM_TRACE]\n"
                    "Internal action trace.\n"
                    "[COHERENCE]\n0.9"
                ),
                "reward": 1.0,
                "coherence": 0.9,
                "state": {"tension": 0.4},
                "entropy": 0.1,
            }
        ],
    }


def test_user_facing_exports_strip_engine_metadata():
    episode = _episode()
    full_story = compile_full_manuscript(episode)
    plain_text = compile_plain_text_export(episode)
    for output in (full_story, plain_text):
        assert "QUANTUM_TRACE" not in output
        assert "COHERENCE" not in output
        assert "Internal action trace" not in output
    assert "Mira enters the vault" in full_story


def test_plain_export_preserves_original_scene_sequence_and_requested_count():
    episode = _episode()
    episode["scenes"][0]["step"] = 3
    episode["requested_scene_count"] = 6
    episode["status"] = "partial"
    output = compile_plain_text_export(episode)
    assert "Scenes: 1 accepted of 6 requested" in output
    assert "SCENE 4 OF 6" in output


def test_manuscript_exports_only_unique_accepted_scenes_and_labels_partial():
    episode = _episode()
    episode["requested_scene_count"] = 3
    episode["status"] = "partial"
    duplicate = dict(episode["scenes"][0])
    episode["scenes"].append(duplicate)
    episode["scenes"].append({
        **duplicate, "step": 1,
        "quality_gate": {"accepted": False},
        "story": "[SCENE 1]\nNARRATIVE:\nRejected draft.\n[QUANTUM_TRACE]\nprivate",
    })
    output = compile_plain_text_export(episode)
    manuscript = compile_full_manuscript(episode)
    assert output.count("Mira enters the vault") == 1
    assert "Partial manuscript: 1 accepted" in output
    assert "missing scene numbers: 2, 3" in output
    assert "Rejected draft" not in manuscript


def test_render_entities_handles_empty_scenes():
    class StreamlitStub:
        @staticmethod
        def markdown(*_args, **_kwargs):
            return None

    import ui.research_views as views
    original = views.st
    views.st = StreamlitStub()
    try:
        render_entities({"characters": {"protagonist": "Mira"}, "scenes": []})
    finally:
        views.st = original


def test_dashboard_handles_empty_scenes():
    class StreamlitStub:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    import ui.research_views as views
    original = views.st
    views.st = StreamlitStub()
    try:
        render_dashboard_tab({
            "status": "partial", "scenes": [],
            "scene_failures": [{"scene_index": 0, "issues": ["truncated_output"]}],
        })
    finally:
        views.st = original
