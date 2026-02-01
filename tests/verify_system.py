import os
import sys
import numpy as np

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.story_environment import StoryEnvironment
from quantum_policy.quantum_policy import QuantumPolicy
from quantum_policy.quantum_action_space import QuantumActionSpace
from explainability.explanation_engine import ExplanationEngine
from evaluation.episode_runner import EpisodeRunner

class MockRenderer:
    """Mocks the LLM renderer for integration testing (v5.4.2)."""
    def render_scene(self, state, actions, rationale, previous_summary=None, language="English", temperature=0.8, mode="Short Story"):
        story = f"MOCK STORY SCENE in {language} (Mode: {mode}) for actions {actions} at tension {state['tension']:.2f}\n[QUANTUM_TRACE]\nMOCK TRACE"
        summary = "Mock summary for continuity."
        prompt_trace = "MOCK PROMPT TRACE"
        coherence_score = 0.95
        return story, summary, prompt_trace, coherence_score

def test_integration():
    print("Starting AetherScribe Lab v5.4.2 Integration Test...")

    # 1. Setup
    DIRECTOR_BRIEF = {
        "initial_story_premise": "Test Premise",
        "genre": "Horror",
        "characters": {"protagonist": "Hero", "antagonist": "Villain", "ally": "Friend"},
        "max_scenes": 3
    }

    env = StoryEnvironment(genre="Horror", director_brief=DIRECTOR_BRIEF, max_steps=3)
    policy = QuantumPolicy(n_qubits=4)
    action_space = QuantumActionSpace()
    explainer = ExplanationEngine()
    renderer = MockRenderer()

    runner = EpisodeRunner(env, policy, action_space, explainer, renderer)

    # 2. Run
    print("Executing v5.4.2 research trajectory...")
    result = runner.run_episode(verbose=False, language="English", mode="Short Story")
    episode_log = result["scenes"]

    # 3. Assertions
    assert len(episode_log) == 3, f"Expected 3 steps, got {len(episode_log)}"
    assert "total_reward" in result, "Reward key mismatch in v5.4.2 payload"
    assert "traits" in result, "Character traits missing from result"
    assert result["mode"] == "Short Story", "Mode mismatch"

    for entry in episode_log:
        assert "story" in entry
        assert "action" in entry
        assert "probs" in entry
        assert "rejections" in entry
        assert "attribution" in entry
        assert "sensitivity" in entry
        assert entry["story"].startswith("MOCK STORY SCENE")
        print(f"Node {entry['step']} verified: Audit trail and telemetry captured.")

    print("\n[SUCCESS] Integration test passed! AetherScribe Lab v5.4.2 is fully operational.")

if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        print(f"\n[FAILURE] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
