"""
run_single_episode.py

The primary interactive entry point for the AetherScribe Research Lab (v3.9.0).
Accepts user input for creative direction and research hyperparameters.
"""

import os
import sys
import uuid
from dotenv import load_dotenv

# Load optional environment variables; the narrative path is keyless by default.
load_dotenv()

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.story_environment import StoryEnvironment
from quantum_policy.quantum_policy import QuantumPolicy
from quantum_policy.quantum_action_space import QuantumActionSpace
from explainability.explanation_engine import ExplanationEngine
from llm.story_renderer import StoryRenderer
from evaluation.episode_runner import EpisodeRunner
from evaluation.database import AetherDatabase
from decision.providers import DecisionRouter
from planning import NarrativePlanner

SUPPORTED_GENRES = ("noir", "fantasy", "thriller", "drama", "comedy", "romance", "horror")
DEFAULT_CHARACTERS = {
    "protagonist": "Mira",
    "antagonist": "Voss",
    "ally": "Nia",
}


def _read_text(prompt, default):
    value = input(prompt).strip()
    return value or default


def _read_float(prompt, default, minimum, maximum):
    try:
        value = float(input(f"{prompt} (default {default}): ") or default)
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _read_int(prompt, default, minimum, maximum):
    try:
        value = int(input(f"{prompt} (default {default}): ") or default)
    except ValueError:
        return default
    return min(maximum, max(minimum, value))

def get_research_input():
    print("\n" + "="*60)
    print("  AETHERSCRIBE RESEARCH LAB: TRAJECTORY CONFIGURATION")
    print("="*60 + "\n")

    premise = _read_text(
        "Enter story premise: ",
        "A study on temporal loops in a closed system.",
    )

    print("\nAvailable genres: Noir, Fantasy, Thriller, Drama, Comedy, Romance, Horror")
    genre = _read_text("Enter genre (default Noir): ", "noir").lower()
    if genre not in SUPPORTED_GENRES:
        print(f"[!] Unknown genre; using noir. Choose from: {', '.join(SUPPORTED_GENRES)}")
        genre = "noir"

    language = _read_text("Output language (default English): ", "English")

    protagonist = _read_text("Protagonist name (default Mira): ", DEFAULT_CHARACTERS["protagonist"])
    antagonist = _read_text("Antagonist name (default Voss): ", DEFAULT_CHARACTERS["antagonist"])
    ally = _read_text("Ally name (default Nia): ", DEFAULT_CHARACTERS["ally"])

    print("\n--- v3.9.0 Research Hyperparameters ---")
    lr = _read_float("Learning Rate", 0.01, 0.0, 1.0)
    eps = _read_float("Quantum Noise / Eps", 0.0, 0.0, 1.0)
    temp = _read_float("LLM Temperature", 0.7, 0.0, 2.0)
    max_scenes = _read_int("Narrative depth / scenes", 5, 1, 12)

    return {
        "brief": {
            "initial_story_premise": premise,
            "genre": genre,
            "characters": {
                "protagonist": protagonist,
                "antagonist": antagonist,
                "ally": ally,
            },
            "max_scenes": max_scenes,
            "language": language
        },
        "params": {
            "lr": lr,
            "eps": eps,
            "temp": temp
        }
    }

def main():
    # 1. Gather Research Input
    config = get_research_input()
    brief = config["brief"]
    params = config["params"]

    # 2. Initialize Core Components
    env = StoryEnvironment(genre=brief["genre"], director_brief=brief, max_steps=brief["max_scenes"])
    policy = QuantumPolicy(n_qubits=4)
    action_space = QuantumActionSpace()
    explainer = ExplanationEngine()
    db = AetherDatabase()

    try:
        renderer = StoryRenderer()
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # 3. Initialize Standardized Runner
    runner = EpisodeRunner(
        env=env,
        policy=policy,
        action_space=action_space,
        explainer=explainer,
        renderer=renderer,
        decision_provider=DecisionRouter(),
        memory_repository=db.memory
        if os.getenv("NARRATIVE_MEMORY_ENABLED", "true").lower() == "true"
        else None,
        memory_enabled=os.getenv("NARRATIVE_MEMORY_ENABLED", "true").lower() == "true",
        memory_limit=int(os.getenv("NARRATIVE_MEMORY_RETRIEVAL_LIMIT", "12")),
        memory_budget=int(os.getenv("NARRATIVE_MEMORY_CONTEXT_TOKEN_BUDGET", "800")),
        story_id=f"story-{uuid.uuid4()}",
        planner=NarrativePlanner(
            enabled=os.getenv("NARRATIVE_PLANNER_ENABLED", "true").lower() == "true"
        ),
        planner_enabled=os.getenv("NARRATIVE_PLANNER_ENABLED", "true").lower() == "true",
        secret_repository=db.secrets
        if os.getenv("NARRATIVE_SECRET_LEDGER_ENABLED", "true").lower() == "true"
        else None,
        secret_enabled=(
            os.getenv("NARRATIVE_SECRET_LEDGER_ENABLED", "true").lower() == "true"
        ),
        reveal_gate_enabled=(
            os.getenv("NARRATIVE_REVEAL_GATE_ENABLED", "true").lower() == "true"
        ),
    )

    # 4. Execute Trajectory
    print(
        f"\n[ORCHESTRATION START] Genre: {brief['genre'].upper()} | "
        f"Depth: {brief['max_scenes']} | "
        f"Characters: {brief['characters']['protagonist']}, "
        f"{brief['characters']['antagonist']}, {brief['characters']['ally']}\n"
    )

    result = runner.run_episode(
        verbose=True,
        language=brief["language"],
        lr=params["lr"],
        eps=params["eps"],
        temperature=params["temp"]
    )

    # 5. Archive Data to SQL Persistence
    episode_id = db.save_episode(
        brief["genre"],
        brief["initial_story_premise"],
        result["total_reward"],
        result["scenes"],
        story_id=result["story_id"],
    )
    if result["scenes"]:
        last_scene = result["scenes"][-1]
        print(
            f"[PROVIDERS] Jev={last_scene.get('decision_audit', {}).get('provider', 'local')} | "
            f"Narrative={last_scene.get('narrative_provider', 'unknown')} "
            f"({last_scene.get('narrative_model', 'unknown')}) | "
            f"Fallback={'yes' if last_scene.get('narrative_fallback') else 'no'}"
        )
        print("\n[MEASUREMENTS]")
        total_input = sum(
            scene.get("narrative_total_input_tokens") or 0 for scene in result["scenes"]
        )
        total_output = sum(
            scene.get("narrative_total_output_tokens") or 0 for scene in result["scenes"]
        )
        total_cost = sum(
            scene.get("narrative_total_cost_usd") or 0 for scene in result["scenes"]
        )
        average_coherence = sum(
            scene.get("coherence", 0) for scene in result["scenes"]
        ) / len(result["scenes"])
        print(
            f"Episode ID: {episode_id} | Scenes: {len(result['scenes'])} | "
            f"Total reward: {result['total_reward']:.3f}"
        )
        print(
            f"Narrative tokens: {total_input} input / {total_output} output | "
            f"Estimated cost: ${total_cost:.6f} | "
            f"Average coherence: {average_coherence:.3f}"
        )
        for scene in result["scenes"]:
            decision = scene.get("decision_audit") or {}
            print(
                f"Scene {scene['step'] + 1}: "
                f"action={scene['action']} | "
                f"selected_probability={scene['selected_probability']:.4f} | "
                f"peak_probability={scene['peak_probability']:.4f} | "
                f"jev={decision.get('provider', 'local')} "
                f"({decision.get('model', 'rules')}) | "
                f"jev_latency_ms={decision.get('latency_ms', 0):.0f} | "
                f"narrative_latency_ms={scene.get('narrative_total_latency_ms') or 0:.0f} | "
                f"coherence={scene['coherence']:.3f} | "
                f"reward={scene['reward']:.3f}"
            )
    print("\n[SUCCESS] Research trajectory archived to: aetherscribe_lab.db")

if __name__ == "__main__":
    main()
