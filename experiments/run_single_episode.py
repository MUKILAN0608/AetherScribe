"""
run_single_episode.py

The primary interactive entry point for the AetherScribe Research Lab (v3.9.0).
Accepts user input for creative direction and research hyperparameters.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY)
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

def get_research_input():
    print("\n" + "="*60)
    print("  AETHERSCRIBE RESEARCH LAB: TRAJECTORY CONFIGURATION")
    print("="*60 + "\n")

    premise = input("Enter Research Premise (Hypothesis): ") or "A study on temporal loops in a closed system."

    print("\nAvailable Genres: Noir, Fantasy, Thriller, Drama, Comedy, Romance, Horror")
    genre = input("Enter Genre Schema (default: Noir): ").lower() or "noir"

    language = input("Output Language (default: English): ") or "English"

    print("\n--- v3.9.0 Research Hyperparameters ---")
    try:
        lr = float(input("Learning Rate (default 0.01): ") or 0.01)
        eps = float(input("Quantum Noise / Eps (default 0.0): ") or 0.0)
        temp = float(input("LLM Temperature (default 0.7): ") or 0.7)
        max_scenes = int(input("Narrative Depth / Scenes (default 5): ") or 5)
    except ValueError:
        print("[!] Invalid numeric input. Reverting to baseline defaults.")
        lr, eps, temp, max_scenes = 0.01, 0.0, 0.7, 5

    return {
        "brief": {
            "initial_story_premise": premise,
            "genre": genre,
            "characters": {"protagonist": "Subject-P", "antagonist": "System-A", "ally": "Support-L"},
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
        renderer=renderer
    )

    # 4. Execute Trajectory
    print(f"\n[ORCHESTRATION START] Genre: {brief['genre'].upper()} | Depth: {brief['max_scenes']}\n")

    result = runner.run_episode(
        verbose=True,
        language=brief["language"],
        lr=params["lr"],
        eps=params["eps"],
        temperature=params["temp"]
    )

    # 5. Archive Data to SQL Persistence
    db.save_episode(brief["genre"], brief["initial_story_premise"], result["total_tension"], result["scenes"])
    print(f"\n[SUCCESS] Research trajectory archived to: aetherscribe_lab.db")

if __name__ == "__main__":
    main()
