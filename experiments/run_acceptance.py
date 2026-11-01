"""Run a real configured-provider episode and write inspectable acceptance artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from decision.providers import DecisionRouter
from environment.story_environment import StoryEnvironment
from evaluation.acceptance import evaluate_episode_acceptance
from evaluation.database import AetherDatabase
from evaluation.episode_runner import EpisodeRunner
from explainability.explanation_engine import ExplanationEngine
from llm.story_renderer import StoryRenderer
from planning import NarrativePlanner
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy
from ui.manuscript import compile_plain_text_export


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genre", default="thriller")
    parser.add_argument("--premise", required=True)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--characters-json", default='{"protagonist":"Mira","antagonist":"Voss","ally":"Nia"}')
    parser.add_argument("--output-dir", default=str(ROOT / "acceptance_runs"))
    args = parser.parse_args()
    characters = json.loads(args.characters_json)
    if args.scenes < 1:
        parser.error("--scenes must be positive")
    brief = {
        "genre": args.genre,
        "initial_story_premise": args.premise,
        "characters": characters,
        "max_scenes": args.scenes,
        "language": "English",
    }
    config = {
        "memory": os.getenv("NARRATIVE_MEMORY_ENABLED", "true").lower() == "true",
        "planner": os.getenv("NARRATIVE_PLANNER_ENABLED", "true").lower() == "true",
        "secrets": os.getenv("NARRATIVE_SECRET_LEDGER_ENABLED", "true").lower() == "true",
    }
    db = AetherDatabase()
    renderer = StoryRenderer()
    runner = EpisodeRunner(
        StoryEnvironment(args.genre, brief, max_steps=args.scenes),
        QuantumPolicy(n_qubits=4), QuantumActionSpace(), ExplanationEngine(), renderer,
        decision_provider=DecisionRouter(),
        memory_repository=db.memory if config["memory"] else None,
        memory_enabled=config["memory"],
        planner=NarrativePlanner(enabled=config["planner"]),
        planner_enabled=config["planner"],
        secret_repository=db.secrets if config["secrets"] else None,
        secret_enabled=config["secrets"],
        reveal_gate_enabled=os.getenv("NARRATIVE_REVEAL_GATE_ENABLED", "true").lower() == "true",
        scene_history_repository=db.scene_history,
        repetition_retry_limit=int(os.getenv("NARRATIVE_MAX_RETRIES", "3")),
    )
    result = runner.run_episode(verbose=False, language="English", render_depth="fast", seed=417)
    result["genre"] = args.genre
    result["premise"] = args.premise
    result["characters"] = characters
    result["completion_conditions"] = result.get("completion_conditions", {})
    acceptance = evaluate_episode_acceptance(result)
    manuscript = compile_plain_text_export(result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"live-{args.genre.lower()}-{args.scenes}-scene"
    manuscript_path = output_dir / f"{stem}.md"
    report_path = output_dir / f"{stem}-report.json"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    report = {
        "acceptance": acceptance,
        "full_episode_complete": result["status"] == "complete",
        "provider": renderer.provider,
        "model": renderer.model,
        "provider_error": renderer.last_error,
        "premise": args.premise,
        "genre": args.genre,
        "characters": characters,
        "scene_contracts": result["scene_contracts"],
        "scene_failures": result["scene_failures"],
        "ending_validation": result["ending_validation"],
        "completion_conditions": result["completion_conditions"],
        "manuscript_path": str(manuscript_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n--- INSPECTED MANUSCRIPT ---\n")
    print(manuscript)
    db.engine.dispose()
    return 0 if acceptance["passed"] and result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
