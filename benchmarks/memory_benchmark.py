"""Reproducible memory-on versus memory-off benchmark.

The benchmark uses a deterministic renderer and decision evaluator so it never
requires paid provider calls. Both conditions replay the exact same VQC action
indices.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.providers import DecisionResponse
from evaluation.database import AetherDatabase
from evaluation.episode_runner import EpisodeRunner
from environment.story_environment import StoryEnvironment
from explainability.explanation_engine import ExplanationEngine
from quantum_policy.quantum_action_space import QuantumActionSpace
from quantum_policy.quantum_policy import QuantumPolicy


class BenchmarkDecisionProvider:
    provider_name = "benchmark-jev"
    model_name = "deterministic-fixture"

    def evaluate(self, request):
        return DecisionResponse(
            selected="approve",
            approved=True,
            rationale="Fixture evaluator approves without changing the locked action.",
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=0.1,
            used_fallback=False,
            error=None,
        )


class BenchmarkRenderer:
    provider = "benchmark-renderer"
    model = "deterministic-fixture"
    last_latency_ms = 0.1
    last_input_tokens = None
    last_output_tokens = None
    last_estimated_cost_usd = None
    last_used_fallback = False

    def render_scene(self, state, actions, rationale, previous_summary=None,
                     language="English", temperature=0.8, mode="Short Story",
                     lite_mode=False, memory_context=None):
        names = ", ".join(f"{role}: {action}" for role, action in actions.items())
        characters = state.get("director_brief", {}).get("characters", {})
        memory_note = (
            f" Continuity records available: {len(memory_context or ())}."
        )
        return (
            f"[SCENE {state.get('step', 0)}]\nNARRATIVE:\n"
            f"{', '.join(characters.values())} preserve the locked actions "
            f"({names}).{memory_note}\n\n"
            "[QUANTUM_TRACE]\nThe fixture preserves the measured action.",
            "The fixture continues the same story.",
            "benchmark prompt",
            0.9,
        )


def _brief():
    return {
        "initial_story_premise": "Mira guards a sealed observatory while Voss seeks its key and Nia conceals the final coordinate.",
        "genre": "thriller",
        "characters": {"protagonist": "Mira", "antagonist": "Voss", "ally": "Nia"},
        "max_scenes": 3,
        "language": "English",
    }


def _run(db, story_id, memory_enabled, locked_action_indices):
    brief = _brief()
    env = StoryEnvironment(genre="thriller", director_brief=brief, max_steps=3)
    runner = EpisodeRunner(
        env,
        QuantumPolicy(n_qubits=4),
        QuantumActionSpace(),
        ExplanationEngine(),
        BenchmarkRenderer(),
        decision_provider=BenchmarkDecisionProvider(),
        memory_repository=db.memory if memory_enabled else None,
        memory_enabled=memory_enabled,
        memory_limit=12,
        memory_budget=800,
        story_id=story_id,
    )
    started = time.perf_counter()
    result = runner.run_episode(
        verbose=False,
        language="English",
        render_depth="fast",
        seed=17,
        locked_action_indices=locked_action_indices,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    db.save_episode(
        brief["genre"], brief["initial_story_premise"], result["total_reward"],
        result["scenes"], story_id=story_id,
    )
    return result, elapsed_ms


def main():
    with tempfile.TemporaryDirectory(prefix="aetherscribe-memory-benchmark-") as directory:
        db = AetherDatabase(f"sqlite:///{Path(directory) / 'benchmark.db'}")
        seed_result, _ = _run(db, "benchmark-seed", False, [6, 7, 13])
        locked_indices = [scene["action_idx"] for scene in seed_result["scenes"]]
        off, off_ms = _run(db, "benchmark-off", False, locked_indices)
        on, on_ms = _run(db, "benchmark-on", True, locked_indices)

        off_actions = [scene["action"] for scene in off["scenes"]]
        on_actions = [scene["action"] for scene in on["scenes"]]
        memory_records = db.get_memory_context("benchmark-on", limit=100).records
        report = {
            "configuration": {
                "scenes": 3,
                "seed": 17,
                "memory_budget": 800,
                "provider": "deterministic-fixture",
            },
            "locked_action_indices": locked_indices,
            "actions_identical": off_actions == on_actions,
            "memory_off": {
                "elapsed_ms": round(off_ms, 3),
                "retrieved_per_scene": [s["memory_retrieval"]["selected"] for s in off["scenes"]],
                "updates": sum(s["memory_updates"] for s in off["scenes"]),
            },
            "memory_on": {
                "elapsed_ms": round(on_ms, 3),
                "retrieved_per_scene": [s["memory_retrieval"]["selected"] for s in on["scenes"]],
                "updates": sum(s["memory_updates"] for s in on["scenes"]),
                "persisted_records": len(memory_records),
            },
            "continuity_checks": {
                "character_names_present": all(
                    all(name in scene["story"] for name in ("Mira", "Voss", "Nia"))
                    for scene in on["scenes"]
                ),
                "locked_action_preserved": on_actions == off_actions,
            },
        }
        print(json.dumps(report, indent=2))
        db.engine.dispose()


if __name__ == "__main__":
    main()
