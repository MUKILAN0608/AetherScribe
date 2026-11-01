from agents.protagonist_agent import ProtagonistAgent
from agents.antagonist_agent import AntagonistAgent
from agents.ally_agent import AllyAgent
from dataclasses import dataclass
from typing import Mapping, Any
import re
from itertools import product


@dataclass(frozen=True)
class NarrativeActionCandidate:
    """One scene-grounded joint intent option available to the VQC."""

    action_id: str
    actions: Mapping[str, str]
    action_type: str
    target: str
    preconditions: tuple[str, ...]
    expected_effect: str
    narrative_purpose: str
    risks: tuple[str, ...]
    reversible: bool
    valid: bool
    invalid_reasons: tuple[str, ...] = ()


def build_scene_action_candidates(
    scene_plan: Any,
    characters: Mapping[str, str],
    *,
    scene_index: int,
    accepted_predecessor: bool,
    tension: float = 0.5,
    precondition_check=None,
    canonical_state=None,
) -> tuple[NarrativeActionCandidate, ...]:
    """Derive VQC alternatives from the current scene contract and story cast."""
    roles = (
        ("Protagonist", "protagonist"),
        ("Antagonist", "antagonist"),
        ("Ally", "ally"),
    )
    cast = {label: str(characters.get(role, "")).strip() for label, role in roles}
    base_preconditions = []
    if scene_index > 0 and not accepted_predecessor:
        base_preconditions.append("the immediately preceding planned scene must be accepted")
    if not cast["Protagonist"]:
        base_preconditions.append("the scene contract must identify a protagonist")

    def clean(value: str) -> str:
        return re.sub(r"^\s*scene\s+\d+\s*:\s*", "", str(value), flags=re.I).strip()

    purpose = clean(scene_plan.purpose)
    protagonist_objective = clean(scene_plan.protagonist_objective or scene_plan.purpose)
    event = clean(scene_plan.event)
    consequence = clean(scene_plan.consequence)
    next_effect = clean(scene_plan.next_scene_effect or scene_plan.consequence)

    protagonist_intents = (
        f"pursues {protagonist_objective}",
        f"responds to {event}",
    )
    antagonist_intents = (
        f"resists {protagonist_objective} through {clean(scene_plan.antagonist_opposition or 'the established conflict')}",
        f"presses against {protagonist_objective}",
    )
    ally_intents = (
        f"supports the objective by {clean(scene_plan.ally_contribution or 'contributing to the established goal')}",
        f"helps bring about {event}",
        f"carries support into {next_effect}",
        f"manages the risk by responding to {consequence}",
    )
    target = scene_plan.location or "the established story setting"
    protagonist_options = range(len(protagonist_intents)) if cast["Protagonist"] else (0,)
    antagonist_options = range(len(antagonist_intents)) if cast["Antagonist"] else (0,)
    ally_options = range(len(ally_intents)) if cast["Ally"] else (0,)
    candidates = []
    for index, (protagonist_index, antagonist_index, ally_index) in enumerate(
        product(protagonist_options, antagonist_options, ally_options)
    ):
        actions = {}
        if cast["Protagonist"]:
            actions["Protagonist"] = protagonist_intents[protagonist_index]
        if cast["Antagonist"]:
            actions["Antagonist"] = antagonist_intents[antagonist_index]
        if cast["Ally"]:
            actions["Ally"] = ally_intents[ally_index]
        unmet = list(base_preconditions)
        if not actions:
            unmet.append("at least one named character must be available to act")
        if precondition_check is not None:
            unmet.extend(
                str(reason) for reason in precondition_check(actions, scene_plan, canonical_state) if reason
            )
        preconditions = tuple(dict.fromkeys((
            "the selected actors are named in the story configuration",
            *( (f"scene {scene_index} predecessor is committed",) if scene_index else ("opening premise is loaded",) ),
        )))
        candidates.append(NarrativeActionCandidate(
            action_id=f"{scene_plan.scene_id or f'scene-{scene_index + 1:04d}'}:intent-{index:02d}",
            actions=actions,
            action_type="scene-grounded joint narrative intent",
            target=target,
            preconditions=preconditions,
            expected_effect=scene_plan.consequence,
            narrative_purpose=purpose,
            risks=(f"current tension={float(tension):.3f}", scene_plan.antagonist_opposition),
            reversible=index % 2 == 0,
            valid=not unmet,
            invalid_reasons=tuple(unmet),
        ))
    return tuple(candidates)

class QuantumActionSpace:
    """
    Research-tier Quantum Action Space (v5.4.2).
    Maps 16 unique joint character actions for the Decision Authority.
    Matches 4-qubit state space (2^4 = 16).
    """
    def __init__(self):
        self.protagonist = ProtagonistAgent()
        self.antagonist = AntagonistAgent()
        self.ally = AllyAgent()

        self.joint_actions = [
            {"Protagonist": "Investigate Anomaly", "Antagonist": "Orchestrate Sabotage", "Ally": "Provide Intelligence"},
            {"Protagonist": "Confront Directly", "Antagonist": "Deploy Manipulation", "Ally": "Support Maneuver"},
            {"Protagonist": "Execute Covert Move", "Antagonist": "Escalate Threat", "Ally": "Mitigate Danger"},
            {"Protagonist": "Negotiate Truce", "Antagonist": "Leverage Corruption", "Ally": "Facilitate Extraction"},
            {"Protagonist": "Analyze Conflict", "Antagonist": "Obstruct Information", "Ally": "Reinforce Resolve"},
            {"Protagonist": "Deploy Countermeasure", "Antagonist": "Initiate Strike", "Ally": "Sacrifice Safety"},
            {"Protagonist": "Seek Truth", "Antagonist": "Weave Deception", "Ally": "Reveal Secret"},
            {"Protagonist": "Fortify Position", "Antagonist": "Infiltrate Systems", "Ally": "Signal Rescue"},
            {"Protagonist": "Breach Perimeter", "Antagonist": "Trigger Ambush", "Ally": "Disable Security"},
            {"Protagonist": "Rally Bystanders", "Antagonist": "Sow Chaos", "Ally": "Protect Civilians"},
            {"Protagonist": "Decipher Code", "Antagonist": "Jam Communications", "Ally": "Maintain Link"},
            {"Protagonist": "Escape Custody", "Antagonist": "Tighten Grip", "Ally": "Distract Guards"},
            {"Protagonist": "Incite Rebellion", "Antagonist": "Exert Control", "Ally": "Lead Vanguard"},
            {"Protagonist": "Retrieve Artifact", "Antagonist": "Guard Vault", "Ally": "Bypass Lock"},
            {"Protagonist": "Heal Wound", "Antagonist": "Inflict Pain", "Ally": "Provide Shelter"},
            {"Protagonist": "Final Stand", "Antagonist": "Unleash Full Might", "Ally": "Last Support"}
        ]
        self.num_actions = len(self.joint_actions)

    def get_joint_action(self, index):
        return self.joint_actions[index % self.num_actions]

    def build_scene_candidates(self, scene_plan, characters, **context):
        """Runtime candidate API; candidates are derived from the current story contract."""
        return build_scene_action_candidates(scene_plan, characters, **context)
