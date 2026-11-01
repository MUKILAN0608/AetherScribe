import numpy as np
import uuid
import logging
import inspect
import os
import re
from dataclasses import asdict, replace
from types import MappingProxyType
from decision.providers import DecisionRequest
from environment.state_representation import StateRepresentation
from quantum_policy.action_registry import joint_action_to_env
from memory import extract_authoritative_candidates
from planning import (
    CharacterKnowledge,
    LockedActionSnapshot,
    NarrativePlanner,
    PlannerInput,
    build_story_plan,
    build_story_contract,
    validate_story_plan,
)
from continuity import StoryContinuity, check_repetition, validate_scene_output, validate_story_ending
from secret_loader import load_secret_package
from agents.orchestration import NarrativeCoordinator, NarrativeEvent
from world_representation import WorldState
from evaluation.scene_lifecycle import SceneLifecycle
from evaluation.narrative_audit import audit_scene_narrative, audit_story
from evaluation.manuscript_repair import repair_manuscript
from future_prediction import FutureWorldPredictor
from temporal_adaptation import TemporalCharacterAdapter
_secret_api = load_secret_package()
RevealContext = _secret_api.RevealContext
RevealDecision = _secret_api.RevealDecision
RevealGate = _secret_api.RevealGate
SecretStatus = _secret_api.SecretStatus
validate_prose = _secret_api.validate_prose


class EpisodeRunner:
    """
    Narrative orchestrator (v6.2): cached quantum diagnostics, configurable render depth.
    """

    def __init__(
        self, env, policy, action_space, explainer, renderer, decision_provider=None,
        memory_repository=None, memory_enabled=True, memory_limit=12, memory_budget=800,
        story_id=None,
        planner=None,
        planner_enabled=False,
        secret_repository=None,
        secret_enabled=False,
        reveal_gate=None,
        reveal_gate_enabled=True,
        scene_history_repository=None,
        repetition_retry_limit=1,
        action_precondition_checker=None,
    ):
        self.env = env
        self.policy = policy
        self.action_space = action_space
        self.explainer = explainer
        self.renderer = renderer
        self.decision_provider = decision_provider
        self.memory_repository = memory_repository
        self.memory_enabled = memory_enabled and memory_repository is not None
        self.memory_limit = memory_limit
        self.memory_budget = memory_budget
        self.story_id = story_id or f"story-{uuid.uuid4()}"
        self.planner = planner
        self.planner_enabled = planner_enabled and planner is not None
        self.secret_repository = secret_repository
        self.secret_enabled = secret_enabled and secret_repository is not None
        self.reveal_gate = reveal_gate or RevealGate()
        self.reveal_gate_enabled = reveal_gate_enabled
        self.scene_history_repository = scene_history_repository
        self.repetition_retry_limit = max(0, int(repetition_retry_limit))
        self.action_precondition_checker = action_precondition_checker
        characters = self.env.director_brief.get("characters", {})
        premise = self.env.director_brief.get("initial_story_premise", "").strip()
        protagonist = characters.get("protagonist", "The protagonist")
        canonical_seed = self.env.director_brief.get("canonical_narrative_state", {}) or {}
        character_profiles = self.env.director_brief.get("character_profiles", {}) or {}
        initial_profiles = {
            name: character_profiles.get(name, character_profiles.get(role, {})) or {}
            for role, name in characters.items() if name
        }
        self.story_continuity = StoryContinuity(
            situation=premise or "The story is beginning.",
            protagonist_goal=str(
                initial_profiles.get(protagonist, {}).get("goal")
                or f"{protagonist} must respond to the central problem."
            ),
            established_facts=tuple(canonical_seed.get("facts", ())),
            character_knowledge={
                name: tuple(profile.get("knowledge", ()))
                for name, profile in initial_profiles.items() if profile.get("knowledge")
            },
            unresolved_conflicts=tuple(
                item.get("description", "") if isinstance(item, dict) else str(item)
                for item in canonical_seed.get("active_conflicts", ())
            ) or ("The central conflict is unresolved.",),
            next_scene_purpose="Establish the setting, immediate goal, and motivating conflict.",
        )
        self.story_plan = build_story_plan(
            premise,
            self.env.genre,
            characters,
            self.env.max_steps,
            world_rules=(
                (self.env.director_brief.get("world_rules"),)
                if isinstance(self.env.director_brief.get("world_rules"), str)
                else tuple(self.env.director_brief.get("world_rules", ()))
            ),
        )
        plan_issues = validate_story_plan(self.story_plan, self.env.max_steps)
        if plan_issues:
            raise ValueError("story plan validation failed before prose generation: " + "; ".join(plan_issues))
        raw_constraints = self.env.director_brief.get("constraints", ()) or ()
        if isinstance(raw_constraints, str):
            raw_constraints = (raw_constraints,)
        self.user_constraints = tuple(str(item) for item in raw_constraints if str(item).strip())
        self.story_contract = build_story_contract(
            self.story_plan,
            tone=str(self.env.director_brief.get("tone", "")),
            characters=characters,
            character_profiles=character_profiles,
            initial_state=canonical_seed,
            constraints=self.user_constraints,
        )
        self.scene_blueprints = tuple({
            "scene_id": scene.scene_id,
            "sequence": scene.scene_index + 1,
            "purpose": scene.purpose,
            "pov_character": scene.pov_character,
            "location": scene.location,
            "time": scene.time,
            "characters_present": list(scene.characters_present),
            "character_goals": dict(scene.character_goals),
            "obstacle": scene.immediate_obstacle,
            "concrete_actions": list(scene.concrete_actions),
            "information_revealed": list(scene.information_revealed),
            "causal_connection": scene.previous_scene_cause,
            "consequence": scene.consequence,
            "knowledge_changes": {key: list(value) for key, value in scene.knowledge_changes.items()},
            "relationship_changes": list(scene.relationship_changes),
            "ending_state": scene.ending_state,
            "setup_payoff_obligations": list(scene.setup_payoff_obligations),
            "continuity_constraints": list(self.story_contract.world_rules),
            "next_scene_effect": scene.next_scene_effect,
        } for scene in self.story_plan.scenes)
        coordinator_seed = dict(canonical_seed)
        if not coordinator_seed.get("active_conflicts"):
            coordinator_seed["active_conflicts"] = [{
                "conflict_id": "central-conflict",
                "description": self.story_contract.main_conflict,
                "status": "active",
                "stakes": self.story_contract.stakes,
            }]
        if not coordinator_seed.get("open_threads"):
            coordinator_seed["open_threads"] = [self.story_contract.central_question]
        if self.story_contract.timeline:
            coordinator_seed.setdefault("timeline", self.story_contract.timeline)
        self.narrative_coordinator = NarrativeCoordinator(
            self.story_id, characters, self.repetition_retry_limit,
            character_profiles=character_profiles, initial_state=coordinator_seed,
            story_contract=self.story_contract.as_dict(),
        )
        self.world_state = WorldState.from_narrative_state(self.narrative_coordinator.state)
        self.future_predictor = FutureWorldPredictor()
        self.character_adapter = TemporalCharacterAdapter()

    def _get_combined_trait_vec(self):
        p_traits = self.action_space.protagonist.traits
        a_traits = self.action_space.antagonist.traits
        l_traits = self.action_space.ally.traits
        return np.array(
            [
                (p_traits["aggression"] + a_traits["hostility"]) / 2.0,
                (p_traits["curiosity"] + a_traits["strategic_depth"]) / 2.0,
                (p_traits["resilience"] + a_traits["manipulation"]) / 2.0,
                (l_traits["loyalty"] + l_traits["supportiveness"]) / 2.0,
            ]
        )

    def run_episode(
        self,
        verbose=True,
        language="English",
        lr=0.01,
        eps=0.0,
        temperature=0.8,
        mode="Short Story",
        render_depth="full",
        on_step=None,
        seed=None,
        locked_action_indices=None,
    ):
        """
        render_depth: 'full' | 'standard' | 'fast'
          - full: scene + summary + coherence LLM + 3 rejection LLM calls
          - standard: scene + summary + coherence; template rejections
          - fast: scene only; heuristic coherence; template rejections
        """
        if seed is not None:
            np.random.seed(seed)
        state = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_log = []
        scene_failures = []
        episode_id = str(uuid.uuid4())
        previous_summary = None
        scene_contracts: list[dict] = []
        secrets_by_scene: dict[int, list] = {}
        scene_history = (
            self.scene_history_repository.list(self.story_id)
            if self.scene_history_repository is not None else []
        )
        trait_vec = self._get_combined_trait_vec()
        attempted_scene_indices: list[int] = []
        full_explain = render_depth == "full"
        lite_render = render_depth == "fast"

        while not done and step < self.env.max_steps:
            attempted_scene_indices.append(step)
            scene_plan = self.story_plan.for_scene(step)
            committed_state = self.narrative_coordinator.state
            candidates = self.action_space.build_scene_candidates(
                scene_plan,
                self.env.director_brief.get("characters", {}),
                scene_index=step,
                accepted_predecessor=(step == 0 or bool(
                    (committed_state.events and committed_state.events[-1].sequence < step + 1)
                    or any(int(item.get("step", -1)) == step - 1 for item in episode_log)
                    or any(int(item.get("scene_index", -1)) == step - 1 for item in scene_failures)
                )),
                tension=float(state.get("tension", 0.5)),
                precondition_check=self.action_precondition_checker,
                canonical_state=committed_state,
            )
            valid_mask = np.asarray([candidate.valid for candidate in candidates], dtype=bool)
            if not np.any(valid_mask):
                scene_failures.append({
                    "scene_index": step,
                    "failure_type": "planning",
                    "issues": ["no_valid_action_candidates"],
                    "retry_count": 0,
                    "provider_error": None,
                    "candidate_issues": [
                        {"candidate_id": item.action_id, "reasons": list(item.invalid_reasons)}
                        for item in candidates
                    ],
                })
                state = {**state, "step": step + 1}
                step += 1
                continue
            state_vec = StateRepresentation.encode(
                state["phase"], state["tension"], state["genre"], state["step"]
            )

            diag = self.policy.compute_step_diagnostics(
                state_vec,
                len(candidates),
                trait_vec=trait_vec,
                eps=eps,
                full_explain=full_explain,
            )
            measured_posterior = np.asarray(diag["action_probs"], dtype=float)
            probs = self.policy.condition_on_valid_actions(measured_posterior, valid_mask)
            entropy = float(-np.sum(np.where(
                probs > 0, probs * np.log2(np.maximum(probs, 1e-12)), 0.0
            )))
            risk_score = diag["risk"]
            attribution = diag["attribution"]
            sensitivity = diag["sensitivity"]

            if locked_action_indices is not None:
                if step >= len(locked_action_indices):
                    raise ValueError("locked_action_indices does not cover every scene.")
                action_idx = int(locked_action_indices[step])
                if action_idx < 0 or action_idx >= len(probs) or not valid_mask[action_idx]:
                    raise ValueError("locked action index is outside the action space.")
            else:
                action_idx = int(np.random.choice(len(probs), p=probs))
            action_candidate = candidates[action_idx]
            joint_action = dict(action_candidate.actions)
            locked_action = MappingProxyType(dict(joint_action))
            env_action = joint_action_to_env(joint_action)
            decision_audit = None
            if self.decision_provider is not None:
                decision_audit = self.decision_provider.evaluate(
                    DecisionRequest(
                        task="Assess whether the locked quantum action satisfies the narrative constraints.",
                        options=("approve", "reject"),
                        context={
                            "locked_action": joint_action,
                            "selected_candidate": asdict(action_candidate),
                            "valid_candidate_count": int(valid_mask.sum()),
                            "state": state,
                            "constraints": self.env.director_brief.get("constraints", []),
                        },
                    )
                )

            rejection_data = []
            max_prob = float(np.max(probs))
            for i, p in enumerate(probs):
                if i == action_idx:
                    continue
                p = float(p)
                if p < 0.05:
                    reason = "Low posterior mass under current quantum state."
                elif p < max_prob * 0.3:
                    reason = "Trait-state interference suppressed this branch."
                elif p < max_prob * 0.7:
                    reason = "Character-trait misalignment with narrative phase."
                else:
                    reason = "Strong alternative; dominated by marginal utility gap."

                rejection_data.append(
                    {
                        "index": i,
                        "action_id": candidates[i].action_id,
                        "action": dict(candidates[i].actions),
                        "valid": candidates[i].valid,
                        "invalid_reasons": list(candidates[i].invalid_reasons),
                        "prob": p,
                        "reason": (
                            "Invalid precondition: " + "; ".join(candidates[i].invalid_reasons)
                            if not candidates[i].valid else reason
                        ),
                    }
                )
            rejection_data.sort(key=lambda x: x["prob"], reverse=True)

            next_state, env_reward, done = self.env.step(env_action)

            memory_result = None
            memory_context = ()
            if self.memory_enabled:
                try:
                    query = " ".join(
                        [state.get("phase", ""), state.get("genre", "")]
                        + [str(value) for value in locked_action.values()]
                    )
                    memory_result = self.memory_repository.retrieve(
                        self.story_id,
                        query=query,
                        budget=self.memory_budget,
                        limit=self.memory_limit,
                    )
                    memory_context = memory_result.records
                except Exception:
                    logging.getLogger("AetherScribeLab.EpisodeRunner").exception(
                        "Narrative memory retrieval failed; continuing without memory."
                    )

            planner_result = None
            planner_error = None
            if self.planner_enabled:
                try:
                    planner_result = self.planner.plan(
                        PlannerInput(
                            story_id=self.story_id,
                            episode_id=episode_id,
                            scene_index=step,
                            current_state={
                                **state,
                                "canonical_version": committed_state.version,
                                "canonical_facts": list(committed_state.facts[-12:]),
                                "active_conflicts": [item.description for item in committed_state.active_conflicts],
                                "open_threads": list(committed_state.open_threads),
                                "world_locations": list(committed_state.locations),
                                "world_objects": dict(committed_state.objects),
                                "timeline": list(committed_state.timeline[-8:]),
                                "story_contract": {
                                    "central_question": self.story_contract.central_question,
                                    "main_conflict": self.story_contract.main_conflict,
                                    "stakes": self.story_contract.stakes,
                                },
                            },
                            memories=tuple(memory_context),
                            character_knowledge=tuple(
                                CharacterKnowledge(
                                    character_id=name,
                                    constraints=tuple(
                                        ("Known: " + "; ".join(profile.knowledge),)
                                        if profile and profile.knowledge
                                        else ("No additional secret knowledge is established.",)
                                    ) + tuple(
                                        "Belief: " + item
                                        for item in (profile.beliefs if profile else ())
                                    ) + self.user_constraints,
                                )
                                for name in dict.fromkeys(self.env.director_brief.get("characters", {}).values())
                                if name
                                for profile in (committed_state.character_states.get(name),)
                            ),
                            locked_action=LockedActionSnapshot(dict(locked_action)),
                            token_budget=self.memory_budget,
                            seed=seed or 0,
                        )
                    )
                except Exception as exc:
                    planner_error = str(exc)
                    logging.getLogger("AetherScribeLab.EpisodeRunner").exception(
                        "Narrative planning failed; continuing without plan."
                    )

            secret_decisions = []
            secret_guidance = []
            hidden_secrets = []
            if self.secret_enabled:
                try:
                    secret_context = RevealContext(
                        scene_index=step,
                        events=frozenset(
                            {state.get("phase", ""), "scene_completed"}
                        ),
                        character_knowledge=frozenset(),
                        revealed_secret_ids=frozenset(),
                        explicit_instructions=frozenset(),
                    )
                    for secret in self.secret_repository.list(self.story_id):
                        if secret.status is SecretStatus.REVEALED:
                            secret_guidance.append(
                                f"Secret {secret.secret_id} is authorized for public continuity."
                            )
                            continue
                        decision = (
                            self.reveal_gate.evaluate(secret, secret_context)
                            if self.reveal_gate_enabled
                            else RevealDecision(False, ("gate_disabled",))
                        )
                        hidden_secrets.append(secret)
                        secret_decisions.append(
                            {
                                "secret_id": secret.secret_id,
                                "allowed": decision.allowed,
                                "reasons": list(decision.reasons),
                            }
                        )
                        if decision.allowed:
                            secret_guidance.append(
                                f"Secret {secret.secret_id} is eligible but not automatically revealed."
                            )
                        else:
                            secret_guidance.append(
                                f"Secret {secret.secret_id} remains concealed; do not reveal it."
                            )
                except Exception as exc:
                    secret_decisions = [{"error": "secret_gate_failed"}]
                    secret_guidance = ["Secret ledger unavailable; preserve all secrets."]
                    hidden_secrets = []
                    logging.getLogger("AetherScribeLab.EpisodeRunner").exception(
                        "Secret gate failed closed; continuing without secret values."
                    )
            secrets_by_scene[step] = list(hidden_secrets)

            agent_context = {
                "protagonist": self.action_space.protagonist,
                "antagonist": self.action_space.antagonist,
                "ally": self.action_space.ally,
            }
            name_map = {
                role.capitalize(): name
                for role, name in self.env.director_brief.get("characters", {}).items()
            }

            rationale = self.explainer.generate_explanation(
                state_vec,
                joint_action,
                rejections=rejection_data,
                attribution=attribution,
                agents=agent_context,
                name_map=name_map,
            )

            if render_depth == "full":
                deep_rejections = self.renderer.analyze_rejections(
                    state=state,
                    chosen_action=joint_action,
                    rejected_actions=rejection_data,
                    rationale=rationale,
                    language=language,
                )
            else:
                deep_rejections = [
                    r["reason"] for r in rejection_data[:2]
                ]

            render_kwargs = {
                "state": state,
                "actions": locked_action,
                "rationale": rationale,
                "previous_summary": previous_summary,
                "language": language,
                "temperature": temperature,
                "mode": mode,
                "lite_mode": lite_render,
            }
            prior_provisional = next((
                item for item in reversed(episode_log)
                if not item.get("quality_gate", {}).get("accepted")
            ), None)
            if prior_provisional:
                provisional_body = re.search(
                    r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)",
                    str(prior_provisional.get("story", "")), re.I | re.S,
                )
                render_kwargs["previous_summary"] = (
                    "UNVERIFIED PRIOR DRAFT (do not treat as fact): "
                    + (provisional_body.group(1).strip()[:500] if provisional_body else "No usable prose.")
                )
            if self.memory_enabled:
                render_kwargs["memory_context"] = memory_context
            renderer_parameters = inspect.signature(self.renderer.render_scene).parameters
            accepts_extra_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in renderer_parameters.values()
            )
            if "planning_guidance" in renderer_parameters or accepts_extra_kwargs:
                scene_plan = self.story_plan.for_scene(step)
                unverified_prior = [
                    re.sub(r"\s+", " ", str(item.get("story", "")))[:220]
                    for item in episode_log
                    if not item.get("quality_gate", {}).get("accepted")
                ]
                render_kwargs["planning_guidance"] = (
                    f"SCENE NUMBER: {scene_plan.scene_index + 1}",
                    f"ARC PHASE: {scene_plan.phase}",
                    f"SCENE PURPOSE: {scene_plan.purpose}",
                    f"CENTRAL QUESTION: {self.story_contract.central_question}",
                    f"MAIN CONFLICT: {self.story_contract.main_conflict}",
                    f"STAKES: {self.story_contract.stakes}",
                    f"CHARACTERS PRESENT: {', '.join(scene_plan.characters_present)}",
                    f"CHARACTER GOALS: {'; '.join(f'{name}: {goal}' for name, goal in scene_plan.character_goals.items())}",
                    f"SCENE CAUSE: {scene_plan.previous_scene_cause}",
                    f"STARTING CONDITION: {committed_state.events[-1].summary if committed_state.events else self.story_plan.premise}",
                    f"VIEWPOINT GOAL: {scene_plan.protagonist_objective or scene_plan.purpose}",
                    f"OBSTACLE: {scene_plan.antagonist_opposition}",
                    f"REQUIRED EVENT: {scene_plan.event}",
                    f"DECISIVE CHOICE: Show a specific motivated action and the observable alternative it rejects.",
                    f"NEW INFORMATION: {scene_plan.new_information or 'Reveal information only when its source and recipient are shown.'}",
                    f"VALIDATION CLUES: {'; '.join(scene_plan.validation_terms)}",
                    f"EXPECTED CONSEQUENCE: {scene_plan.consequence}",
                    f"SETUP PAYOFF: {'; '.join(scene_plan.setup_payoff_obligations)}",
                    f"ENDING REQUIREMENT: {self.story_contract.intended_ending if step == self.env.max_steps - 1 else 'Do not resolve the central question early; carry forward only consequences already established.'}",
                    f"UNVERIFIED PRIOR SCENE: {unverified_prior[-1] if unverified_prior else 'None'}",
                    f"CONSTRAINTS: {'; '.join(self.story_contract.constraints + self.story_contract.world_rules)}",
                    f"KNOWLEDGE CONSTRAINTS: {'; '.join(planner_result.knowledge_constraints) if planner_result else 'Characters may use only established knowledge.'}",
                    *(
                        planner_result.rendering_guidance
                        if planner_result is not None else ()
                    ),
                )
            if self.secret_enabled:
                render_kwargs["secret_guidance"] = secret_guidance
            if "story_continuity" in inspect.signature(self.renderer.render_scene).parameters:
                render_kwargs["story_continuity"] = self.story_continuity
            # Build each scene contract from the current plan and committed state.
            # It remains advisory for generation; deterministic gates below decide acceptance.
            scene_contract = {
                "scene_id": scene_plan.scene_id,
                "sequence": scene_plan.scene_index + 1,
                "phase": scene_plan.phase,
                "purpose": scene_plan.purpose,
                "pov_character": scene_plan.pov_character,
                "characters_present": list(scene_plan.characters_present),
                "character_goals": dict(scene_plan.character_goals),
                "starting_condition": " ".join(filter(None, (
                    committed_state.events[-1].summary if committed_state.events else self.story_plan.premise,
                    *(
                        "Prior scene generation failed; no unsupported event or fact was established. Continue from the approved plan and canonical state."
                        for failure in scene_failures
                        if int(failure.get("scene_index", -1)) == step - 1
                    ),
                    *(
                        "Provisional scene context (unverified; do not treat as canon): "
                        + str(item.get("story", ""))[:700]
                        for item in episode_log
                        if not item.get("quality_gate", {}).get("accepted")
                    ),
                ))),
                "viewpoint_goal": scene_plan.protagonist_objective or scene_plan.purpose,
                "obstacle": scene_plan.antagonist_opposition,
                "scene_cause": scene_plan.previous_scene_cause,
                "concrete_event": scene_plan.event,
                "decisive_choice": scene_plan.event,
                "immediate_consequence": scene_plan.consequence,
                "ending_state": scene_plan.ending_state,
                "setup_payoff_obligations": list(scene_plan.setup_payoff_obligations),
                "knowledge_changes": {key: list(value) for key, value in scene_plan.knowledge_changes.items()},
                "relationship_changes": list(scene_plan.relationship_changes),
                "next_scene_condition": scene_plan.next_scene_effect or scene_plan.unresolved_objective,
                "intended_state_changes": [
                    candidate.expected_effect for candidate in candidates if candidate.valid
                ][:4],
                "required_characters": list(self.env.director_brief.get("characters", {}).values()),
                "starting_state_version": committed_state.version,
                "starting_facts": list(committed_state.facts[-5:]),
                "character_goals": {
                    name: profile.goal for name, profile in committed_state.character_states.items()
                },
                "character_knowledge": {
                    name: list(profile.knowledge) for name, profile in committed_state.character_states.items()
                },
                "character_beliefs": {
                    name: list(profile.beliefs) for name, profile in committed_state.character_states.items()
                },
                "character_relationships": {
                    name: dict(profile.relationships) for name, profile in committed_state.character_states.items()
                },
                "character_conditions": {
                    name: {"condition": profile.condition, "location": profile.location,
                           "emotional_state": profile.emotional_state}
                    for name, profile in committed_state.character_states.items()
                },
                "established_objects": sorted(committed_state.objects),
                "active_conflicts": [item.description for item in committed_state.active_conflicts],
                "location": scene_plan.location,
                "required_event": scene_plan.event,
                "required_evidence": list(scene_plan.validation_terms),
                "dependencies": [self.story_plan.for_scene(step - 1).scene_id] if step else [],
                "provisional_dependencies": [
                    item.get("scene_id") for item in scene_contracts
                    if item.get("status") == "provisional"
                ],
                "unresolved_dependencies": [
                    self.story_plan.for_scene(step - 1).scene_id
                    for failure in scene_failures
                    if step > 0 and int(failure.get("scene_index", -1)) == step - 1
                ],
                "dependency_policy": "accepted, provisional, or explicitly unresolved predecessor; only accepted state is canonical",
                "allowed_revelations": [item for item in secret_guidance if "eligible" in item],
                "forbidden_revelations": [item for item in secret_guidance if "remains concealed" in item],
                "expected_ending_state": scene_plan.consequence,
                "next_scene_effect": scene_plan.next_scene_effect,
                "continuity_constraints": list(self.story_plan.world_rules),
                "resolution_conditions": (
                    [self.story_plan.narrative_question, self.story_plan.central_conflict,
                     scene_plan.consequence]
                    if scene_plan.phase == "resolution" else []
                ),
                "action_candidates": [
                    {
                        "action_id": item.action_id,
                        "actions": dict(item.actions),
                        "action_type": item.action_type,
                        "target": item.target,
                        "preconditions": list(item.preconditions),
                        "expected_effect": item.expected_effect,
                        "narrative_purpose": item.narrative_purpose,
                        "risks": list(item.risks),
                        "reversible": item.reversible,
                        "valid": item.valid,
                        "invalid_reasons": list(item.invalid_reasons),
                    }
                    for item in candidates
                ],
                "status": "pending",
                "lifecycle": {"status": "pending", "history": []},
            }
            scene_contracts.append(scene_contract)
            lifecycle = SceneLifecycle(scene_contract["scene_id"])
            contract_issues = self.narrative_coordinator.validate_scene_contract(scene_contract)
            if contract_issues:
                lifecycle.transition("failed", reason=";".join(contract_issues))
                scene_contract["status"] = "failed"
                scene_contract["lifecycle"] = lifecycle.snapshot()
                scene_failures.append({
                    "scene_index": step,
                    "failure_type": "planning",
                    "issues": [f"invalid_scene_contract:{issue}" for issue in contract_issues],
                    "retry_count": 0,
                    "provider_error": None,
                    "validation_errors": [{
                        "code": "invalid_scene_contract", "category": "structure",
                        "scene_id": scene_contract["scene_id"], "severity": "error",
                        "message": "The scene blueprint failed coordinator validation before generation.",
                        "evidence": list(contract_issues),
                        "expected": "Complete contract anchored to the current canonical version",
                        "actual": list(contract_issues),
                        "repair_suggestion": "Repair the planner/coordinator contract before requesting prose.",
                        "retry_permitted": False,
                    }],
                })
                state = {**state, "step": step + 1}
                step += 1
                continue
            def _contract_value(value):
                if isinstance(value, MappingProxyType) or isinstance(value, dict):
                    return "; ".join(
                        f"{key}: {', '.join(map(str, item)) if isinstance(item, (list, tuple, set)) else str(item)}"
                        for key, item in value.items()
                    ) or "None"
                if isinstance(value, (list, tuple, set)):
                    return "; ".join(map(str, value)) or "None"
                return str(value) if value else "None"

            contract_lines = tuple(
                f"{key.replace('_', ' ').upper()}: {value if isinstance(value, str) else '; '.join(map(str, value)) if value else 'None'}"
                for key, value in (
                    ("scene id", scene_contract["scene_id"]),
                    ("order", f"{scene_contract['sequence']} of {len(self.story_plan.scenes)}"),
                    ("purpose", scene_contract["purpose"]),
                    ("starting condition", scene_contract["starting_condition"]),
                    ("viewpoint goal", scene_contract["viewpoint_goal"]),
                    ("obstacle", scene_contract["obstacle"]),
                    ("phase", scene_contract["phase"]),
                    ("required characters", scene_contract["required_characters"]),
                    ("location", scene_contract["location"]),
                    ("required event", scene_contract["required_event"]),
                    ("decisive choice", scene_contract["decisive_choice"]),
                    ("required evidence", scene_contract["required_evidence"]),
                    ("causal dependency", scene_contract["dependencies"] or ["user premise"]),
                    ("starting facts", scene_contract["starting_facts"] or ["user premise"]),
                    ("expected ending state", scene_contract["expected_ending_state"]),
                    ("immediate consequence", scene_contract["immediate_consequence"]),
                    ("next scene condition", scene_contract["next_scene_condition"]),
                    ("next scene effect", scene_contract["next_scene_effect"]),
                    ("forbidden revelations", scene_contract["forbidden_revelations"] or ["unsupported facts"]),
                )
            )
            contract_lines = tuple(
                f"{key.replace('_', ' ').upper()}: {_contract_value(value)}"
                for key, value in (
                    ("character goals", scene_contract["character_goals"]),
                    ("character knowledge", scene_contract["character_knowledge"]),
                    ("character beliefs", scene_contract["character_beliefs"]),
                    ("character conditions", scene_contract["character_conditions"]),
                    ("character relationships", scene_contract["character_relationships"]),
                    ("current timeline", committed_state.timeline[-3:]),
                    ("canonical facts", scene_contract["starting_facts"]),
                    ("established objects", scene_contract["established_objects"]),
                    ("active conflicts", scene_contract["active_conflicts"]),
                    ("world rules", scene_contract["continuity_constraints"]),
                )
            ) + contract_lines
            if "planning_guidance" in inspect.signature(self.renderer.render_scene).parameters or accepts_extra_kwargs:
                render_kwargs["planning_guidance"] = tuple(render_kwargs.get("planning_guidance", ())) + contract_lines
            repetition_warning = None
            retry_count = 0
            scene_plan = self.story_plan.for_scene(step)
            action_beat = f"{scene_plan.phase}:{scene_plan.purpose}"
            secret_validation = {"valid": True, "violations": []}
            final_quality = None
            story_text = ""
            lifecycle.transition("generating")
            while True:
                if lifecycle.status == "retrying":
                    lifecycle.transition("generating")
                renderer_parameters = inspect.signature(self.renderer.render_scene).parameters
                accepts_extra_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in renderer_parameters.values()
                )
                if repetition_warning and (
                    "repetition_warning" in renderer_parameters or accepts_extra_kwargs
                ):
                    render_kwargs["repetition_warning"] = repetition_warning
                if "generation_seed" in renderer_parameters or accepts_extra_kwargs:
                    base_seed = int(seed) if seed is not None else 0
                    render_kwargs["generation_seed"] = base_seed + step * 100 + retry_count
                story_text, previous_summary, prompt_trace, coherence_score = self.renderer.render_scene(
                    **render_kwargs,
                )
                lifecycle.transition("generated")
                lifecycle.transition("validating")
                final_quality = validate_scene_output(
                    story_text,
                    step,
                    self.env.director_brief.get("characters", {}),
                    scene_history,
                    action_beat,
                    min_words=int(os.getenv("NARRATIVE_MIN_SCENE_WORDS", "0")),
                    max_words=int(os.getenv("NARRATIVE_MAX_SCENE_WORDS", "0")) or None,
                    required_terms=scene_plan.validation_terms,
                    premise=self.env.director_brief.get("initial_story_premise", ""),
                    forbidden_internal_terms=tuple(
                        str(value) for pair in locked_action.items() for value in pair if value
                    ),
                )
                failure_issues = list(final_quality.issues)
                known_story_context = " ".join((
                    self.story_plan.premise,
                    *[str(name) for name in self.env.director_brief.get("characters", {}).values() if name],
                    *committed_state.facts,
                    *(event.summary for event in committed_state.events),
                    *(
                        str(item.get("story", ""))
                        for item in scene_history if item.get("accepted", True)
                    ),
                    *(fact for profile in committed_state.character_states.values()
                      for fact in profile.knowledge),
                ))
                narrative_findings = audit_scene_narrative(
                    story_text, step, known_context=known_story_context
                )
                failure_issues.extend(finding["code"] for finding in narrative_findings)
                final_scene_issues = ()
                if step == self.env.max_steps - 1:
                    _ending_ok, final_scene_issues = validate_story_ending(
                        story_text,
                        self.env.director_brief.get("characters", {}),
                        premise=self.story_plan.premise,
                    )
                    failure_issues.extend(
                        f"episode_resolution:{issue}" for issue in final_scene_issues
                    )
                provider_metadata = getattr(self.renderer, "last_provider_metadata", {}) or {}
                if provider_metadata.get("finish_reason") == "length":
                    failure_issues.append("provider_truncated_output")
                if getattr(self.renderer, "last_used_fallback", False):
                    failure_issues.append("template_fallback_not_real_generation")
                if self.secret_enabled and hidden_secrets:
                    validation = validate_prose(story_text, hidden_secrets)
                    secret_validation = {
                        "valid": validation.valid,
                        "violations": [
                            {"secret_id": item.secret_id, "evidence": item.evidence}
                            for item in validation.violations
                        ],
                    }
                    if not validation.valid:
                        failure_issues.append("protected_secret_leak")
                narrative_match = re.search(
                    r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
                    story_text, re.IGNORECASE | re.DOTALL,
                )
                candidate_prose = narrative_match.group(1).strip() if narrative_match else ""
                scene_plan = self.story_plan.for_scene(step)
                event = NarrativeEvent(
                    event_id=f"{episode_id}:{scene_plan.scene_id or step}",
                    scene_id=scene_plan.scene_id or f"scene-{step + 1:04d}",
                    sequence=step + 1,
                    summary=candidate_prose[:300],
                    participants=tuple(scene_plan.characters_present),
                    location=scene_plan.location,
                    time=scene_plan.time,
                    # Prose summaries remain event context, not canonical facts.
                    # Facts/knowledge require explicit deterministic evidence.
                    facts=(),
                    decisions={
                        self.env.director_brief.get("characters", {}).get(role.lower(), role): action
                        for role, action in locked_action.items()
                    },
                    consequences=(scene_plan.consequence,),
                    phase=scene_plan.phase,
                )
                plan_contract = {
                    **dict(scene_contract),
                    "scene_id": scene_plan.scene_id or f"scene-{step + 1:04d}",
                    "purpose": scene_plan.purpose,
                }
                coordination_review = self.narrative_coordinator.review(event, plan_contract, retry_count)
                if not coordination_review.accepted:
                    failure_issues.extend(f"agent_review:{reason}" for reason in (coordination_review.revision.reasons if coordination_review.revision else ()))
                if not failure_issues:
                    break
                if retry_count >= self.repetition_retry_limit:
                    lifecycle.transition("failed", reason=";".join(failure_issues))
                    break
                lifecycle.transition("revision_required", reason=";".join(failure_issues))
                retry_count += 1
                lifecycle.transition("retrying")
                corrections = []
                if "provider_truncated_output" in failure_issues or "truncated_output" in failure_issues:
                    corrections.append(
                        "The previous response hit its output limit. Write exactly three complete sentences and no more than 80 words. "
                        "Finish every sentence and preserve the planned event and consequence."
                    )
                if "scene_above_maximum_length" in failure_issues:
                    maximum_scene_words = int(os.getenv("NARRATIVE_MAX_SCENE_WORDS", "130"))
                    retry_word_limit = min(
                        maximum_scene_words - 5,
                        max(75, int(maximum_scene_words * 0.70)),
                    )
                    corrections.append(
                        f"Cut the passage to {retry_word_limit} words maximum. "
                        "Keep only the action that changes the situation and its consequence."
                    )
                if "repeated_scene_opening" in failure_issues:
                    rejected_opening = (
                        final_quality.repetition.matched_opening
                        or re.split(r"(?<=[.!?])\s+", candidate_prose.strip(), maxsplit=1)[0]
                    )[:180]
                    character_names = [name for name in self.env.director_brief.get("characters", {}).values() if name]
                    rejected_subject = next(
                        (name for name in character_names if rejected_opening.casefold().startswith(str(name).casefold())),
                        None,
                    )
                    alternate_names = [name for name in character_names if name != rejected_subject]
                    required_opening_name = alternate_names[(retry_count - 1) % len(alternate_names)] if alternate_names else None
                    retry_voice = (
                        f"The FIRST WORD of the first sentence MUST be {required_opening_name}. Show {required_opening_name} taking a distinct action."
                        if required_opening_name else
                        "Begin with dialogue or a concrete setting change, not the rejected subject."
                    )
                    corrections.append(
                        f"REQUIRED: replace the opening sentence completely; {retry_voice}. "
                        f"The rejected opening was: {rejected_opening!r}. Do not reuse its wording, subject, action, image, or rhythm. "
                        "Keep all other scene requirements and produce a complete scene meeting the minimum length."
                    )
                if "repeated_sentence" in failure_issues or "similar_scene" in failure_issues:
                    corrections.append(
                        "Use new actions, dialogue, images, and sentence openings. Do not reuse any sentence or exchange from an earlier scene."
                    )
                if "protected_secret_leak" in failure_issues:
                    corrections.append("Remove all protected values and aliases without naming them.")
                if "template_fallback_not_real_generation" in failure_issues:
                    corrections.append("The provider returned template fallback, not model prose; do not accept it.")
                if "abstract_development_without_event" in failure_issues:
                    corrections.append(
                        "Do not summarize resolve, tension, sacrifice, or consequences. Dramatize the specific action, "
                        "show the other character's response, and state the concrete change it causes."
                    )
                if "unsupported_family_relationship" in failure_issues:
                    corrections.append(
                        "Do not introduce an unestablished relative or family motive. Keep the established cast and relationships."
                    )
                if "unseeded_plot_device" in failure_issues:
                    corrections.append(
                        "Remove the unseeded plot device. Resolve the beat using only objects and mechanisms already established in the premise or accepted scenes."
                    )
                if "unexplained_location_transition" in failure_issues:
                    corrections.append(
                        "Keep the scene in the established location or show a clear physical transition from the prior accepted scene."
                    )
                if "unsupported_character_knowledge" in failure_issues:
                    corrections.append(
                        "Do not give a character prior knowledge without showing when and how they learned it; dramatize the discovery in this scene if appropriate."
                    )
                if any(issue.startswith("episode_resolution:") for issue in failure_issues):
                    corrections.append(
                        "This is the final scene. Show an earned outcome for the central conflict, "
                        "name the protagonist and opposing character, and state a concrete lasting consequence. "
                        "Answer the central question with an event that already occurred. Do not leave the decision or outcome awaiting, "
                        "offer unresolved alternatives, defer the outcome to a future scene, or say its impact is unknown. "
                        "Do not end on a question or introduce a new conflict."
                    )
                general_correction = final_quality.correction_instruction or "Correct the scene validation issues."
                focused_corrections = " ".join(corrections)
                repetition_warning = (
                    f"RETRY {retry_count} OF {self.repetition_retry_limit}. "
                    f"{focused_corrections} {general_correction}"
                ).strip()

            continuity_validation = final_quality.continuity
            repetition = final_quality.repetition
            if failure_issues or not final_quality.accepted or not secret_validation["valid"] or getattr(self.renderer, "last_used_fallback", False):
                failure_issues = list(dict.fromkeys((*failure_issues, *final_quality.issues)))
                if not secret_validation["valid"]:
                    failure_issues.append("protected_secret_leak")
                if getattr(self.renderer, "last_used_fallback", False):
                    failure_issues.append("template_fallback_not_real_generation")
                scene_failures.append({
                    "scene_index": step,
                    "failure_type": (
                        "provider" if not candidate_prose.strip()
                        else "provider_fallback" if getattr(self.renderer, "last_used_fallback", False)
                        else "validation"
                    ),
                    "issues": list(dict.fromkeys(failure_issues)),
                    "validation_errors": [
                        *[asdict(error) for error in getattr(final_quality, "errors", ())],
                        *[{
                            "code": finding["code"], "category": "narrative_quality",
                            "scene_id": finding["scene_id"], "severity": "error",
                            "message": finding["message"], "evidence": finding["evidence"],
                            "expected": "Dramatized event and observable consequence",
                            "actual": finding["message"],
                            "repair_suggestion": finding["repair_suggestion"],
                            "retry_permitted": True,
                        } for finding in narrative_findings],
                    ],
                    "retry_count": retry_count,
                    "provider_error": getattr(self.renderer, "last_error", None),
                    "finish_reason": (getattr(self.renderer, "last_provider_metadata", {}) or {}).get("finish_reason"),
                    "output_tokens": getattr(self.renderer, "scene_output_tokens", None)
                    or getattr(self.renderer, "last_output_tokens", None),
                    "input_tokens": getattr(self.renderer, "scene_input_tokens", None)
                    or getattr(self.renderer, "last_input_tokens", None),
                    "last_provider_call_output_tokens": getattr(self.renderer, "last_output_tokens", None),
                    "last_provider_call_input_tokens": getattr(self.renderer, "last_input_tokens", None),
                    "agent_proposals": [asdict(proposal) for proposal in coordination_review.proposals],
                    "agent_disagreements": [asdict(item) for item in coordination_review.disagreements],
                })
                if candidate_prose.strip():
                    lifecycle.transition("provisional", reason="draft retained after validation retries")
                    scene_contract["status"] = "provisional"
                else:
                    scene_contract["status"] = "failed"
                scene_contract["lifecycle"] = lifecycle.snapshot()
                if candidate_prose.strip():
                    # Preserve the best generated prose with its failed gates, but never
                    # commit it to canonical state or present it as accepted.
                    reader_story = f"[SCENE {step + 1}]\nNARRATIVE:\n{candidate_prose}"
                    provisional_entry = {
                        "step": step,
                        "state": state,
                        "action": joint_action,
                        "action_idx": action_idx,
                        "action_candidate": {
                            **asdict(action_candidate),
                            "actions": dict(action_candidate.actions),
                        },
                        "candidate_selection": {
                            "method": "stochastic sampling from VQC posterior conditioned on valid candidates",
                            "valid_candidate_count": int(valid_mask.sum()),
                            "invalid_candidate_count": int((~valid_mask).sum()),
                            "candidate_ids": [candidate.action_id for candidate in candidates],
                            "candidate_validity": valid_mask.tolist(),
                            "measured_posterior": measured_posterior.tolist(),
                            "valid_conditioned_posterior": probs.tolist(),
                            "selected_candidate_id": action_candidate.action_id,
                        },
                        "probs": probs.tolist(),
                        "selected_probability": float(probs[action_idx]),
                        "peak_probability": float(np.max(probs)),
                        "rejections": rejection_data,
                        "deep_rejections": [],
                        "attribution": attribution,
                        "sensitivity": sensitivity,
                        "risk": float(risk_score),
                        "rationale": action_beat,
                        "weights": self.policy.weights.tolist(),
                        "action_idx": action_idx,
                        "story": reader_story,
                        "scene_plan": {
                            "scene_id": scene_plan.scene_id,
                            "title": scene_plan.title or f"Scene {step + 1}",
                            "purpose": scene_plan.purpose,
                            "phase": scene_plan.phase,
                        },
                        "coherence": float(coherence_score) if isinstance(coherence_score, (int, float)) else None,
                        "entropy": entropy,
                        "reward": 0.0,
                        "quality_gate": {
                            "accepted": False,
                            "status": "provisional",
                            "issues": list(dict.fromkeys(failure_issues)),
                        },
                        "generation_status": "generated_provisional",
                        "validation_status": "failed",
                        "validation_errors": scene_failures[-1]["validation_errors"],
                        "retry_count": retry_count,
                        "diagnostics": scene_failures[-1],
                    }
                    episode_log.append(provisional_entry)
                    scene_history.append({
                        "episode_id": episode_id,
                        "scene_index": step,
                        "story": reader_story,
                        "summary": previous_summary or "",
                        "outcome": previous_summary or "",
                        "opening": re.sub(
                            r"\W+", " ",
                            re.split(r"(?<=[.!?])\s+", candidate_prose.strip(), maxsplit=1)[0].lower(),
                        ).strip()[:180],
                        "ending": re.split(r"(?<=[.!?])\s+", candidate_prose.strip())[-1].lower(),
                        "beat": action_beat,
                        "objective": scene_plan.purpose,
                        "accepted": False,
                    })
                    if on_step is not None:
                        on_step(provisional_entry, step + 1, self.env.max_steps)
                # Provisional prose is context only. Keep later contracts tied to
                # accepted canonical state, while allowing explicitly listed gaps.
                state = next_state
                step += 1
                continue

            if dict(locked_action) != joint_action:
                raise RuntimeError("Locked quantum action changed during narrative rendering.")

            narrative_match = re.search(
                r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
                story_text,
                re.IGNORECASE | re.DOTALL,
            )
            decision_trace_match = re.search(
                r"\[QUANTUM_TRACE\]\s*(.*?)(?=\[COHERENCE\]|$)",
                story_text,
                re.IGNORECASE | re.DOTALL,
            )
            narrative_text = narrative_match.group(1).strip() if narrative_match else ""
            reader_story = f"[SCENE {step + 1}]\nNARRATIVE:\n{narrative_text}"
            decision_trace = decision_trace_match.group(1).strip() if decision_trace_match else ""

            coordination_commit = self.narrative_coordinator.commit(
                event, plan_contract, expected_version=scene_contract["starting_state_version"]
            )
            if not coordination_commit.accepted or coordination_commit.update is None:
                raise RuntimeError("coordinator rejected a scene after final validation")
            scene_contract["status"] = "accepted"
            lifecycle.transition("accepted")
            scene_contract["lifecycle"] = lifecycle.snapshot()
            self.world_state = self.world_state.apply_committed(event, coordination_commit.update)
            adaptation_records = self.character_adapter.apply_committed_event(event)
            candidate_indices = np.argsort(probs)[-3:][::-1]
            prediction_candidates = tuple(
                " / ".join(f"{role}: {action}" for role, action in candidates[int(candidate)].actions.items())
                for candidate in candidate_indices
            )
            prediction_report = self.future_predictor.predict(
                {"version": self.world_state.version,
                 "entities": tuple(sorted(self.world_state.entities)),
                 "facts": self.narrative_coordinator.state.facts},
                prediction_candidates,
            )

            memory_updates = 0
            memory_rejections = []
            if self.memory_enabled and final_quality.accepted:
                memory_candidates = extract_authoritative_candidates(
                    self.story_id,
                    scene={
                        "authoritative_facts": []
                    },
                    action={"description": "VQC selected: " + " | ".join(
                        f"{role}: {action}" for role, action in locked_action.items()
                    )},
                    event={
                        "type": "scene_transition",
                        "description": f"Accepted scene {step + 1} in {state.get('phase', 'unknown')} phase.",
                    },
                    episode_id=episode_id,
                    scene_index=step,
                )
                for memory_candidate in memory_candidates:
                    try:
                        self.memory_repository.save(memory_candidate)
                        memory_updates += 1
                    except Exception as exc:
                        memory_rejections.append(str(exc))

            measured_coherence = (
                float(coherence_score)
                if isinstance(coherence_score, (int, float)) else None
            )
            composite_reward = (
                env_reward * measured_coherence
                if final_quality.accepted and measured_coherence is not None
                else env_reward if final_quality.accepted else 0.0
            )
            total_reward += composite_reward
            if final_quality.accepted:
                self.policy.update(composite_reward, state_vec, action_idx, lr=lr)

            log_entry = {
                "step": step,
                "state": state,
                "action": joint_action,
                "env_action": env_action,
                "action_idx": action_idx,
                "action_candidate": {
                    **asdict(action_candidate),
                    "actions": dict(action_candidate.actions),
                },
                "candidate_selection": {
                    "method": "stochastic sampling from VQC posterior conditioned on valid candidates",
                    "valid_candidate_count": int(valid_mask.sum()),
                    "invalid_candidate_count": int((~valid_mask).sum()),
                    "candidate_ids": [candidate.action_id for candidate in candidates],
                    "candidate_validity": valid_mask.tolist(),
                    "measured_posterior": measured_posterior.tolist(),
                    "valid_conditioned_posterior": probs.tolist(),
                    "selected_candidate_id": action_candidate.action_id,
                },
                "selected_probability": float(probs[action_idx]),
                "peak_probability": float(np.max(probs)),
                "sampling_mode": "stochastic-conditioned-on-valid-candidates",
                "random_seed": seed,
                "decision_audit": (
                    {
                        "selected": decision_audit.selected,
                        "approved": decision_audit.approved,
                        "rationale": decision_audit.rationale,
                        "provider": decision_audit.provider,
                        "model": decision_audit.model,
                        "latency_ms": decision_audit.latency_ms,
                        "used_fallback": decision_audit.used_fallback,
                        "error": decision_audit.error,
                    }
                    if decision_audit is not None else None
                ),
                "probs": probs.tolist(),
                "entropy": float(entropy),
                "risk": risk_score,
                "attribution": attribution,
                "rejections": rejection_data,
                "deep_rejections": deep_rejections,
                "sensitivity": sensitivity,
                "coherence": measured_coherence,
                "rationale": rationale,
                "story": reader_story,
                "decision_trace": decision_trace,
                "summary": previous_summary,
                "narrative_provider": getattr(self.renderer, "provider", None),
                "narrative_model": getattr(self.renderer, "model", None),
                "narrative_latency_ms": getattr(self.renderer, "last_latency_ms", None),
                "narrative_input_tokens": getattr(self.renderer, "last_input_tokens", None),
                "narrative_output_tokens": getattr(self.renderer, "last_output_tokens", None),
                "narrative_estimated_cost_usd": getattr(self.renderer, "last_estimated_cost_usd", None),
                "narrative_total_latency_ms": getattr(self.renderer, "scene_latency_ms", None),
                "narrative_total_input_tokens": getattr(self.renderer, "scene_input_tokens", None),
                "narrative_total_output_tokens": getattr(self.renderer, "scene_output_tokens", None),
                "narrative_total_cost_usd": getattr(self.renderer, "scene_estimated_cost_usd", None),
                "narrative_fallback": getattr(self.renderer, "last_used_fallback", False),
                "story_id": self.story_id,
                "episode_id": episode_id,
                "memory_enabled": self.memory_enabled,
                "memory_retrieval": (
                    {
                        "selected_ids": [record.id for record in memory_context],
                        "selected": len(memory_context),
                        "diagnostics": {
                            "candidates": memory_result.diagnostics.candidates,
                            "selected": memory_result.diagnostics.selected,
                            "deduplicated": memory_result.diagnostics.deduplicated,
                            "truncated": memory_result.diagnostics.truncated,
                            "budget": memory_result.diagnostics.budget,
                            "used": memory_result.diagnostics.used,
                        },
                    }
                    if memory_result is not None else
                    {"selected_ids": [], "selected": 0, "diagnostics": None}
                ),
                "memory_updates": memory_updates,
                "memory_rejections": memory_rejections,
                "planner": {
                    "enabled": self.planner_enabled,
                    "provider": getattr(planner_result, "provider", None),
                    "model": getattr(planner_result, "model", None),
                    "latency_ms": getattr(planner_result, "latency_ms", None),
                    "confidence": getattr(planner_result, "confidence", None),
                    "validation_status": getattr(planner_result, "validation_status", "error" if planner_error else None),
                    "scene_objective": getattr(planner_result, "scene_objective", None),
                    "pacing": getattr(planner_result, "pacing", None),
                    "relevant_memory_ids": list(getattr(planner_result, "relevant_memory_ids", ())),
                    "continuity_risks": list(getattr(planner_result, "continuity_risks", ())),
                    "secret_reveal_guidance": getattr(planner_result, "secret_reveal_guidance", None),
                    "error": planner_error,
                },
                "scene_plan": {
                    "scene_number": self.story_plan.for_scene(step).scene_index + 1,
                    "scene_id": self.story_plan.for_scene(step).scene_id or f"scene-{step + 1:04d}",
                    "title": self.story_plan.for_scene(step).title,
                    "pov_character": self.story_plan.for_scene(step).pov_character,
                    "location": self.story_plan.for_scene(step).location,
                    "time": self.story_plan.for_scene(step).time,
                    "phase": self.story_plan.for_scene(step).phase,
                    "purpose": self.story_plan.for_scene(step).purpose,
                    "protagonist_objective": self.story_plan.for_scene(step).protagonist_objective,
                    "antagonist_opposition": self.story_plan.for_scene(step).antagonist_opposition,
                    "ally_contribution": self.story_plan.for_scene(step).ally_contribution,
                    "event": self.story_plan.for_scene(step).event,
                    "new_information": self.story_plan.for_scene(step).new_information,
                    "validation_terms": list(self.story_plan.for_scene(step).validation_terms),
                    "consequence": self.story_plan.for_scene(step).consequence,
                    "next_scene_effect": self.story_plan.for_scene(step).next_scene_effect,
                    "contract": scene_contract,
                },
                "coordination": {
                    "accepted": coordination_commit.accepted,
                    "state_version": coordination_commit.state.version,
                    "state_update_id": coordination_commit.update.update_id,
                    "agents": [proposal.agent for proposal in coordination_commit.proposals],
                    "disagreements": [item.__dict__ for item in coordination_commit.disagreements],
                },
                "world_state_version": self.world_state.version,
                "predictions": [
                    {"future_id": future.future_id, "label": future.label,
                     "proposed_action": future.proposed_action,
                     "events": list(future.predicted_events),
                     "status": future.status, "plausibility": future.plausibility,
                     "risks": list(future.risks)}
                    for future in prediction_report.futures
                ],
                "character_adaptation": [
                    {"character_id": record.character_id, "event_id": record.event_id,
                     "metrics": dict(record.metrics), "grounded_facts": list(record.grounded_facts)}
                    for record in adaptation_records
                ],
                "secret_gate": {
                    "enabled": self.secret_enabled,
                    "gate_enabled": self.reveal_gate_enabled,
                    "decisions": secret_decisions,
                    "policy": "fail-closed",
                },
                "secret_validation": secret_validation,
                "continuity": {
                    "valid": continuity_validation.valid,
                    "issues": list(final_quality.issues),
                    "situation": self.story_continuity.situation,
                    "goal": self.story_continuity.protagonist_goal,
                    "purpose": self.story_continuity.next_scene_purpose,
                },
                "quality_gate": {
                    "accepted": final_quality.accepted,
                    "issues": list(final_quality.issues),
                },
                "generation_status": "generated_accepted",
                "validation_status": "accepted",
                "repetition": {
                    "repeated": repetition.repeated,
                    "reasons": list(repetition.reasons),
                    "similarity": repetition.similarity,
                    "retry_count": retry_count,
                },
                "prompt_trace": prompt_trace,
                "weights": self.policy.weights.tolist(),
                "reward": float(composite_reward),
            }
            episode_log.append(log_entry)
            if self.scene_history_repository is not None:
                body = narrative_text
                sentences = [item.strip() for item in body.split(".") if item.strip()]
                opening = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0]
                opening = re.sub(r"\W+", " ", opening.lower()).strip()[:180]
                self.scene_history_repository.add(
                    story_id=self.story_id,
                    episode_id=episode_id,
                    scene_index=step,
                    story=reader_story,
                    summary=previous_summary or "",
                    outcome=previous_summary or "",
                    opening=opening,
                    ending=sentences[-1].lower() if sentences else "",
                    beat="|".join(f"{key}:{value}" for key, value in locked_action.items()),
                    objective=self.story_continuity.next_scene_purpose,
                    accepted=final_quality.accepted,
                )
            body = narrative_text
            sentences = [item.strip() for item in body.split(".") if item.strip()]
            opening = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0]
            opening = re.sub(r"\W+", " ", opening.lower()).strip()[:180]
            scene_history.append({
                "episode_id": episode_id,
                "scene_index": step,
                "story": reader_story,
                "summary": previous_summary or "",
                "outcome": previous_summary or "",
                "opening": opening,
                "ending": sentences[-1].lower() if sentences else "",
                "beat": action_beat,
                "objective": self.story_plan.for_scene(step).purpose,
                "accepted": final_quality.accepted,
            })
            if final_quality.accepted:
                next_continuity = self.story_continuity.after_scene(
                    previous_summary,
                    dict(locked_action),
                    (
                        "Advance the established conflict through a concrete consequence."
                        if step + 1 < self.env.max_steps
                        else "Deliver a consequential scene ending consistent with the established conflict."
                    ),
                )
                committed_state = self.narrative_coordinator.state
                self.story_continuity = replace(
                    next_continuity,
                    established_facts=tuple(committed_state.facts),
                    character_knowledge={
                        name: tuple(profile.knowledge)
                        for name, profile in committed_state.character_states.items()
                        if profile.knowledge
                    },
                    unresolved_conflicts=tuple(
                        conflict.description for conflict in committed_state.active_conflicts
                        if conflict.status == "active"
                    ) or next_continuity.unresolved_conflicts,
                )

            if on_step is not None:
                on_step(log_entry, step + 1, self.env.max_steps)

            if verbose:
                print(f"[Scene {step + 1}] reward={composite_reward:.3f} tension={state['tension']:.3f}")

            state = next_state
            step += 1

        generated_indices = {int(scene.get("step", -1)) for scene in episode_log}
        for plan in self.story_plan.scenes:
            if any(contract["sequence"] == plan.scene_index + 1 for contract in scene_contracts):
                continue
            scene_contracts.append({
                "scene_id": plan.scene_id,
                "sequence": plan.scene_index + 1,
                "phase": plan.phase,
                "purpose": plan.purpose,
                "starting_condition": self.story_plan.premise,
                "viewpoint_goal": plan.protagonist_objective or plan.purpose,
                "obstacle": plan.antagonist_opposition,
                "concrete_event": plan.event,
                "decisive_choice": plan.event,
                "immediate_consequence": plan.consequence,
                "next_scene_condition": plan.next_scene_effect or plan.unresolved_objective,
                "intended_state_changes": [],
                "required_characters": list(self.env.director_brief.get("characters", {}).values()),
                "starting_state_version": None,
                "starting_facts": [],
                "character_goals": dict(self.story_plan.character_goals),
                "character_knowledge": {},
                "character_beliefs": {},
                "character_relationships": {},
                "character_conditions": {},
                "established_objects": [],
                "active_conflicts": [],
                "location": plan.location,
                "required_event": plan.event,
                "required_evidence": list(plan.validation_terms),
                "dependencies": [self.story_plan.for_scene(plan.scene_index - 1).scene_id] if plan.scene_index else [],
                "dependency_policy": "pending scene contracts use generated status only; accepted dependencies remain canonical",
                "allowed_revelations": [],
                "forbidden_revelations": ["unsupported facts"],
                "expected_ending_state": plan.consequence,
                "next_scene_effect": plan.next_scene_effect,
                "continuity_constraints": list(self.story_plan.world_rules),
                "resolution_conditions": (
                    [self.story_plan.narrative_question, self.story_plan.central_conflict, plan.consequence]
                    if plan.phase == "resolution" else []
                ),
                "status": "pending",
                "lifecycle": {
                    "scene_id": plan.scene_id,
                    "status": "pending",
                    "history": [],
                },
            })
        scene_contracts.sort(key=lambda contract: contract["sequence"])
        manuscript_repairs = []
        if len({int(scene.get("step", -1)) for scene in episode_log}) == self.env.max_steps:
            def _audit_complete_story(items, requested):
                return audit_story(
                    items, requested, self.env.director_brief.get("characters", {}),
                    story_contract=self.story_contract.as_dict(),
                )

            def _repair_one_scene(scene, manuscript, findings):
                index = int(scene["step"])
                plan = self.story_plan.for_scene(index)
                context_lines = []
                for item in manuscript:
                    prose = re.search(
                        r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)",
                        str(item.get("story", "")), re.I | re.S,
                    )
                    if prose:
                        context_lines.append(f"Scene {int(item['step']) + 1}: {prose.group(1).strip()[:240]}")
                issue_text = "; ".join(str(item.get("message", item.get("code", "repair"))) for item in findings)
                guidance = (
                    f"Manuscript-level revision for scene {index + 1}/{self.env.max_steps}. "
                    f"Purpose: {plan.purpose}. Event: {plan.event}. Consequence: {plan.consequence}. "
                    f"Required cast: {', '.join(str(value) for value in self.env.director_brief.get('characters', {}).values())}. "
                    f"Preserve established plot facts and repair these whole-story findings: {issue_text}. "
                    "Write the scene as dramatized action with a goal, obstacle, decision, and changed consequence."
                )
                repaired_text, _, _, _ = self.renderer.render_scene(
                    state=scene["state"],
                    actions=scene.get("action", {}),
                    rationale="Bounded whole-manuscript revision; retain the locked action.",
                    previous_summary="\n".join(context_lines)[:2600],
                    language=language,
                    mode=mode,
                    lite_mode=True,
                    planning_guidance=[guidance],
                    repetition_warning=(
                        "Repair the supplied manuscript findings in a fresh complete scene. "
                        "Do not add unsupported people, relationships, or events."
                    ),
                    generation_seed=(int(seed) if seed is not None else 0) + 10000 + index,
                )
                narrative = re.search(
                    r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|\[COHERENCE\]|$)",
                    repaired_text, re.I | re.S,
                )
                body = narrative.group(1).strip() if narrative else repaired_text.strip()
                if not body:
                    return None
                return {"story": f"[SCENE {index + 1}]\nNARRATIVE:\n{body}"}

            def _validate_manuscript_repair(candidate, index):
                scene = episode_log[index]
                prose_match = re.search(
                    r"NARRATIVE:\s*(.*?)(?=\[QUANTUM_TRACE\]|$)",
                    str(candidate.get("story", "")), re.I | re.S,
                )
                body = prose_match.group(1).strip() if prose_match else ""
                quality = validate_scene_output(
                    body, index, self.env.director_brief.get("characters", {}),
                    [item for item in scene_history if not (
                        item.get("episode_id") == episode_id
                        and int(item.get("scene_index", -1)) == index
                    )],
                    "|".join(f"{role}:{action}" for role, action in scene.get("action", {}).items()),
                    min_words=int(os.getenv("NARRATIVE_MIN_SCENE_WORDS", "0")),
                    max_words=int(os.getenv("NARRATIVE_MAX_SCENE_WORDS", "0")) or None,
                    required_terms=self.story_plan.for_scene(index).validation_terms,
                    premise=self.env.director_brief.get("initial_story_premise", ""),
                    forbidden_internal_terms=tuple(
                        str(value) for pair in scene.get("action", {}).items() for value in pair if value
                    ),
                )
                issues = list(quality.issues)
                if audit_scene_narrative(body, index, known_context=self.story_plan.premise):
                    issues.append("whole_story_repair_failed_narrative_audit")
                if index == self.env.max_steps - 1:
                    valid_ending, ending_issues = validate_story_ending(
                        body,
                        self.env.director_brief.get("characters", {}),
                        premise=self.story_plan.premise,
                    )
                    if not valid_ending:
                        issues.extend(f"ending:{issue}" for issue in ending_issues)
                if self.secret_enabled and secrets_by_scene.get(index):
                    secret_check = validate_prose(body, secrets_by_scene[index])
                    if not secret_check.valid:
                        issues.append("protected_secret_leak")
                if getattr(self.renderer, "last_used_fallback", False):
                    issues.append("template_fallback_not_real_generation")
                if (getattr(self.renderer, "last_provider_metadata", {}) or {}).get("finish_reason") == "length":
                    issues.append("provider_truncated_output")
                return not issues, list(dict.fromkeys(issues))

            episode_log, whole_story_audit, manuscript_repairs = repair_manuscript(
                episode_log,
                self.env.max_steps,
                audit=_audit_complete_story,
                repair_scene=_repair_one_scene,
                validate_scene=_validate_manuscript_repair,
            )
        final_scene_present = bool(episode_log and episode_log[-1].get("step") == self.env.max_steps - 1)
        final_scene_accepted = bool(
            final_scene_present and episode_log[-1].get("quality_gate", {}).get("accepted")
        )
        ending_valid, ending_issues = (
            validate_story_ending(
                episode_log[-1]["story"],
                self.env.director_brief.get("characters", {}),
                premise=self.story_plan.premise,
            )
            if final_scene_present else (
                False,
                ("final_scene_not_generated",),
            )
        )
        all_accepted = (
            len(episode_log) == self.env.max_steps
            and all(scene.get("quality_gate", {}).get("accepted") for scene in episode_log)
        )
        assembled_scene_count = len(generated_indices)
        whole_story_audit = audit_story(
            episode_log, self.env.max_steps, self.env.director_brief.get("characters", {}),
            story_contract=self.story_contract.as_dict(),
        )
        complete = all_accepted and ending_valid and whole_story_audit["passed"]
        return {
            "id": str(uuid.uuid4()),
            "episode_id": episode_id,
            "scenes": episode_log,
            "total_reward": float(total_reward),
            "mode": mode,
            "render_depth": render_depth,
            "traits": trait_vec.tolist(),
            "characters": self.env.director_brief.get("characters", {}),
            "story_id": self.story_id,
            "requested_scene_count": self.env.max_steps,
            "planned_scene_count": len(self.story_plan.scenes),
            "story_bible": {
                "premise": self.story_plan.premise,
                "genre": self.story_plan.genre,
                "narrative_question": self.story_plan.narrative_question,
                "central_conflict": self.story_plan.central_conflict,
                "stakes": self.story_plan.stakes,
                "character_goals": dict(self.story_plan.character_goals),
                "world_rules": list(self.story_plan.world_rules),
                "open_threads": list(self.story_plan.open_threads),
                "turning_points": list(self.story_plan.turning_points),
            },
            "story_plan": [
                {
                    "scene_id": plan.scene_id or f"scene-{plan.scene_index + 1:04d}",
                    "sequence": plan.scene_index + 1,
                    "title": plan.title or f"Scene {plan.scene_index + 1}",
                    "phase": plan.phase,
                    "pov_character": plan.pov_character,
                    "location": plan.location,
                    "time": plan.time,
                    "objective": plan.purpose,
                    "change": plan.story_change or plan.consequence,
                    "next_scene_reason": plan.next_scene_effect,
                }
                for plan in self.story_plan.scenes
            ],
            "story_contract": self.story_contract.as_dict(),
            "scene_blueprints": [dict(blueprint) for blueprint in self.scene_blueprints],
            "scene_contracts": scene_contracts,
            "world_state": {
                "story_id": self.world_state.story_id,
                "version": self.world_state.version,
                "entities": {key: entity.__dict__ for key, entity in self.world_state.entities.items()},
                "committed_event_ids": list(self.world_state.committed_event_ids),
            },
            "world_usda": self.world_state.to_usda(),
            "world_visualization": __import__("world_representation").OmniverseAdapter.status(),
            "prediction_method": "transparent-rule-based",
            "narrative_state_version": self.narrative_coordinator.state.version,
            "state_updates": [asdict(update) for update in self.narrative_coordinator.audit],
            "character_profiles": {key: asdict(profile) for key, profile in self.character_adapter.profiles.items()},
            "adaptation_records": [asdict(record) for record in self.character_adapter.records],
            "policy_weights": self.policy.weights.tolist(),
            "accepted_scene_count": sum(
                1 for scene in episode_log if scene.get("quality_gate", {}).get("accepted")
            ),
            "generated_scene_count": assembled_scene_count,
            "assembled_scene_count": assembled_scene_count,
            "provisional_scene_count": sum(
                1 for scene in episode_log if scene.get("generation_status") == "generated_provisional"
            ),
            "failed_scene_count": len(scene_failures),
            "attempted_scene_count": len(attempted_scene_indices),
            "generation_complete": assembled_scene_count == self.env.max_steps,
            "validation_complete": (
                len(scene_failures) == 0 and assembled_scene_count == self.env.max_steps
                and ending_valid and whole_story_audit["passed"]
            ),
            "planned_scene_indices": list(range(self.env.max_steps)),
            "attempted_scene_indices": attempted_scene_indices,
            "blocked_scene_indices": [],
            "pending_scene_indices": [index for index in range(self.env.max_steps) if index not in generated_indices],
            "failed_scene_indices": [failure["scene_index"] for failure in scene_failures],
            "completion_conditions": {
                "all_planned_scenes_accepted": all_accepted,
                "final_scene_accepted": final_scene_accepted,
                "ending_validation_passed": ending_valid,
                "whole_story_audit_passed": whole_story_audit["passed"],
            },
            "story_complete": complete,
            "status": "complete" if complete else (
                "generated" if assembled_scene_count == self.env.max_steps
                else "partial" if episode_log else "failed"
            ),
            "manuscript_status": "complete" if complete else (
                "generated" if assembled_scene_count == self.env.max_steps
                else "partial" if episode_log else "failed"
            ),
            "ending_validation": {"valid": ending_valid, "issues": list(ending_issues)},
            "whole_story_audit": whole_story_audit,
            "manuscript_repairs": manuscript_repairs,
            "scene_failures": scene_failures,
        }
