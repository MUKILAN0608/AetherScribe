import numpy as np
import uuid
import logging
import inspect
import os
import re
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
)
from continuity import StoryContinuity, check_repetition, validate_scene_output, validate_story_ending
from secret_loader import load_secret_package
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
        characters = self.env.director_brief.get("characters", {})
        premise = self.env.director_brief.get("initial_story_premise", "").strip()
        protagonist = characters.get("protagonist", "The protagonist")
        self.story_continuity = StoryContinuity(
            situation=premise or "The story is beginning.",
            protagonist_goal=f"{protagonist} must respond to the central problem.",
            unresolved_conflicts=("The central conflict is unresolved.",),
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
        scene_history = (
            self.scene_history_repository.list(self.story_id)
            if self.scene_history_repository is not None else []
        )
        trait_vec = self._get_combined_trait_vec()
        full_explain = render_depth == "full"
        lite_render = render_depth == "fast"

        while not done:
            state_vec = StateRepresentation.encode(
                state["phase"], state["tension"], state["genre"], state["step"]
            )

            diag = self.policy.compute_step_diagnostics(
                state_vec,
                self.action_space.num_actions,
                trait_vec=trait_vec,
                eps=eps,
                full_explain=full_explain,
            )
            probs = diag["action_probs"]
            entropy = diag["entropy"]
            risk_score = diag["risk"]
            attribution = diag["attribution"]
            sensitivity = diag["sensitivity"]

            if locked_action_indices is not None:
                if step >= len(locked_action_indices):
                    raise ValueError("locked_action_indices does not cover every scene.")
                action_idx = int(locked_action_indices[step])
                if action_idx < 0 or action_idx >= len(probs):
                    raise ValueError("locked action index is outside the action space.")
            else:
                action_idx = int(np.random.choice(len(probs), p=probs))
            joint_action = self.action_space.get_joint_action(action_idx)
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
                        "action": self.action_space.get_joint_action(i),
                        "prob": p,
                        "reason": reason,
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
                            current_state=state,
                            memories=tuple(memory_context),
                            character_knowledge=(
                                CharacterKnowledge(
                                    character_id=role,
                                    constraints=tuple(
                                        self.env.director_brief.get("constraints", [])
                                    ),
                                )
                                for role in self.env.director_brief.get(
                                    "characters", {}
                                )
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
            if self.memory_enabled:
                render_kwargs["memory_context"] = memory_context
            renderer_parameters = inspect.signature(self.renderer.render_scene).parameters
            accepts_extra_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in renderer_parameters.values()
            )
            if "planning_guidance" in renderer_parameters or accepts_extra_kwargs:
                scene_plan = self.story_plan.for_scene(step)
                render_kwargs["planning_guidance"] = (
                    f"SCENE NUMBER: {scene_plan.scene_index + 1}",
                    f"ARC PHASE: {scene_plan.phase}",
                    f"SCENE PURPOSE: {scene_plan.purpose}",
                    f"PROTAGONIST OBJECTIVE: {scene_plan.protagonist_objective or scene_plan.purpose}",
                    f"ANTAGONIST OPPOSITION: {scene_plan.antagonist_opposition or 'Maintain the established conflict without inventing unsupported facts.'}",
                    f"ALLY CONTRIBUTION: {scene_plan.ally_contribution or 'Provide relevant support grounded in the established story.'}",
                    f"REQUIRED EVENT: {scene_plan.event}",
                    f"NEW INFORMATION: {scene_plan.new_information or 'Add only information supported by a concrete event.'}",
                    f"VALIDATION CLUES: {'; '.join(scene_plan.validation_terms)}",
                    f"EXPECTED CONSEQUENCE: {scene_plan.consequence}",
                    f"NEXT SCENE EFFECT: {scene_plan.next_scene_effect or scene_plan.unresolved_objective}",
                    *(
                        planner_result.rendering_guidance
                        if planner_result is not None else ()
                    ),
                )
            if self.secret_enabled:
                render_kwargs["secret_guidance"] = secret_guidance
            if "story_continuity" in inspect.signature(self.renderer.render_scene).parameters:
                render_kwargs["story_continuity"] = self.story_continuity
            repetition_warning = None
            retry_count = 0
            scene_plan = self.story_plan.for_scene(step)
            action_beat = f"{scene_plan.phase}:{scene_plan.purpose}"
            secret_validation = {"valid": True, "violations": []}
            final_quality = None
            story_text = ""
            while True:
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
                )
                failure_issues = list(final_quality.issues)
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
                if not failure_issues:
                    break
                if retry_count >= self.repetition_retry_limit:
                    break
                retry_count += 1
                corrections = []
                if "provider_truncated_output" in failure_issues or "truncated_output" in failure_issues:
                    corrections.append(
                        "The previous response hit its output limit. Write a shorter complete scene, "
                        "finish every sentence, and preserve the planned evidence and consequence."
                    )
                if "scene_above_maximum_length" in failure_issues:
                    corrections.append(
                        f"Cut the passage to {max(45, int(os.getenv('NARRATIVE_MAX_SCENE_WORDS', '130')) - 15)} words maximum. "
                        "Keep only the action that changes the situation and its consequence."
                    )
                if "repeated_sentence" in failure_issues or "similar_scene" in failure_issues:
                    corrections.append(
                        "Use new actions, dialogue, images, and sentence openings. Do not reuse any sentence or exchange from an earlier scene."
                    )
                if "protected_secret_leak" in failure_issues:
                    corrections.append("Remove all protected values and aliases without naming them.")
                if "template_fallback_not_real_generation" in failure_issues:
                    corrections.append("The provider returned template fallback, not model prose; do not accept it.")
                repetition_warning = final_quality.correction_instruction or "Correct the scene validation issues."
                if corrections:
                    repetition_warning += " " + " ".join(corrections)
                repetition_warning += f" Retry {retry_count} of {self.repetition_retry_limit}."

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
                    "issues": list(dict.fromkeys(failure_issues)),
                    "retry_count": retry_count,
                    "provider_error": getattr(self.renderer, "last_error", None),
                    "finish_reason": (getattr(self.renderer, "last_provider_metadata", {}) or {}).get("finish_reason"),
                    "output_tokens": getattr(self.renderer, "last_output_tokens", None),
                })
                # A rejected scene must not advance accepted memory or block
                # later planned scenes from being attempted.
                if step + 1 < self.env.max_steps:
                    step += 1
                    continue
                break

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

            memory_updates = 0
            memory_rejections = []
            if self.memory_enabled and final_quality.accepted:
                candidates = extract_authoritative_candidates(
                    self.story_id,
                    scene={
                        "authoritative_facts": [
                            f"{role} performed {action.replace('_', ' ')}"
                            for role, action in locked_action.items()
                        ]
                    },
                    action={"description": " | ".join(
                        f"{role}: {action}" for role, action in locked_action.items()
                    )},
                    event={
                        "type": "scene_transition",
                        "description": f"Scene {step + 1} completed in {state.get('phase', 'unknown')} phase.",
                    },
                    episode_id=episode_id,
                    scene_index=step,
                )
                for candidate in candidates:
                    try:
                        self.memory_repository.save(candidate)
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
                "selected_probability": float(probs[action_idx]),
                "peak_probability": float(np.max(probs)),
                "sampling_mode": "stochastic",
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
                },
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
                self.scene_history_repository.add(
                    story_id=self.story_id,
                    episode_id=episode_id,
                    scene_index=step,
                    story=reader_story,
                    summary=previous_summary or "",
                    outcome=previous_summary or "",
                    opening=body[:180].lower(),
                    ending=sentences[-1].lower() if sentences else "",
                    beat="|".join(f"{key}:{value}" for key, value in locked_action.items()),
                    objective=self.story_continuity.next_scene_purpose,
                    accepted=final_quality.accepted,
                )
            body = narrative_text
            sentences = [item.strip() for item in body.split(".") if item.strip()]
            scene_history.append({
                "episode_id": episode_id,
                "scene_index": step,
                "story": reader_story,
                "summary": previous_summary or "",
                "outcome": previous_summary or "",
                "opening": body[:180].lower(),
                "ending": sentences[-1].lower() if sentences else "",
                "beat": action_beat,
                "objective": self.story_plan.for_scene(step).purpose,
            })
            if final_quality.accepted:
                self.story_continuity = self.story_continuity.after_scene(
                    previous_summary,
                    dict(locked_action),
                    (
                        "Advance the established conflict through a concrete consequence."
                        if step + 1 < self.env.max_steps
                        else "Deliver a consequential scene ending consistent with the established conflict."
                    ),
                )

            if on_step is not None:
                on_step(log_entry, step + 1, self.env.max_steps)

            if verbose:
                print(f"[Scene {step + 1}] reward={composite_reward:.3f} tension={state['tension']:.3f}")

            state = next_state
            step += 1

        ending_valid, ending_issues = (
            validate_story_ending(
                episode_log[-1]["story"],
                self.env.director_brief.get("characters", {}),
            )
            if episode_log else (False, ("no_accepted_scenes",))
        )
        all_accepted = (
            len(episode_log) == self.env.max_steps
            and all(scene.get("quality_gate", {}).get("accepted") for scene in episode_log)
        )
        complete = all_accepted and ending_valid
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
            "accepted_scene_count": sum(
                1 for scene in episode_log if scene.get("quality_gate", {}).get("accepted")
            ),
            "failed_scene_count": len(scene_failures),
            "story_complete": complete,
            "status": "complete" if complete else ("partial" if episode_log else "failed"),
            "ending_validation": {"valid": ending_valid, "issues": list(ending_issues)},
            "scene_failures": scene_failures,
        }
