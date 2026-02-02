import numpy as np
import uuid
from environment.state_representation import StateRepresentation

class EpisodeRunner:
    """
    Elite Narrative Orchestrator (v5.4.2).
    Coordinates Quantum decisions with Decision Governance and High-Fidelity Telemetry.
    """
    def __init__(self, env, policy, action_space, explainer, renderer):
        self.env = env
        self.policy = policy
        self.action_space = action_space
        self.explainer = explainer
        self.renderer = renderer

    def _get_combined_trait_vec(self):
        """Synthesizes character traits for quantum biasing."""
        p_traits = self.action_space.protagonist.traits
        a_traits = self.action_space.antagonist.traits
        l_traits = self.action_space.ally.traits
        return np.array([
            (p_traits['aggression'] + a_traits['hostility']) / 2.0,
            (p_traits['curiosity'] + a_traits['strategic_depth']) / 2.0,
            (p_traits['resilience'] + a_traits['manipulation']) / 2.0,
            (l_traits['loyalty'] + l_traits['supportiveness']) / 2.0
        ])

    def run_episode(self, verbose=True, language="English", lr=0.01, eps=0.0, temperature=0.8, mode="Short Story"):
        """
        Executes a trajectory capturing full candidate superposition and rejection logic (v5.4.2).
        """
        state = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_log = []
        previous_summary = None
        trait_vec = self._get_combined_trait_vec()

        while not done:
            # 1. State Encoding
            state_vec = StateRepresentation.encode(state['phase'], state['tension'], state['genre'], state['step'])

            # 2. Quantum Decision Superposition
            probs = self.policy.get_action_probs(state_vec, self.action_space.num_actions, trait_vec=trait_vec, eps=eps)
            entropy = self.policy.get_entropy(state_vec, trait_vec=trait_vec, eps_noise=eps)

            # Governance: Risk & Attribution
            risk_score = self.policy.estimate_narrative_risk(state_vec, trait_vec=trait_vec, eps_noise=eps)
            attribution = self.policy.get_influence_attribution(state_vec, trait_vec=trait_vec)

            # Measure / Collapse
            action_idx = np.random.choice(len(probs), p=probs)
            joint_action = self.action_space.get_joint_action(action_idx)

            # Governance: Why-Not Analysis (Simplified for Research Clarity)
            rejection_data = []
            max_prob = np.max(probs)
            for i, p in enumerate(probs):
                if i != action_idx:
                    # Translate technical quantum terms into simple narrative logic
                    if p < 0.05:
                        reason = "Weak Potential (The story path was too unlikely to pursue)"
                    elif p < max_prob * 0.3:
                        reason = "Logic Conflict (This path contradicted character motivations)"
                    elif p < max_prob * 0.7:
                        reason = "Character Mismatch (Character traits didn't align with this choice)"
                    else:
                        reason = "Alternative Path (A strong choice, but the current one felt more 'right')"

                    rejection_data.append({
                        "index": i,
                        "action": self.action_space.get_joint_action(i),
                        "prob": float(p),
                        "reason": reason
                    })

            # Sort by probability to show the most "tempting" rejected actions first
            rejection_data.sort(key=lambda x: x['prob'], reverse=True)

            # 3. Environment Transition
            next_state, env_reward, done = self.env.step(joint_action)

            # 4. Rationale & Sensitivity
            expectations = self.policy.circuit(state_vec, self.policy.weights, trait_vec=trait_vec, eps=eps)

            # Pass attribution and agent objects for research-grade narratives (v5.4.2)
            agent_context = {
                "protagonist": self.action_space.protagonist,
                "antagonist": self.action_space.antagonist,
                "ally": self.action_space.ally
            }

            # Map roles to names for explanation
            name_map = {role.capitalize(): name for role, name in self.env.director_brief.get("characters", {}).items()}

            rationale = self.explainer.generate_explanation(
                state_vec, joint_action, rejections=rejection_data, attribution=attribution, agents=agent_context, name_map=name_map
            )
            sensitivity = self.policy.get_feature_sensitivity(state_vec, trait_vec=trait_vec, eps_noise=eps)

            # 5. Semantic Rendering (Decision-First)
            deep_rejections = self.renderer.analyze_rejections(
                state=state,
                chosen_action=joint_action,
                rejected_actions=rejection_data,
                rationale=rationale,
                language=language
            )

            story_text, previous_summary, prompt_trace, coherence_score = self.renderer.render_scene(
                state=state,
                actions=joint_action,
                rationale=rationale,
                previous_summary=previous_summary,
                language=language,
                temperature=temperature,
                mode=mode
            )

            # 6. Policy Update
            composite_reward = env_reward * coherence_score
            total_reward += composite_reward
            self.policy.update(composite_reward, state_vec, action_idx, lr=lr)

            # 7. Log Telemetry
            episode_log.append({
                "step": step,
                "state": state,
                "action": joint_action,
                "action_idx": int(action_idx),
                "probs": probs.tolist(),
                "entropy": float(entropy),
                "risk": risk_score,
                "attribution": attribution,
                "rejections": rejection_data,
                "deep_rejections": deep_rejections,
                "sensitivity": sensitivity,
                "coherence": float(coherence_score),
                "rationale": rationale,
                "story": story_text,
                "summary": previous_summary,
                "prompt_trace": prompt_trace,
                "weights": self.policy.weights.tolist(),
                "reward": float(composite_reward)
            })

            state = next_state
            step += 1

        return {
            "id": str(uuid.uuid4()),
            "scenes": episode_log,
            "total_reward": float(total_reward),
            "mode": mode,
            "traits": trait_vec.tolist()
        }
