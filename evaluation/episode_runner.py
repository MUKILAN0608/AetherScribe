import numpy as np
import uuid
from environment.state_representation import StateRepresentation
from quantum_policy.action_registry import joint_action_to_env


class EpisodeRunner:
    """
    Narrative orchestrator (v6.2): cached quantum diagnostics, configurable render depth.
    """

    def __init__(self, env, policy, action_space, explainer, renderer):
        self.env = env
        self.policy = policy
        self.action_space = action_space
        self.explainer = explainer
        self.renderer = renderer

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
    ):
        """
        render_depth: 'full' | 'standard' | 'fast'
          - full: scene + summary + coherence LLM + 3 rejection LLM calls
          - standard: scene + summary + coherence; template rejections
          - fast: scene only; heuristic coherence; template rejections
        """
        state = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_log = []
        previous_summary = None
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

            action_idx = int(np.random.choice(len(probs), p=probs))
            joint_action = self.action_space.get_joint_action(action_idx)
            env_action = joint_action_to_env(joint_action)

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

            story_text, previous_summary, prompt_trace, coherence_score = self.renderer.render_scene(
                state=state,
                actions=joint_action,
                rationale=rationale,
                previous_summary=previous_summary,
                language=language,
                temperature=temperature,
                mode=mode,
                lite_mode=lite_render,
            )

            composite_reward = env_reward * coherence_score
            total_reward += composite_reward
            self.policy.update(composite_reward, state_vec, action_idx, lr=lr)

            log_entry = {
                "step": step,
                "state": state,
                "action": joint_action,
                "env_action": env_action,
                "action_idx": action_idx,
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
                "reward": float(composite_reward),
            }
            episode_log.append(log_entry)

            if on_step is not None:
                on_step(log_entry, step + 1, self.env.max_steps)

            if verbose:
                print(f"[Scene {step + 1}] reward={composite_reward:.3f} tension={state['tension']:.3f}")

            state = next_state
            step += 1

        return {
            "id": str(uuid.uuid4()),
            "scenes": episode_log,
            "total_reward": float(total_reward),
            "mode": mode,
            "render_depth": render_depth,
            "traits": trait_vec.tolist(),
            "characters": self.env.director_brief.get("characters", {}),
        }
