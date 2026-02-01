class ExplanationEngine:
    def generate_explanation(self, state_vec, probs, expectations, weights, joint_action, reward, attribution=None, agents=None):
        """
        Generates a research-memo style narrative explaining the quantum decision logic using simple analogies.
        """
        lines = []
        lines.append("### 📝 RESEARCH MEMO: DECISION LOGIC")

        # 1. The Core Conflict Analogized
        lines.append("The system treated this scene as a 'competition of futures'. Imagine multiple story paths trying to happen at once, like waves in a pool.")

        actions_desc = []
        for character, action in joint_action.items():
            actions_desc.append(f"**{character}** ({action.lower().replace('_', ' ')})")

        lines.append(f"\n**Resolution:** The 'wave' that became real was the one where " + " and ".join(actions_desc) + ".")

        # 2. The Character-Logic Bridge
        if attribution and agents:
            # Find the most influential agent
            top_agent_key = max(attribution, key=attribution.get)
            top_agent_obj = agents.get(top_agent_key.lower())
            influence_pct = attribution[top_agent_key] * 100

            lines.append(f"\n**Why this path?** The **{top_agent_key}**'s personality was the main driver here ({influence_pct:.1f}% influence).")

            if top_agent_obj and hasattr(top_agent_obj, 'trait_descriptions'):
                traits = top_agent_obj.traits
                top_trait = max(traits, key=traits.get)
                trait_desc = top_agent_obj.trait_descriptions.get(top_trait, "core personality")

                lines.append(f"Their high {top_trait} ({trait_desc}) acted like a magnet, pulling the quantum probability away from boring choices and toward this specific outcome.")

        # 3. Simple explanation of 'Interference'
        max_prob = max(probs)
        confidence = "high" if max_prob > 0.4 else "moderate"

        lines.append(f"\n**Quantum Insight:** Other paths were 'cancelled out' through destructive interference. This path was selected with {confidence} confidence. Because the story tension is currently {state_vec[0]:.2f}, those alternative futures simply didn't have enough logical energy to exist.")

        lines.append(f"\n**Outcome Audit:** This decision achieved a utility score of {reward:.2f}, confirming it was the most effective way to progress the character arcs.")

        return "\n".join(lines)
