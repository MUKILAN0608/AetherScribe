class ExplanationEngine:
    def generate_explanation(self, state_vec, joint_action, rejections=None, attribution=None, agents=None, name_map=None):
        """
        Generates a detailed research-grade explanation of the decision logic and rejections.
        Uses actual character names from name_map.
        """
        lines = []
        lines.append("### 🖋️ STORY DECISION: WHY THIS HAPPENED")

        # 1. Simple Conflict Description
        lines.append("The story was at a crossroads. Many different things could have happened, but the characters' personalities pushed the plot in one specific direction.")

        actions_desc = []
        for role, action in joint_action.items():
            char_name = name_map.get(role, role) if name_map else role
            actions_desc.append(f"**{char_name}** decided to {action.lower().replace('_', ' ')}")

        lines.append(f"\n**What happened:** " + " and ".join(actions_desc) + ".")

        # 2. Rejection Analysis (Research Focus)
        if rejections:
            lines.append("\n#### ❌ WHY OTHER PATHS FAILED")
            # Take the top 2 rejections
            for rej in rejections[:2]:
                rej_actions = []
                for role, act in rej['action'].items():
                    name = name_map.get(role, role) if name_map else role
                    rej_actions.append(f"{name}: {act.replace('_', ' ')}")
                lines.append(f"- **Path {rej['index']}** ({', '.join(rej_actions)}): {rej['reason']}")

        # 3. Character Motivation
        if attribution and agents:
            top_role = max(attribution, key=attribution.get)
            top_agent_obj = agents.get(top_role.lower())
            top_char_name = name_map.get(top_role, top_role) if name_map else top_role

            lines.append(f"\n**The Driver:** **{top_char_name}** was the main person making things happen in this scene.")

            if top_agent_obj and hasattr(top_agent_obj, 'traits'):
                traits = top_agent_obj.traits
                top_trait = max(traits, key=traits.get)

                lines.append(f"Because they are very **{top_trait.replace('_', ' ')}**, they naturally chose an action that fits their personality rather than doing something out of character.")

        # 4. Simple Tension Explanation
        tension = state_vec[0]
        if tension > 0.7:
            tension_desc = "The story is very intense right now, so the characters are making bold, risky moves."
        elif tension < 0.3:
            tension_desc = "Things are relatively calm, so the characters are acting more carefully."
        else:
            tension_desc = "The pressure is building, leading the characters to take clear actions to reach their goals."

        lines.append(f"\n**The Mood:** {tension_desc}")

        lines.append(f"\n**Result:** This choice keeps the story moving forward in a way that makes sense for everyone involved.")

        return "\n".join(lines)
