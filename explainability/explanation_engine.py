class ExplanationEngine:
    def generate_explanation(self, state_vec, joint_action, rejections=None, attribution=None, agents=None, name_map=None):
        """
        Generates a very simple explanation for children or non-experts.
        Uses actual character names.
        """
        lines = []
        lines.append("### 🖋️ STORY DECISION: WHY THIS HAPPENED")

        # 1. Simple Conflict Description
        lines.append("The story was at a choice point. Many different things could have happened, but the characters' feelings pushed the story in one way.")

        actions_desc = []
        for role, action in joint_action.items():
            char_name = name_map.get(role, role) if name_map else role
            actions_desc.append(f"**{char_name}** chose to {action.lower().replace('_', ' ')}")

        lines.append(f"\n**What happened:** " + " and ".join(actions_desc) + ".")

        # 2. Rejection Analysis (Simplified)
        if rejections:
            lines.append("\n#### ❌ WHY OTHER CHOICES WERE SKIPPED")
            for rej in rejections[:2]:
                rej_actions = []
                for role, act in rej['action'].items():
                    name = name_map.get(role, role) if name_map else role
                    rej_actions.append(f"{name}: {act.replace('_', ' ')}")
                lines.append(f"- **Choice {rej['index']}** ({', '.join(rej_actions)}): {rej['reason']}")

        # 3. Character Motivation
        if attribution and agents:
            top_role = max(attribution, key=attribution.get)
            top_agent_obj = agents.get(top_role.lower())
            top_char_name = name_map.get(top_role, top_role) if name_map else top_role

            lines.append(f"\n**The Leader:** **{top_char_name}** was the main person making things happen.")

            if top_agent_obj and hasattr(top_agent_obj, 'traits'):
                traits = top_agent_obj.traits
                top_trait = max(traits, key=traits.get)

                lines.append(f"Because they are very **{top_trait.replace('_', ' ')}**, they naturally chose an action that fits who they are.")

        # 4. Simple Tension Explanation
        tension = state_vec[0]
        if tension > 0.7:
            tension_desc = "The story is very exciting and fast right now!"
        elif tension < 0.3:
            tension_desc = "Everything is calm and quiet."
        else:
            tension_desc = "The story is getting more interesting."

        lines.append(f"\n**The Feeling:** {tension_desc}")

        lines.append(f"\n**Result:** This choice makes the most sense for the story.")

        return "\n".join(lines)
