class ExplanationEngine:
    """Explainable decisions using character names only — never role labels."""

    def _name_for(self, role_key: str, name_map: dict) -> str:
        if not name_map:
            return role_key
        return (
            name_map.get(role_key)
            or name_map.get(role_key.capitalize())
            or name_map.get(role_key.lower())
            or role_key
        )

    def generate_explanation(
        self,
        state_vec,
        joint_action,
        rejections=None,
        attribution=None,
        agents=None,
        name_map=None,
    ):
        name_map = name_map or {}
        tension = float(state_vec[0]) if state_vec is not None and len(state_vec) else 0.5

        lines = []
        lines.append(
            f"Quantum policy measurement at narrative tension {tension:.2f} "
            f"collapsed to the following joint action."
        )

        chosen_parts = []
        for role, action in joint_action.items():
            name = self._name_for(role, name_map)
            chosen_parts.append(f"{name} — {action.replace('_', ' ')}")
        lines.append("Measured outcome: " + "; ".join(chosen_parts) + ".")

        if rejections:
            lines.append("Superposed branches not selected (posterior mass):")
            for rej in rejections[:3]:
                alt_parts = []
                for role, act in rej["action"].items():
                    name = self._name_for(role, name_map)
                    alt_parts.append(f"{name} — {act.replace('_', ' ')}")
                lines.append(
                    f"  · Branch P={rej['prob']:.0%}: "
                    + "; ".join(alt_parts)
                    + f". {rej.get('reason', 'Suppressed by VQC interference.')}"
                )

        if attribution:
            top_role = max(attribution, key=attribution.get)
            driver = self._name_for(top_role, name_map)
            influence = attribution.get(top_role, 0.0)
            lines.append(
                f"Trait-channel attribution peaks on {driver} "
                f"(quantum influence {influence:.0%})."
            )

            if agents:
                agent_obj = agents.get(top_role.lower())
                if agent_obj and hasattr(agent_obj, "traits"):
                    traits = agent_obj.traits
                    trait = max(traits, key=traits.get)
                    lines.append(
                        f"{driver}'s dominant disposition — {trait.replace('_', ' ')} — "
                        f"aligned with this choice."
                    )

        if tension > 0.7:
            pressure = "The story is in a high-pressure climax phase."
        elif tension < 0.3:
            pressure = "The story is establishing setting and relationships."
        else:
            pressure = "The story is escalating toward a turning point."
        lines.append(pressure)

        return "\n".join(lines)
