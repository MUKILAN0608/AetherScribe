from .base_agent import BaseAgent

class AllyAgent(BaseAgent):
    """
    Research-tier Ally agent (v5.4.2).
    Features a numerical trait matrix for quantum decision biasing.
    """
    def __init__(self, name="Ally"):
        super().__init__(name)
        # Trait Matrix: [Loyalty, Supportiveness, Risk_Tolerance]
        self.traits = {
            "loyalty": 0.9,
            "supportiveness": 0.7,
            "risk_tolerance": 0.4
        }
        self.trait_descriptions = {
            "loyalty": "faithfulness to commitments or obligations",
            "supportiveness": "tendency to provide help and encouragement",
            "risk_tolerance": "willingness to accept uncertainty for potential gain"
        }

    @property
    def actions(self):
        return [
            "provide_critical_intelligence",
            "execute_support_maneuver",
            "mitigate_external_threat",
            "facilitate_strategic_extraction",
            "reinforce_moral_resolve",
            "sacrifice_personal_safety"
        ]
