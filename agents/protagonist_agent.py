from .base_agent import BaseAgent

class ProtagonistAgent(BaseAgent):
    """
    Research-tier Protagonist agent (v5.4.2).
    Features a numerical trait matrix for quantum decision biasing.
    """
    def __init__(self, name="Protagonist"):
        super().__init__(name)
        # Trait Matrix: [Aggression, Curiosity, Resilience]
        self.traits = {
            "aggression": 0.4,
            "curiosity": 0.8,
            "resilience": 0.6
        }
        self.trait_descriptions = {
            "aggression": "tendency to use force or direct confrontation",
            "curiosity": "desire to uncover secrets and explore the unknown",
            "resilience": "ability to withstand pressure and bounce back from setbacks"
        }

    @property
    def actions(self):
        return [
            "investigate_technical_anomaly",
            "confront_adversary_directly",
            "execute_covert_maneuver",
            "negotiate_high_stakes_truce",
            "analyze_internal_conflict",
            "deploy_emergency_countermeasure"
        ]
