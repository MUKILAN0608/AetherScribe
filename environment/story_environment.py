from .genre_dynamics import GENRE_TENSION_MULTIPLIER

class StoryEnvironment:
    """
    Narrative world with multi-agent relationship dynamics and synergy optimization.
    Optimized for Aether Scribe v5.4.2 High-Transparency Research Edition.
    """

    phases = ["intro", "rising", "climax"]

    ACTION_IMPACT = {
        "investigate_technical_anomaly": 0.05,
        "confront_adversary_directly": 0.15,
        "execute_covert_maneuver": 0.08,
        "negotiate_high_stakes_truce": -0.10,
        "analyze_internal_conflict": -0.05,
        "deploy_emergency_countermeasure": 0.12,
        "orchestrate_systemic_sabotage": 0.15,
        "deploy_psychological_manipulation": 0.10,
        "escalate_physical_threat": 0.20,
        "leverage_hidden_corruption": 0.08,
        "obstruct_vital_information": 0.05,
        "initiate_decisive_strike": 0.25,
        "provide_critical_intelligence": -0.05,
        "execute_support_maneuver": 0.05,
        "mitigate_external_threat": -0.12,
        "facilitate_strategic_extraction": -0.15,
        "reinforce_moral_resolve": -0.05,
        "sacrifice_personal_safety": 0.18
    }

    def __init__(self, genre, director_brief=None, max_steps=5):
        self.genre = genre.lower()
        self.director_brief = director_brief or {}
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.phase_index = 0
        self.tension = 0.3
        self.current_step = 0

        # v5.4.2 Relationship Matrix: [Trust, Hostility]
        # Standardized range: [0.0, 1.0]
        self.relationships = {
            "protagonist_ally": {"trust": 0.6, "enmity": 0.1},
            "protagonist_antagonist": {"trust": 0.0, "enmity": 0.7},
            "ally_antagonist": {"trust": 0.0, "enmity": 0.5}
        }
        return self._get_state()

    def _get_state(self):
        return {
            "genre": self.genre,
            "phase": self.phase,
            "tension": round(self.tension, 3),
            "step": self.current_step,
            "relationships": self.relationships, # v5.4.2
            "director_brief": self.director_brief
        }

    @property
    def phase(self):
        return self.phases[self.phase_index]

    def _update_relationships(self, actions):
        """
        Interpersonal Physics: Evolves trust and enmity based on joint actions (env keys).
        """
        p_act = actions.get("Protagonist") or actions.get("protagonist")
        a_act = actions.get("Antagonist") or actions.get("antagonist")
        l_act = actions.get("Ally") or actions.get("ally")

        # P-L Relationship (Trust building/erosion)
        if l_act in ["provide_critical_intelligence", "execute_support_maneuver"]:
            self.relationships["protagonist_ally"]["trust"] = min(1.0, self.relationships["protagonist_ally"]["trust"] + 0.1)
        if l_act == "sacrifice_personal_safety":
            self.relationships["protagonist_ally"]["trust"] = min(1.0, self.relationships["protagonist_ally"]["trust"] + 0.2)

        # P-A Relationship (Hostility escalation)
        if p_act == "confront_adversary_directly" or a_act == "initiate_decisive_strike":
            self.relationships["protagonist_antagonist"]["enmity"] = min(1.0, self.relationships["protagonist_antagonist"]["enmity"] + 0.15)
        if a_act == "deploy_psychological_manipulation":
            self.relationships["protagonist_antagonist"]["enmity"] = min(1.0, self.relationships["protagonist_antagonist"]["enmity"] + 0.1)

    def step(self, actions, perturbation=0.0):
        """
        Processes joint actions and evolves state + relationships.
        v5.4.2: Integrated Synergy-Aware reward logic.
        """
        self._update_relationships(actions)

        # Calculate Tension
        raw_impact = sum(self.ACTION_IMPACT.get(act, 0.0) for act in actions.values())
        genre_factor = GENRE_TENSION_MULTIPLIER.get(self.genre, 1.0)

        # Interpersonal Friction increases tension
        friction = (self.relationships["protagonist_antagonist"]["enmity"] +
                    self.relationships["protagonist_ally"]["enmity"]) * 0.05

        tension_delta = (raw_impact + perturbation + friction) * genre_factor
        self.tension = min(max(self.tension + tension_delta, 0.0), 1.0)

        # Phase Evolution
        if self.tension > 0.75: self.phase_index = 2
        elif self.tension > 0.45: self.phase_index = 1
        else: self.phase_index = 0

        self.current_step += 1

        # v5.4.2 Multi-Objective Reward + Synergy Bonus
        r_tension = self.tension
        r_progress = (self.current_step / self.max_steps) * 0.5
        r_stability = 0.2

        # Synergy Bonus: Reward high trust with Ally
        r_synergy = self.relationships["protagonist_ally"]["trust"] * 0.3

        composite_reward = r_tension + r_progress + r_stability + r_synergy
        done = self.current_step >= self.max_steps

        return self._get_state(), composite_reward, done
