from agents.protagonist_agent import ProtagonistAgent
from agents.antagonist_agent import AntagonistAgent
from agents.ally_agent import AllyAgent

class QuantumActionSpace:
    """
    Research-tier Quantum Action Space (v5.4.2).
    Maps 16 unique joint character actions for the Decision Authority.
    Matches 4-qubit state space (2^4 = 16).
    """
    def __init__(self):
        self.protagonist = ProtagonistAgent()
        self.antagonist = AntagonistAgent()
        self.ally = AllyAgent()

        self.joint_actions = [
            {"Protagonist": "Investigate Anomaly", "Antagonist": "Orchestrate Sabotage", "Ally": "Provide Intelligence"},
            {"Protagonist": "Confront Directly", "Antagonist": "Deploy Manipulation", "Ally": "Support Maneuver"},
            {"Protagonist": "Execute Covert Move", "Antagonist": "Escalate Threat", "Ally": "Mitigate Danger"},
            {"Protagonist": "Negotiate Truce", "Antagonist": "Leverage Corruption", "Ally": "Facilitate Extraction"},
            {"Protagonist": "Analyze Conflict", "Antagonist": "Obstruct Information", "Ally": "Reinforce Resolve"},
            {"Protagonist": "Deploy Countermeasure", "Antagonist": "Initiate Strike", "Ally": "Sacrifice Safety"},
            {"Protagonist": "Seek Truth", "Antagonist": "Weave Deception", "Ally": "Reveal Secret"},
            {"Protagonist": "Fortify Position", "Antagonist": "Infiltrate Systems", "Ally": "Signal Rescue"},
            {"Protagonist": "Breach Perimeter", "Antagonist": "Trigger Ambush", "Ally": "Disable Security"},
            {"Protagonist": "Rally Bystanders", "Antagonist": "Sow Chaos", "Ally": "Protect Civilians"},
            {"Protagonist": "Decipher Code", "Antagonist": "Jam Communications", "Ally": "Maintain Link"},
            {"Protagonist": "Escape Custody", "Antagonist": "Tighten Grip", "Ally": "Distract Guards"},
            {"Protagonist": "Incite Rebellion", "Antagonist": "Exert Control", "Ally": "Lead Vanguard"},
            {"Protagonist": "Retrieve Artifact", "Antagonist": "Guard Vault", "Ally": "Bypass Lock"},
            {"Protagonist": "Heal Wound", "Antagonist": "Inflict Pain", "Ally": "Provide Shelter"},
            {"Protagonist": "Final Stand", "Antagonist": "Unleash Full Might", "Ally": "Last Support"}
        ]
        self.num_actions = len(self.joint_actions)

    def get_joint_action(self, index):
        return self.joint_actions[index % self.num_actions]
