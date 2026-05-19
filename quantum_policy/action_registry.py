"""Maps display labels from QuantumActionSpace to environment physics keys."""

DISPLAY_TO_ENV = {
    "Investigate Anomaly": "investigate_technical_anomaly",
    "Confront Directly": "confront_adversary_directly",
    "Execute Covert Move": "execute_covert_maneuver",
    "Negotiate Truce": "negotiate_high_stakes_truce",
    "Analyze Conflict": "analyze_internal_conflict",
    "Deploy Countermeasure": "deploy_emergency_countermeasure",
    "Orchestrate Sabotage": "orchestrate_systemic_sabotage",
    "Deploy Manipulation": "deploy_psychological_manipulation",
    "Escalate Threat": "escalate_physical_threat",
    "Leverage Corruption": "leverage_hidden_corruption",
    "Obstruct Information": "obstruct_vital_information",
    "Initiate Strike": "initiate_decisive_strike",
    "Provide Intelligence": "provide_critical_intelligence",
    "Support Maneuver": "execute_support_maneuver",
    "Mitigate Danger": "mitigate_external_threat",
    "Facilitate Extraction": "facilitate_strategic_extraction",
    "Reinforce Resolve": "reinforce_moral_resolve",
    "Sacrifice Safety": "sacrifice_personal_safety",
    # Extended action set (approximate nearest canonical impact)
    "Seek Truth": "investigate_technical_anomaly",
    "Weave Deception": "deploy_psychological_manipulation",
    "Reveal Secret": "provide_critical_intelligence",
    "Fortify Position": "reinforce_moral_resolve",
    "Infiltrate Systems": "execute_covert_maneuver",
    "Signal Rescue": "execute_support_maneuver",
    "Breach Perimeter": "confront_adversary_directly",
    "Trigger Ambush": "initiate_decisive_strike",
    "Disable Security": "execute_covert_maneuver",
    "Rally Bystanders": "reinforce_moral_resolve",
    "Sow Chaos": "orchestrate_systemic_sabotage",
    "Protect Civilians": "mitigate_external_threat",
    "Decipher Code": "investigate_technical_anomaly",
    "Jam Communications": "obstruct_vital_information",
    "Maintain Link": "provide_critical_intelligence",
    "Escape Custody": "execute_covert_maneuver",
    "Tighten Grip": "escalate_physical_threat",
    "Distract Guards": "execute_support_maneuver",
    "Incite Rebellion": "confront_adversary_directly",
    "Exert Control": "deploy_psychological_manipulation",
    "Lead Vanguard": "sacrifice_personal_safety",
    "Retrieve Artifact": "investigate_technical_anomaly",
    "Guard Vault": "leverage_hidden_corruption",
    "Bypass Lock": "execute_covert_maneuver",
    "Heal Wound": "mitigate_external_threat",
    "Inflict Pain": "escalate_physical_threat",
    "Provide Shelter": "facilitate_strategic_extraction",
    "Final Stand": "confront_adversary_directly",
    "Unleash Full Might": "initiate_decisive_strike",
    "Last Support": "sacrifice_personal_safety",
}


def to_env_key(display_label: str) -> str:
    if display_label in DISPLAY_TO_ENV:
        return DISPLAY_TO_ENV[display_label]
    normalized = display_label.lower().replace(" ", "_")
    return normalized


def joint_action_to_env(joint_action: dict) -> dict:
    return {role: to_env_key(act) for role, act in joint_action.items()}
