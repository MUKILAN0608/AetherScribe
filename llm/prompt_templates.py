from environment.genre_dynamics import GENRE_CONSTRAINTS

def build_prompt(state, actions, rationale, scene_index, previous_summary=None, language="English", mode="Short Story"):
    """
    Elite Decision-First Narrative Architect (v5.4.2).
    Harden character naming and ultra-concise logic generation.
    """
    director_brief = state.get("director_brief", {})
    genre = state['genre']
    characters = director_brief.get("characters", {})
    specific_constraints = GENRE_CONSTRAINTS.get(genre.lower(), [])

    constraints_str = "\n".join([f"- {c}" for c in specific_constraints])
    continuity_str = f"- Previous State: {previous_summary}" if previous_summary else "- State: Initial Trajectory Collapse."

    # Force Character Names (Anti-Hallucination Protocol v5.4.2)
    p_name = characters.get("protagonist", "The Protagonist")
    a_name = characters.get("antagonist", "The Antagonist")
    l_name = characters.get("ally", "The Ally")

    persona_instruction = (
        f"ROLE: PROTAGONIST | NAME: {p_name}\n"
        f"ROLE: ANTAGONIST | NAME: {a_name}\n"
        f"ROLE: ALLY | NAME: {l_name}"
    )

    return f"""
AETHER SCRIBE | DIGITAL NARRATIVE ARCHITECTURE (v5.4.2)
ROLE: ELITE NARRATIVE ARCHITECT & QUANTUM SYSTEMS DIRECTOR
OUTPUT LANGUAGE: {language}
ARCHITECTURAL MODE: {mode}

────────────────────────────────────────
STRICT PERSONA ENFORCEMENT:
{persona_instruction}

(CRITICAL: You MUST use these specific names consistently. NEVER use role placeholders like 'The Protagonist'. If names are assigned, they are absolute. Fail-safe: If names are missing, authorize genre-appropriate identities and stick to them.)
────────────────────────────────────────

────────────────────────────────────────
DECISION-FIRST SCENECRAFT PROTOCOL:
1. OUTPUT: Focus strictly on the logic of the narrative shift. Do NOT generate prose blocks.
2. DURATION: Generate exactly ONE high-density sentence summarizing the outcome.
3. ADAPTATION: The result must strictly resolve the joint actions Decided by the Quantum Engine.
4. TONE: Maintain the atmospheric precision of the {genre} domain.

────────────────────────────────────────
NARRATIVE CONTEXT:
- Seed: {director_brief.get("initial_story_premise", "N/A")}
- Domain: {genre}
- Amplitude (Tension): {state['tension']:.2f}
- Continuity: {continuity_str}

GENRE CONSTRAINTS:
{constraints_str}

────────────────────────────────────────
QUANTUM-LOCKED ACTIONS:
{actions}

────────────────────────────────────────
ALGORITHMIC RATIONALE:
{rationale}

────────────────────────────────────────
MANDATORY DECISION-FIRST FORMAT:
[SCENE {scene_index}]

RESULT: <A high-density 1-sentence summary of the outcome, strictly using authorized character names.>

[QUANTUM_TRACE]

<A high-level technical analysis explaining exactly why the engine chose this narrative line over alternatives.
Explain how the current Tension ({state['tension']:.2f}) influenced the Quantum RL policy to select these specific joint actions to optimize trajectory utility.>
"""
