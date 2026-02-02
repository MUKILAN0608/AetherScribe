def build_prompt(state, actions, rationale, scene_index, previous_summary=None, language="English"):
    """
    Simplified Storytelling Architect (v5.5.0).
    Enforces strict character naming and clear, simple story progression.
    """
    director_brief = state.get("director_brief", {})
    genre = state['genre']
    characters = director_brief.get("characters", {})

    # Character Name Enforcement
    p_name = characters.get("protagonist", "The Protagonist")
    a_name = characters.get("antagonist", "The Antagonist")
    l_name = characters.get("ally", "The Ally")

    persona_instruction = (
        f"CHARACTER NAMES:\n"
        f"- Protagonist: {p_name}\n"
        f"- Antagonist: {a_name}\n"
        f"- Ally: {l_name}"
    )

    return f"""
STORYTELLING INSTRUCTIONS
LANGUAGE: {language}
GENRE: {genre}

{persona_instruction}

STRICT RULE: Use ONLY the names listed above. Do NOT invent new names. Do NOT use generic titles like "The Protagonist".

YOUR TASK:
1. Write exactly ONE clear, descriptive sentence showing what happens in this scene.
2. The scene MUST follow these specific character actions: {actions}
3. The story must follow the logic of this reason: {rationale}
4. Keep the story consistent with what happened before: {previous_summary if previous_summary else "This is the start of the story."}

FORMAT:
[SCENE {scene_index}]
RESULT: <Your 1-sentence story update here>

[QUANTUM_TRACE]
<A simple, 1-sentence explanation of why the story took this turn based on the characters' goals and the current mood.>
"""
