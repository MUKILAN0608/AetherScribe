def build_prompt(state, actions, rationale, scene_index, previous_summary=None, language="English", mode="Short Story"):
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
STYLE: VERY EASY ENGLISH (Use simple words, like for a child)

{persona_instruction}

STRICT RULE: Use ONLY the character names listed above. NEVER use role titles like "Protagonist" or "Antagonist".

YOUR TASK:
1. Write exactly ONE clear, very simple sentence showing what happens in this scene.
2. Use very EASY ENGLISH and basic vocabulary.
3. Use ONLY character names. Do NOT use roles like "The Protagonist".
4. The scene MUST follow these specific actions: {actions}
5. The story must follow the logic of this reason: {rationale}
6. Keep the story consistent with what happened before: {previous_summary if previous_summary else "This is the start of the story."}

FORMAT:
[SCENE {scene_index}]
RESULT: <Your very simple 1-sentence story update here>

[QUANTUM_TRACE]
<A simple, one-sentence explanation of why this happened using easy words.>
"""
