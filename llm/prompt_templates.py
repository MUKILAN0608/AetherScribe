import re

MODE_SPECS = {
    "Short Story": {
        "length": (
            "Write 3–4 sentences of accomplished literary prose (minimum 120 words, up to 160). "
            "Every sentence must earn its place."
        ),
        "register": (
            "Precise, mature, and vivid — comparable to a strong literary magazine submission."
        ),
    },
    "Long Story": {
        "length": (
            "Write a full scene of 5 to 8 sentences (minimum 180 words, up to 280). "
            "One sentence per line in the NARRATIVE block. Each sentence should be 20–40 words — "
            "rich, specific, and developed. Never output a single-sentence scene."
        ),
        "register": (
            "Publication-grade literary fiction: layered atmosphere, interiority, dialogue or gesture, "
            "and consequential action. Powerful and immersive, never thin or telegraphic."
        ),
    },
}

GENRE_STYLE = {
    "Thriller": (
        "Propulsive clarity, mounting jeopardy, sharp concrete detail. "
        "Stakes must feel immediate and irreversible."
    ),
    "Horror": (
        "Slow dread, precise sensory unease, restraint before revelation. "
        "Dread through implication as much as event."
    ),
    "Comedy": (
        "Timing, irony, and character truth; wit from situation and voice, not forced jokes."
    ),
    "Romance": (
        "Emotional granularity, tension between desire and obstacle, intimate physical detail."
    ),
    "Drama": (
        "Moral weight, interior conflict, dialogue and gesture that reveal character fracture."
    ),
    "Fantasy": (
        "World-consistent wonder, mythic texture grounded in cause and effect; no generic fantasy clichés."
    ),
}

CRAFT_RULES = """
CRAFT (mandatory):
- Show through action, gesture, and sensory detail; avoid abstract summary ("he felt sad" → embody it).
- Use strong, specific verbs; cut filler (very, really, suddenly, just, began to).
- Vary sentence length for rhythm; at least one concrete image per scene.
- Dialogue only if it reveals power or emotion; make it sparse and sharp.
- No meta-narration, no addressing the reader, no "this scene", no bullet lists.
- Forbidden register: childish, textbook, fanfiction, or AI-generic phrasing ("little did they know", "with determination").
- Character names exactly as given; never "the protagonist" or role labels.

PREMISE LOCK (mandatory):
- Stay strictly within the STORY PREMISE — same world, conflict, and stakes.
- Do not invent new main characters, locations, or subplots absent from the premise.
- Each scene must read as the next beat of the same story, not a digression.
"""


def _format_actions(actions, characters):
    lines = []
    for role, act in actions.items():
        key = role.lower()
        name = characters.get(key) or characters.get(role) or role
        lines.append(f"{name}: {act.replace('_', ' ')}")
    return "; ".join(lines)


def _clean_rationale(rationale: str) -> str:
    if not rationale:
        return "The scene follows the most coherent causal path given character incentives."
    text = re.sub(r"#{1,6}\s*", "", rationale)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"[_`]", "", text)
    text = re.sub(r"🖋️|❌|✓", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:450]


def build_prompt(
    state,
    actions,
    rationale,
    scene_index,
    previous_summary=None,
    language="English",
    mode="Short Story",
):
    director_brief = state.get("director_brief", {})
    genre = state.get("genre", "Drama")
    characters = director_brief.get("characters", {})
    tension = state.get("tension", 0.5)
    phase = state.get("phase", "rising")

    p_name = characters.get("protagonist", "Lead")
    a_name = characters.get("antagonist", "Opponent")
    l_name = characters.get("ally", "Ally")

    mode_key = mode if mode in MODE_SPECS else "Long Story"
    spec = MODE_SPECS[mode_key]
    genre_note = GENRE_STYLE.get(genre, "Genre-appropriate literary tone.")

    action_block = _format_actions(actions, characters)
    premise = director_brief.get("initial_story_premise", "").strip()
    continuity = (
        previous_summary
        if previous_summary
        else "Opening beat: ground the reader in the premise's world and central conflict."
    )
    rationale_clean = _clean_rationale(rationale)

    tension_note = (
        "climax pressure — compress time, raise physical and moral cost"
        if tension > 0.72
        else "rising pressure — escalate complication without resolving the core conflict"
        if tension > 0.42
        else "establishing pressure — orient the reader, plant tension beneath ordinary surfaces"
    )

    return f"""You are the principal literary architect for a narrative research laboratory.
Your prose is published in peer-reviewed creative trajectories and must read as authored, not generated.

LANGUAGE: {language}
GENRE: {genre} — {genre_note}
SCENE: {scene_index} | ARC: {phase} | DRAMATIC PRESSURE: {tension_note}

STORY PREMISE (anchor every line to this; do not deviate):
{premise if premise else "Follow the established conflict and character goals."}

CHARACTERS (use these exact names only):
- {p_name}
- {a_name}
- {l_name}

EVENTS THAT MUST OCCUR ON THE PAGE (depict all; do not substitute or omit):
{action_block}

CAUSAL LOGIC (weave in; do not quote):
{rationale_clean}

STORY SO FAR:
{continuity}

{CRAFT_RULES}

STYLE:
- {spec["register"]}
- {spec["length"]}

OUTPUT FORMAT:

[SCENE {scene_index}]
NARRATIVE:
<5–8 sentences; one sentence per line; minimum 180 words total; no blank lines>

[QUANTUM_TRACE]
<One sentence: in-world cause of events, character names only, literary tone.>

[COHERENCE]
<Internal quality note; one sentence.>
"""
