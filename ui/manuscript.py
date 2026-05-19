import re
from typing import List, Tuple

import numpy as np


def parse_scene_story(story_text: str) -> Tuple[str, str]:
    narrative = ""
    trace = ""

    if "NARRATIVE:" in story_text:
        parts = story_text.split("NARRATIVE:", 1)[1]
        if "[QUANTUM_TRACE]" in parts:
            narrative, rest = parts.split("[QUANTUM_TRACE]", 1)
            trace = rest.split("[COHERENCE]")[0].strip()
        else:
            narrative = parts.strip()
    else:
        content = story_text.split("[QUANTUM_TRACE]")
        block = content[0]
        narrative = re.sub(r"^\[SCENE\s+\d+\]\s*", "", block, flags=re.IGNORECASE)
        narrative = narrative.replace("RESULT:", "").strip()
        trace = content[1].strip() if len(content) > 1 else ""

    return narrative.strip(), trace


def compile_full_manuscript(episode: dict) -> str:
    paragraphs = []
    for scene in episode["scenes"]:
        prose, _ = parse_scene_story(scene["story"])
        if prose:
            paragraphs.append(prose)
    return "\n\n".join(paragraphs)


def compile_plain_text_export(episode: dict) -> str:
    metrics = episode_metrics(episode)
    lines = [
        "AETHER SCRIBE — NARRATIVE TRAJECTORY EXPORT",
        f"Genre: {episode.get('genre', 'N/A')}",
        f"Scenes: {metrics['scene_count']}",
        "",
        "PREMISE",
        episode.get("premise", ""),
        "",
        "=" * 48,
        "",
    ]
    for i, scene in enumerate(episode["scenes"], 1):
        prose, _ = parse_scene_story(scene["story"])
        lines.extend([f"SCENE {i}", "", prose, ""])
    return "\n".join(lines)


def episode_metrics(episode: dict) -> dict:
    scenes = episode["scenes"]
    coherences = [s.get("coherence", 0.0) for s in scenes]
    rewards = [s["reward"] for s in scenes]
    return {
        "mean_coherence": float(np.mean(coherences)) if coherences else 0.0,
        "mean_utility": float(np.mean(rewards)) if rewards else 0.0,
        "cumulative_utility": float(episode.get("total_reward", sum(rewards))),
        "final_tension": float(scenes[-1]["state"]["tension"]) if scenes else 0.0,
        "scene_count": len(scenes),
        "mean_entropy": float(np.mean([s["entropy"] for s in scenes])) if scenes else 0.0,
    }


def relationship_note(role: str, relationships: dict) -> str:
    if role == "Antagonist":
        e = relationships["protagonist_antagonist"]["enmity"]
        if e > 0.75:
            return "High antagonism"
        if e > 0.45:
            return "Escalating conflict"
        return "Latent rivalry"
    if role == "Ally":
        t = relationships["protagonist_ally"]["trust"]
        if t > 0.75:
            return "Strong alliance"
        if t > 0.45:
            return "Developing trust"
        return "Provisional support"
    return "Narrative driver"


def scene_insight(scene: dict, name_map: dict) -> str:
    """Reader-facing decision insight without exposing engine internals."""
    deep = scene.get("deep_rejections") or []
    if deep and isinstance(deep[0], str) and len(deep[0]) > 20:
        return deep[0]

    trace_narrative, trace = parse_scene_story(scene["story"])
    if trace and "policy" not in trace.lower() and "quantum" not in trace.lower():
        return trace

    top_role = max(scene["attribution"], key=scene["attribution"].get)
    lead = name_map.get(top_role, top_role)
    tension = scene["state"]["tension"]
    if tension > 0.7:
        phase = "climax-phase"
    elif tension > 0.4:
        phase = "rising-action"
    else:
        phase = "establishment"
    return (
        f"This scene advances the {phase} arc with {lead} as the primary narrative driver, "
        f"consistent with the evolving character dynamics."
    )


def generate_manuscript_markdown(episode: dict) -> str:
    metrics = episode_metrics(episode)
    lines = [
        f"# Narrative Trajectory: {episode['genre']}",
        "",
        "## Abstract",
        episode["premise"],
        "",
        "## Metrics",
        f"- Mean coherence: {metrics['mean_coherence']:.3f}",
        f"- Cumulative narrative utility: {metrics['cumulative_utility']:.3f}",
        f"- Scenes: {metrics['scene_count']}",
        "",
        "## Manuscript",
        "",
        compile_full_manuscript(episode),
    ]
    return "\n".join(lines)


def generate_manuscript_latex(episode: dict) -> str:
    metrics = episode_metrics(episode)
    latex = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{times}",
        "\\title{Narrative Trajectory: " + episode["genre"] + "}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}",
        episode["premise"][:800].replace("&", "\\&"),
        "\\end{abstract}",
        "\\section*{Trajectory Summary}",
        f"Scenes: {metrics['scene_count']}; "
        f"Mean coherence: {metrics['mean_coherence']:.3f}; "
        f"Cumulative utility: {metrics['cumulative_utility']:.3f}.",
        "\\section{Rendered Narrative}",
    ]
    for s in episode["scenes"]:
        prose, _ = parse_scene_story(s["story"])
        safe = prose.replace("&", "\\&").replace("%", "\\%")
        latex.append(f"\\subsection*{{Scene {s['step'] + 1}}} {safe}")
    latex.append("\\end{document}")
    return "\n".join(latex)


def trajectory_steps(episode: dict) -> List[str]:
    return [f"S{s['step'] + 1}" for s in episode["scenes"]]
