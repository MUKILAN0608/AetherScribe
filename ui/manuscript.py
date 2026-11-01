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
    for scene in _manuscript_scenes(episode):
        prose, _ = parse_scene_story(scene["story"])
        if prose:
            sequence = int(scene.get("step", len(paragraphs))) + 1
            title = (scene.get("scene_plan") or {}).get("title") or f"Scene {sequence}"
            status = "PROVISIONAL — NEEDS REVISION" if not scene.get("quality_gate", {}).get("accepted", False) else "ACCEPTED"
            paragraphs.append(f"## {title} · {status}\n\n{prose}")
    return "\n\n".join(paragraphs)


def compile_plain_text_export(episode: dict) -> str:
    metrics = episode_metrics(episode)
    requested = int(episode.get("requested_scene_count", episode.get("scenes_count", metrics["scene_count"])))
    provisional_count = sum(
        1 for scene in _manuscript_scenes(episode)
        if not scene.get("quality_gate", {}).get("accepted", False)
    )
    lines = [
        "AETHER SCRIBE — NARRATIVE TRAJECTORY EXPORT",
        f"Genre: {episode.get('genre', 'N/A')}",
        f"Status: {episode.get('status', 'partial')}",
        f"Scenes: {episode.get('generated_scene_count', len(_manuscript_scenes(episode)))} generated; "
        f"{metrics['scene_count']} accepted; {provisional_count} provisional of {requested} requested",
        "",
        "PREMISE",
        episode.get("premise", ""),
        "",
        "=" * 48,
        "",
    ]
    blocked = episode.get("blocked_scene_indices", [])
    if episode.get("status", "partial") != "complete":
        missing = sorted(set(range(requested)) - {int(scene.get("step", -1)) for scene in _manuscript_scenes(episode)})
        lines.extend([
            f"Manuscript status: {episode.get('manuscript_status', episode.get('status', 'partial'))}; "
            f"{episode.get('generated_scene_count', len(_manuscript_scenes(episode)))} generated, "
            f"{metrics['scene_count']} accepted, {provisional_count} provisional, "
            f"{len(missing)} missing; scene validation failures: {episode.get('failed_scene_count', 0)}. "
            f"Missing scene numbers: "
            + (", ".join(str(index + 1) for index in missing) or "none"),
            "Completion conditions: " + ", ".join(
                key.replace("_", " ")
                for key, passed in (episode.get("completion_conditions") or {}).items()
                if not passed
            ) if episode.get("completion_conditions") else "",
            "",
        ])
    for i, scene in enumerate(_manuscript_scenes(episode), 1):
        prose, _ = parse_scene_story(scene["story"])
        sequence = int(scene.get("step", i - 1)) + 1
        requested = int(episode.get("requested_scene_count", episode.get("scenes_count", len(episode["scenes"]))))
        title = (scene.get("scene_plan") or {}).get("title") or f"Scene {sequence}"
        scene_status = "ACCEPTED" if scene.get("quality_gate", {}).get("accepted", False) else "PROVISIONAL — NEEDS REVISION"
        lines.extend([f"SCENE {sequence} OF {requested} — {title} [{scene_status}]", "", prose, ""])
    return "\n".join(lines)


def episode_metrics(episode: dict) -> dict:
    scenes = _accepted_scenes(episode)
    coherences = [
        float(s["coherence"])
        for s in scenes
        if isinstance(s.get("coherence"), (int, float))
    ]
    rewards = [s["reward"] for s in scenes]
    return {
        "mean_coherence": float(np.mean(coherences)) if coherences else None,
        "mean_utility": float(np.mean(rewards)) if rewards else 0.0,
        "cumulative_utility": float(episode.get("total_reward", sum(rewards))),
        "final_tension": float(scenes[-1]["state"]["tension"]) if scenes else 0.0,
        "scene_count": len(scenes),
        "mean_entropy": float(np.mean([s["entropy"] for s in scenes])) if scenes else 0.0,
    }


def _accepted_scenes(episode: dict) -> list[dict]:
    """Return unique accepted scene records in planned order for reader exports."""
    scenes = []
    seen_steps = set()
    for scene in sorted(episode.get("scenes", []), key=lambda item: int(item.get("step", 0))):
        quality = scene.get("quality_gate")
        if isinstance(quality, dict) and not quality.get("accepted", False):
            continue
        step = int(scene.get("step", len(scenes)))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        scenes.append(scene)
    return scenes


def _manuscript_scenes(episode: dict) -> list[dict]:
    """Return every actual generated scene entry, including provisional drafts."""
    scenes = []
    seen_steps = set()
    for scene in sorted(episode.get("scenes", []), key=lambda item: int(item.get("step", 0))):
        step = int(scene.get("step", len(scenes)))
        if step in seen_steps or not str(scene.get("story", "")).strip():
            continue
        seen_steps.add(step)
        scenes.append(scene)
    return scenes


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
    generated = len(_manuscript_scenes(episode))
    provisional = sum(
        1 for scene in _manuscript_scenes(episode)
        if not scene.get("quality_gate", {}).get("accepted", False)
    )
    lines = [
        f"# Narrative Trajectory: {episode['genre']}",
        "",
        "## Abstract",
        episode["premise"],
        "",
        "## Metrics",
        f"- Mean coherence: {metrics['mean_coherence']:.3f}" if metrics["mean_coherence"] is not None else "- Mean coherence: not measured",
        f"- Cumulative narrative utility: {metrics['cumulative_utility']:.3f}",
        f"- Status: {episode.get('manuscript_status', episode.get('status', 'partial'))}",
        f"- Scenes: {generated} generated; {metrics['scene_count']} accepted; {provisional} provisional",
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
        f"Mean coherence: {metrics['mean_coherence']:.3f}" if metrics["mean_coherence"] is not None else "Mean coherence: not measured",
        f"Cumulative utility: {metrics['cumulative_utility']:.3f}.",
        "\\section{Rendered Narrative}",
    ]
    for s in _manuscript_scenes(episode):
        prose, _ = parse_scene_story(s["story"])
        safe = prose.replace("&", "\\&").replace("%", "\\%")
        status = "Accepted" if s.get("quality_gate", {}).get("accepted", False) else "Provisional -- needs revision"
        latex.append(f"\\subsection*{{Scene {s['step'] + 1} ({status})}} {safe}")
    latex.append("\\end{document}")
    return "\n".join(latex)


def trajectory_steps(episode: dict) -> List[str]:
    return [f"S{s['step'] + 1}" for s in episode["scenes"]]
