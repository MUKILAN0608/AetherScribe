"""Explainable decision formatting — character names only, no role labels."""

import html
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from ui.manuscript import parse_scene_story


def build_name_map(characters: Dict[str, str]) -> Dict[str, str]:
    """Maps internal role keys and Title-case roles to display names."""
    mapping = {}
    for role, name in characters.items():
        mapping[role.lower()] = name
        mapping[role.capitalize()] = name
        mapping[role] = name
        if role.lower() == "protagonist":
            mapping["Protagonist"] = name
        elif role.lower() == "antagonist":
            mapping["Antagonist"] = name
        elif role.lower() == "ally":
            mapping["Ally"] = name
    return mapping


def action_to_readable(action: str) -> str:
    if not action:
        return ""
    if "_" not in action and action[0].isupper():
        return action.strip()
    return action.replace("_", " ").strip()


def format_joint_action(
    joint_action: Dict[str, str], name_map: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Returns list of (character_name, action_readable)."""
    rows = []
    for role, act in joint_action.items():
        name = (
            name_map.get(role)
            or name_map.get(role.capitalize())
            or name_map.get(role.lower())
            or role
        )
        rows.append((name, action_to_readable(act)))
    return rows


def clean_rationale_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"[_`]", "", text)
    text = re.sub(r"\b(?:Protagonist|Antagonist|Ally)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def distill_vqc_rationale(raw: str) -> Dict[str, str]:
    """Split engine rationale into UI sections without duplicating branch lists."""
    raw = raw or ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    measurement = ""
    driver = ""
    disposition = ""
    arc = ""

    for line in lines:
        low = line.lower()
        if low.startswith("measured outcome"):
            continue
        if "superposed branches" in low or low.startswith("· branch") or low.startswith("branch p="):
            continue
        if "quantum policy measurement" in low:
            measurement = line
        elif "trait-channel attribution" in low:
            driver = line
        elif "dominant disposition" in low:
            disposition = line
        elif "climax phase" in low or "establishing setting" in low or "escalating toward" in low:
            arc = line

    summary_parts = [p for p in (measurement, driver, disposition, arc) if p]
    return {
        "measurement": measurement,
        "driver": driver,
        "disposition": disposition,
        "arc": arc,
        "summary": " ".join(summary_parts) if summary_parts else clean_rationale_text(raw)[:400],
    }


def _to_past_verb(verb: str) -> str:
    v = verb.lower()
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    if v.endswith("e"):
        return v + "d"
    if v.endswith("s") and not v.endswith("ss"):
        return v + "ed"
    if v.endswith(("sh", "ch", "x", "z")):
        return v + "ed"
    return v + "ed"


def _action_phrase_simple(name: str, action: str) -> str:
    act = action_to_readable(action).lower()
    parts = act.split()
    if len(parts) >= 2:
        past = _to_past_verb(parts[0])
        obj = " ".join(parts[1:])
        return f"{name} {past} the {obj}"
    if parts:
        return f"{name} {_to_past_verb(parts[0])}"
    return name


def build_quantum_trace(chosen: List[Tuple[str, str]], fallback: str = "") -> str:
    if not chosen:
        return clean_rationale_text(fallback)
    if len(chosen) == 1:
        n, a = chosen[0]
        return f"The scene turned on the moment when {_action_phrase_simple(n, a)}."
    parts = [_action_phrase_simple(n, a) for n, a in chosen]
    if len(parts) == 2:
        return f"The scene turned when {parts[0]}, and {parts[1]}."
    return (
        f"The scene turned when {parts[0]}, "
        f"while {', '.join(parts[1:-1])}, and {parts[-1]}."
    )


def _joint_summary(actions: List[Tuple[str, str]]) -> str:
    return "; ".join(f"{n}: {a}" for n, a in actions)


def explain_why_not_chosen(
    rej_actions: List[Tuple[str, str]],
    chosen_actions: List[Tuple[str, str]],
    template_reason: str,
    rej_probability: float,
    selected_posterior: float,
) -> str:
    """Full explanation grounded in VQC posteriors (no truncated LLM filler)."""
    alt = _joint_summary(rej_actions)
    picked = _joint_summary(chosen_actions)
    reason = clean_rationale_text(template_reason) or (
        "This branch carried lower narrative utility under the current quantum state."
    )
    lead = ""
    if rej_probability > selected_posterior + 0.005:
        lead = (
            "This branch ranked higher in the VQC posterior than the measured outcome, "
            "but the policy draws one joint action at random from the full distribution—"
            "so a lower-probability branch can still be selected. "
        )
    return (
        f"{lead}"
        f"Alternate ({rej_probability:.0%} posterior): {alt}. "
        f"Measured ({selected_posterior:.0%} posterior): {picked}. "
        f"{reason}"
    )


def _peak_joint_action(scene: dict, max_idx: int) -> Dict[str, str]:
    if int(scene.get("action_idx", -1)) == max_idx:
        return scene.get("action") or {}
    for rej in scene.get("rejections") or []:
        if int(rej.get("index", -1)) == max_idx:
            return rej.get("action") or {}
    return scene.get("action") or {}


def _llm_reason_usable(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    if text.count(":") >= 3:
        return False
    if text.endswith("...") or text.endswith(","):
        return False
    return True


def primary_driver_name(scene: dict, name_map: Dict[str, str]) -> str:
    attr = scene.get("attribution") or {}
    if not attr:
        return ""
    top_role = max(attr, key=attr.get)
    return (
        name_map.get(top_role)
        or name_map.get(top_role.capitalize())
        or name_map.get(top_role.lower())
        or ""
    )


def build_scene_decision_record(
    scene: dict, name_map: Dict[str, str], scene_total: int
) -> Dict[str, Any]:
    """Structured explainability payload for one scene."""
    n = scene["step"] + 1
    chosen = format_joint_action(scene.get("action") or {}, name_map)
    rejections = []

    probs = scene.get("probs") or []
    action_idx = int(scene.get("action_idx", 0))
    posterior = float(probs[action_idx]) if probs and 0 <= action_idx < len(probs) else 0.0
    max_posterior = float(max(probs)) if probs else 0.0
    max_idx = int(np.argmax(probs)) if probs else action_idx

    rejections_raw = (scene.get("rejections") or [])[:3]
    deep = scene.get("deep_rejections") or []
    for i, rej in enumerate(rejections_raw):
        alt_rows = format_joint_action(rej.get("action") or {}, name_map)
        template_reason = rej.get("reason", "")
        rej_prob = float(rej.get("prob", 0))
        why_not = explain_why_not_chosen(
            alt_rows, chosen, template_reason, rej_prob, posterior
        )
        if i < len(deep) and isinstance(deep[i], str) and _llm_reason_usable(deep[i]):
            extra = clean_rationale_text(deep[i])
            if extra and extra not in why_not:
                why_not = f"{why_not} {extra}"
        rejections.append(
            {
                "actions": alt_rows,
                "probability": rej_prob,
                "why_not": why_not,
            }
        )

    prose, trace_raw = parse_scene_story(scene.get("story", ""))
    trace = build_quantum_trace(chosen, trace_raw)
    driver = primary_driver_name(scene, name_map)
    vqc = distill_vqc_rationale(scene.get("rationale", ""))

    peak_action = format_joint_action(_peak_joint_action(scene, max_idx), name_map)
    peak_label = _joint_summary(peak_action) if peak_action else "—"
    sampled_note = ""
    if probs and posterior < max_posterior - 0.005:
        sampled_note = (
            f"The VQC distribution peaked at {max_posterior:.0%} ({peak_label}), "
            f"but this scene stochastically measured the selected branch at {posterior:.0%}."
        )

    clarity = 0.0
    if len(probs) >= 2:
        sorted_p = sorted(float(p) for p in probs)
        clarity = float(sorted_p[-1] - sorted_p[-2])

    sensitivity = scene.get("sensitivity") or []
    attr = scene.get("attribution") or {}
    attr_named = {name_map.get(k, k): float(v) for k, v in attr.items()}
    total_attr = sum(attr_named.values()) or 1.0
    attr_named = {k: v / total_attr for k, v in attr_named.items()}

    return {
        "index": n,
        "total": scene_total,
        "prose": prose,
        "trace": trace,
        "chosen": chosen,
        "rejections": rejections,
        "driver": driver,
        "vqc": vqc,
        "coherence": float(scene.get("coherence", 0)),
        "tension": float(scene["state"]["tension"]),
        "utility": float(scene.get("reward", 0)),
        "clarity": clarity,
        "phase": scene["state"].get("phase", ""),
        "entropy": float(scene.get("entropy", 0)),
        "risk": float(scene.get("risk", 0)),
        "posterior": posterior,
        "max_posterior": max_posterior,
        "peak_label": peak_label,
        "sampled_note": sampled_note,
        "action_idx": action_idx,
        "attribution": attr_named,
        "sensitivity": [float(x) for x in sensitivity[:4]],
    }
