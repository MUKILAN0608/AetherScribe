"""Research UI: manuscript, explainable decisions, dashboard."""

import html
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from ui.charts import (
    attribution_pie,
    bonds_chart,
    clarity_chart,
    control_heatmap,
    entropy_chart,
    has_chart_data,
    line_chart,
    narrative_shape,
    pressure_utility,
    reward_chart,
    tension_chart,
)
from ui.charts import chart_with_title
from ui.decisions import build_name_map, build_scene_decision_record
from ui.quantum_panel import render_quantum_explain_panel
from ui.manuscript import (
    compile_full_manuscript,
    compile_plain_text_export,
    episode_metrics,
    generate_manuscript_latex,
    generate_manuscript_markdown,
    parse_scene_story,
    relationship_note,
    trajectory_steps,
)


def render_kpi_strip(ep: dict, extended: bool = False):
    m = episode_metrics(ep)
    scenes = ep["scenes"]
    mean_entropy = float(np.mean([s.get("entropy", 0) for s in scenes])) if scenes else 0.0
    mean_risk = float(np.mean([s.get("risk", 0) for s in scenes])) if scenes else 0.0
    posts = []
    for s in scenes:
        probs = s.get("probs") or []
        idx = int(s.get("action_idx", 0))
        if probs and 0 <= idx < len(probs):
            posts.append(float(probs[idx]))
    mean_posterior = float(np.mean(posts)) if posts else 0.0

    if extended:
        st.markdown(
            f"""
<div class="kpi-grid-extended">
  <div class="kpi-card accent"><div class="kpi-value">{m['mean_coherence']:.2f}</div><div class="kpi-label">Coherence</div></div>
  <div class="kpi-card accent"><div class="kpi-value">{m['cumulative_utility']:.2f}</div><div class="kpi-label">Utility</div></div>
  <div class="kpi-card"><div class="kpi-value">{m['final_tension']:.2f}</div><div class="kpi-label">Tension</div></div>
  <div class="kpi-card quantum"><div class="kpi-value quantum">{mean_posterior:.0%}</div><div class="kpi-label">Mean posterior</div></div>
  <div class="kpi-card quantum"><div class="kpi-value quantum">{mean_entropy:.2f}</div><div class="kpi-label">Policy entropy</div></div>
  <div class="kpi-card quantum"><div class="kpi-value quantum">{mean_risk:.2f}</div><div class="kpi-label">Narrative risk</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-value">{m['mean_coherence']:.2f}</div><div class="kpi-label">Coherence</div></div>
  <div class="kpi-card"><div class="kpi-value">{m['cumulative_utility']:.2f}</div><div class="kpi-label">Utility</div></div>
  <div class="kpi-card"><div class="kpi-value">{m['final_tension']:.2f}</div><div class="kpi-label">Tension</div></div>
  <div class="kpi-card"><div class="kpi-value">{m['scene_count']}</div><div class="kpi-label">Scenes</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_entities(ep: dict):
    name_map = build_name_map(ep["characters"])
    rel = ep["scenes"][-1]["state"]["relationships"]
    cards = []
    for role in ("protagonist", "antagonist", "ally"):
        display = name_map.get(role, "—")
        note = relationship_note(role.capitalize(), rel)
        cards.append(
            f'<div class="entity-card">'
            f'<div class="entity-name">{html.escape(display)}</div>'
            f'<div class="entity-meta">{html.escape(note)}</div></div>'
        )
    st.markdown('<div class="entity-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_decision_card(rec: dict):
    chosen_html = "".join(
        f'<div class="chosen-action"><b>{html.escape(n)}</b> — {html.escape(a)}</div>'
        for n, a in rec["chosen"]
    )
    alt_html = ""
    for alt in rec["rejections"]:
        actions = "; ".join(
            f"<b>{html.escape(n)}</b> — {html.escape(a)}" for n, a in alt["actions"]
        )
        alt_html += (
            f'<div class="alt-action">'
            f'<span style="color:#6e7681;">{alt["probability"]:.0%} likelihood</span><br>'
            f"{actions}<br>"
            f'<span style="color:#8b949e;font-style:italic;">{html.escape(alt["why_not"])}</span>'
            f"</div>"
        )

    rationale = html.escape(rec["rationale"]) if rec["rationale"] else "—"
    driver = html.escape(rec["driver"]) if rec["driver"] else ""

    driver_line = (
        f'<p style="font-size:0.85rem;color:#8b949e;margin:0.5rem 0 0;">'
        f"Primary driver: <b>{driver}</b></p>"
        if driver
        else ""
    )

    st.markdown(
        f"""
<div class="decision-card">
  <h4>Scene {rec['index']} of {rec['total']}</h4>
  <div class="decision-section-title">Selected action</div>
  {chosen_html}
  <div class="decision-section-title">Why this action</div>
  <div class="insight-panel">{rationale}</div>
  {driver_line}
  <div class="decision-section-title">Alternatives not selected</div>
  {alt_html if alt_html else '<p style="color:#6e7681;font-size:0.85rem;">—</p>'}
  <div class="decision-metrics">
    <span>Coherence {rec['coherence']:.2f}</span>
    <span>Tension {rec['tension']:.2f}</span>
    <span>Utility {rec['utility']:.2f}</span>
    <span>Clarity {rec['clarity']:.2f}</span>
    <span>Phase {html.escape(rec['phase'])}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_quantum_explain_panel(rec: dict):
    chosen_html = "".join(
        f'<div class="chosen-action"><b>{html.escape(n)}</b> — {html.escape(a)}</div>'
        for n, a in rec["chosen"]
    )
    alt_html = ""
    for alt in rec["rejections"]:
        actions = "; ".join(
            f"<b>{html.escape(n)}</b> — {html.escape(a)}" for n, a in alt["actions"]
        )
        alt_html += (
            f'<div class="alt-action">'
            f'<span style="color:#58a6ff;">Posterior P={alt["probability"]:.0%}</span> — {actions}<br>'
            f'<span style="color:#8b949e;font-style:italic;">{html.escape(alt["why_not"])}</span>'
            f"</div>"
        )
    attr_parts = " · ".join(
        f"{html.escape(name)} {weight:.0%}"
        for name, weight in sorted(rec["attribution"].items(), key=lambda x: -x[1])
    )
    trace_block = ""
    if rec.get("trace"):
        trace_block = (
            f'<div class="decision-section-title">Quantum trace (narrative)</div>'
            f'<p style="color:#8b949e;font-size:0.9rem;">{html.escape(rec["trace"])}</p>'
        )
    st.markdown(
        f"""
<div class="decision-card" style="margin-top:1rem;">
  <h4 style="margin:0 0 0.75rem;color:#58a6ff;">Quantum decision — Scene {rec['index']}</h4>
  <div class="decision-metrics" style="margin-bottom:0.75rem;">
    <span>Posterior P={rec['posterior']:.0%}</span>
    <span>Entropy {rec['entropy']:.2f}</span>
    <span>Risk {rec['risk']:.2f}</span>
    <span>Margin ΔP {rec['clarity']:.2f}</span>
    <span>Coherence {rec['coherence']:.2f}</span>
  </div>
  <div class="decision-section-title">Measured joint action</div>
  {chosen_html}
  <div class="decision-section-title">VQC rationale</div>
  <div class="insight-panel">{html.escape(rec['rationale'] or '—')}</div>
  <div class="decision-section-title">Trait-channel attribution</div>
  <p style="font-size:0.85rem;color:#c9d1d9;">{attr_parts or '—'}</p>
  <div class="decision-section-title">Branches not measured</div>
  {alt_html if alt_html else '<p style="color:#6e7681;font-size:0.85rem;">—</p>'}
  {trace_block}
</div>
""",
        unsafe_allow_html=True,
    )


def _format_prose_lines(prose: str) -> str:
    """Display prose as multi-line; preserve LLM line breaks when present."""
    if not prose:
        return ""
    if "\n" in prose.strip():
        lines = [ln.strip() for ln in prose.strip().split("\n") if ln.strip()]
    else:
        lines = re.split(r"(?<=[.!?…])\s+", prose.strip())
        lines = [ln.strip() for ln in lines if ln.strip()]
    return "\n".join(lines[:10])


def _attribution_values(scene: dict, role_keys: list) -> list:
    attr = scene.get("attribution") or {}
    vals = []
    for role in role_keys:
        v = attr.get(role)
        if v is None:
            v = attr.get(role.lower())
        if v is None:
            v = attr.get(role.capitalize())
        vals.append(float(v or 0.0))
    return vals


def render_story_tab(ep: dict):
    """Professional manuscript with per-scene quantum explainability."""
    st.markdown(
        f"""
<div class="abstract-block">
  <span class="abstract-label">Premise</span>
  {html.escape(ep['premise'])}
</div>
""",
        unsafe_allow_html=True,
    )
    render_entities(ep)

    name_map = build_name_map(ep["characters"])
    total = len(ep["scenes"])
    recs = [build_scene_decision_record(s, name_map, total) for s in ep["scenes"]]

    tab_full, tab_scenes = st.tabs(["Full story", "Scenes & quantum trace"])

    with tab_full:
        full_text = compile_full_manuscript(ep)
        full_display = _format_prose_lines(full_text.replace("\n\n", "\n"))
        st.markdown(
            f'<div class="manuscript-hero manuscript-full">{html.escape(full_display)}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download full story",
            compile_plain_text_export(ep),
            file_name=f"story_{ep['id'][:8]}.txt",
            use_container_width=True,
        )

    with tab_scenes:
        for rec in recs:
            lines = _format_prose_lines(rec["prose"])
            st.markdown(
                f"""
<div class="scene-block">
  <div class="scene-index">Scene {rec['index']} of {rec['total']}</div>
  <p class="scene-prose-lines">{html.escape(lines)}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            with st.expander(
                f"Quantum decision · Scene {rec['index']}",
                expanded=(rec["index"] == 1),
            ):
                render_quantum_explain_panel(rec)


def _chart_in_panel(title: str, fig, values=None):
    if values is not None and not has_chart_data(values):
        st.markdown(
            f'<p class="chart-empty-title">{html.escape(title)}</p>',
            unsafe_allow_html=True,
        )
        st.caption("No data for this scene run yet.")
        return
    st.plotly_chart(
        chart_with_title(fig, title),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def render_dashboard_tab(ep: dict):
    m = episode_metrics(ep)
    name_map = build_name_map(ep["characters"])
    steps = trajectory_steps(ep)
    scenes = ep["scenes"]
    tension = [s["state"]["tension"] for s in scenes]
    utility = [s["reward"] for s in scenes]
    coherence = [s.get("coherence", 0) for s in scenes]
    entropy = [s.get("entropy", 0) for s in scenes]
    risk = [s.get("risk", 0) for s in scenes]
    posteriors = []
    for s in scenes:
        probs = s.get("probs") or []
        idx = int(s.get("action_idx", 0))
        posteriors.append(float(probs[idx]) if probs and 0 <= idx < len(probs) else 0.0)
    gaps = []
    for s in scenes:
        probs = s.get("probs") or []
        if len(probs) >= 2:
            sp = sorted(probs, reverse=True)
            gaps.append(sp[0] - sp[1])
        else:
            gaps.append(0.0)
    role_keys = ["Protagonist", "Antagonist", "Ally"]
    agent_names = [name_map.get(r.lower(), r) for r in role_keys]

    st.markdown(
        f"""
<div class="dash-hero">
  <h2>Quantum narrative dashboard</h2>
  <p>{html.escape(ep['genre'])} · {len(scenes)} scenes · cumulative utility {m['cumulative_utility']:.2f}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    render_kpi_strip(ep, extended=True)

    st.markdown('<p class="dash-section-title">Trajectory dynamics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _chart_in_panel("Dramatic tension", tension_chart(steps, tension), tension)
    with c2:
        _chart_in_panel("Narrative coherence", line_chart(steps, coherence, color="#3fb950"), coherence)
    with c3:
        _chart_in_panel("Scene utility", reward_chart(steps, utility), utility)
    with c4:
        _chart_in_panel("Decision margin ΔP", clarity_chart(steps, gaps), gaps)

    st.markdown('<p class="dash-section-title">Quantum policy metrics</p>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        _chart_in_panel(
            "VQC posterior (measured action)",
            line_chart(steps, posteriors, color="#58a6ff", y_range=[0, 1]),
            posteriors,
        )
    with q2:
        _chart_in_panel("Policy entropy", entropy_chart(steps, entropy), entropy)
    with q3:
        _chart_in_panel("Narrative risk", line_chart(steps, risk, color="#f85149"), risk)
    with q4:
        _chart_in_panel(
            "Pressure vs utility",
            pressure_utility(tension, utility, steps),
            utility,
        )

    st.markdown('<p class="dash-section-title">Character &amp; attribution</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        vals = _attribution_values(scenes[-1], role_keys)
        _chart_in_panel("Final trait attribution", attribution_pie(agent_names, vals), vals)
    with r2:
        trust = [s["state"]["relationships"]["protagonist_ally"]["trust"] for s in scenes]
        enmity = [s["state"]["relationships"]["protagonist_antagonist"]["enmity"] for s in scenes]
        _chart_in_panel(
            "Relationship bonds",
            bonds_chart(steps, trust, enmity, name_map.get("ally", ""), name_map.get("antagonist", "")),
            trust + enmity,
        )
    with r3:
        attr = [_attribution_values(s, role_keys) for s in scenes]
        flat = [v for row in attr for v in row]
        _chart_in_panel("Control heatmap", control_heatmap(steps, agent_names, attr), flat)

    if len(scenes) >= 2:
        shape = narrative_shape(steps, scenes)
        if shape is not None:
            st.markdown('<p class="dash-section-title">Narrative state space (PCA)</p>', unsafe_allow_html=True)
            _chart_in_panel("Trajectory embedding", shape)

    st.markdown('<p class="dash-section-title">Quantum decision log</p>', unsafe_allow_html=True)
    rows = []
    for scene in scenes:
        rec = build_scene_decision_record(scene, name_map, len(scenes))
        rows.append(
            {
                "Scene": rec["index"],
                "Posterior": f"{rec['posterior']:.0%}",
                "Entropy": round(rec["entropy"], 2),
                "Risk": round(rec["risk"], 2),
                "Coherence": round(rec["coherence"], 2),
                "ΔP": round(rec["clarity"], 2),
                "Measured action": "; ".join(f"{n}: {a}" for n, a in rec["chosen"]),
                "Driver": rec["driver"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Posterior": st.column_config.TextColumn(width="small"),
            "Measured action": st.column_config.TextColumn(width="large"),
        },
    )


def render_export_tab(ep: dict):
    st.markdown("#### Exports")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Markdown",
            generate_manuscript_markdown(ep),
            file_name=f"trajectory_{ep['id'][:8]}.md",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "LaTeX",
            generate_manuscript_latex(ep),
            file_name=f"trajectory_{ep['id'][:8]}.tex",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Plain text",
            compile_plain_text_export(ep),
            file_name=f"trajectory_{ep['id'][:8]}.txt",
            use_container_width=True,
        )
    with st.expander("Full data (JSON)"):
        st.download_button(
            "JSON",
            json.dumps(ep, indent=2, default=str),
            file_name=f"trajectory_{ep['id'][:8]}.json",
            use_container_width=True,
        )
