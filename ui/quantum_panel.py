"""Structured quantum decision panel for the manuscript UI."""

import html

import streamlit as st


def render_quantum_explain_panel(rec: dict):
    vqc = rec.get("vqc") or {}
    t = "div"

    metric_defs = [
        (f'{rec["posterior"]:.0%}', "Selected branch P"),
        (f'{rec.get("max_posterior", rec["posterior"]):.0%}', "Peak branch P"),
        (f'{rec["entropy"]:.2f}', "Entropy"),
        (f'{rec["risk"]:.2f}', "Risk"),
        (f'{rec["clarity"]:.2f}', "Margin ΔP"),
        (f'{rec["coherence"]:.2f}', "Coherence"),
    ]
    metric_parts = []
    for val, lbl in metric_defs:
        metric_parts.append(
            f'<{t} class="qx-metric"><span class="qx-metric-val">{val}</span>'
            f'<span class="qx-metric-lbl">{lbl}</span></{t}>'
        )
    metrics_html = f'<{t} class="qx-metrics">' + "".join(metric_parts) + f"</{t}>"

    note = rec.get("sampled_note") or ""
    note_html = (
        f'<p class="qx-note">{html.escape(note)}</p>' if note else ""
    )

    legend = (
        "Posteriors come from the variational quantum circuit (VQC). "
        "The measured row is the joint action actually used in the story; "
        "alternates are other branches in the same superposition. "
        "A higher alternate posterior is not an error—it means stochastic sampling, not argmax."
    )

    chosen_rows = "".join(
        f"<tr><td class='qx-name'>{html.escape(n)}</td>"
        f"<td class='qx-action'>{html.escape(a)}</td></tr>"
        for n, a in rec["chosen"]
    )

    alt_cards = []
    for i, alt in enumerate(rec["rejections"], start=1):
        action_lines = "<br>".join(
            f"• <b>{html.escape(n)}</b> — {html.escape(a)}" for n, a in alt["actions"]
        )
        alt_cards.append(
            f'<{t} class="qx-alt-card">'
            f'<{t} class="qx-alt-head">'
            f'<span class="qx-alt-num">Alternate path {i}</span>'
            f'<span class="qx-alt-prob">Posterior {alt["probability"]:.0%}</span>'
            f"</{t}>"
            f'<{t} class="qx-alt-actions">{action_lines}</{t}>'
            f'<{t} class="qx-alt-reason">'
            f'<span class="qx-alt-reason-lbl">Why this path was not chosen</span>'
            f"<p class='qx-alt-reason-text'>{html.escape(alt['why_not'])}</p>"
            f"</{t}></{t}>"
        )
    alt_html = (
        "".join(alt_cards)
        if alt_cards
        else "<p class='qx-p qx-empty'>No alternate branches above the 5% posterior threshold.</p>"
    )

    attr_bars = []
    for name, weight in sorted(rec["attribution"].items(), key=lambda x: -x[1]):
        pct = min(100, max(0, int(round(weight * 100))))
        attr_bars.append(
            f'<{t} class="qx-attr-row"><span class="qx-attr-name">{html.escape(name)}</span>'
            f'<{t} class="qx-attr-bar"><{t} class="qx-attr-fill" style="width:{pct}%"></{t}></{t}>'
            f'<span class="qx-attr-pct">{weight:.0%}</span></{t}>'
        )
    attr_html = "".join(attr_bars) if attr_bars else "<p class='qx-p qx-empty'>—</p>"

    vqc_parts = [
        vqc.get(k) for k in ("measurement", "driver", "disposition", "arc") if vqc.get(k)
    ]
    vqc_body = "".join(f"<p class='qx-p'>{html.escape(p)}</p>" for p in vqc_parts)
    if not vqc_body:
        vqc_body = f"<p class='qx-p'>{html.escape(vqc.get('summary', '—'))}</p>"

    html_block = (
        f'<{t} class="qx-card">'
        f'<{t} class="qx-header">Quantum decision · Scene {rec["index"]} of {rec["total"]}</{t}>'
        f'<p class="qx-legend">{html.escape(legend)}</p>'
        f"{metrics_html}"
        f"{note_html}"
        f'<{t} class="qx-section"><{t} class="qx-label">Chosen path (measured)</{t}>'
        f"<table class='qx-table'><tbody>{chosen_rows}</tbody></table></{t}>"
        f'<{t} class="qx-section"><{t} class="qx-label">Policy rationale</{t}>{vqc_body}</{t}>'
        f'<{t} class="qx-section"><{t} class="qx-label">Character influence</{t}>{attr_html}</{t}>'
        f'<{t} class="qx-section"><{t} class="qx-label">Other paths not chosen</{t}>{alt_html}</{t}>'
        f'<{t} class="qx-section"><{t} class="qx-label">Narrative trace</{t}>'
        f'<p class="qx-trace">{html.escape(rec.get("trace") or "—")}</p></{t}>'
        f"</{t}>"
    )
    st.markdown(html_block, unsafe_allow_html=True)
